"""

intervention_power_laws.py
--------------------------

Author: Michael Dickens <michael@mdickens.me>
Created: 2024-11-15

Work with Disease Control Priorities 3 data.

Source: https://www.dcp-3.org/chapter/2561/cost-effectiveness-analysis
$/DALY data: https://www.dcp-3.org/sites/default/files/chapters/Annex%207A.%20Details%20of%20Interventions%20in%20Figs.pdf

The file 'DCP3 cost per DALY.txt' was created by extracting just the $/DALY
numbers from the table in Annex 07A of the DCP3 report.

"""

import csv
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats


def pareto_goodness_of_fit(samples):
    """Perform a goodness-of-fit test using the method from
    Suarez-Espinoza et al. (2018),
    "A goodness-of-fit test for the Pareto distribution."
    http://soche.cl/chjs/volumes/09/01/Suarez-Espinosa_etal(2018).pdf

    Return the Kendall correlation coefficient between the samples and the
    estimators. A smaller correlation indicates a worse fit. There is no known
    analytic solution for the p-value given the correlation, but roughly
    speaking, a correlation of < 0.95 is potentially a bad fit, and < 0.8 is
    almost certainly a bad fit. Larger samples should find higher correlations.
    See Suarez-Espinoza et al. (2018) for a table of empirically-determined
    p-values.

    This test can detect thin-tailed distributions but it's still bad at
    distinguishing between Pareto and lognormal distributions (both give high
    correlations).
    """
    samples = sorted(samples)
    estimators = []

    # This is slow. I tried writing a vectorized version but it was hard to
    # read and barely faster.
    for i in range(int(0.9 * len(samples))):
        estimator = sum([
            (samples[m] - samples[i]) / (len(samples) - (i + 1))
            for m in range(i + 1, len(samples))
        ])
        estimators.append(estimator)

    R = stats.kendalltau(samples[:len(estimators)], estimators)
    return R.statistic


def alpha_to_rra(alpha):
    """
    Convert a Pareto distribution alpha parameter to a relative risk
    aversion coefficient.
    """
    return 1/alpha


def fit_power_law(source_name, entries):
    """
    Fit a list of DALYs/$ data to a power law (Pareto distribution).
    """
    entries = np.array(entries)
    entry_means = stats.gmean(entries, axis=1)
    entry_means.sort()

    # fit entry_means to a Pareto distribution
    fitparams = stats.pareto.fit(entry_means)
    alpha, loc, scale = fitparams
    lognorm_fitparams = stats.lognorm.fit(entry_means)

    # test goodness of fit
    ks = stats.kstest(entry_means, 'pareto', args=fitparams)
    lognorm_ks = stats.kstest(entry_means, 'lognorm', args=lognorm_fitparams)

    print(f"DALYs/$\n\talpha: {alpha:.02f}\n\tRRA: {alpha_to_rra(alpha):.02f}\n\tloc: {loc:.2g}, scale: {scale:.2g}")

    # pvalue tests the null hypothesis that the data follows a Pareto
    # distribution. A low p-value is evidence that the data is not
    # Pareto-distributed. Qualitative descriptions are roughly based on how
    # much Bayesian evidence a particular p-value provides.
    pval_description = "looks"
    if ks.pvalue < 0.25:
        pval_description = "perhaps"
    elif ks.pvalue < 0.05:
        pval_description = "probably not"
    elif ks.pvalue < 0.01:
        pval_description = "almost certainly not"

    # Test how much evidence there is that the distribution is Pareto vs. lognormal
    bayes_factor = ks.pvalue / lognorm_ks.pvalue
    bayes_description = "much more"
    if bayes_factor < 3:
        bayes_description = "more"
    if bayes_factor < 1:
        bayes_description = "but less"
    if bayes_factor < 1/3:
        bayes_description = "but much less"

    print(f"Goodness of fit:\n\tp = {ks.pvalue:.1g} ({pval_description} Pareto-distributed, {bayes_description} Pareto than lognormal)")

    # show scatterplot and fitted curve
    cdf = np.linspace(0, 1, len(entry_means))
    plt.title("DALYs/$ Cumulative Distribution Function")
    plt.xscale("log")
    plt.xlabel("DALYs/$ (log scale)")
    plt.ylabel("Cumulative probability")
    plt.scatter(entry_means, cdf)
    plt.plot(entry_means, stats.pareto.cdf(entry_means, *fitparams))
    plt.plot(entry_means, stats.lognorm.cdf(entry_means, *lognorm_fitparams), color='green')
    plt.gcf().set_size_inches(10, 8)
    plt.savefig(f"data/{source_name}-curve-fit.png", dpi=150)
    plt.show()


def dcp3_power_law():
    with open("data/DCP3 cost per DALY.txt") as f:
        text = f.read()
        text_entries = text.split(',')
        entries = []
        for entry in text_entries:
            if '–' in entry:
                entry = [float(x) for x in entry.split('–')]
            else:
                entry = [float(entry), float(entry)]

            # flip from $/DALY to DALYs/$
            entry = 1 / np.array(entry)
            entries.append(entry)

    fit_power_law("DCP3", entries)


def aim_power_law():
    with open("data/Ambitious-Impact-CEEs.tsv") as f:
        reader = csv.DictReader(f, delimiter="\t")
        entries = []
        for row in reader:
            if row["Cost-effectiveness (DALY/$)"] is None:
                continue
            entry = float(row["Cost-effectiveness (DALY/$)"])
            entries.append([entry, entry])

    # The AIM cost-effectiveness estimates (which are sorted by DALYs/$
    # descending) include one (relatively) super-ineffective outlier that's
    # ~20x worse than the second-worst.
    entries = entries[:-1]

    fit_power_law("AIM", entries)


def cost_effectiveness_with_error(true_alpha, estimate_error=0.2, num_samples=10000, show_output=True, show_plot=True):
    """
    Simulate the effect of estimation error on cost-effectiveness analysis.
    Assume true DALYs/$ are pulled from a Pareto distribution, and then some
    random multiplicative error is applied.

    Error causes the estimated fit to systematically over-estimate alpha.

    Note: The goodness-of-fit test strongly rejects that the result follows a
    Pareto distribution, but the estimates in DCP3 do fit a Pareto dist, which
    means this model of how errors occur must be wrong. The error gets worse at
    higher true alpha and higher estimation error. (Doesn't fit lognorm either)
    """
    true_DALYs_per_dollar = stats.lomax.rvs(true_alpha, loc=0, scale=1, size=num_samples)
    true_DALYs_per_dollar.sort()
    estimation_errors = stats.norm.rvs(loc=0, scale=estimate_error, size=num_samples)
    estimated_DALYs_per_dollar = true_DALYs_per_dollar * np.exp(estimation_errors)
    estimated_DALYs_per_dollar.sort()

    # fit estimated DALYs/$ to a Pareto distribution
    alpha, loc, scale = stats.lomax.fit(estimated_DALYs_per_dollar)
    ks = stats.kstest(estimated_DALYs_per_dollar, 'lomax', args=(alpha, loc, scale))
    if show_output:
        print(f"true alpha {true_alpha:.2f}  -->  {alpha:.02f} estimated alpha (goodness-of-fit: p = {ks.pvalue:1.1g})")
        # print(f"\tloc = {loc}, scale = {scale}")

    # show scatterplot and fitted curve
    if show_plot:
        cdf = np.linspace(0, 1, len(estimated_DALYs_per_dollar))
        plt.title("Estimated DALYs/$ Cumulative Distribution Function")
        plt.xscale("log")
        plt.xlabel("DALYs/$ (log scale)")
        plt.ylabel("Cumulative probability")
        plt.scatter(estimated_DALYs_per_dollar, cdf, s=1)
        plt.plot(estimated_DALYs_per_dollar, stats.lomax.cdf(estimated_DALYs_per_dollar, alpha, loc, scale), color='orange')
        plt.show()

    return alpha


def sim_goodness_of_fit():
    for true_alpha in np.linspace(0.8, 1.8, 6):
        cost_effectiveness_with_error(true_alpha, 1, num_samples=10000, show_plot=False)


def sim_goodness_of_fit_given_error():
    # for error in np.linspace(0.5, 1, 11):
    for error in np.linspace(0.8, 1, 5):
        print(f"At error {error:.2f}:\n\t", end='')
        cost_effectiveness_with_error(1.2, error, num_samples=5000, show_plot=False)


def sim_bias():
    """
    Simulate what happens on average when we get 93 cost-effectiveness
    samples (which is the size of the DCP3 data set) with some noise in the
    cost-effectiveness estimate. How well do the estimates fit a Pareto
    distribution, and is the alpha estimate biased?
    """
    true_alpha = 1.2
    error = 1
    est_alphas = [cost_effectiveness_with_error(true_alpha, error, num_samples=93, show_output=False, show_plot=False) for _ in range(1000)]

    print(f"at error {error}, mean estimate: {np.mean(est_alphas):.02f} (true alpha = {true_alpha}). stderr: {np.std(est_alphas) / np.sqrt(len(est_alphas)):.02f}")
    print(f"p-value of bias: {stats.ttest_1samp(est_alphas, true_alpha).pvalue:.02f} (t = {stats.ttest_1samp(est_alphas, true_alpha).statistic:.01f})")


# dcp3_power_law()
aim_power_law()
# sim_goodness_of_fit()
# sim_goodness_of_fit_given_error()
# sim_bias()
