"""

dcp3.py
-------

Author: Michael Dickens <michael@mdickens.me>
Created: 2024-11-15

Work with Disease Control Priorities 3 data.

Source: https://www.dcp-3.org/chapter/2561/cost-effectiveness-analysis
$/DALY data: https://www.dcp-3.org/sites/default/files/chapters/Annex%207A.%20Details%20of%20Interventions%20in%20Figs.pdf

The file 'DCP3 cost per DALY.txt' was created by extracting just the $/DALY
numbers from the table in Annex 07A of the DCP3 report.

"""

from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

def alpha_to_rra(alpha):
    """
    Convert a Pareto distribution alpha parameter to a relative risk
    aversion coefficient.
    """
    return 1/alpha


def dcp3_power_law():
    """
    Fit the DCP3 DALYs/$ data to a power law (Pareto distribution).
    """
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

    entries = np.array(entries)
    entry_means = stats.gmean(entries, axis=1)
    entry_means.sort()

    # fit entry_means to a Pareto distribution
    fitparams = stats.pareto.fit(entry_means)
    alpha, loc, scale = fitparams

    # test goodness of fit
    ks = stats.kstest(entry_means, 'pareto', args=fitparams)
    lognorm_ks = stats.kstest(entry_means, 'lognorm', args=stats.lognorm.fit(entry_means))

    print(f"DCP3 DALYs/$\n\talpha: {alpha:.02f}\n\tRRA: {alpha_to_rra(alpha):.02f}\n\tloc: {loc:.2g}, scale: {scale:.2g}")

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
    plt.title("DCP3 DALYs/$ Cumulative Distribution Function")
    plt.xscale("log")
    plt.xlabel("DALYs/$ (log scale)")
    plt.ylabel("Cumulative probability")
    plt.scatter(entry_means, cdf)
    plt.plot(entry_means, stats.pareto.cdf(entry_means, *fitparams))
    plt.gcf().set_size_inches(10, 8)
    plt.savefig("data/dcp3-curve-fit.png", dpi=150)
    plt.show()

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
        print(f"true alpha {true_alpha:.2f} --> estimated alpha {alpha:.02f} (difference {alpha - true_alpha:.02f})")
        # print(f"\tloc = {loc}, scale = {scale}")
        # print(f"\tgoodness-of-fit: p = {ks.pvalue}")

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


def run_dcp3_sims():
    """
    Simulate what happens on average when we get 93 cost-effectiveness
    samples (which is the size of the DCP3 data set) with some noise in the
    cost-effectiveness estimate. How well do the estimates fit a Pareto
    distribution, and is the alpha estimate biased?
    """
    true_alpha = 1.1
    est_alphas = [cost_effectiveness_with_error(true_alpha, 1, num_samples=93, show_output=False, show_plot=False) for _ in range(10000)]

    print(f"mean estimate: {np.mean(est_alphas):.02f} (true alpha = {true_alpha}). stdev: {np.std(est_alphas):.02f}")
    print(f"p-value of bias: {stats.ttest_1samp(est_alphas, true_alpha).pvalue:.02f} (t = {stats.ttest_1samp(est_alphas, true_alpha).statistic:.01f})")
