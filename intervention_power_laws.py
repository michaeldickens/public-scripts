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
    Fit the DCP3 DALYs/$ data to a Pareto distribution.
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
    alpha, loc, scale = stats.pareto.fit(entry_means)

    # test goodness of fit
    ks = stats.kstest(entry_means, 'pareto', args=(alpha, loc, scale))

    print(f"DCP3 DALYs/$\n\talpha: {alpha:.02f}\n\tRRA: {alpha_to_rra(alpha):.02f}\n\tloc: {loc}, scale: {scale}")

    # pvalue tests the null hypothesis that the data follows a Pareto distribution.
    # A low p-value is evidence that the data is not Pareto-distributed.
    description = "probably"
    if ks.pvalue < 0.25:
        description = "perhaps not"
    elif ks.pvalue < 0.05:
        description = "probably not"
    elif ks.pvalue < 0.01:
        description = "almost certainly not"
    print(f"Goodness of fit:\n\tp = {ks.pvalue} ({description} Pareto-distributed)")

    # show scatterplot and fitted curve
    empirical_cdf = np.linspace(0, 1, len(entry_means))
    plt.title("DCP3 DALYs/$ Cumulative Distribution Function")
    plt.xscale("log")
    plt.xlabel("DALYs/$ (log scale)")
    plt.ylabel("Cumulative probability")
    plt.scatter(entry_means, empirical_cdf)
    plt.plot(entry_means, stats.pareto.cdf(entry_means, alpha, loc, scale))
    plt.gcf().set_size_inches(10, 8)
    plt.savefig("data/dcp3-curve-fit.png", dpi=150)
    plt.show()

def cost_effectiveness_with_error(true_alpha):
    """
    Simulate the effect of estimation error on cost-effectiveness analysis.
    Assume true DALYs/$ are pulled from a Pareto distribution, and then some
    random multiplicative error is applied.

    Error causes the estimated fit to systematically over-estimate alpha.

    Note: The goodness-of-fit test strongly rejects that the result follows a
    Pareto distribution, but the estimates in DCP3 do fit a Pareto dist, which
    means this model of how errors occur must be wrong. The error gets worse at
    higher true alpha and higher estimation error.
    """
    true_DALYs_per_dollar = stats.pareto.rvs(true_alpha, loc=0, scale=1, size=100000)
    estimation_errors = stats.norm.rvs(loc=0, scale=0.2, size=100000)
    estimated_DALYs_per_dollar = true_DALYs_per_dollar * np.exp(estimation_errors)

    # fit estimated DALYs/$ to a Pareto distribution
    alpha, loc, scale = stats.pareto.fit(estimated_DALYs_per_dollar)
    # ks = stats.kstest(estimated_DALYs_per_dollar, 'pareto', args=(alpha, loc, scale))
    print(f"true alpha {true_alpha:.02f} --> estimated alpha {alpha:.02f} (difference {alpha - true_alpha:.02f})")
    # print(f"Estimated DALYs/$\n\talpha: {alpha:.02f}\n\tRRA: {alpha_to_rra(alpha):.02f}\n\tloc: {loc}, scale: {scale}\n\tGoodness of fit: p = {ks.pvalue}")


dcp3_power_law()
