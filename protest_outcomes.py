"""

protest_outcomes.py
-------------------

Author: Michael Dickens
Created: 2024-12-04

Code for my protest outcomes meta-analysis.

Study sources:

Madestam, A., Shoag, D., Veuger, S., & Yanagizawa-Drott, D. (2013). Do
Political Protests Matter? Evidence from the Tea Party Movement.
https://doi.org/10.1093/qje/qjt021

Klein Teeselink, B., & Melios, G. (2021). Weather to Protest: The Effect of
Black Lives Matter Protests on the 2020 Presidential Election.
https://dx.doi.org/10.2139/ssrn.3809877

Larreboure, M., & Gonzalez, F. (2021). The Impact of the Women’s March on the
U.S. House Election. https://mlarreboure.com/womenmarch.pdf

Wasow, O. (2020). Agenda Seeding: How 1960s Black Protests Moved Elites, Public Opinion and Voting. https://doi.org/10.1017/S000305542000009X

"""

from collections import namedtuple
from scipy import stats
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


class Outcome(namedtuple("Outcome", ["mean", "stderr", "n"])):
    @property
    def stdev(self):
        return self.stderr * np.sqrt(self.n)

    @property
    def variance(self):
        return self.stderr**2 * self.n

    @property
    def tstat(self):
        return self.mean / self.stderr

    @property
    def pval(self):
        return 2 * (1 - stats.t.cdf(self.tstat, self.n - 1))


def pooled_outcome(outcomes):
    """
    Calculate pooled outcome using a random-effects model.

    Borenstein et al. (2010). A basic introduction to fixed-effect and
    random-effects models for meta-analysis.
    https://de.meta-analysis.com/download/Intro_Models.pdf

    Parameters:
    -----------
    outcomes : array-like
        Outcome objects from each experiment

    Returns:
    --------
    Outcome
        Pooled outcome object with weighted mean, standard error, and
        total sample size
    """
    means = np.asarray([x.mean for x in outcomes])
    stderrs = np.asarray([x.stderr for x in outcomes])
    ns = np.asarray([x.n for x in outcomes])
    df = len(outcomes) - 1

    # Calculate weights as if using a fixed-effects model
    weights_fixed = np.asarray([1 / x.stderr**2 for x in outcomes])

    # Calculate fixed-effect weighted mean
    mean_fixed = np.sum(weights_fixed * means) / np.sum(weights_fixed)

    # Calculate between-study variance (tau^2)
    Q = np.sum(weights_fixed * (means - mean_fixed) ** 2)
    C = np.sum(weights_fixed) - np.sum(weights_fixed**2) / np.sum(weights_fixed)
    tau_squared = (Q - df) / C
    tau = np.sqrt(tau_squared)

    # Recalculate weights incorporating between-study variance
    weights = [1 / (x.stderr**2 + tau_squared) for x in outcomes]

    # Calculate pooled mean and standard error
    mean_random = np.sum(weights * means) / np.sum(weights)
    stderr_random = 1 / np.sqrt(np.sum(weights))

    print(f"Mean: {mean_random:.2f}; Between-study stderr (tau): {tau:.2f}; within-study stderr: {1 / np.sqrt(np.sum(weights_fixed)):.2f}")
    print(f"Implied probability of negative effect: {1 - stats.norm.cdf(mean_random / tau)}")

    return Outcome(mean_random, stderr_random, sum(ns))


def vote_share_per_protester():
    tea_party = Outcome(mean=18.81, stderr=7.85, n=2758)
    blm = Outcome(mean=3.3, stderr=0.6, n=3053)
    womens_march = Outcome(mean=9.62, stderr=3.44, n=2936)
    outcomes = [tea_party, blm, womens_march]

    result = pooled_outcome(outcomes)
    print("Vote Share per Protester")
    print(f"\tMean: {result.mean:.2f}")
    print(f"\tStandard Error: {result.stderr:.2f}")
    print(f"\tSample Size: {result.n}")
    print(f"\tp-value: {result.pval:.3f}")
    print()


def protest_effect():
    """
    Some studies reported multiple effects for a protest so I used vote
    share if reported, and favorability if not.
    """
    tea_party = Outcome(mean=1.55, stderr=0.69, n=2758)
    blm = Outcome(mean=2.7, stderr=1.2, n=3053)
    earth_day = Outcome(5.72, 3.2, n=5223)  # n in Table 1
    civil_rights_violent = Outcome(mean=-5.54, stderr=2.48, n=2207)  # n in Table 3

    outcomes = [tea_party, blm, earth_day]

    result = pooled_outcome(outcomes)
    print("Protest Effect")
    print(f"\tMean: {result.mean:.2f}")
    print(f"\tStandard Error: {result.stderr:.2f}")
    print(f"\tSample Size: {result.n}")
    print(f"\tp-value: {result.pval:.3f}")
    print()


vote_share_per_protester()
protest_effect()
