"""

protest_outcomes.py
-------------------

Author: Michael Dickens
Created: 2024-12-04

Code for my protest outcomes meta-analysis.

Studies included:

Madestam, A., Shoag, D., Veuger, S., & Yanagizawa-Drott, D. (2013). Do
Political Protests Matter? Evidence from the Tea Party Movement.
https://doi.org/10.1093/qje/qjt021

Klein Teeselink, B., & Melios, G. (2021). Weather to Protest: The Effect of
Black Lives Matter Protests on the 2020 Presidential Election.
https://dx.doi.org/10.2139/ssrn.3809877

Larreboure, M., & Gonzalez, F. (2021). The Impact of the Women’s March on the
U.S. House Election. https://mlarreboure.com/womenmarch.pdf

Wasow, O. (2020). Agenda Seeding: How 1960s Black Protests Moved Elites,
Public Opinion and Voting. https://doi.org/10.1017/S000305542000009X

"""

from collections import namedtuple
from scipy import stats
import math
import numpy as np


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
        return 2 * (1 - stats.t.cdf(abs(self.tstat), self.n - 1))

    @property
    def likelihood_ratio(self):
        """
        The probability density of getting this mean given that it's the
        true mean, divided by the probability density of getting this mean
        given that the true mean is 0.
        """
        return stats.norm.pdf(self.mean, self.mean, self.stderr) / stats.norm.pdf(
            self.mean, 0, self.stderr
        )


def round_pval(pval):
    return math.ceil(pval * 1000) / 1000


def pooled_outcome(outcomes):
    """
    Calculate pooled outcome using a random-effects model.

    When combining outcomes from multiple studies, you can use either a
    fixed-effects model or a random-effects model. A fixed-effects model
    assumes that all protests are equally effective, and it estimates the
    expected effectiveness of protests given the observed data. A
    random-effects model assumes that the effectiveness of protests can vary,
    and it estimates the effectiveness of the *average* protest.

    For an explanation of the math, see:

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

    # Calculate weights as if using a fixed-effects model.
    weights_fixed = np.asarray([1 / x.stderr**2 for x in outcomes])

    # Calculate fixed-effect weighted mean.
    mean_fixed = np.sum(weights_fixed * means) / np.sum(weights_fixed)

    # Calculate between-study variance (tau^2).
    Q = np.sum(weights_fixed * (means - mean_fixed) ** 2)
    C = np.sum(weights_fixed) - np.sum(weights_fixed**2) / np.sum(weights_fixed)
    tau_squared = (Q - df) / C

    if tau_squared < 0:
        # The effect sizes look similar enough that we should use a
        # fixed-effects model, which is equivalent to a random-effects model
        # with tau = 0.
        tau_squared = 0

    tau = np.sqrt(tau_squared)

    # Recalculate weights incorporating between-study variance.
    weights = [1 / (x.stderr**2 + tau_squared) for x in outcomes]

    # Calculate pooled mean and standard error.
    mean_random = np.sum(weights * means) / np.sum(weights)
    stderr_random = 1 / np.sqrt(np.sum(weights))

    return Outcome(mean_random, stderr_random, sum(ns)), tau


def print_stats_with_fraud(outcomes, p_fraud):
    probabilistic_pvals = []

    # Iterate through every possible combination with respect to which studies
    # are fraudulent.
    for flags in range(int(2 ** len(outcomes))):
        # Create a list of flags representing whether each study is fraudulent
        outcome_flags = [(flags & (1 << i)) != 0 for i in range(len(outcomes))]
        probability = np.prod(
            [p_fraud if flag else 1 - p_fraud for flag in outcome_flags]
        )

        # Exclude fraudulent studies.
        adj_outcomes = []
        for outcome, flag in zip(outcomes, outcome_flags):
            if not flag:
                adj_outcomes.append(outcome)

        # Calculate the pooled outcome. If there are no non-fraudulent studies,
        # the p-value is 0.5 on the assumption that fraudulent studies can get
        # any result they want.
        if len(adj_outcomes) == 0:
            pval = 0.5
        elif len(adj_outcomes) == 1:
            result = adj_outcomes[0]
            pval = result.pval
        else:
            result, _ = pooled_outcome(adj_outcomes)
            pval = result.pval

        probabilistic_pvals.append(probability * pval)

    # Calculate probability-weighted p-value. Note that there is not a simple
    # way to combine the distributions, but we can calculate the overall
    # probability of getting a result at least this extreme given the null
    # hypothesis.
    pval = np.sum(probabilistic_pvals)
    print(f"\tp-value: {pval:.4g}")


def print_stats(
    outcomes, add_null_clones=False, additional_unpublished_studies=0, p_fraud=0
):
    """
    Print the results of a meta-analysis.

    Parameters:
    -----------
    outcomes : array-like
        Outcome objects from each experiment
    add_null_clones : bool, optional
        Robustness check for publication bias. Create a "null clone" of each
        outcome that's identical except with mean=0.
    additional_unpublished_studies : int, optional
        Use this if you want to add more unpublished studies.
    p_fraud : float, optional
        Robustness check for fraud. Assume every study independently has this
        probability of being fraudulent. For a fraudulent study, the mean is
        set to zero and the standard error and sample size are unchanged.
    """
    if add_null_clones:
        dummy_clones = [
            Outcome(mean=0, stderr=outcome.stderr, n=outcome.n) for outcome in outcomes
        ]
        outcomes = outcomes + dummy_clones

    avg_stderr = np.mean([x.stderr for x in outcomes])
    avg_n = np.mean([x.n for x in outcomes])
    dummy_null_outcome = Outcome(mean=0, stderr=avg_stderr, n=avg_n)
    outcomes = outcomes + [
        dummy_null_outcome for _ in range(additional_unpublished_studies)
    ]

    if p_fraud > 0:
        return print_stats_with_fraud(outcomes, p_fraud)

    result, tau = pooled_outcome(outcomes)

    print(f"\tMean: {result.mean:.2f}")
    print(f"\tStandard Error: {result.stderr:.2f}")
    print(f"\tp-value: {result.pval:.4g}")
    print(f"\tlikelihood ratio: {result.likelihood_ratio:.4g}")
    p_backfire = 1 - stats.norm.cdf(result.mean / tau) if tau > 1e-10 else 0
    print(f"\timplied P(protest backfires): {p_backfire:.4g}")

    print()


def check_difference(base_outcome, alt_outcome):
    """
    Check if the difference between two outcomes is significant.

    Parameters:
    -----------
    base_outcome : Outcome
        The base outcome to compare against.
    alt_outcome : Outcome
        The alternative outcome to compare with.

    Returns:
    --------
    Outcome
        The difference between the two outcomes.
    """
    return Outcome(
        mean=alt_outcome.mean - base_outcome.mean,
        stderr=math.sqrt(base_outcome.stderr**2 + alt_outcome.stderr**2),
        n=base_outcome.n + alt_outcome.n,
    )


def vote_share_per_protester(add_null_clones=False, p_fraud=0):
    """
    Expected change in vote share per protester. Vote share is measured as

        number of voters / turnout

    For example, if an additional one protester changes 10 votes and turnout
    is 50%, then vote share per protester is 10 / 0.5 = 20.
    """
    tea_party = Outcome(mean=18.81, stderr=7.85, n=2758)
    blm = Outcome(mean=3.3, stderr=0.6, n=3053)
    womens_march = Outcome(mean=9.62, stderr=3.44, n=2936)

    outcomes = [tea_party, blm, womens_march]
    print("Vote Share per Protester")
    print_stats(outcomes, add_null_clones=add_null_clones, p_fraud=p_fraud)


def protest_effect(add_null_clones=False, p_fraud=0):
    """
    Total effect of a protest on societal-level outcomes.

    Some studies reported multiple effects for a protest. I used vote share as
    the outcome measure if it was reported. For the Earth Day study, which did
    not report vote share, I used favorability toward government spending on
    environmental protection.
    """
    # Convert a 5-point scale to a percentage agreement scale. This is kind of
    # arbitrary but my logic is:
    #
    # On a 5-point scale, the average "agree" score is 4.5 and "disagree" is
    # 1.5, which is a 3-point swing. Therefore a 3-point change on a 5-point
    # scale is equivalent to a 100% change on a binary scale. Divide by 3, then
    # multiply by 100 to get a percentage.
    special_favors_scalar = 100 / 3

    tea_party_votes = Outcome(mean=1.55, stderr=0.69, n=2758)
    tea_party_favorability = Outcome(mean=5.7, stderr=2.5, n=2758)
    blm_votes = Outcome(mean=2.7, stderr=1.2, n=3053)
    blm_special_favors = Outcome(
        mean=0.242 * special_favors_scalar, stderr=0.360 * special_favors_scalar, n=2556
    )
    womens_march_votes = Outcome(mean=4.95, stderr=1.28, n=2940)
    earth_day_favorability_1 = Outcome(5.72, 3.2, n=5223)  # n in Table 1
    civil_rights_violent = Outcome(mean=-5.54, stderr=2.48, n=2207)  # n in Table 3

    vote_outcomes = [tea_party_votes, blm_votes, womens_march_votes, earth_day_favorability_1]
    vote_outcomes_rainfall_only = [tea_party_votes, blm_votes, earth_day_favorability_1]
    single_hypothesis_outcomes = [
        tea_party_votes,
        blm_votes,
        womens_march_votes,
        earth_day_favorability_1,
        civil_rights_violent,
    ]
    favorability_outcomes = [
        tea_party_favorability,
        blm_special_favors,
        earth_day_favorability_1,
    ]

    print("Protest Effects – Primary Outcomes")
    print_stats(vote_outcomes, add_null_clones=add_null_clones, p_fraud=p_fraud)
    print("Protest Effects – Single Hypothesis")
    print_stats(single_hypothesis_outcomes, add_null_clones=add_null_clones, p_fraud=p_fraud)
    print("Protest Effects – Favorability")
    print_stats(favorability_outcomes, add_null_clones=add_null_clones, p_fraud=p_fraud)

    diff = check_difference(pooled_outcome(vote_outcomes)[0], civil_rights_violent)
    print(f"Nonviolent vs. Violent Difference: p-value {diff.pval:.4g}, likelihood ratio {diff.likelihood_ratio:.4g}")


def get_digit_counts(digits):
    """
    Count the number of occurrences of each digit in a list.

    Parameters:
    -----------
    digits : list of int
        List of digits (0-9) to count.

    Returns:
    --------
    np.ndarray
        Array where index i gives the number of occurrences of digit i.
    """
    digit_counts = [0 for _ in range(10)]
    for digit in digits:
        digit_counts[digit] += 1
    return np.array(digit_counts)


def digit_fraud_check(first_digits, last_digits):
    # test that the first digits follow Benford's law
    benford_frequencies = np.array(
        [0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046]
    )
    benford_counts = benford_frequencies * len(first_digits)
    first_digit_counts = get_digit_counts(first_digits)[1:]
    res = stats.chisquare(first_digit_counts, benford_counts)
    pval = round_pval(res.pvalue)
    print(f"\tFirst-digit Benford's Law p-value: {pval:.3f}")

    # test that last digits follow a uniform distribution
    last_digit_counts = get_digit_counts(last_digits)
    res = stats.chisquare(last_digit_counts)
    pval = round_pval(res.pvalue)
    print(f"\tLast-digit uniformity p-value:     {pval:.3f}")

    # As a power check, test what p-values we get if we reverse the first and
    # last digits. We want to see that the first digits do *not* follow a
    # uniform distribution and the last digits do *not* follow Benford's law.
    # If we get high p-values, that means the fraud check is underpowered.

    # remove all zeros
    filtered_last_digits = [x for x in last_digits if x != 0]
    reverse_benford_counts = benford_frequencies * len(filtered_last_digits)
    reverse_last_digit_counts = get_digit_counts(filtered_last_digits)[1:]
    last_pval = round_pval(
        stats.chisquare(reverse_last_digit_counts, reverse_benford_counts).pvalue
    )
    first_pval = round_pval(
        stats.chisquare(
            first_digit_counts, [len(first_digits) / 9 for _ in range(9)]
        ).pvalue
    )

    print(f"\tPower check p-values:             {first_pval:.3f}, {last_pval:.3f}")


def fraud_checks():
    # Using numbers from Table IV, V, VI, VII; all columns; Rainy Protest, % of
    # Pop. Protesting Scaling; mean and standard error. First digits Table V
    # only include columns (1, 2, 7, 8) due to correlations between survey
    # answers. Not including Table III because I doubt you'd fabricate that
    # table even if you were doing fraud. Last digits additionally include R^2
    # for Tables V and VII, but not IV and VI due to insufficient significant
    # figures (= last digit is less random).
    tea_party_first_digits = [
        int(x) for x in ("739362723152521362131413141617161921413131214693")
    ]
    tea_party_last_digits = [
        int(x)
        for x in (
            "7012574723144508757661805664377440914534591528277"
            "168111059944492134425201602"
        )
    ]
    print("Tea Party:")
    digit_fraud_check(tea_party_first_digits, tea_party_last_digits)

    # Using numbers from Tables 2, 3, 4; all panels; line items
    # Attendees/Population, Rain prob., lambda, rho; mean and standard error;
    # Model 1 only because model results are correlated which could create a
    # false positive.
    blm_first_digits = [
        int(x)
        for x in (
            "362124514615413145123551361421314"
            "612345938135151461241717121824252"
            "23451852332727"
        )
    ]
    blm_last_digits = [
        int(x)
        for x in (
            "367224169544335656078816348341174"
            "415826853445139057678700767513320"
            "471139986458"
        )
    ]
    print("\nBLM:")
    digit_fraud_check(blm_first_digits, blm_last_digits)

    # Using numbers from Table 2, 3, 4, 5; Panel A, B, C, Rainy protest
    # indicator, Rainfall, LASSO-chosen weather variable, Avg. dependent
    # variable (when distinct); mean and standard error. Last digits
    # additionally include R^2.
    womens_march_first_digits = [
        int(x)
        for x in (
            "1312443151415182931515951316245512321271157524137"
            "21951755257641286111724214318413542242"
        )
    ]
    womens_march_last_digits = [
        int(x)
        for x in (
            "8048373912089525800172408068195060027574141808976"
            "1984724661710603343914312846864339435626455"
        )
    ]
    print("\nWomen's March:")
    digit_fraud_check(womens_march_first_digits, womens_march_last_digits)


vote_share_per_protester(add_null_clones=True)
protest_effect(add_null_clones=True)
# fraud_checks()
