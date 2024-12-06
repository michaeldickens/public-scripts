"""

rct_observational.py
--------------------

Author: Michael Dickens <michael@mdickens.me>
Created: 2024-11-23

A method to combine RCT data with observational data.

"""

import numpy as np
from scipy import integrate, optimize
from scipy.stats import norm
import matplotlib.pyplot as plt


def observational_evidence_with_bias(obs_mean, obs_stderr):
    bias_prior_mean = 0
    bias_prior_stdev = obs_mean

    def likelihood_function(x):
        return integrate.quad(
            lambda bias: norm.pdf(x + bias, loc=obs_mean, scale=obs_stderr) * norm.pdf(bias, loc=bias_prior_mean, scale=bias_prior_stdev),
            -6 * bias_prior_stdev, 6 * bias_prior_stdev
        )[0]

    return likelihood_function


def combine_evidence(rct_mean, rct_stderr, obs_mean, obs_stderr, bias_scaling=1):
    # Set the bias due to confounding variables such that it's centered as zero
    # and the standard deviation equals the observational mean. That means
    # there's a reasonable probability that the observational data is fully
    # explained by confounding variables.
    bias_prior_mean = 0
    bias_prior_stdev = obs_mean * bias_scaling

    def likelihood_function(x):
        rct_prob = norm.pdf(x, loc=rct_mean, scale=rct_stderr)
        obs_prob = integrate.quad(
            lambda bias: norm.pdf(x + bias, loc=obs_mean, scale=obs_stderr) * norm.pdf(bias, loc=bias_prior_mean, scale=bias_prior_stdev),
            -6 * bias_prior_stdev, 6 * bias_prior_stdev
        )[0]
        return rct_prob * obs_prob

    return likelihood_function


def scale_to_1(probability_densities, bin_width):
    return probability_densities / np.sum(probability_densities) / bin_width


# RCT and observational evidence agree, but RCT is only marginally significant
# and observational is much stronger
wide_likelihood = combine_evidence(1, 0.5, 1, 0.2, bias_scaling=1)
narrow_likelihood = combine_evidence(1, 0.5, 1, 0.2, bias_scaling=0.5)
obs_likelihood_function = observational_evidence_with_bias(1, 0.2)

# graph the likelihood function
x = np.linspace(-1, 3, 50)
bin_width = (x[-1] - x[0]) / len(x)
domain = (x[0], x[-1])
rct_likelihood = scale_to_1([norm.pdf(xi, loc=1, scale=0.5) for xi in x], bin_width)
# obs_likelihood = scale_to_1([obs_likelihood_function(xi) for xi in x], bin_width)
wide_smart_likelihood = scale_to_1([wide_likelihood(xi) for xi in x], bin_width)
narrow_smart_likelihood = scale_to_1([narrow_likelihood(xi) for xi in x], bin_width)

# fit smart_likelihood to a normal distribution
# fit_mean, fit_std = optimize.curve_fit(lambda x, m, s: norm.pdf(x, m, s), x, smart_likelihood)[0]
# fit_likelihood = norm.pdf(x, fit_mean, fit_std)

for likelihood_table, name in [
        (rct_likelihood, "RCT"),
        (wide_smart_likelihood, "Combined (wide prior)"),
        (narrow_smart_likelihood, "Combined (narrow prior)")
]:
    mean = np.sum(x * likelihood_table * bin_width)
    print(f"{name}: mean = {mean:.2f}, stdev = {np.sum((x - mean)**2 * likelihood_table)**0.5:.2f}")

plt.plot(x, rct_likelihood, label="RCT")
# plt.plot(x, obs_likelihood, label="Observational")
plt.plot(x, wide_smart_likelihood, label="Combined (wide prior)")
plt.plot(x, narrow_smart_likelihood, label="Combined (narrow prior)")
# plt.plot(x, fit_likelihood, label="Combined (Fit)")
plt.legend()
plt.show()
