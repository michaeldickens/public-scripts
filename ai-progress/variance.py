"""

variance.py
-----------

Author: Michael Dickens <michael@mdickens.me>
Created: 2022-03-22

How does the true standard deviation of a growth rate compare to the estimated standard deviation?

"""

import numpy as np

def gen_samples(mu, s_t, s_slice, num_years, points_per_year):
    '''Generate samples with:

    - mu = mean growth rate
    - s_t = standard deviation of growth
    - s_slice = standard deviation of point-in-time samples
    - num_years = number of years
    - points_per_year = number of samples to include per year
    '''
    pop_log_mean = 1
    samples = []
    rvs = np.random.normal(mu, s_t, num_years)  # true growth
    for t in range(num_years + 1):
        # generate samples around the true growth
        samples.append([np.exp(x) for x in np.random.normal(pop_log_mean, pop_log_mean * s_slice, points_per_year)])
        if t < num_years:
            pop_log_mean += rvs[t]

    return samples


def estimate_stdev(samples):
    '''Naively estimate the standard deviation of the growth rate.'''
    means = list(map(np.mean, samples))
    growths = [np.log(y/x) for x, y in zip(means, means[1:])]
    return np.std(growths)


def estimate_slice_stdev(samples):
    '''Estimate the standard deviation of point-in-time samples using pooled
    variance.
    '''
    means = list(map(np.mean, samples))
    growths = [np.log(y/x) for x, y in zip(means, means[1:])]
    growth_rate = np.mean(growths)
    return np.sqrt(np.mean([
        np.var([np.log(x) - (growth_rate * (i + 1)) for x in sample])
        for i, sample in enumerate(samples)
    ]))


est_stdev = 0

num_trials = 100
for i in range(num_trials):
    samples = gen_samples(0.07, 0.074, 0.019, 10, 10)
    # est_stdev += (1 / num_trials) * (estimate_stdev(samples) - estimate_slice_stdev(samples))
    est_stdev += (1 / num_trials) * estimate_stdev(samples)
    # est_stdev += (1 / num_trials) * estimate_slice_stdev(samples)

print(est_stdev)
