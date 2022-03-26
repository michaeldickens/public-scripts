"""

variance.py
-----------

Author: Michael Dickens <michael@mdickens.me>
Created: 2022-03-22

"""

import numpy as np

def gen_samples(mu, s_t, s_slice, num_years, points_per_year):
    pop_mean = 1
    samples = []
    rvs = np.random.normal(mu, s_t, num_years)
    for t in range(num_years + 1):
        samples.append(list(map(np.exp, np.random.normal(pop_mean, s_slice, points_per_year))))
        if t < num_years:
            pop_mean *= np.exp(rvs[t])

    return samples


def estimate_stdev(samples):
    means = list(map(np.mean, samples))
    growths = [y/x - 1 for x, y in zip(means, means[1:])]
    return np.std(growths)


def estimate_slice_stdev(samples):
    mu0 = np.mean(samples[0])
    stdevs = [np.std(sample) * mu0 / np.mean(sample) for sample in samples]
    return np.mean(stdevs)


est_stdev = 0

num_trials = 100
for i in range(num_trials):
    samples = gen_samples(0.1, 0.2, 1.5, 10, 10)
    # est_stdev += (1 / num_trials) * (estimate_stdev(samples) - estimate_slice_stdev(samples))
    # est_stdev += (1 / num_trials) * estimate_stdev(samples)
    est_stdev += (1 / num_trials) * estimate_slice_stdev(samples)

print(est_stdev)
