"""

mission_hedging.py
------------------

Author: Michael Dickens <michael@mdickens.me>
Created: 2022-02-14

"""

import multiprocessing
import numpy as np

def gen_returns():
    '''Returns for [index, hedge, co2]'''

    # Choose the industry that the hedge most resembles:
    # - NoDur shows the most stable high performance
    # - Hlth shows high performance but high vol
    # - HiTec shows negative idiosyncratic return
    #
    # All are long/short to remove market beta and levered up to match market
    # stdev.
    means = [
        0.08,  # index return
        {      # beta-neutral hedge return
            'NoDur':  0.07,
            'Hlth':   0.04,
            'HiTec': -0.02,
        }['Hlth'],
        0.1,  # co2 increase
    ]

    stdevs = [0.18, 0.18, 0.05]
    correlations = [
        [1, 0  , 0  ],
        [0, 1  , 0.5],
        [0, 0.5, 1  ],
    ]
    covariances = [
        [correl * stdev1 * stdev2 for correl, stdev2 in zip(row, stdevs)]
        for row, stdev1 in zip(correlations, stdevs)
    ]

    return np.random.multivariate_normal(means, covariances, size=1)[0]


def utility(money, co2):
    return np.log(money) * co2


def simulate(num_years, prop_hedge):
    money = 1
    total_co2 = 1

    for _ in range(num_years):
        index, hedge, co2 = gen_returns()
        money *= prop_hedge * np.exp(hedge) + (1 - prop_hedge) * np.exp(index)
        # total_co2 += co2            # linear (normal)
        total_co2 *= np.exp(co2)  # exponential (lognormal)

    return utility(money, total_co2)

def two_sigma_CI(fun, sample_size):
    pool = multiprocessing.Pool(8)
    samples = pool.map(fun, range(sample_size))
    # samples = list(map(fun, range(sample_size)))  # single-threaded version
    mean = np.mean(samples)
    stderr = np.std(samples) / np.sqrt(sample_size)
    return (mean - 2 * stderr, mean + 2 * stderr)

def simulate0(_):
    return simulate(num_years, 0)

def simulate1(_):
    return simulate(num_years, 0.1)

def simulate2(_):
    return simulate(num_years, 0.2)

def simulate3(_):
    return simulate(num_years, 0.5)

def simulate4(_):
    return simulate(num_years, 1)

num_years = 1
sample_size = 100000

if __name__ == '__main__':
    print("   Unhedged: {:.3f} to {:.3f}".format(*two_sigma_CI(simulate0, sample_size)))
    print(" 10% hedged: {:.3f} to {:.3f}".format(*two_sigma_CI(simulate1, sample_size)))
    print(" 20% hedged: {:.3f} to {:.3f}".format(*two_sigma_CI(simulate2, sample_size)))
    print(" 50% hedged: {:.3f} to {:.3f}".format(*two_sigma_CI(simulate3, sample_size)))
    print("100% hedged: {:.3f} to {:.3f}".format(*two_sigma_CI(simulate4, sample_size)))
