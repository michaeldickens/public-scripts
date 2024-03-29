"""ulcer_sim.py
------------

Author: Michael Dickens <michael@mdickens.me>
Created: 2023-09-13

Generate simulated investment returns that target a particular return and ulcer
index.

TODO: Right now it has a ~100% chance of blowing up. But that might be a
feature, not a bug.

TODO: Test that my `scaled_sigma` calculations are doing the right thing. Write
a test function that computes `scaled_sigma` at a fixed value and then
generates a bunch of returns, and see if it does in fact have the drawdown
probability that it's supposed to have.

"""

from typing import List
import numpy as np
from scipy.stats import norm


def ulcer_index(return_series: List[float]):
    sum_squared = 0
    peak = 1
    price = 1
    for ret in return_series:
        price *= 1 + ret
        if price > peak:
            peak = price
        else:
            sum_squared += (100 * (price / peak) - 1)**2

    return np.sqrt(sum_squared / len(return_series))


def simulate_periods(num_periods: int, mu: float, sigma: float, ulcer: float):
    """
    Simulate `num_periods` periods such that, in expectation, the log
    returns have mean `mu` and standard deviation `sigma`, and the returns have
    ulcer index `ulcer`.
    """
    # baseline stdev that we can increase during drawdowns to average out to
    # the target stdev. need to figure out how to calculate the correct
    # starting sigma
    base_sigma = sigma
    np.seterr('raise')

    return_series = []
    peak = 1
    peak_period = 0
    price = 1
    for period in range(num_periods):
        if price == 0:
            return_series.append(-1)
            return return_series

        drawdown = min(0, (price / peak) - 1)
        scaled_sigma = base_sigma
        if drawdown < 0:
            dd_length = period - peak_period
            expected_price = peak * np.exp(mu * dd_length)
            deviation = np.log(price / expected_price)
            # Choose a sigma such that a drawdown of this magnitude has a 25% chance of happening again
            target_stderr = norm.ppf(0.25)  # around -0.67

            # set `scaled_sigma` such that a log-price change of `deviation` is
            # `target_sigma` standard deviations below the mean
            scaled_stderr = deviation / target_stderr
            scaled_sigma = scaled_stderr / np.sqrt(dd_length)

        rv = np.random.normal(mu, scaled_sigma)
        ret = np.exp(rv) - 1
        return_series.append(ret)
        price *= 1 + ret
        if price > peak:
            peak = price
            peak_period = period

    return return_series


rets = simulate_periods(10000, 0.05, 0.15, 20)
print("{:.1f}".format(100 * np.mean(rets)))
print("{:.1f}".format(100 * np.std(rets)))
print(ulcer_index(rets))
import ipdb; ipdb.set_trace()
