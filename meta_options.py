"""
meta_options.py
---------------

Author: Michael Dickens <michael@mdickens.me>
Created: 2021-11-26

Monte Carlo simulation of the value of meta-options. Takes a different approach
from Ben Kuhn's script: https://github.com/benkuhn/option-val

"""

import unittest
import numpy as np

from collections import defaultdict
from functools import reduce

class MetaOpt:
    utility_fun = np.log
    startup_volatility = 0.70  # ~4x S&P 500 vol
    startup_growth_rate = 0.05  # geometric mean
    market_certainty_equivalent = 0.04  # geometric mean
    vesting_years = 4
    annual_equity_value = 50000
    annual_bigco_cash_value = 50000

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            assert hasattr(self, k)
            setattr(self, k, v)

    def sim_year(self, price):
        ret = np.random.normal(loc=startup_growth_rate, scale=startup_volatility)
        return price * np.exp(ret)

    def value_given_quit_year(self, startup_rets, quit_year):
        """
        quit_year: Quit at the end of this year. (0 = quit immediately)
        """
        # Compounded value of vested shares
        initial_vested_value = self.annual_equity_value * quit_year
        vested_value = reduce(lambda x, r: x * np.exp(r), startup_rets, initial_vested_value)

        # Value of excess bigco salary if invested in the market
        cash_value = 0
        paychecks_per_year = 24
        for _ in range(paychecks_per_year * quit_year, paychecks_per_year * self.vesting_years):
            # Assumes you get paid at the end of each half-month
            cash_value *= np.exp(self.market_certainty_equivalent / paychecks_per_year)
            cash_value += self.annual_bigco_cash_value / paychecks_per_year

        return vested_value + cash_value

    def utility_given_quit_year(self, startup_rets, quit_year):
        return self.utility_fun(self.value_given_quit_year(startup_rets, quit_year))

    """
    Return: Tuple of (optimal quitting year, startup's total return at quitting time)
    """
    def post_hoc_optimal_quitting_year(self, startup_rets=None):
        if startup_rets is None:
            startup_rets = [self.sim_year(price) for _ in range(self.vesting_years)]

        utility_if_quit = [
            self.utility_given_quit_year(startup_rets, y)
            for y in range(self.vesting_years + 1)
        ]

        quit_year = np.argmax(utility_if_quit)
        total_ret = reduce(lambda x, r: x * np.exp(r), startup_rets[:(quit_year + 1)], 1)
        return (quit_year, total_ret)

    def average_monte_carlo(self, num_iterations=10000):
        quitting_total_rets_by_year = defaultdict(list)

        for i in range(num_iterations):
            quit_year, startup_total_ret = self.post_hoc_optimal_quitting_year()

        # This tells us the average total ret at which it was optimal to quit.
        # Unfortunately, that's not the same thing as the prior choice of total
        # ret cutoff that maximizes utility.
        average_quitting_total_ret_by_year = [np.mean(xs) for xs in quitting_total_rets_by_year]


class TestMetaOpt(unittest.TestCase):
    def test_value_given_quit_year(self):
        meta_opt = MetaOpt()
        self.assertAlmostEqual(
            meta_opt.value_given_quit_year([0.1, 0.1, 0.1, 0.1], 4),
            50000 * 4 * np.exp(0.4)
        )
        self.assertEqual(
            int(meta_opt.value_given_quit_year([0.1, 0.1, 0.1, 0.1], 0)),
            216707
        )

    def test_post_hoc_optimal_quitting_year(self):
        meta_opt = MetaOpt()
        self.assertEqual(meta_opt.post_hoc_optimal_quitting_year([0.00, 0.00, 0.00, 0.05])[0], 0)
        self.assertEqual(meta_opt.post_hoc_optimal_quitting_year([0.00, 0.00, 0.00, 0.10])[0], 4)
