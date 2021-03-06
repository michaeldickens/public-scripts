import csv
from multiprocessing import Pool
from datetime import datetime
from dateutil.relativedelta import relativedelta
from functools import reduce
import math
import random
import unittest

import numpy as np
from scipy import optimize, stats


def read_historical_data():
    with open("rolling-1-year-returns.csv", "r") as infile:
        reader = csv.reader(infile)
        rows = []
        header = None
        for row in reader:
            if not header:
                header = row
            else:
                rows.append(row)

    # columns = [Date, Equities, Alt Annual Rebalanced, Alt Daily Rebalanced]
    return {datetime.strptime(r[0], "%Y-%m"): (float(r[1]), float(r[3])) for r in rows}


cg_tax_rate = 0.20


def run_conventional(donation_prop, market_rets, volatile_asset_rets):
    """Simulate the behavior of a conventional investment where you buy the market,
    donate some each year, and sell + pay taxes at the end.
    """
    core_holding_gain = reduce(lambda accum, ret: accum * (1 + ret), market_rets, 1)
    cap_gain_prop = (core_holding_gain - 1) / core_holding_gain

    total_donation = 0
    wealth = 1
    for market_ret in market_rets:
        donation = wealth * donation_prop
        wealth -= donation
        total_donation += donation
        wealth *= 1 + market_ret
        if wealth <= 0:
            return (total_donation, 0)

    wealth *= 1 - max(0, cg_tax_rate * cap_gain_prop)

    return (total_donation, wealth)


def run_donation_trick(donation_trick_prop, market_rets, volatile_asset_rets):
    """Simulate the behavior of the donation trick:

    1. Invest your core holding in the market.
    2. With some of your money, go long and short the same volatile asset.
       Donate whichever one goes up, and sell whichever one goes down,
       accumulating capital losses.
    3. At the end, sell the core holding and pay taxes."""
    overhead_cost = 0.00

    total_donation = 0
    core_holding_size = 1
    market_price = 1
    cap_losses_by_year = []
    loser = 0
    for market_ret, volatile_asset_ret in zip(market_rets, volatile_asset_rets):
        cap_loss = 0

        # If taxes_prop < 0, we can't add negative taxes to our wealth, but we can
        # record a cap loss.
        cap_gain_prop =  (market_price - 1) / market_price
        taxes_prop = cg_tax_rate * cap_gain_prop
        if taxes_prop < 0:
            taxes_prop = 0
            cap_loss -= taxes_prop

        # Sell an amount of the core holding such that, after tax:
        # 2*volatile_holding_size / total wealth = donation_trick_prop
        sale_size = (
            (loser * (donation_trick_prop - 1) + core_holding_size * donation_trick_prop)
            /
            ((donation_trick_prop - 1) * taxes_prop + 1)
        )
        if sale_size >= 0:
            core_holding_size -= sale_size
            volatile_holding_size = (loser + (1 - taxes_prop) * sale_size) / 2
        else:
            # In rare cases, the loser position is too large, so we need to buy
            # more of the core holding. Instead of tracking a new cost basis,
            # buy back in at the current price but the original basis, and then
            # add the difference to our cap losses.
            # TODO: test this
            core_holding_size -= sale_size
            volatile_holding_size = (loser + sale_size) / 2
            cap_loss -= cap_gain_prop * sale_size

        volatile_asset_long = volatile_holding_size * (1 + volatile_asset_ret - overhead_cost)
        volatile_asset_short = volatile_holding_size * (1 - volatile_asset_ret - overhead_cost)
        winner = max(volatile_asset_long, volatile_asset_short)
        loser = min(volatile_asset_long, volatile_asset_short)

        total_donation += winner

        cap_loss += volatile_holding_size - loser
        market_price *= 1 + market_ret
        core_holding_size *= 1 + market_ret
        cap_losses_by_year.append(cap_loss)
        if core_holding_size <= 0:
            return (total_donation, 0)

    # Cap losses can only carry forward for 5 years
    total_cap_loss = sum(cap_losses_by_year[-5:])

    cap_gain_prop = (market_price - 1) / market_price
    core_holding_size -= max(0, cg_tax_rate * (core_holding_size * cap_gain_prop - total_cap_loss))
    wealth = core_holding_size + loser

    return (total_donation, wealth)


def value_to_preserve_wealth(returns_dict, holding_years, step_fun):
    """Find how much to donate each year to make your wealth stay stable on
    average."""
    def expected_wealth_growth(arg_array):
        total_growth = 0
        for start_date in returns_dict:
            # For every valid start date, make a list of returns for the next
            # `holding_years` years, and calculate the behavior of investing
            # and donating over those years.
            if start_date + relativedelta(years=holding_years - 1) not in returns_dict:
                continue
            rets = [returns_dict[start_date + relativedelta(years=i)] for i in range(holding_years)]
            market_rets = [t[0] for t in rets]
            volatile_asset_rets = [t[1] for t in rets]
            _, new_wealth = step_fun(arg_array[0], market_rets, volatile_asset_rets)
            total_growth += np.log(new_wealth)

        return total_growth / len(returns_dict)

    # Need to set an upper bound. Otherwise, the optimizer will try to set
    # donation_prop so large that the donor will sometimes go bankrupt, which
    # breaks the optimizer. Pay attention to make sure the "optimal" result
    # isn't hitting the upper bound.
    opt = optimize.minimize(lambda p: expected_wealth_growth(p)**2, x0=0.001, bounds=[(0, 0.25)])

    if not opt.success:
        print(opt)
        raise Exception("optimization failed")

    return opt.x[0]


def donation_trick_ev(
        returns_dict, holding_years, conventional_donation_prop=None, donation_trick_prop=None,
        verbose=True
):
    # Assume core holding starts with a 0% capital gain.

    if conventional_donation_prop is None and donation_trick_prop is None:
        # Calculate the ex-post optimal donation proportion for both
        # conventional investing and the donation trick. "Optimal" is the
        # proportion that preserves wealth on average.
        conventional_donation_prop = value_to_preserve_wealth(
            returns_dict, holding_years, run_conventional)
        donation_trick_prop = value_to_preserve_wealth(
            returns_dict, holding_years, run_donation_trick)
        if verbose:
            print("conventional prop: {:.1f}%".format(100 * conventional_donation_prop))
            print("volatile prop    : {:.1f}%".format(100 * donation_trick_prop))

    elif conventional_donation_prop is None or donation_trick_prop is None:
        raise ValueError(
            "conventional_donation_prop and donation_trick_prop must both be None or non-None"
        )

    conventional_results = []
    donation_trick_results = []
    for period in returns_dict:
        market_ret, volatile_asset_ret = returns_dict[period]

        # Ignore wealth growth b/c we chose donation_trick_prop to keep wealth stable
        conventional_results.append(run_conventional(conventional_donation_prop, [market_ret], [volatile_asset_ret]))
        donation_trick_results.append(run_donation_trick(donation_trick_prop, [market_ret], [volatile_asset_ret]))

    average_conventional_donation = np.mean([pair[0] for pair in conventional_results])
    average_trick_donation = np.mean([pair[0] for pair in donation_trick_results])
    average_conventional_wealth = np.mean([pair[1] for pair in conventional_results])
    average_trick_wealth = np.mean([pair[1] for pair in donation_trick_results])

    if verbose:
        print("avg conventional donation: {:.2f}%".format(100 * average_conventional_donation))
        print("avg trick donation       : {:.2f}%".format(100 * average_trick_donation))

    return (
        average_conventional_donation,
        average_trick_donation,
        average_conventional_wealth,
        average_trick_wealth
    )


class PoolHelper:
    """Python can't pass closures across a process pool, so I have to use this
    helper class to store state.
    """
    def __init__(self, returns_dict, holding_years, fuzz_factor):
        self.returns_dict = returns_dict
        self.holding_years = holding_years
        self.fuzz_factor = fuzz_factor
        self.optimal_conventional_donation_prop = value_to_preserve_wealth(
            returns_dict, holding_years, run_conventional)
        self.optimal_donation_trick_prop = value_to_preserve_wealth(
            returns_dict, holding_years, run_donation_trick)


    def run_single_sim(self, ignore_me):
        conventional_donation_prop = np.random.normal(
            loc=self.optimal_conventional_donation_prop,
            scale=self.fuzz_factor * self.optimal_conventional_donation_prop,
        )
        donation_trick_prop = np.random.normal(
            loc=self.optimal_donation_trick_prop,
            scale=self.fuzz_factor * self.optimal_donation_trick_prop,
        )

        return donation_trick_ev(
            self.returns_dict, self.holding_years, conventional_donation_prop,
            donation_trick_prop, verbose=False
        )

def donation_trick_ev_with_estimation_error(returns_dict, holding_years, fuzz_factor=0.2):
    pool_helper = PoolHelper(returns_dict, holding_years, fuzz_factor)
    with Pool() as pool:
        # 100,000 iterations takes about 80 seconds (560 seconds of CPU time)
        sim_results = list(pool.map(pool_helper.run_single_sim, range(100000)))

    conventional_donation_ev = np.mean([pair[0] for pair in sim_results])
    trick_donation_ev = np.mean([pair[1] for pair in sim_results])
    conventional_wealth_ev = np.mean([pair[2] for pair in sim_results])
    trick_wealth_ev = np.mean([pair[3] for pair in sim_results])

    print("conventional donation EV: {:.2f}%".format(100 * conventional_donation_ev))
    print("trick donation EV       : {:.2f}%".format(100 * trick_donation_ev))
    print("conventional wealth EV: {:.2f}%".format(100 * conventional_wealth_ev))
    print("trick wealth EV       : {:.2f}%".format(100 * trick_wealth_ev))

    print("trick - conventional total: {:.3f}%".format(
        100 * (
            holding_years * (trick_donation_ev - conventional_donation_ev)
            +
            (trick_wealth_ev - conventional_wealth_ev)
        )
    ))

    return (conventional_donation_ev, trick_donation_ev, conventional_wealth_ev, trick_wealth_ev)


def simulated_sample():
    # Note: I wrote this function when I had different plans for how to
    # simulate. I'm not using it right now, but I might use it later.
    sigma = 0.16
    mu = 0.05
    leverage = 3

    market_ret = np.exp(np.random.normal(
        loc=(mu + sigma**2/2)*leverage - (sigma*leverage)**2/2,
        scale=sigma*leverage
    ))
    net_gain = market_ret - 1
    deduction = abs(net_gain)

    # 20% tax rate on half the portfolio
    tax_savings = 0.5 * 0.20 * deduction
    return tax_savings


class TestDonationTrick(unittest.TestCase):
    def assertAlmostEqual2(self, pair, expected):
        self.assertAlmostEqual(pair[0], expected[0])
        self.assertAlmostEqual(pair[1], expected[1])

    def assertStages(self, prop, stages, places=None):
        """Run a simulator in stages, checking that each stage is correct.

        stages: A list of lists where each inner list contains
          [market ret, volatile asset ret, expected donation, expected wealth]
        """
        market_rets = []
        volatile_asset_rets = []
        for i, stage in enumerate(stages):
            market_rets.append(stage[0])
            volatile_asset_rets.append(stage[1])
            actual = run_donation_trick(prop, market_rets, volatile_asset_rets)
            self.assertAlmostEqual(actual[0], stage[2], places, "year {} total donation".format(i+1))
            self.assertAlmostEqual(actual[1], stage[3], places, "year {} wealth".format(i+1))

    def test_run_conventional_1y(self):
        self.assertEqual(run_conventional(1, [0.1], None), (1, 0))
        self.assertEqual(run_conventional(0, [0.1], None), (0, 1.08))
        self.assertAlmostEqual2(run_conventional(0.1, [0.1], None), (0.1, 0.99 - 0.09*0.2))
        self.assertAlmostEqual2(run_conventional(0.1, [-0.1], None), (0.1, 0.81))

    def test_run_conventional(self):
        self.assertAlmostEqual2(run_conventional(0, [1, 0.1, 0.5], None), (0, 2.84))
        self.assertAlmostEqual2(run_conventional(0.2, [0.25, 0.25, 0.25], None), (0.6, 0.9024))

    def test_run_donation_trick_1y(self):
        self.assertEqual(run_donation_trick(1, [0], [0]), (0.5, 0.5))
        self.assertEqual(run_donation_trick(0.5, [0], [0]), (0.25, 0.75))
        self.assertEqual(run_donation_trick(0, [0.1], [0]), (0, 1.08))
        self.assertAlmostEqual2(run_donation_trick(0.1, [0], [0.5]), (0.075, 0.925))
        self.assertAlmostEqual2(run_donation_trick(0.1, [0.1], [0]), (0.05, 1.022))
        self.assertAlmostEqual2(run_donation_trick(0.2, [0.1], [0.5]), (0.15, 0.924))
        self.assertAlmostEqual2(run_donation_trick(0.1, [-0.1], [0]), (0.05, 0.86))

    def test_run_donation_trick(self):
        self.assertAlmostEqual2(run_donation_trick(0, [1, 0.1, 0.5], [0, 0, 0]), (0, 2.84))

        self.assertStages(0.2, [
            [-0.1, 0.5, 0.15, 0.77],
            [0, 0.5, 0.2655, 0.6545],
        ])

        # TODO: unit test for two years where market for first year is
        # positive. (hard to calculate results by hand in this case, so hard to
        # know if it's right)


returns_dict = read_historical_data()
# donation_trick_ev(returns_dict, holding_years=5)
donation_trick_ev_with_estimation_error(returns_dict, holding_years=5)
