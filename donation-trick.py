import csv
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

        volatile_asset_long = volatile_holding_size * (1 + volatile_asset_ret)
        volatile_asset_short = volatile_holding_size * (1 - volatile_asset_ret)
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


def donation_trick_ev(returns_dict, holding_years):
    # Assume core holding starts with a 0% capital gain.
    conventional_donation_prop = value_to_preserve_wealth(
        returns_dict, holding_years, run_conventional)
    donation_trick_prop = value_to_preserve_wealth(
        returns_dict, holding_years, run_donation_trick)
    print("conventional prop: {:.1f}%".format(100 * conventional_donation_prop))
    print("volatile prop    : {:.1f}%".format(100 * donation_trick_prop))

    total_conventional_donation = 0
    total_trick_donation = 0
    for period in returns_dict:
        market_ret, volatile_asset_ret = returns_dict[period]


        # Ignore wealth growth b/c we chose donation_trick_prop to keep wealth stable
        total_conventional_donation += run_conventional(conventional_donation_prop, [market_ret], [volatile_asset_ret])[0]
        total_trick_donation += run_donation_trick(donation_trick_prop, [market_ret], [volatile_asset_ret])[0]

    average_conventional_donation = total_conventional_donation / len(returns_dict)
    average_trick_donation = total_trick_donation / len(returns_dict)

    print("avg conventional donation: {:.2f}%".format(100 * average_conventional_donation))
    print("avg trick donation       : {:.2f}%".format(100 * average_trick_donation))


def simulated_sample():
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
donation_trick_ev(returns_dict, holding_years=10)
