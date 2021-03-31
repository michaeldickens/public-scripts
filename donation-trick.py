import csv
from datetime import datetime
from functools import reduce
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
        assert(wealth >= 0)

    wealth *= 1 - max(0, cg_tax_rate * cap_gain_prop)

    return (total_donation, wealth)


def run_donation_trick(donation_trick_prop, market_rets, volatile_asset_rets):
    net_cap_gain = 0

    # TODO: If core holding declines >50% while volatile holding is flat, then
    # theoretically we should use some of the volatile holding to buy more of
    # the core holding, which means now we have multiple lots with different
    # cost bases. But that makes the math way more complicated and it's pretty
    # rare, so let's assume if that does happen, we will buy back in at the
    # current price but the original basis, and then deduct the difference as a
    # loss. That should usually be correct b/c when we need to sell, we will
    # sell the highest-basis positions first.
    total_donation = 0
    core_holding_lots = [(1, 1)]  # pairs of (basis, current value)
    for market_ret, volatile_asset_ret in zip(market_rets, volatile_asset_rets):
        # sort from least to most cap gain
        core_holding_lots.sort(key=lambda pair: pair[1] / pair[0])
        wealth = sum([pair[1] for pair in core_holding_lots])

        # Sell off lots of core holding to buy volatile holding
        volatile_holding_size = wealth * donation_trick_prop / 2
        sold_holdings = 0
        for i in range(len(core_holding_lots)):
            # TODO: This is still wrong. Correct approach is to sell holdings
            # until we get to the correct volatile holding size relative to
            # *current* wealth, not relative to wealth before paying cap gains
            # tax. But that's a lot more complicated to calculate.

            cap_gain_on_lot = core_holding_lots[i][1] - core_holding_lots[i][0]
            owed_tax = max(0, cg_tax_rate * cap_gain_on_lot)
            if core_holding_lots[i][1] - owed_tax + sold_holdings < volatile_holding_size:
                if cap_gain_on_lot < 0:
                    # If incurred a loss, carry it forward. If incurred a gain,
                    # pay it now.
                    net_cap_gain += cap_gain_on_lot
                sold_holdings += core_holding_lots[i][1] - owed_tax
                core_holding_lots[i][0] = 0
                core_holding_lots[i][1] = 0
            else:
                tax_prop = owed_tax / core_holding_lots[i][1]
                # TODO

        volatile_asset_long = volatile_holding_size * (1 + volatile_asset_ret)
        volatile_asset_short = volatile_holding_size * (1 - volatile_asset_ret)
        winner = max(volatile_asset_long, volatile_asset_short)
        loser = min(volatile_asset_long, volatile_asset_short)

        total_donation += winner

        # wealth *= 1 - donation_trick_prop
        # net_cap_gain += wealth * market_ret
        # net_cap_gain -= volatile_holding_size - loser
        # wealth *= 1 + market_ret
        # assert(wealth >= 0)
        # wealth += loser

    wealth -= cg_tax_rate * max(0, net_cap_gain)

    return (total_donation, wealth)


def value_to_preserve_wealth(returns_dict, step_fun):
    returns_list = list(returns_dict.values())

    def expected_wealth_growth(arg_array):
        total_growth = 0
        for market_ret, volatile_asset_ret in returns_list:
            _, new_wealth = step_fun(arg_array[0], [market_ret], [volatile_asset_ret])
            total_growth += np.log(new_wealth)

        return total_growth / len(returns_list)

    opt = optimize.minimize(lambda p: expected_wealth_growth(p)**2, x0=0.01)
    if not opt.success:
        print(opt)
        raise Exception("optimization failed")

    return opt.x[0]


def donation_trick_ev(returns_dict):
    # Assume core holding starts with a 0% capital gain.
    conventional_donation_prop = value_to_preserve_wealth(
        returns_dict, run_conventional)
    donation_trick_prop = value_to_preserve_wealth(
        returns_dict, run_donation_trick)
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

    def test_run_conventional_1y(self):
        self.assertEqual(run_conventional(1, [0.1], None), (1, 0))
        self.assertEqual(run_conventional(0, [0.1], None), (0, 1.08))
        self.assertAlmostEqual2(run_conventional(0.1, [0.1], None), (0.1, 0.99 - 0.09*0.2))
        self.assertAlmostEqual2(run_conventional(0.1, [-0.1], None), (0.1, 0.81))

    def test_run_conventional_3y(self):
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

    def test_run_donation_trick_3y(self):
        self.assertAlmostEqual2(run_donation_trick(0, [1, 0.1, 0.5], [0, 0, 0]), (0, 2.84))
        self.assertAlmostEqual2(run_donation_trick(0.5, [0.5, 0.5, 0.5], [0, 0, 0]), (, ))



returns_dict = read_historical_data()
# donation_trick_ev(returns_dict)
