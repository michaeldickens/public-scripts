import csv
from datetime import datetime
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


def run_conventional(donation_prop, market_ret, volatile_asset_ret):
    wealth = 1
    donation = wealth * donation_prop
    wealth -= donation
    wealth_pre_gain = wealth
    wealth *= (1 + market_ret)
    wealth -= max(0, cg_tax_rate * (wealth - wealth_pre_gain))
    wealth = max(0, wealth)  # don't let wealth go below 0

    return (donation, wealth)


def run_donation_trick(volatile_asset_prop, market_ret, volatile_asset_ret):
    # TODO: what if short position loses more than 100%? should probably make
    # up the difference by paying out of wealth
    wealth = 1
    volatile_asset_long = wealth * volatile_asset_prop / 2 * (1 + volatile_asset_ret)
    volatile_asset_short = wealth * volatile_asset_prop / 2 * (1 - volatile_asset_ret)
    winner = max(volatile_asset_long, volatile_asset_short)
    loser = min(volatile_asset_long, volatile_asset_short)
    wealth -= volatile_asset_prop
    wealth_pre_gain = wealth
    wealth *= (1 + market_ret)

    core_gain = wealth - wealth_pre_gain
    volatile_asset_gain = loser - volatile_asset_prop / 2  # will be negative
    wealth -= max(0, cg_tax_rate * (core_gain + volatile_asset_gain))

    # Donate the winner. Add the loser's remaining value back into `wealth`
    donation = winner
    wealth += loser

    wealth = max(0, wealth)  # don't let wealth go below 0

    return (donation, wealth)


def value_to_preserve_wealth(returns_dict, step_fun):
    returns_list = list(returns_dict.values())

    def expected_wealth_growth(arg_array):
        total_growth = 0
        for market_ret, volatile_asset_ret in returns_list:
            _, new_wealth = step_fun(arg_array[0], market_ret, volatile_asset_ret)
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
    volatile_asset_prop = value_to_preserve_wealth(
        returns_dict, run_donation_trick)
    print("conventional prop: {:.1f}%".format(100 * conventional_donation_prop))
    print("volatile prop    : {:.1f}%".format(100 * volatile_asset_prop))

    total_conventional_donation = 0
    total_trick_donation = 0
    for period in returns_dict:
        market_ret, volatile_asset_ret = returns_dict[period]

        # Ignore wealth growth b/c we chose volatile_asset_prop to keep wealth stable
        total_conventional_donation += run_conventional(conventional_donation_prop, market_ret, volatile_asset_ret)[0]
        total_trick_donation += run_donation_trick(volatile_asset_prop, market_ret, volatile_asset_ret)[0]

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
    def assertPairAlmostEqual(self, pair, expected):
        self.assertAlmostEqual(pair[0], expected[0])
        self.assertAlmostEqual(pair[1], expected[1])

    def test_run_conventional(self):
        self.assertEqual(run_conventional(1, 0.1, None), (1, 0))
        self.assertEqual(run_conventional(0, 0.1, None), (0, 1.08))
        self.assertAlmostEqual(run_conventional(0.1, 0.1, None)[1], 0.99 - 0.09*0.2)

    def test_run_donation_trick(self):
        self.assertEqual(run_donation_trick(1, 0, 0), (0.5, 0.5))
        self.assertEqual(run_donation_trick(0.5, 0, 0), (0.25, 0.75))
        self.assertEqual(run_donation_trick(0, 0.1, 0), (0, 1.08))
        self.assertPairAlmostEqual(run_donation_trick(0.1, 0, 0.5), (0.075, 0.925))
        self.assertPairAlmostEqual(run_donation_trick(0.1, 0.1, 0), (0.05, 1.022))
        self.assertPairAlmostEqual(run_donation_trick(0.2, 0.1, 0.5), (0.15, 0.924))


returns_dict = read_historical_data()
donation_trick_ev(returns_dict)
