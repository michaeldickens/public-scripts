import random
from functools import reduce
from pprint import pprint
from unittest import TestCase

import numpy as np
from scipy import integrate, optimize, stats


class LeverageEnvironment:
    rf = 0.00  # not currently used
    rra = 2.0

    mean = 0.07  # log(arithmetic mean)
    sigma = 0.13
    market_sigma = 0.156

    def __init__(self,  rra=None, mean=None, sigma=None, market_sigma=None):
        if rra is not None:
            self.rra = rra
        if mean is not None:
            self.mean = mean
        if sigma is not None:
            self.sigma = sigma
        if market_sigma is not None:
            self.market_sigma = market_sigma

    def utility(self, consumption):
        if self.rra == 1:
            return np.log(consumption)
        return (consumption**(1 - self.rra) - 1) / (1 - self.rra)

    def expected_utility(self, leverage=1):
        # mean and standard deviation of a normal random variable scale linearly
        # when multiplied by a constant
        sigma = leverage * self.sigma
        mu = leverage * self.mean - sigma**2/2  # median / geometric mean

        # Formula taken from slide 12 of
        # http://web.stanford.edu/class/cme241/lecture_slides/UtilityTheoryForRisk.pdf
        if self.rra == 1:
            return mu
        return (np.exp(mu*(1 - self.rra) + sigma**2/2 * (1 - self.rra)**2) - 1) / (1 - self.rra)

    def expected_utility_old(self, leverage):
        # mean and standard deviation of a normal random variable scale linearly
        # when multiplied by a constant
        mean = self.mean * leverage
        sigma = self.sigma * leverage

        def integrand(w):
            inner = mean - sigma**2/2 + sigma * w
            # `self.utility()` is inlined here to prevent float overflow
            if self.rra == 1:
                util = inner
            else:
                util = (np.exp(inner * (1 - self.rra)) - 1) / (1 - self.rra)
            return util * stats.norm.pdf(w, 0, 1)

        return integrate.quad(integrand, -10, 10)[0]

    def expected_utility_general(self, leverage):
        '''
        Works for any utility function
        '''
        mean = self.mean * leverage
        sigma = self.sigma * leverage

        def integrand(w):
            # Formula adapted from Domian, Racine, and Wilson (2003),
            # "Leveraged Stock Portfolios over Long Holding Periods: A
            # Continuous Time Model". `w` is a standard normal random variable.
            # Can also be computed by using a normal random variable with
            # mu=(mean - sigma^2/2), sigma=sigma.
            consumption = np.exp(mean - sigma**2/2 + sigma * w)
            return self.utility(consumption) * stats.norm.pdf(w, 0, 1)

        return integrate.quad(integrand, -10, 10)[0]

    def optimal_leverage(self):
        return optimize.minimize_scalar(
            lambda leverage: -self.expected_utility_general(leverage),
            bracket=[0.00001, 1, 999999]
        )

    def print_optimal_leverage(self):
        opt_result = self.optimal_leverage()
        print("Leverage {} gives utility {}".format(opt_result.x, -opt_result.fun))

    def risk_free_return_for_utility(self, u):
        '''Find the risk-free return rate that will give a particular utility.'''
        if self.rra == 1:
            return u
        return np.log((u * (1 - self.rra) + 1)**(1/(1 - self.rra)))

    def certainty_equivalent_return(self, leverage=None):
        '''
        Find the risk-free return that has the same utility as the
        leveraged distribution given by mean, sigma, and leverage.
        If leverage is not specified, use optimal leverage.

        Alternatively, can compute analytically:
        Certainty Equivalent = (leverage * mean) - (leverage * sigma)**2/2 * rra
        See slide 12 of
        http://web.stanford.edu/class/cme241/lecture_slides/UtilityTheoryForRisk.pdf
        '''
        if leverage is None:
            leverage_result = self.optimal_leverage()
            leverage = leverage_result.x
            expected_utility = -leverage_result.fun
        else:
            expected_utility = self.expected_utility_general(leverage)

        return {
            'leverage': leverage,
            'utility': expected_utility,
            'certainty-equivalent': self.risk_free_return_for_utility(expected_utility),
        }

    def expected_utility_after_n_years(self, num_years: int, pure_time_preference: float) -> float:
        mean = self.mean * num_years
        sigma = self.sigma * np.sqrt(num_years)
        pseudo_asset = LeverageEnvironment(
            rra=self.rra, mean=mean, sigma=sigma, market_sigma=self.market_sigma)
        return (
            (1 - pure_time_preference)**num_years
            * pseudo_asset.expected_utility(leverage=1)
        )

    def optimal_consumption_rate(self, discount_rate, leverage):
        mean = self.mean * leverage
        sigma = self.sigma * leverage
        rra = self.rra

        return discount_rate**(1/rra) * np.exp((mean + sigma**2/2)*(1 - rra)/rra - (1 - rra)*sigma**2/2)

    def samuelson_share(self):
        # equal to `optimal_leverage`
        return self.mean / (self.sigma**2 * self.rra)


def optimal_allocation(asset1_mean, asset1_stdev, asset2_mean, asset2_stdev, correlation, eta=1):
    # maximize Sharpe ratio
    def sharpe(asset1_prop):
        asset2_prop = 1 - asset1_prop
        joint_mean = asset1_mean * asset1_prop + asset2_mean * asset2_prop
        joint_stdev = np.sqrt(
            (asset1_stdev * asset1_prop)**2
            + (asset2_stdev * asset2_prop)**2
            + 2 * correlation * asset1_stdev * asset1_prop * asset2_stdev * asset2_prop
        )

        return joint_mean / joint_stdev

    opt = optimize.minimize_scalar(lambda asset1_prop: -sharpe(asset1_prop), bracket = [-1, 2])
    print(opt.x)


class TestUtilityLeverage(TestCase):
    def test_risk_free_return_for_utility_finds_fixed_point(self):
        for rra in [0.5, 1.0, 2.0]:
            obj = LeverageEnvironment(rra=rra)

            for i in range(100):
                fixed_point = random.randrange(0, 200)/100  # up to 200% return
                utility = obj.utility(np.exp(fixed_point))
                r = obj.risk_free_return_for_utility(utility)
                self.assertAlmostEqual(fixed_point, r)

    def test_expected_utility_after_one_year(self):
        obj = LeverageEnvironment()

        for i in range(20):
            for j in range(30):
                mean = i / 100  # 0% to 20%
                sigma = (j + 1) / 100  # 1% to 30%
                one_year_utility = obj.expected_utility(leverage=1)
                n_year_utility = obj.expected_utility_after_n_years(
                    num_years=1, pure_time_preference=0)
                self.assertAlmostEqual(one_year_utility, n_year_utility)

    def test_optimal_leverage_equals_samuelson_share(self):
        # TODO: For sufficiently large rra/small mean/large sigma,
        # `optimal_leverage` fails to find a solution
        for rra in [0.5, 1.0, 2.0, 3.0]:
            for mean in [0.05, 0.2]:
                for sigma in [0.08, 0.13, 0.17]:
                    obj = LeverageEnvironment(mean=mean, sigma=sigma, rra=rra)
                    optimal_leverage = obj.optimal_leverage().x
                    samuelson_share = obj.samuelson_share()
                    self.assertAlmostEqual(optimal_leverage, samuelson_share, places=5)


if __name__ == "__main__":
    rra = 1
    print(LeverageEnvironment(rra=rra, mean=0.047, sigma=0.085).certainty_equivalent_return())
    print(LeverageEnvironment(rra=rra, mean=0.05, sigma=0.156).certainty_equivalent_return(leverage=1))
