import random
from functools import reduce
from pprint import pprint
from unittest import TestCase

import numpy as np
from scipy import integrate, optimize, stats


class LeverageEnvironment:
    rf = 0.00  # not currently used
    rra = 2.0

    # Log space parameters. MLE of mu is simply the sample mu: the mean of the
    # log returns for each year (a.k.a. geometric mean).
    mu = 0.07
    sigma = 0.13
    market_sigma = 0.16

    def __init__(self,  rra=None, mu=None, sigma=None, market_sigma=None):
        if rra is not None:
            self.rra = rra
        if mu is not None:
            self.mu = mu              # mean of normal distribution / geometric mean of log-normal
        if sigma is not None:
            self.sigma = sigma
        if market_sigma is not None:
            self.market_sigma = market_sigma
        self.alpha = self.mu + self.sigma**2/2  # log of arithmetic mean

    def utility(self, consumption):
        if self.rra == 1:
            return np.log(consumption)
        return (consumption**(1 - self.rra) - 1) / (1 - self.rra)

    def expected_utility(self, leverage=1, starting_capital=1):
        # mean and standard deviation of a normal random variable scale linearly
        # when multiplied by a constant

        # mu = median or geometric mean
        # mu could also be written as leverage * self.alpha - sigma**2/2
        mu = leverage * self.mu - leverage * (leverage - 1) * self.sigma**2 / 2
        sigma = leverage * self.sigma

        # See page 5 of https://www.gordoni.com/lifetime_portfolio_selection.pdf
        # Or see slide 12 of
        # http://web.stanford.edu/class/cme241/lecture_slides/UtilityTheoryForRisk.pdf
        # Expected utility follows from these two facts:
        # 1. k * Lognorm(mu, sigma) = Lognorm(mu + log k, sigma)
        # 2. E[Lognorm(mu, sigma)^n] = exp(n*mu + n^2*sigma^2 / 2)
        if self.rra == 1:
            return mu + np.log(starting_capital)
        return (
            np.exp(
                (mu + np.log(starting_capital)) * (1 - self.rra)
                + sigma**2/2 * (1 - self.rra)**2
            ) - 1
        ) / (1 - self.rra)

    def expected_utility_old(self, leverage, starting_capital=1):
        # mean and standard deviation of a normal random variable scale linearly
        # when multiplied by a constant
        mean = self.alpha * leverage
        sigma = self.sigma * leverage

        def integrand(w):
            inner = mean - sigma**2/2 + sigma * w
            # `self.utility()` is inlined here to prevent float overflow
            if self.rra == 1:
                util = inner + np.log(starting_capital)
            else:
                util = ((starting_capital**(1 - self.rra)) * np.exp(inner * (1 - self.rra)) - 1) / (1 - self.rra)
            return util * stats.norm.pdf(w, 0, 1)

        return integrate.quad(integrand, -10, 10)[0]

    def expected_utility_general(self, leverage, starting_capital=1):
        '''
        Works for any utility function
        '''
        mean = self.alpha * leverage
        sigma = self.sigma * leverage

        def integrand(w):
            # Formula adapted from Domian, Racine, and Wilson (2003),
            # "Leveraged Stock Portfolios over Long Holding Periods: A
            # Continuous Time Model". `w` is a standard normal random variable.
            # Can also be computed by using a normal random variable with
            # mu=(mean - sigma^2/2), sigma=sigma.
            consumption = starting_capital * np.exp(mean - sigma**2/2 + sigma * w)
            return self.utility(consumption) * stats.norm.pdf(w, 0, 1)

        return integrate.quad(integrand, -10, 10)[0]

    def optimal_leverage(self, starting_capital=1):
        return optimize.minimize_scalar(
            lambda leverage: -self.expected_utility(
                leverage, starting_capital=starting_capital
            ),
            bracket=[0.00001, 1, 999999]
        )

    def print_optimal_leverage(self):
        opt_result = self.optimal_leverage()
        print("Leverage {} gives utility {}".format(opt_result.x, -opt_result.fun))

    def risk_free_return_for_utility(self, u, starting_capital=1):
        '''Find the risk-free return rate that will give a particular utility.'''
        if self.rra == 1:
            return u - np.log(starting_capital)
        return np.log((u * (1 - self.rra) + 1)**(1/(1 - self.rra))) - np.log(starting_capital)

    def certainty_equivalent_return(self, leverage=None, starting_capital=1):
        '''
        Find the risk-free return that has the same utility as the
        leveraged distribution given by mean, sigma, and leverage.
        If leverage is not specified, use optimal leverage.

        Alternatively, can compute analytically:
        Certainty Equivalent = (leverage * alpha) - (leverage * sigma)**2/2 * rra
        (alpha is log of arithmetic mean)
        See slide 12 of
        http://web.stanford.edu/class/cme241/lecture_slides/UtilityTheoryForRisk.pdf
        '''
        if leverage is None:
            leverage_result = self.optimal_leverage(starting_capital=starting_capital)
            leverage = leverage_result.x
            expected_utility = -leverage_result.fun
        else:
            expected_utility = self.expected_utility(leverage, starting_capital=starting_capital)

        return {
            'leverage': leverage,
            'utility': expected_utility,
            'certainty-equivalent': self.risk_free_return_for_utility(
                expected_utility, starting_capital=starting_capital
            ),
        }

    def expected_utility_after_n_years(self, num_years: int, pure_time_preference: float) -> float:
        sigma = self.sigma * np.sqrt(num_years)
        mu = self.alpha * num_years - sigma**2/2
        pseudo_asset = LeverageEnvironment(
            rra=self.rra, mu=mu, sigma=sigma, market_sigma=self.market_sigma)
        return (
            (1 - pure_time_preference)**num_years
            * pseudo_asset.expected_utility(leverage=1)
        )

    def optimal_consumption_rate(self, discount_rate, leverage):
        mean = self.alpha * leverage
        sigma = self.sigma * leverage
        rra = self.rra

        return discount_rate**(1/rra) * np.exp((mean + sigma**2/2)*(1 - rra)/rra - (1 - rra)*sigma**2/2)

    def samuelson_share(self):
        # equal to `optimal_leverage`
        return self.alpha / (self.sigma**2 * self.rra)


def optimal_allocation(asset1_mean, asset1_stdev, asset2_mean, asset2_stdev, correlation, risk_free_rate=0):
    '''
    Find the allocation between two assets that maximizes Sharpe ratio. This
    allocation (with varying amounts of leverage) maximizes utility for any
    RRA.
    '''
    def sharpe(asset1_prop):
        asset2_prop = 1 - asset1_prop
        joint_mean = asset1_mean * asset1_prop + asset2_mean * asset2_prop
        joint_stdev = np.sqrt(
            (asset1_stdev * asset1_prop)**2
            + (asset2_stdev * asset2_prop)**2
            + 2 * correlation * asset1_stdev * asset1_prop * asset2_stdev * asset2_prop
        )

        return (joint_mean - risk_free_rate) / joint_stdev

    # This is the point where the derivative of the Sharpe is 0. Calculated using sympy.
    optimal_asset1_prop = (
        asset2_stdev*(
            -asset1_mean*asset2_stdev + asset2_mean*correlation*asset1_stdev
            - correlation*risk_free_rate*asset1_stdev + risk_free_rate*asset2_stdev)
        /
        (
            asset1_mean*correlation*asset1_stdev*asset2_stdev - asset1_mean*asset2_stdev**2
            + asset2_mean*correlation*asset1_stdev*asset2_stdev - asset2_mean*asset1_stdev**2
            - 2*correlation*risk_free_rate*asset1_stdev*asset2_stdev
            + risk_free_rate*asset1_stdev**2 + risk_free_rate*asset2_stdev**2
        )
    )

    print(optimal_asset1_prop, sharpe(optimal_asset1_prop))


class TestLeverageEnvironment(TestCase):
    # Run these with the command
    #   python3 -m nose leverage.py
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
            for mu in [0.05, 0.2]:
                for sigma in [0.08, 0.13, 0.17]:
                    obj = LeverageEnvironment(mu=mu, sigma=sigma, rra=rra)
                    optimal_leverage = obj.optimal_leverage().x
                    samuelson_share = obj.samuelson_share()
                    self.assertAlmostEqual(optimal_leverage, samuelson_share, places=5)

    def test_expected_utility_methods_are_identical(self):
        for rra in [0.5, 1.0, 2.0, 3.0]:
            for mu in [0.05, 0.2]:
                for sigma in [0.08, 0.13, 0.17]:
                    for leverage in [0.5, 1, 2]:
                        for starting_capital in [0.5, 1, 2]:
                            env = LeverageEnvironment(mu=mu, sigma=sigma, rra=rra)
                            ev1 = env.expected_utility(leverage=leverage, starting_capital=starting_capital)
                            ev2 = env.expected_utility_old(leverage=leverage, starting_capital=starting_capital)
                            self.assertAlmostEqual(ev1, ev2)


if False and __name__ == "__main__":
    # edit these
    expected_sell_price = 30  # arithmetic mean
    affirm_volatility = 0.60
    relative_risk_aversion = 1.5
    years_to_liquidity = 0.6
    current_tax_rate = 0.4865
    strike_price = 1.30  # fake numbers
    valuation = 15.00

    # TODO: This is wrong in a way that seems like it matters. Savings isn't a
    # log-normal distribution with mean savings_later/spending_now. It's a
    # log-normal distribution with mean savings_later, minus a constant factor
    # savings_now. But also you need to account for your other savings. that
    # you didn't spend on exercise.

    # don't edit these
    spending_now = (valuation - strike_price) * current_tax_rate + strike_price
    if years_to_liquidity >= 1:
        # convert income to long-term capital gain
        savings_later = (expected_sell_price - valuation) * (0.51 - 0.20)
    else:
        # convert income to short-term capital gain (avoid CA and FICA taxes)
        savings_later = (expected_sell_price - valuation) * (0.123 + 0.0235)
    return_to_exercise = (1 + savings_later / spending_now)**(1/years_to_liquidity) - 1

    # print("If you invest cash:")
    # market_res = LeverageEnvironment(
    #     rra=relative_risk_aversion, mu=0.033, sigma=0.094
    # ).certainty_equivalent_return(leverage=0.1)
    # print(market_res)

    # print("If you hold AFRM:")
    # afrm_res = LeverageEnvironment(
    #     # rra=relative_risk_aversion, mu=0.15 - affirm_volatility**2 / 2, sigma=affirm_volatility
    #     rra=relative_risk_aversion, mu=0.04 - 0.35**2 / 2, sigma=0.35
    # ).certainty_equivalent_return(leverage=1)
    # print(afrm_res)

    print(LeverageEnvironment(rra=2, mu=0.1, sigma=0.16).certainty_equivalent_return(leverage=1))

    # print("If you spend cash to exercise:")
    # exercise_res = LeverageEnvironment(
    #     rra=relative_risk_aversion,
    #     mu=return_to_exercise - (affirm_volatility**2)/2,
    #     sigma=affirm_volatility
    # ).certainty_equivalent_return(leverage=1)
    # print(exercise_res)

    # print("Should exercise?", "YES" if exercise_res['utility'] > market_res['utility'] else "NO")


if __name__ == "__main__":
    afrm_price = 99.00
    afrm_buy_price = 15.38
    afrm_arithmetic_mean = -0.02
    afrm_volatility = 1.00
    relative_risk_aversion = 1.5
    years_to_liquidity = 0.3
    stcg_rate = 0.37
    ltcg_rate = 0.20

    print("Market:")
    market_res = LeverageEnvironment(
        rra=relative_risk_aversion, mu=0.07, sigma=0.11
    ).certainty_equivalent_return(leverage=1.5)
    print(market_res)

    print("AFRM:")
    afrm_res = LeverageEnvironment(
        rra=relative_risk_aversion,
        mu=afrm_arithmetic_mean - afrm_volatility**2 / 2,
        sigma=afrm_volatility
    ).certainty_equivalent_return(leverage=1)
    print(afrm_res)

    # -48% difference in annual CE return = -15% 3-month return vs. a tax-based
    # -return of 27%. 8% net return from holding
