import random
from functools import reduce
from unittest import TestCase

import numpy as np
from scipy import integrate, optimize, stats


class Environment:
    # TODO: these parameter values are all quick guesses, need refinement
    eta = 1.2
    consumption_min = 100  # dollars per year
    alpha = 1.09  # Newman, (2006). Power laws, Pareto distributions and Zipf's law
    pop_size = 7e9
    mu = 0.05  # inflation-adjusted
    sigma = 0.16
    risk_free = 0.00  # inflation-adjusted

    def personal_utility(self, consumption: float) -> float:
        return consumption**(1 - self.eta) / (1 - self.eta)

    def consumption(self, person_index: float) -> float:
        return self.consumption_min / (1 - (person_index/self.pop_size))**(1/self.alpha)

    def utility(self, donor_wealth: float) -> float:
        if donor_wealth > 1e30:
            # special case to prevent scipy from breaking on big numbers
            last_person = self.pop_size - 1
        elif donor_wealth < 1e-10:
            # special case to prevent scipy from breaking on small numbers
            last_person = 0
        else:
            # could improve performance by caching this function, since we call it with the same values repeatedly
            root_results = optimize.root_scalar(
                lambda last_person: donor_wealth - integrate.quad(
                    lambda i: self.consumption(last_person) - self.consumption(i),
                    0,
                    last_person,
                )[0],
                method='brentq',  # scipy docs say this method is usually the best
                bracket=[0, self.pop_size-1],
            )
            if not root_results.converged:
                raise Exception("root finder did not converge: {}".format(root_results.flag))
            last_person = root_results.root

        integral_result = integrate.quad(
            lambda i: (
                self.personal_utility(self.consumption(last_person))
                - self.personal_utility(self.consumption(i))),
            0,
            last_person
        )
        return integral_result[0]

    def investment_return_pdf(self, rate: float) -> float:
        '''rate: 0 = go to zero, 1 = no change'''
        return stats.lognorm.pdf(rate, self.sigma, loc=self.mu)

    def expected_future_utility(self, leverage: float, donor_wealth: float) -> float:
        '''Expected utility after letting savings compound before donating'''
        integral_result = integrate.quad(
            lambda r: (
                self.utility(((1 + r - self.risk_free)**leverage + self.risk_free)*donor_wealth)
                * self.investment_return_pdf(r)
            ),
            # <0.01x or >10x is suuuuper unlikely; need to cut off so
            # r^leverage isn't something dumb like 10^million
            -1,
            10,
        )
        return integral_result[0]

    def optimal_leverage(self, donor_wealth: float) -> float:
        # note: this has triple-nested integrals nested inside a minimization, which I think is O(n^2 log n)

        # only works if local optimum is global optimum, which I feel like it is but not 100% sure
        try:
            minimize_result = optimize.minimize_scalar(
                lambda leverage: -self.expected_future_utility(leverage, donor_wealth),
                bracket=[0, 2, 100],
            )
        except ValueError as e:
            if str(e) == 'Not a bracketing interval.':
                # this error happens because the upper end of the bracket is
                # still not enough leverage, so the optimal amount of leverage
                # is unknown but very large
                return np.inf
            else:
                raise e

        if not minimize_result.success:
            raise Exception("minimizer failed: {}".format(minimize_result.message))
        return minimize_result.x


class UtilityLeverage:
    rf = 0.00  # TODO: not currently used
    rra = 2.0

    mu = 0.08
    sigma = 0.13
    market_sigma = 0.156

    def __init__(self, rra=None, mu=None, sigma=None, market_sigma=None):
        if rra is not None:
            self.rra = rra
        if mu is not None:
            self.mu = mu
        if sigma is not None:
            self.sigma = sigma
        if market_sigma is not None:
            self.market_sigma = market_sigma

    def utility(self, consumption):
        if self.rra == 1:
            return np.log(consumption)
        return consumption**(1 - self.rra) / (1 - self.rra)

    def expected_utility(self, leverage, starting_capital=1):
        # mean and standard deviation of a normal random variable scale linearly
        # when multiplied by a constant
        mu = self.mu * leverage
        sigma = self.sigma * leverage

        def integrand(w):
            # Formula adapted from Domian, Racine, and Wilson (2003),
            # "Leveraged Stock Portfolios over Long Holding Periods: A
            # Continuous Time Model". `w` is a standard normal random variable.
            # Can also be computed by using a normal random variable with
            # mean=(mu - sigma^2/2), sigma=sigma.
            inner = mu - sigma**2/2 + sigma * w
            # `self.utility()` is inlined here to prevent float overflow
            if self.rra == 1:
                util = np.log(starting_capital) + inner
            else:
                util = starting_capital**(1 - self.rra) * np.exp(inner * (1 - self.rra)) / (1 - self.rra)
            return util * stats.norm.pdf(w, 0, 1)

        return integrate.quad(integrand, -20, 20)[0]

    def expected_utility_old(self, leverage):
        # my old version that's off by one. tbd why
        return integrate.quad(
            # lambda r: self.utility(np.exp(r * leverage)) * stats.norm.pdf(r, self.mu, self.sigma),

            # `self.utility()` is inclined here to prevent float overflow
            lambda r: (
                (r * leverage if self.rra == 1 else np.exp((r * leverage - self.sigma**2*leverage**2/2) * (1 - self.rra)) / (1 - self.rra))
                * stats.norm.pdf(r, self.mu, self.sigma)
            ),
            -1, 100
        )[0]

    def optimal_leverage(self):
        return optimize.minimize_scalar(
            lambda leverage: -self.expected_utility(leverage),
            bracket=[0.00001, 1, 999999]
        )

    def print_optimal_leverage(self):
        opt_result = self.optimal_leverage()
        print("Leverage {} gives utility {}".format(opt_result.x, -opt_result.fun))

    def risk_free_return_for_utility(self, u):
        '''Find the risk-free return rate that will give a particular utility.'''
        if self.rra == 1:
            return u
        return np.log((u * (1 - self.rra))**(1/(1 - self.rra)))

    def equivalent_risk_free_return(self):
        '''
        Find the risk-free return that has the same utility as the
        optimally-leveraged distribution given by mu and sigma.

        Note: Apparently, in the literature this is called a
        "certainty equivalent."
        '''
        leverage_result = self.optimal_leverage()
        optimal_leverage = leverage_result.x
        expected_utility = -leverage_result.fun
        return {
            'leverage': optimal_leverage,
            'utility': expected_utility,
            'risk-free': self.risk_free_return_for_utility(expected_utility),
        }

    def expected_utility_after_n_years(self, num_years: int, pure_time_preference: float) -> float:
        mu = self.mu * num_years
        sigma = self.sigma * np.sqrt(num_years)
        pseudo_asset = UtilityLeverage(
            rra=self.rra, mu=mu, sigma=sigma, market_sigma=self.market_sigma)
        return (
            (1 - pure_time_preference)**num_years
            * pseudo_asset.expected_utility(leverage=1)
        )

    def optimal_consumption_rate(self, discount_rate, leverage):
        mu = self.mu * leverage
        sigma = self.sigma * leverage
        rra = self.rra

        return discount_rate**(1/rra) * np.exp((mu + sigma**2/2)*(1 - rra)/rra - (1 - rra)*sigma**2/2)


def samuelson_share(expected_return, stdev, eta=1):
    mu = np.log(1 + expected_return)

    opt = optimize.minimize_scalar(
        # whole things is squared so the minimum is at 0 instead of negative
        lambda sigma: (((np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2)) - stdev**2)**2,
        bracket=[0.0000, 0.1, 10],
    )
    sigma = opt.x

    alpha = mu + sigma**2 / 2
    samuelson_share = alpha / (sigma**2 * eta)
    return samuelson_share, expected_return * samuelson_share * 100


class TestUtilityLeverage(TestCase):
    def test_risk_free_return_for_utility_finds_fixed_point(self):
        for rra in [0.5, 1.0, 2.0]:
            obj = UtilityLeverage(rra=rra)

            for i in range(100):
                fixed_point = random.randrange(0, 200)/100  # up to 200% return
                utility = obj.utility(np.exp(fixed_point))
                r = obj.risk_free_return_for_utility(utility)
                self.assertAlmostEqual(fixed_point, r)

    def test_expected_utility_after_one_year(self):
        obj = UtilityLeverage()

        for i in range(20):
            for j in range(30):
                mu = i / 100  # 0% to 20%
                sigma = (j + 1) / 100  # 1% to 30%
                one_year_utility = obj.expected_utility(leverage=1)
                n_year_utility = obj.expected_utility_after_n_years(
                    num_years=1, pure_time_preference=0)
                self.assertAlmostEqual(one_year_utility, n_year_utility)


# TODO: these are wrong b/c they don't account for the fact the growth rate is volatile.
# 2020-05-11

# TODO: wrong because discount rate goes outside the utility function. but I'm
# confused b/c u(1) = 0 and applying any discount makes it still 0.
# we want the value of r such that u(c*e^r)*e^-delta = u(c) for any value of c

# # typical investor
# obj = UtilityLeverage(rra=2.0, mu=0.03, sigma=0.13)
# print(obj.risk_free_return_for_utility(obj.expected_utility(leverage=1)))

# # altruist with same portfolio
# obj = UtilityLeverage(rra=1.0, mu=0.03, sigma=0.13)
# print(obj.risk_free_return_for_utility(obj.expected_utility(leverage=1)))

# # altruist with better portfolio
# obj = UtilityLeverage(rra=1.0, mu=0.04, sigma=0.12)
# print(obj.risk_free_return_for_utility(obj.expected_utility(leverage=1)))

# # altruist with better portfolio, optimally leveraged
# obj = UtilityLeverage(rra=1.0, mu=0.04, sigma=0.12)
# print(obj.equivalent_risk_free_return())

# for i in range(100):
#     obj = UtilityLeverage(rra=1.0, mu=i/1000.0, sigma=0.12)
#     print("{}\t".format(obj.equivalent_risk_free_return()['risk-free'],)

# obj = UtilityLeverage(rra=1.5, mu=0.03, sigma=0.13)
# print(obj.optimal_consumption_rate(discount_rate=0.02, leverage=obj.optimal_leverage().x))


# print(samuelson_share(0.016, 0.04))
print(samuelson_share(0.045, 0.12))
print(samuelson_share(0.09, 0.13))
print(samuelson_share(0.04, 0.11))
print(samuelson_share(0.07, 0.10))