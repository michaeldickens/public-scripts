"""

daf-utility.py
--------------

Author: Michael Dickens <mdickens93@gmail.com>
Created: 2020-10-13

"""

import random
from dataclasses import dataclass

import numpy as np

@dataclass
class Portfolio():
    rra: float = 2.5
    mu: float = 0.03
    sigma: float = 0.15
    leverage: float = 1
    fee: float = 0
    tax_rate: float = 0
    risk_of_ruin: float = 0

    def utility_function(self, x):
        if self.rra == 1:
            return np.log(x)
        return (x**(1 - self.rra) - 1) / (1 - self.rra)

    def marginal_utility(self, ret):
        return np.exp(ret)**(-self.rra)

    def random_outcome(self):
        '''Return utility'''
        wealth = 1
        ret = np.random.normal(loc=self.mu, scale=self.sigma)
        if random.random() < self.risk_of_ruin:
            wealth = 0
        wealth *= 1 - self.fee
        old_wealth = wealth
        wealth *= np.exp(self.leverage * ret)
        wealth -= (wealth - old_wealth) * self.tax_rate
        return wealth * self.marginal_utility(ret)

    def random_outcome_v2(self, external_wealth=100):
        '''This version doesn't assume utility of wealth is linear.

        TODO: I'm getting results that don't make sense
        '''
        wealth = 100
        ret = np.random.normal(loc=self.mu, scale=self.sigma)
        if random.random() < self.risk_of_ruin:
            wealth = 0
        wealth *= 1 - self.fee
        old_wealth = wealth
        wealth *= np.exp(self.leverage * ret)
        wealth -= (wealth - old_wealth) * self.tax_rate

        optimal_leverage = (self.mu + self.sigma**2 / 2) / (self.sigma**2 * self.rra)

        external_wealth *= np.exp(optimal_leverage * ret)
        total_utility = self.utility_function(wealth + external_wealth)
        external_utility = self.utility_function(external_wealth)
        return total_utility - external_utility

    def monte_carlo(self, num_simulations=50000):
        '''
        Monte Carlo simulation to find expected utility.

        This is obsolete now that I've written an analytic solution.
        '''
        # All sims use the same seed, that way it's as if they are seeing the same outcomes
        seed = 2
        random.seed(seed)
        np.random.seed(seed)

        simulations = [self.random_outcome_v2() for _ in range(num_simulations)]
        expected_utility = np.mean(simulations)
        stderr = np.std(simulations) / np.sqrt(num_simulations)
        print("{} expected utility (stderr: {})".format(expected_utility, stderr))

    def expected_utility(self, external_wealth=1):
        '''Analytically compute expected utility'''
        gross_utility = external_wealth**(-self.rra) * np.exp((self.leverage - self.rra) * self.mu + (self.leverage - self.rra)**2 * self.sigma**2 / 2)
        return (gross_utility * (1 - self.tax_rate) + self.tax_rate) * (1 - self.fee) * (1 - self.risk_of_ruin)

    def expected_utility_v2(self, external_wealth=100):
        '''Don't assume linear utility'''

        # Technically, the way I combine portfolios assumes the overall
        # portfolio has a fixed amount of leverage, not that each individual
        # portfolio does. That means the investor's leverage must vary
        # continuously

        wealth = 1 - self.fee

        self.alpha = self.mu + self.sigma**2 / 2
        optimal_leverage = self.alpha / (self.sigma**2 * self.rra)
        total_wealth = external_wealth + wealth
        # TODO: changing external leverage from optimal_leverage to 1 results
        # in decreasing `total_ev - external_ev` instead of increasing it (for
        # some inputs). seems wrong
        external_leverage = optimal_leverage
        net_leverage = (external_leverage * external_wealth + self.leverage * wealth) / total_wealth
        alpha = net_leverage * self.alpha + np.log(total_wealth)
        sigma = net_leverage * self.sigma
        external_alpha = external_leverage * self.alpha + np.log(external_wealth)
        external_sigma = external_leverage * self.sigma

        external_ev = np.exp((1 - self.rra)*external_alpha - self.rra * (1 - self.rra) * external_sigma**2 / 2) / (1 - self.rra)
        ruin_ev = external_ev
        success_ev = np.exp((1 - self.rra)*alpha - self.rra * (1 - self.rra) * sigma**2 / 2) / (1 - self.rra)
        total_ev = ruin_ev * self.risk_of_ruin + success_ev * (1 - self.risk_of_ruin)

        print(total_ev, total_ev - external_ev)


daf = Portfolio(fee=0.006, risk_of_ruin=0.01)

# This assumes you never realize capital gains, but that you have to pay taxes on dividends and dividends are consistent (even in down years), so the tax behaves like a fixed fee
# taxable = Portfolio(rra=1.2, mu=0.03, sigma=0.10, leverage=1.3, fee=0.03 / 2 * 0.2, tax_rate=0, risk_of_ruin=0.03)
taxable = Portfolio(fee=0.02 * 0.15, risk_of_ruin=0.03)

# print("taxable:", taxable.monte_carlo())
# print("DAF:", daf.monte_carlo())

def get_both(portfolio):
    external_wealth = 10000
    print("v1:", portfolio.expected_utility(external_wealth))
    portfolio.expected_utility_v2(external_wealth)

get_both(Portfolio(fee=0.006, risk_of_ruin=0.01, leverage=1))
get_both(Portfolio(fee=0.003, risk_of_ruin=0.03, leverage=2.5))
