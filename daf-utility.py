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
    rra: float
    mu: float
    sigma: float
    leverage: float
    fee: float
    tax_rate: float
    deep_risk: float

    def marginal_utility(self, ret):
        return np.exp(ret)**(-self.rra)

    def run_simulation(self):
        '''Return utility'''
        wealth = 100
        ret = np.random.normal(loc=self.mu, scale=self.sigma)
        if random.random() < self.deep_risk:
            wealth = 0
        wealth *= 1 - self.fee
        old_wealth = wealth
        wealth *= np.exp(self.leverage * ret)
        wealth -= (wealth - old_wealth) * self.tax_rate
        return wealth * self.marginal_utility(ret)

    def monte_carlo(self, num_simulations=50000):
        seed = 0
        random.seed(seed)
        np.random.seed(seed)

        simulations = [self.run_simulation() for _ in range(num_simulations)]
        expected_utility = np.mean(simulations)
        stderr = np.std(simulations) / np.sqrt(num_simulations)
        print("{} expected utility (stderr: {})".format(expected_utility, stderr))

daf = Portfolio(rra=1.2, mu=0.03, sigma=0.10, leverage=1, fee=0.006, tax_rate=0, deep_risk=0.01)

# This assumes you never realize capital gains, but that you have to pay taxes on dividends and dividends are consistent (even in down years), so the tax behaves like a fixed fee
taxable = Portfolio(rra=1.2, mu=0.03, sigma=0.10, leverage=1.3, fee=0.03 / 2 * 0.2, tax_rate=0, deep_risk=0.02)

daf.monte_carlo()
taxable.monte_carlo()

# expected utility monotonically decreases with increasing fees, taxes, deep risk
# monotonically increases with increasing leverage
