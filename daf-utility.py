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
    risk_of_ruin: float

    def marginal_utility(self, ret):
        return np.exp(ret)**(-self.rra)

    def run_simulation(self):
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

    def monte_carlo(self, num_simulations=50000):
        seed = 0
        random.seed(seed)
        np.random.seed(seed)

        simulations = [self.run_simulation() for _ in range(num_simulations)]
        expected_utility = np.mean(simulations)
        stderr = np.std(simulations) / np.sqrt(num_simulations)
        print("{} expected utility (stderr: {})".format(expected_utility, stderr))

    def expected_utility(self):
        gross_utility = np.exp((self.leverage - self.rra) * self.mu + (self.leverage - self.rra)**2 * self.sigma**2 / 2)
        return (gross_utility * (1 - self.tax_rate) + self.tax_rate) * (1 - self.fee) * (1 - self.risk_of_ruin)

daf = Portfolio(rra=1.2, mu=0.03, sigma=0.10, leverage=1, fee=0.006, tax_rate=0, risk_of_ruin=0.01)

# This assumes you never realize capital gains, but that you have to pay taxes on dividends and dividends are consistent (even in down years), so the tax behaves like a fixed fee
# taxable = Portfolio(rra=1.2, mu=0.03, sigma=0.10, leverage=1.3, fee=0.03 / 2 * 0.2, tax_rate=0, risk_of_ruin=0.03)
taxable = Portfolio(rra=1.2, mu=0.03, sigma=0.10, leverage=1, fee=0.003, tax_rate=0, risk_of_ruin=0.01)

print("DAF:", daf.expected_utility())
print("taxable:", taxable.expected_utility())

# expected utility monotonically decreases with increasing fees, taxes, deep risk
# monotonically increases with increasing leverage
