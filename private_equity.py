"""

private_equity.py
-----------------

Author: Michael Dickens <michael@mdickens.me>
Created: 2022-08-09

"""

import numpy as np

class PrivateEquitySim:
    public_mu = 0.05
    public_sigma = 0.15
    pe_component_mu = 0.005
    pe_component_sigma = 0.2
    pe_beta = 1.5
    rra = 1
    num_years = 500
    rebalance_years = 10
    discount_rate = 0.03
    consumption_rate = discount_rate / rra

    def utility(self, consumption):
        if self.rra == 1:
            return np.log(consumption)
        return (consumption**(1 - self.rra) - 1) / (1 - self.rra)

    def target_allocation(self):
        pe_mu = self.public_mu * self.pe_beta + self.pe_component_mu
        pe_sigma = np.sqrt((self.public_sigma * self.pe_beta)**2 + self.pe_component_sigma**2)

        public_allocation = (self.public_mu + self.public_sigma**2/2) / (self.public_sigma * self.rra)
        pe_allocation = (pe_mu + pe_sigma**2/2) / (pe_sigma * self.rra)

        return (public_allocation, pe_allocation)

    def gen_one_year_return(self):
        public_component = np.random.normal(self.public_mu, self.public_sigma)
        pe_component = np.random.normal(self.pe_component_mu, self.pe_component_sigma)
        public_equity = public_component
        private_equity = self.pe_beta * public_component + pe_component

        return (public_equity, private_equity)

    def run_sample(self, target_allocation=None):
        if target_allocation is None:
            target_allocation = self.target_allocation()
        allocation = [x for x in target_allocation]
        total_utility = 0

        for y in range(self.num_years):
            consumption = sum(allocation) * self.consumption_rate
            allocation[0] -= consumption
            total_utility += (1 - self.discount_rate)**y * self.utility(consumption)

            rets = self.gen_one_year_return()
            allocation[0] *= np.exp(rets[0])
            allocation[1] *= np.exp(rets[1])
            if y % self.rebalance_years == self.rebalance_years - 1:
                total = sum(allocation)
                target_total = sum(target_allocation)
                allocation[0] = total * target_allocation[0] / target_total
                allocation[1] = total * target_allocation[1] / target_total

        return total_utility

    def monte_carlo(self):
        results = [self.run_sample() for _ in range(1000)]
        return np.mean(results)


sim = PrivateEquitySim()
print(sim.target_allocation())
print(sim.monte_carlo())
