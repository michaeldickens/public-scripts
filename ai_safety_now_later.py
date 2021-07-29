"""

ai_safety_now_later.py
----------------------

Author: Michael Dickens <michael@mdickens.me>
Created: 2021-07-29

"""

from typing import Callable, List, Tuple

import numpy as np
from scipy import integrate, optimize, stats


class AISafetyModel:
    # Monetary numbers are in millions of dollars
    starting_capital = 10000
    required_research_median = 10000
    required_research_sigma = np.log(5) # log(10) = 2.3, so sigma=2.3 means 10x is 1 stdev
    timeline_median = 3  # decades
    timeline_sigma = np.log(2)
    investment_return = 0.08  # market return minus research cost growth
    num_decades = 20

    def p_spending_is_sufficient(
            self,
            decade: int,
            spending: float,
    ):
        return stats.lognorm.cdf(
            spending, self.required_research_sigma, scale=self.required_research_median
        )

    def p_sufficient_research(
            self,
            spending_schedule: List[float],
            verbose=False,
    ) -> float:
        def total_spending_as_of(decade):
            capital = self.starting_capital
            spending = 0
            for y in range(decade + 1):
                spending += capital * spending_schedule[y]
                capital *= (1 - spending_schedule[y]) * (1 + self.investment_return)**10
            return spending

        schedule2 = [0 for _ in range(self.num_decades)]
        for i in range(len(spending_schedule)):
            schedule2[i] = spending_schedule[i]
        spending_schedule = schedule2

        numerator = 0
        denominator = 0
        for decade in range(self.num_decades):
            p_agi = stats.lognorm.pdf(decade + 0.5, self.timeline_sigma, scale=self.timeline_median)
            numerator += (
                p_agi * self.p_spending_is_sufficient(decade, total_spending_as_of(decade))
            )
            if verbose:
                print("{}: P(AGI) = {:.3f}, total spending = {:>6.0f}, P(sufficient) = {:.3f}".format(
                    2020 + 10*decade,
                    p_agi,
                    total_spending_as_of(decade),
                    self.p_spending_is_sufficient(decade, total_spending_as_of(decade)))
                )
            denominator += p_agi

        return numerator / denominator

    def optimal_spending_schedule(self):
        num_to_opt = self.num_decades
        bounds_constraint = optimize.LinearConstraint(
            np.identity(num_to_opt),
            lb=[0 for _ in range(num_to_opt)],
            ub=[1 for _ in range(num_to_opt)],
        )
        opt = optimize.minimize(
            lambda schedule: -self.p_sufficient_research(schedule),
            [0.1 for _ in range(num_to_opt)],
            constraints=[bounds_constraint],
        )
        # If spending=1 at any point, all the numbers after it are noise
        for i in range(len(opt.x)):
            if opt.x[i] == 1:
                for j in range(i + 1, len(opt.x)):
                    opt.x[j] = 0

        print(opt)
        self.p_sufficient_research(opt.x, verbose=True)


AISafetyModel().optimal_spending_schedule()
