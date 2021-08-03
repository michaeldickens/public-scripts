"""

ai_safety_now_later.py
----------------------

Author: Michael Dickens <michael@mdickens.me>
Created: 2021-07-29

Input variables are initialized at the top of the AISafetyModel class. These
can be freely changed.

"""

from typing import Callable, List, Tuple

import numpy as np
from scipy import integrate, optimize, stats


class AISafetyModel:
    # starting_capital and required_research_median are in billions of dollars
    # (although technically it doesn't matter as long as they use the same
    # scale).
    starting_capital = 5
    required_research_median = 10
    required_research_sigma = np.log(5) # log(n) means n times larger is a
                                        # 1-sigma event, n^2 times larger is a
                                        # 2-sigma event, etc.
    timeline_median = 3  # decades
    timeline_sigma = np.log(2)
    investment_return = 0.03  # market return minus research cost growth
    num_decades = 20  # The model solves the optimization problem over this
                      # many decades, ignoring any decades after.
    p_agi_matters = 1  # Allow for the possibility that AGI turns out not to be
                       # dangerous, and it's friendly even without any AI
                       # safety research being done.

    def p_spending_is_sufficient(
            self,
            spending: float,
    ):
        '''
        Return the probability that `spending` is enough research spending to avert
        unfriendly AI.
        '''
        return stats.lognorm.cdf(
            spending, self.required_research_sigma, scale=self.required_research_median
        )

    def run_agi_spending(self, decade, spending_schedule):
        '''
        Helper function to find remaining capital and total spending up to `decade`
        if following `spending_schedule`.
        '''
        capital = self.starting_capital
        spending = 0
        for y in range(decade + 1):
            spending += capital * spending_schedule[y]
            capital *= (1 - spending_schedule[y]) * (1 + self.investment_return)**10
        return capital, spending

    def p_sufficient_research(
            self,
            spending_schedule,
            verbose=False,
    ) -> float:
        '''
        Return the probability that, given a particular spending schedule,
        enough research will be done to avert unfriendly AI.
        '''

        def total_spending_as_of(decade):
            return self.run_agi_spending(decade, spending_schedule)[1]

        spending_schedule = (
            list(spending_schedule)
            +
            [0 for _ in range(self.num_decades - len(spending_schedule))]
        )

        numerator = 0
        denominator = 0
        for decade in range(self.num_decades):
            p_agi = stats.lognorm.pdf(decade + 0.5, self.timeline_sigma, scale=self.timeline_median)
            numerator += (
                p_agi * self.p_spending_is_sufficient(total_spending_as_of(decade))
            )
            if verbose:
                print("{}: P(AGI) = {:.3f}, spending = {:>3.0f}%, cumulative spending = {:>6.0f}, P(sufficient|AGI) = {:.3f}".format(
                    2020 + 10*decade,
                    p_agi,
                    100 * spending_schedule[decade],
                    total_spending_as_of(decade),
                    self.p_spending_is_sufficient(total_spending_as_of(decade)))
                )
            denominator += p_agi

        return numerator / denominator

    def post_agi_utility(self, decade, spending_schedule):
        '''
        Return the expected utility of any capital left over after the
        development of AGI.
        '''
        capital = self.run_agi_spending(decade, spending_schedule)[0]
        discount = 0.001

        investment_ret = 0.1

        return np.log(capital)

    def expected_utility(self, spending_schedule):
        '''
        Return the expected utility of a given spending schedule.
        '''
        def total_spending_as_of(decade):
            return self.run_agi_spending(decade, spending_schedule)[1]

        spending_schedule = (
            list(spending_schedule)
            +
            [0 for _ in range(self.num_decades - len(spending_schedule))]
        )

        numerator = 0
        denominator = 0
        for decade in range(self.num_decades):
            p_agi_now = stats.lognorm.pdf(
                decade + 0.5, self.timeline_sigma, scale=self.timeline_median)
            p_sufficient = self.p_spending_is_sufficient(total_spending_as_of(decade))
            utility = p_agi_now * (
                (
                    self.p_agi_matters
                    * (0 + p_sufficient * self.post_agi_utility(decade, spending_schedule))
                )
                +
                (1 - self.p_agi_matters) * self.post_agi_utility(decade, spending_schedule)
            )
            numerator += utility
            denominator += p_agi_now

        return numerator / denominator

    def optimal_spending_schedule(self):
        '''
        Calculate and print the spending schedule that maximizes expected utility.
        '''
        num_to_opt = self.num_decades
        bounds_constraint = optimize.LinearConstraint(
            np.identity(num_to_opt),
            lb=[0 for _ in range(num_to_opt)],
            ub=[1 for _ in range(num_to_opt)],
        )
        opt = optimize.minimize(
            lambda schedule: -self.expected_utility(schedule),
            [0.1 for _ in range(num_to_opt)],
            constraints=[bounds_constraint],
        )

        # If spending=1 at any point, the optimizer will put in random values
        # afterward because those values don't actually affect anything. Set
        # them all to 0 to make the output easier to read.
        for i in range(len(opt.x)):
            if opt.x[i] == 1:
                for j in range(i + 1, len(opt.x)):
                    opt.x[j] = 0

        print(opt)
        print()
        self.p_sufficient_research(opt.x, verbose=True)


AISafetyModel().optimal_spending_schedule()
