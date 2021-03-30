"""

three-pronged-choice.py
-----------------------

Author: Michael Dickens <mdickens93@gmail.com>
Created: 2020-09-04

"""

import numpy as np
from scipy import optimize


def utility_of_consumption(eta: float, c: float) -> float:
    if eta == 1:
        return np.log(c)
    else:
        return (c**(1 - eta) + 1) / (1 - eta)


def final_utility(delta_f, r, eta, budget):
    if eta == 1:
        return (delta_f * np.log(budget * delta_f) + r - delta_f) / delta_f**2
    else:
        return eta*(budget / eta)**(1 - eta) / (1 - eta) / (r*eta - r + delta_f)**eta


def optimize_three_pronged_choice_fix_me(p, delta_f, r, eta, budget_0):
    # At attempt at dynamic programming. Calculate final spending and work
    # backward

    # TODO: this will always spend $0 on XRR, because that means the previous
    # period had the lowest possible discount rate
    def optimize_final_budget(budget_f):
        undiscounted_utility = final_utility(delta_f, r, eta, budget_f)
        discount_rate = delta_f
        budget = budget_f

        while discount_rate < delta_0:
            accum_spending = delta_0 / discount_rate - 1
            prev_accum_spending = final_accum_spending - xrr
            prev_discount_rate = delta_0 / (1 + prev_accum_spending)
            utility_fn = lambda consumption, xrr: (
                utility_of_consumption(eta, consumption)
                + (1 - prev_discount_rate) * undiscounted_utility
            )

            initial_guess = [budget * 0.1, budget * 0.1]
            opt = optimize.minimize(utility_fn, initial_guess, constraints=TODO)


def optimize_single_period(p, delta_f, r, eta, budget):
    '''
    See notebook (date 9/4) for explanation

    p: initial probability of extinction
    delta_f: final probability of extinction (different letter used to make it more visually distinct)
    r: total interest rate between initial and final choice
    eta: elasticity of marginal utility of consumption
    budget: total budget, normalized such that 1 unit of spending halves x-risk
    '''
    def utility_fn(input_vector):
        consumption = input_vector[0] * budget
        xrr = input_vector[1] * budget
        discount_rate = p / (1 + xrr)
        undiscounted_final_utility = final_utility(
            delta_f, r, eta, np.exp(r) * (budget - consumption - xrr)
        )
        return utility_of_consumption(eta, consumption) + (1 - discount_rate) * undiscounted_final_utility

    initial_guess = [0.1, 0.1]
    spending_sum_constraint = optimize.LinearConstraint(
        [1, 1],
        lb=[0],
        ub=[1],
    )
    bounds_constraint = optimize.LinearConstraint(
        np.identity(2),  # identity matrix
        lb=[0, 0],
        ub=[1, 1],
    )

    # opt = optimize.minimize(
    #     lambda x: -utility_fn(x), initial_guess,
    #     # constraints=[spending_sum_constraint, bounds_constraint]
    #     constraints=[spending_sum_constraint]
    # )

    # shitty maximize
    x = initial_guess
    prev_max = utility_fn(x)
    improving = True
    step_size = [0.00001, 0.0001]
    while improving:
        improving = False
        for i in range(2):
            if x[i] - step_size[i] < 0:
                continue
            x[i] -= step_size[i]
            if utility_fn(x) > prev_max:
                improving = True
                prev_max = utility_fn(x)
            else:
                x[i] += step_size[i]

            if x[i] + step_size[i] > 1:
                continue
            x[i] += step_size[i]
            if utility_fn(x) > prev_max:
                improving = True
                prev_max = utility_fn(x)
            else:
                x[i] -= step_size[i]

    print("{} -> {} or {}".format(x, prev_max, utility_fn(x)))


def optimize_single_period_no_present_consumption(p, delta_f, r, eta, budget, magic_risk_vanish):
    '''
    No choice about how much to consume now.

    magic_risk_vanish: If True, x-risk magically vanishes and drops to its
      lowest rate. If False, x-risk permanently stays at whatever level you set
      it to (bounded below by delta_f).
    '''
    def utility_fn(xrr_prop):
        if xrr_prop < 0:
            return -1e99
        xrr = xrr_prop * budget
        discount_rate = p / (1 + xrr)
        if not magic_risk_vanish and discount_rate >= delta_f:
            final_discount = discount_rate
        else:
            final_discount = delta_f
        undiscounted_final_utility = final_utility(
            final_discount, r, eta, np.exp(r) * (budget - xrr)
        )
        return (1 - discount_rate) * undiscounted_final_utility

    opt = optimize.minimize_scalar(lambda x: -utility_fn(x))
    print(opt)


def optimize_n_periods(num_periods, r, eta, pure_discount, epsilon, beta, years_per_period=100):
    def utility_fn(xs):
        c = [xs[2*i] for i in range(num_periods)]
        s = [xs[2*i + 1] for i in range(num_periods)]

        # scale by investment return
        for i in range(1, num_periods):
            c[i] *= (1 + r)**(i * years_per_period)
            s[i] *= (1 + r)**(i * years_per_period)

        # utility_floor = -utility_of_consumption(eta, 0.01)
        utility_floor = 0

        # Note: This doesn't really make sense because higher consumption at
        # time t decreases the probability of there being consumption at time
        # t. First one should really be t-1 instead, but then the two-period
        # model is too sparse
        initial_x_risk = 0.001
        x_risk = [
            min(1, initial_x_risk * sum(c[0:(i+1)])**epsilon * sum(s[0:(i+1)])**beta)
            for i in range(num_periods)
        ]

        return sum([
            np.product([
                ((1 - pure_discount) * (1 - x_risk[j]))**years_per_period
                for j in range(i+1)
            ])
            * (utility_of_consumption(eta, c[i]) + utility_floor)
            for i in range(num_periods)
        ])

    bounds_constraint = optimize.LinearConstraint(
        np.identity(2 * num_periods),  # identity matrix
        lb=[0.0, 0] * num_periods,
        ub=[1] * (2 * num_periods),
    )

    sum_constraint = optimize.LinearConstraint(
        np.array([1] * (2 * num_periods)),
        lb=0,
        ub=1,
    )

    opt = optimize.minimize(
        lambda xs: -utility_fn(xs),
        x0=np.array([1.0 / (2 * num_periods)] * (2 * num_periods)),
        constraints=[bounds_constraint, sum_constraint],
    )
    print(opt)


optimize_n_periods(2, r=0.05, eta=1.5, pure_discount=0.01, epsilon=0.5, beta=0.7)
print("\n")
optimize_n_periods(2, r=0.05, eta=1.5, pure_discount=0.00001, epsilon=0.5, beta=0.7)


# Some observations:
# 1. present consumption always goes close to 0 (hence why I wrote a new function to ignore it)
# 2. if magic_risk_vanish=False and budget is too small to get x-risk down to delta_f, it's best to spend almost all money on x-risk reduction because it continues to provide linear returns
