'''

ramsey.py
---------

Author: Michael Dickens
Created: 2020-06-24

'''

import math

import numpy as np
from scipy import optimize
from scipy.special import gamma, gammaincc


def utility_function(eta: float, c: float) -> float:
    if eta == 1:
        return np.log(c)
    else:
        return (c**(1 - eta) + 1) / (1 - eta)


def consumption_with_declining_discount(alpha, delta, r, eta):
    '''
    alpha: Rate of decline of discount rate
    delta: Minimum discount rate
    r: Interest rate
    eta: Elasticity of marginal utility of consumption

    example values: alpha=0.01, delta=0.001, r=0.05, eta=1
    '''
    consumption_coef =  (
        ((delta + r*eta - r) / eta)**(1 - alpha/eta)
        # TODO: unclear if the gamma(1 - alpha/eta) term should be there,
        # Python's definition of gammaincc is weird
        / (gamma(1 - alpha/eta) * gammaincc(1 - alpha/eta, (delta + r*eta - r)/eta))
    )
    def consumption(t):
        return (
            consumption_coef
            * (1 + t)**(-alpha/eta)
            # ((delta + r*eta - r)/eta) / math.exp((r - delta)/eta)
            * math.exp((r - delta)/eta * (1 + t))
        )

    prev_capital = 1

    for i in range(1, 1 + 1000000):
        t = i / 10000
        dt = t / i
        consum = consumption(t)
        capital = prev_capital * (1 + dt * r) - dt * consum
        capital_growth_rate = (capital - prev_capital) / prev_capital / dt
        consumption_rate = consum / capital
        consumption_growth_rate = (consum - consumption(t-dt)) / consum / dt
        # TODO: This starts to become inaccurate over time, I think due to floating point rounding issues
        if i % int(100/dt) == 2 or (i % int(10/dt) == 2 and i*dt < 100):
            print("{}\t{}\t{}\t{}\t{}".format(t, capital, capital_growth_rate, consumption_rate, consumption_growth_rate))

        prev_capital = capital


def optimize_three_pronged_choice_multiperiod(alpha, delta, r, eta, num_periods=300):
    '''
    Optimize where each period has a three-pronged choice (invest, consume, reduce discount rate).

    alpha: Rate at which it becomes harder to reduce the discount rate (negative)
    delta: Minimum discount rate
    r: Interest rate
    eta: Elasticity of marginal utility of consumption
    '''
    # num_periods=300 takes 10 minutes
    def total_utility(xs):
        capital = 1
        spending_on_discount = 1
        discount_factor = 1
        utility = 0
        for i in range(num_periods):
            utility += discount_factor * utility_function(eta, capital * xs[num_periods + i])
            discount_factor *= 1 - delta * spending_on_discount**(-alpha)
            spending_on_discount += capital * xs[i]
            capital *= 1 - xs[i] - xs[num_periods + i]
            capital *= 1 + r

        return -utility  # negative so we can minimize it

    # first num_periods entries are spending on discount rate reduction
    # second num_periods are spending on general consumption
    initial_guess = np.array([0.5 / num_periods for _ in range(2 * num_periods)])

    # spend no more than 100% in each period
    spending_sum_constraint = optimize.LinearConstraint(
        np.array([[1 if i == j or i == j + num_periods else 0 for i in range(2*num_periods)]
                  for j in range(num_periods)]),
        lb = [0] * num_periods, ub = [1] * num_periods
    )

    # cannot spend less than 0% or more than 100% on each thing
    bounds_constraint = optimize.LinearConstraint(
        np.identity(2 * num_periods),  # identity matrix
        lb=[0] * 2 * num_periods, ub=[1] * 2 * num_periods
    )

    opt = optimize.minimize(total_utility, initial_guess, constraints=[spending_sum_constraint, bounds_constraint])
    for i in range(num_periods):
        print("x({}) = {}, c({}) = {}".format(i, opt.x[i], i, opt.x[num_periods + i]))


def optimize_three_pronged_choice_fixed(alpha, delta, r, eta, num_periods=500):
    '''
    Optimize where each period has a three-pronged choice (invest, consume,
    reduce discount rate). As a simplifying assumption require that the
    proportion of capital consumed in each period remains fixed, and likewise
    for spending on reducing the discount rate.

    '''
    def total_utility(xs):
        capital = 100
        spending_on_discount = 1
        discount_factor = 1
        utility = 0
        for i in range(num_periods):
            utility += discount_factor * utility_function(eta, capital * xs[1])

            # what if at a certain point, the discount rate becomes fixed at a
            # low value and you can't reduce it anymore?
            # if i < 100:
            #     discount_factor *= 1 - delta * spending_on_discount**(-alpha)
            # else:
            #     discount_factor *= 1 - 0.01/100
            discount_factor *= 1 - delta * spending_on_discount**(-alpha)
            spending_on_discount += capital * xs[0]
            capital *= 1 - xs[0] - xs[1]
            capital *= 1 + r

        return -utility  # negative so we can minimize it

    initial_guess = np.array([0.5, 0.5])

    # no more than 100% each period
    spending_sum_constraint = optimize.LinearConstraint(
        np.array([1, 1]).reshape(1, -1), lb=[0], ub=[1]
    )

    # cannot spend less than 0% or more than 100% on each thing
    bounds_constraint = optimize.LinearConstraint(np.array([[1, 0], [0, 1]]), lb=[0, 0], ub=[1, 1])

    opt = optimize.minimize(total_utility, initial_guess, constraints=[spending_sum_constraint, bounds_constraint])
    print("x(t) = {:.4f}%, c(t) = {:.4f}%".format(100 * opt.x[0], 100 * opt.x[1]))
    print("{}".format(-total_utility(opt.x)))


def optimize_discount_multiperiod(alpha, delta, r, eta, num_periods=100):
    g = (r - delta) / eta
    utility_lower_limit = abs(utility_function(eta, 1))

    def total_utility(xs):
        capital = 1
        spending_on_discount = 1
        discount_factor = 1
        utility = 0
        for i in range(num_periods):
            utility += discount_factor * (utility_lower_limit + utility_function(eta, (1 + g)**i))
            # utility += discount_factor * 1
            discount_factor *= 1 - delta * spending_on_discount**(-alpha)
            spending_on_discount += capital * xs[i]
            capital *= 1 - xs[i]
            capital *= 1 + r

        return -utility  # negative so we can minimize it

    # first num_periods entries are spending on discount rate reduction
    # second num_periods are spending on general consumption
    initial_guess = np.array([0.1] * num_periods)

    # cannot spend less than 0% or more than 100% in each period
    bounds_constraint = optimize.LinearConstraint(
        np.identity(num_periods),  # identity matrix
        lb=[0] * num_periods, ub=[1] * num_periods
    )

    opt = optimize.minimize(total_utility, initial_guess, constraints=[bounds_constraint])
    for i in range(num_periods):
        print("x({}) = {}".format(i, opt.x[i]))
    print("total utility = {}".format(-total_utility(opt.x)))
    copy = [x if i != 0 else 1 for i, x in enumerate(opt.x)]
    print("fun utility = {}".format(-total_utility(copy)))


def optimize_discount_fixed(alpha, delta, r, eta, num_periods=10000):
    g = (r - delta) / eta
    utility_lower_limit = abs(utility_function(eta, 1))

    def total_utility(xs):
        capital = 100
        spending_on_discount = 1
        discount_factor = 1
        utility = 0
        for i in range(num_periods):
            # add a constant to utility function so it's not negative, b/c that messes up the optimizer
            utility += discount_factor * (utility_lower_limit + utility_function(eta, (1 + g)**i))
            discount_factor *= 1 - delta * spending_on_discount**(-alpha)
            spending_on_discount += capital * xs[0]
            capital *= 1 - xs[0]
            capital *= 1 + r

        return -utility  # negative so we can minimize it

    initial_guess = np.array([0.01])

    # cannot spend less than 0% or more than 100%
    bounds_constraint = optimize.LinearConstraint([1], lb=[0], ub=[1])

    opt = optimize.minimize(total_utility, initial_guess, constraints=[bounds_constraint])
    print("x(t) = {:.4f}%".format(100 * opt.x[0]))
    print("{}".format(-total_utility(opt.x)))
    for x in [0.1, 1, 10, 100]:
        print("{}: {}".format(x, -total_utility([x/100])))


def optimize_discount_transient(alpha, delta, r, eta, num_periods=1000):
    '''
    Optimize spending on reducing the discount rate when reductions are
    transient--that is, spending only determines the discount rate of the
    next period.
    '''
    g = (r - delta) / eta
    utility_lower_limit = abs(utility_function(eta, 1))

    def total_utility(xs):
        capital = 100
        discount_factor = 1
        utility = 0
        for i in range(num_periods):
            # add a constant to utility function so it's not negative, b/c that messes up the optimizer
            utility += discount_factor * (utility_lower_limit + utility_function(eta, (1 + g)**i))
            discount_factor *= 1 - delta * (1 + capital * xs[0])**(-alpha)
            capital *= 1 - xs[0]
            capital *= 1 + r

        return -utility  # negative so we can minimize it

    initial_guess = np.array([0.01])

    # cannot spend less than 0% or more than 100%
    bounds_constraint = optimize.LinearConstraint([1], lb=[0], ub=[1])

    # note: it would be more efficient to solve this using dynamic programming,
    # but that doesn't seem necessary given the current performance
    opt = optimize.minimize(total_utility, initial_guess, constraints=[bounds_constraint])
    print("x(t) = {:.4f}%".format(100 * opt.x[0]))
    print("{}".format(-total_utility(opt.x)))
    for x in [0.1, 1, 10, 100]:
        print("{}: {}".format(x, -total_utility([x/100])))


def optimize_consumption_cumulative(delta, r, eta, num_periods=100):
    def total_utility(xs):
        capital = 100
        cumulative_consumption = 0
        utility = 1
        for t in range(num_periods):
            consumption = capital * xs[t]
            cumulative_consumption += consumption
            utility += (1 - delta)**t * utility_function(eta, cumulative_consumption)
            capital -= consumption
            capital *= 1 + r

        return -utility  # negative so we can minimize it

    initial_guess = np.array([1 / num_periods] * num_periods)

    # cannot spend less than 0% or more than 100% on any period
    bounds_constraint = optimize.LinearConstraint(
        np.identity(num_periods),  # identity matrix
        lb=[0] * num_periods, ub=[1] * num_periods
    )

    opt = optimize.minimize(total_utility, initial_guess, constraints=[bounds_constraint])
    for i in range(num_periods):
        print("x({}) = {}".format(i, opt.x[i]))
    print("total utility = {}".format(-total_utility(opt.x)))
    copy = [x if i != 0 else 1 for i, x in enumerate(opt.x)]
    print("fun utility = {}".format(-total_utility(copy)))


# optimize_three_pronged_choice_multiperiod(alpha=1, delta=0.001, r=0.05, eta=1)
# optimize_three_pronged_choice_fixed(alpha=1, delta=0.02, r=0.05, eta=1)
# optimize_discount_fixed(alpha=1, delta=0.02, r=0.05, eta=1.35)
optimize_consumption_cumulative(delta=0.02, r=0.05, eta=1.35)
