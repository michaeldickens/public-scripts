"""

x-risk-now-later.py
-------------------

Author: Michael Dickens <mdickens93@gmail.com>
Created: 2020-07-19

"""

from typing import Callable

from scipy import optimize
import sympy


def updated_x_risk(initial_x_risk: float, spending: float) -> float:
    return initial_x_risk / (1 + spending)


def compare(pair):
    x, y = pair
    if x == y:
        print("{} = {}".format(x, y))
    else:
        print("{} {} {} {} (ratio {})".format(x, "<" if x < y else ">", y, "Later" if x < y else "Now", x / y))


def permanent_reduction_binary(
        period_utility: Callable[[float], float],
        convergence_value: Callable[[float], float],
        initial_x_risk: float,
        exogenous_spending: float,
        interest_rate: float,
        periods_til_later=100,
        budget=1,
) -> (float, float):
    '''
    Calculate the utility of spending now vs. later on reducing x-risk where
    spending permanently reduces x-risk.

    After the first `periods_til_later` periods, there will be
    `exogenous_spending` spending on x-risk.

    period_utility(t): utility at time t
    convergence_value(x_risk): convergence point of
    `sum([(1 - x_risk)**t * period_utility(t) for t in range(infinity)])`

    return: (utility of spending now, utility of spending later)
    '''

    spending_now = budget
    spending_later = spending_now * (1 + interest_rate)**periods_til_later
    x_risk_now = updated_x_risk(initial_x_risk, spending_now)

    utility_now = (
        sum([(1 - x_risk_now)**t * period_utility(t) for t in range(periods_til_later)])
        + ((1 - x_risk_now)**periods_til_later
           * convergence_value(updated_x_risk(initial_x_risk, spending_now + exogenous_spending)))
    )

    utility_later = (
        # from now til later, discount at the initial rate
        sum([(1 - initial_x_risk)**t * period_utility(t) for t in range(periods_til_later)])

        # from later til forever, discount at the reduced rate
        + ((1 - initial_x_risk)**periods_til_later
           * convergence_value(updated_x_risk(initial_x_risk, spending_later + exogenous_spending)))
    )

    return (utility_now, utility_later)


def permanent_reduction_continuous(
        period_utility: Callable[[float], float],
        convergence_value: Callable[[float], float],
        initial_x_risk: float,
        exogenous_spending: float,
        interest_rate: float,
        periods_til_later=100,
        budget=1,
) -> (float, float):
    '''
    Choose the optimal amount to spend now.

    After the first `periods_til_later` periods, there will be
    some exogenous spending on x-risk.

    period_utility(t): utility at time t
    convergence_value(x_risk): convergence point of
      `sum([(1 - x_risk)**t * period_utility(t) for t in range(infinity)])`
    initial_x_risk: baseline level of x-risk before any spending
    exogenous_spending: amount of spending on x-risk that will occur at period `periods_til_later`

    return: (optimal spending, total utility)
    '''
    def total_utility(spending_now):
        spending_later = (budget - spending_now) * (1 + interest_rate)**periods_til_later
        x_risk_now = updated_x_risk(initial_x_risk, spending_now)
        x_risk_later = updated_x_risk(initial_x_risk, spending_now + spending_later + exogenous_spending)
        utility = (
            sum([(1 - x_risk_now)**t * period_utility(t) for t in range(periods_til_later)])
            + (1 - x_risk_later)**periods_til_later * convergence_value(x_risk_later)
        )
        return utility

    bounds_constraint = optimize.LinearConstraint([1], lb=[0], ub=[budget])
    opt = optimize.minimize(lambda x: -total_utility(x), [budget/2], constraints=[bounds_constraint])
    print("0: {}".format(total_utility(0)))
    print("{}: {}".format(budget, total_utility(budget)))
    return opt.x[0], total_utility(opt.x[0])


def temporary_reduction_binary(
        period_utility: Callable[[float], float],
        convergence_value: Callable[[float], float],
        initial_x_risk: float,
        exogenous_spending: float,
        interest_rate: float,
        periods_til_later=100,
        budget=1,
) -> (float, float):
    '''
    X-risk will be high for the first `periods_til_later` periods, then it will
    be low due to exogenous spending. Spending only reduces x-risk for the next
    `periods_til_later` periods.

    initial_x_risk: baseline x-risk from the current period until `periods_til_later`
    exogenous_spending: spending that occurs repeatedly except in the first
      `periods_til_later` periods, thus serving to pin x-risk at a lower level and
      make marginal spending less valuable

    return: (utility of spending now, utility of spending later)
    '''
    spending_now = budget
    spending_later = spending_now * (1 + interest_rate)**periods_til_later
    x_risk_now = updated_x_risk(initial_x_risk, spending_now)
    x_risk_later = updated_x_risk(initial_x_risk, spending_later + exogenous_spending)
    long_term_x_risk = updated_x_risk(initial_x_risk, exogenous_spending)

    utility_now = (
        # from now til later
        sum([(1 - x_risk_now)**t * period_utility(t) for t in range(periods_til_later)])

        # from later til forever
        + (1 - x_risk_now)**periods_til_later * convergence_value(long_term_x_risk)
    )

    utility_later = (
        # from now til later
        sum([(1 - initial_x_risk)**t * period_utility(t) for t in range(periods_til_later)])

        # from later til 2*later
        + ((1 - initial_x_risk)**periods_til_later
           * sum([(1 - x_risk_later)**t * period_utility(periods_til_later + t)
                  for t in range(periods_til_later)]))

        # from 2*later til forever
        + ((1 - initial_x_risk)**periods_til_later
           * (1 - x_risk_later)**periods_til_later
           * convergence_value(long_term_x_risk))
    )

    print("now1 {}, now2 {} | now-later {}, later-2x {}, 2x-forever {}".format(
        # from now til later
        sum([(1 - x_risk_now)**t * period_utility(t) for t in range(periods_til_later)]),

        # from later til forever
        (1 - x_risk_now)**periods_til_later * convergence_value(long_term_x_risk),


        # from now til later
        sum([(1 - initial_x_risk)**t * period_utility(t) for t in range(periods_til_later)]),

        # from later til 2*later
        ((1 - initial_x_risk)**periods_til_later
           * sum([(1 - x_risk_later)**t * period_utility(periods_til_later + t)
                  for t in range(periods_til_later)])),

        # from 2*later til forever
        ((1 - initial_x_risk)**periods_til_later
           * (1 - x_risk_later)**periods_til_later
           * convergence_value(long_term_x_risk))
    ))

    print(utility_now - utility_later)

    return (utility_now, utility_later)


def temporary_reduction_continuous(
        period_utility: Callable[[float], float],
        convergence_value: Callable[[float], float],
        initial_x_risk: float,
        exogenous_spending: float,
        interest_rate: float,
        periods_til_later=100,
        budget=1,
) -> (float, float):
    '''
    Unlike in `permanent_reduction`, `exogenous_spending` occurs repeatedly
    except for in the first `periods_til_later` periods, thus serving to pin
    x-risk at a lower level and make marginal spending less valuable.
    '''

    def total_utility(spending_now):
        spending_later = (budget - spending_now) * (1 + interest_rate)**periods_til_later
        x_risk_now = updated_x_risk(initial_x_risk, spending_now)
        x_risk_later = updated_x_risk(initial_x_risk, spending_later + exogenous_spending)
        long_term_x_risk = updated_x_risk(initial_x_risk, exogenous_spending)
        utility = (
            # from now til later
            sum([(1 - x_risk_now)**t * period_utility(t) for t in range(periods_til_later)])

            # from later til 2*later
            + ((1 - x_risk_now)**periods_til_later
               * sum([(1 - x_risk_later)**t * period_utility(periods_til_later + t)
                      for t in range(periods_til_later)])
            )

            # from 2*later til forever
            + ((1 - x_risk_now)**periods_til_later
               * (1 - x_risk_later)**periods_til_later
               * convergence_value(long_term_x_risk))
        )
        return utility

    bounds_constraint = optimize.LinearConstraint([1], lb=[0], ub=[budget])
    opt = optimize.minimize(lambda x: -total_utility(x), [budget/2], constraints=[bounds_constraint])
    print("0: {}".format(total_utility(0)))
    print("{}: {}".format(budget, total_utility(budget)))
    return opt.x[0], total_utility(opt.x[0])


def breakeven_interest_rate(budget=1, exogenous_spending=0):
    period_utility = lambda t: 1
    convergence_value = lambda x_risk: 1 / x_risk
    initial_x_risk = 0.002
    f = temporary_reduction_binary
    def func(interest_rate):
        now, later = f(period_utility, convergence_value, initial_x_risk, exogenous_spending, interest_rate, budget=budget)
        return (now - later)**2

    # for some reason this gives the wrong answer for some initial guesses, but
    # gets it right if the initial guess is close enough to 0. maybe step sizes
    # are too big?
    opt = optimize.minimize(func, [0.001])
    res = f(period_utility, convergence_value, initial_x_risk, exogenous_spending, opt.x[0], budget=budget)
    print("Breakeven rate {} gives utility {} (diff {:.0e})".format(opt.x[0], res, (res[0] - res[1])/res[0]))
    # print(f(consumption, convergence_value, initial_x_risk, exogenous_spending, initial_x_risk / (1 - initial_x_risk), budget=budget))


# constant utility
# compare(permanent_reduction_binary(lambda t: 1, lambda x_risk: 1 / x_risk, 0.001, 0, 0.02))

# linear utility (exponentially-increasing consumption with logarithmic utility of consumption)
#compare(permanent_reduction_binary(lambda t: t, lambda x_risk: (1 - x_risk) / x_risk**2, 0.001, 500, 0.02))

# print(temporary_reduction_binary(lambda t: 1, lambda x_risk: 1 / x_risk, 0.001, 100, 0.02))

# compare(permanent_reduction_with_exogenous_spending(lambda t: 1, lambda x_risk: 1 / x_risk, 0.001, 100, 0.02))

# print(temporary_reduction_continuous(lambda t: 1, lambda x_risk: 1 / x_risk, 0.001, 100, 0.02, budget=500))

# breakeven_interest_rate(budget=0.01, exogenous_spending=0)
# breakeven_interest_rate(budget=0.01, exogenous_spending=0.01)
# breakeven_interest_rate(budget=0.01, exogenous_spending=0.1)
# breakeven_interest_rate(budget=0.01, exogenous_spending=1)
# breakeven_interest_rate(budget=0.01, exogenous_spending=10)
# breakeven_interest_rate(budget=0.01, exogenous_spending=100)

# breakeven_interest_rate(budget=100, exogenous_spending=0)
# breakeven_interest_rate(budget=100, exogenous_spending=100)
# breakeven_interest_rate(budget=100, exogenous_spending=1000)
# breakeven_interest_rate(budget=100, exogenous_spending=10000)
# breakeven_interest_rate(budget=100, exogenous_spending=100000)
# breakeven_interest_rate(budget=100, exogenous_spending=1000000)

temporary_reduction_binary(lambda t: 1, lambda x_risk: 1 / x_risk, initial_x_risk=0.001, exogenous_spending=0, interest_rate=0.03, budget=1)
temporary_reduction_binary(lambda t: 1, lambda x_risk: 1 / x_risk, initial_x_risk=0.001, exogenous_spending=0, interest_rate=0.03, budget=0.1)
temporary_reduction_binary(lambda t: 1, lambda x_risk: 1 / x_risk, initial_x_risk=0.001, exogenous_spending=0, interest_rate=0.03, budget=0.01)
temporary_reduction_binary(lambda t: 1, lambda x_risk: 1 / x_risk, initial_x_risk=0.001, exogenous_spending=0, interest_rate=0.03, budget=0.001)
