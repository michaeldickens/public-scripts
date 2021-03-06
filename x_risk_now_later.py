"""

x-risk-now-later.py
-------------------

Author: Michael Dickens <mdickens93@gmail.com>
Created: 2020-07-19

"""

from typing import Callable, Tuple

from scipy import integrate, optimize, stats
import sympy

from leverage import LeverageEnvironment


def updated_x_risk(initial_x_risk: float, spending: float) -> float:
    return initial_x_risk / (1 + spending)


def compare(pair):
    x, y = pair
    if x == y:
        print("{} = {}".format(x, y))
    else:
        print("{} {} {} {} (relative difference {})".format(
            x, "<" if x < y else ">", y, "Later" if x < y else "Now", abs(x - y) / x))


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
    `exogenous_spending` spending
    on x-risk.

    Doubling cost is normalized to 1. Budget and exogenous spending should be
    put relative to doubling cost.

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

    print("now {} + {}, later {} + {}".format(
        sum([(1 - x_risk_now)**t * period_utility(t) for t in range(periods_til_later)]),
        ((1 - x_risk_now)**periods_til_later
         * convergence_value(updated_x_risk(initial_x_risk, spending_now + exogenous_spending))),

        # from now til later, discount at the initial rate
        sum([(1 - initial_x_risk)**t * period_utility(t) for t in range(periods_til_later)]),

        # from later til forever, discount at the reduced rate
        ((1 - initial_x_risk)**periods_til_later
           * convergence_value(updated_x_risk(initial_x_risk, spending_later + exogenous_spending)))

    ))

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
    print("{}: {}".format(opt.x[0], total_utility(opt.x[0])))
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


def permanent_reduction_with_movement_growth_binary_finite(
        period_utility: Callable[[float], float],
        convergence_value: Callable[[float], float],
        initial_x_risk: float,
        movement_growth_rate: float,
        interest_rate: float,
        periods_til_later=100,
        budget=1,
):
    '''
    Calculate the utility of spending now vs. later on reducing x-risk where
    spending permanently reduces x-risk, and where spending now results in
    increasing spending in future periods due to movement growth.

    Specifically, every $1 spent now will cause additional spending to occur
    in the future such that at time t, `(1 + movement_growth_rate)^t` total
    spending has occurred. Spending stops after `periods_til_later` periods
    have passed.

    Doubling cost is normalized to 1. Budget and exogenous spending should be
    put relative to doubling cost.

    period_utility(t): utility at time t
    convergence_value(x_risk): convergence point of
    `sum([(1 - x_risk)**t * period_utility(t) for t in range(infinity)])`

    return: (utility of spending now, utility of spending later)
    '''

    spending_now = budget
    spending_later = spending_now * (1 + interest_rate)**periods_til_later
    x_risk_now = updated_x_risk(initial_x_risk, spending_now)

    utility_now = 0
    discount_factor = 1
    for t in range(periods_til_later):
        utility_now += discount_factor * period_utility(t)
        discount_factor *= 1 - updated_x_risk(initial_x_risk, spending_now * (1 + movement_growth_rate)**t)
    utility_now_100 = utility_now
    utility_now += (
        discount_factor
        * convergence_value(updated_x_risk(
            initial_x_risk, spending_now * (1 + movement_growth_rate)**periods_til_later))
    )

    utility_later = 0
    discount_factor = 1
    for t in range(periods_til_later):
        utility_later += discount_factor * period_utility(t)
        discount_factor *= 1 - initial_x_risk
    utility_later_100 = utility_later
    for t in range(periods_til_later):
        utility_later += discount_factor * period_utility(periods_til_later + t)
        discount_factor *= 1 - updated_x_risk(initial_x_risk, spending_later * (1 + movement_growth_rate)**t)
    utility_later_200 = utility_later
    utility_later += (
        discount_factor
        * convergence_value(updated_x_risk(
            initial_x_risk, spending_later * (1 + movement_growth_rate)**(periods_til_later)))
    )
    print("{:.2f}%:\n\tspending now {}, spending later {}\n\t{}: now 100 {}, now forever {}\n\t{}: later 100 {}, later 200 {}, later forever {}".format(
        interest_rate * 100,
        spending_now, spending_later,
        utility_now, utility_now_100, utility_now - utility_later_100,
        utility_later, utility_later_100, utility_later_200 - utility_later_100, utility_later - utility_later_200
    ))

    return (utility_now, utility_later)


def permanent_reduction_with_movement_growth_binary(
        period_utility: Callable[[float], float],
        convergence_value: Callable[[float], float],
        initial_x_risk: float,
        movement_growth_rate: float,
        interest_rate: float,
        periods_til_later=100,
        budget=1,
):
    '''
    Calculate the utility of spending now vs. later on reducing x-risk where
    spending permanently reduces x-risk, and where spending now results in
    increasing spending in future periods due to movement growth.

    Specifically, every $1 spent now will cause additional spending to occur
    in the future such that at time t, `(1 + movement_growth_rate)^t` total
    spending has occurred. Spending continues occurring indefinitely.

    This function simulates indefinite expenditures by calculating over a
    long but finite time horizon. This could give inaccurate results when
    x-risk is very small.

    Doubling cost is normalized to 1. Budget and exogenous spending should be
    put relative to doubling cost.

    period_utility(t): utility at time t
    convergence_value(x_risk): convergence point of
    `sum([(1 - x_risk)**t * period_utility(t) for t in range(infinity)])`

    return: (utility of spending now, utility of spending later)
    '''

    spending_now = budget
    spending_later = spending_now * (1 + interest_rate)**periods_til_later
    x_risk_now = updated_x_risk(initial_x_risk, spending_now)
    very_long_time = 10000

    utility_now = 0
    discount_factor = 1
    for t in range(very_long_time):
        utility_now += discount_factor * period_utility(t)
        discount_factor *= 1 - updated_x_risk(initial_x_risk, spending_now * (1 + movement_growth_rate)**t)

    utility_later = 0
    discount_factor = 1
    for t in range(periods_til_later):
        utility_later += discount_factor * period_utility(t)
        discount_factor *= 1 - initial_x_risk
    for t in range(very_long_time - periods_til_later):
        utility_later += discount_factor * period_utility(periods_til_later + t)
        discount_factor *= 1 - updated_x_risk(initial_x_risk, spending_later * (1 + movement_growth_rate)**t)

    return (utility_now, utility_later)

def permanent_reduction_binary_with_extra_exogenous_spending(
        period_utility: Callable[[float], float],
        convergence_value: Callable[[float], float],
        initial_x_risk: float,
        exogenous_spending: float,
        interest_rate: float,
        periods_til_later=100,
        extra_exogenous_spending=0,
        extra_periods=100,
        budget=1,
) -> (float, float):
    '''
    After the first round of exogenous spending, a second round of
    `extra_exogenous_spending` spending occurs after an additional
    `extra_periods` periods.

    Note: I wrote this kind of quickly and haven't checked it, so
    the implementation could be wrong.
    '''

    spending_now = budget
    spending_later = spending_now * (1 + interest_rate)**periods_til_later
    x_risk_now = updated_x_risk(initial_x_risk, spending_now)


    utility_now = (
        sum([(1 - x_risk_now)**t * period_utility(t) for t in range(periods_til_later)])

        + ((1 - x_risk_now)**periods_til_later
           * sum([(1 - updated_x_risk(initial_x_risk, spending_now + exogenous_spending))**t * period_utility(t)
                  for t in range(extra_periods)])
        )

        + ((1 - x_risk_now)**periods_til_later
           * (1 - updated_x_risk(initial_x_risk, spending_now + exogenous_spending))**extra_periods
           * convergence_value(updated_x_risk(initial_x_risk, spending_now + exogenous_spending + extra_exogenous_spending)))
    )

    utility_later = (
        # from now til later, discount at the initial rate
        sum([(1 - initial_x_risk)**t * period_utility(t) for t in range(periods_til_later)])

        + ((1 - initial_x_risk)**periods_til_later
           * sum([(1 - updated_x_risk(initial_x_risk, spending_later + exogenous_spending))**t * period_utility(t)
                  for t in range(extra_periods)])
           )

        # from later til forever, discount at the reduced rate
        + ((1 - initial_x_risk)**periods_til_later
           * (1 - updated_x_risk(initial_x_risk, spending_later + exogenous_spending))**extra_periods
           * convergence_value(updated_x_risk(initial_x_risk, spending_later + exogenous_spending + extra_exogenous_spending)))
    )

    return (utility_now, utility_later)


def permanent_reduction_binary_with_exogenous_target(
        period_utility: Callable[[float], float],
        convergence_value: Callable[[float], float],
        initial_x_risk: float,
        exogenous_spending_target: float,
        interest_rate: float,
        periods_til_later=100,
        budget=1,
):
    '''
    After `periods_til_later` periods, society will spend enough such that total spending equals
    `exogenous_spending_target`. If you spend more, society will spend less.
    '''

    spending_now = budget
    spending_later = spending_now * (1 + interest_rate)**periods_til_later
    x_risk_now = updated_x_risk(initial_x_risk, spending_now)

    utility_now = (
        sum([(1 - x_risk_now)**t * period_utility(t) for t in range(periods_til_later)])
        + ((1 - x_risk_now)**periods_til_later
           * convergence_value(updated_x_risk(initial_x_risk, max(spending_now, exogenous_spending_target))))
    )

    utility_later = (
        # from now til later, discount at the initial rate
        sum([(1 - initial_x_risk)**t * period_utility(t) for t in range(periods_til_later)])

        # from later til forever, discount at the reduced rate
        + ((1 - initial_x_risk)**periods_til_later
           * convergence_value(updated_x_risk(initial_x_risk, exogenous_spending_target + spending_later)))
    )

    return (utility_now, utility_later)


def expected_utility_of_giving(constant_func: Callable[[float], Tuple[float, float]],
                               mean: float,
                               stdev: float,
):
    expected_utility_now = 0
    expected_utility_later = 0

    # integrate
    for i in range(-99, 100):
        z_score = i / 10
        interest_rate = mean + z_score * stdev
        utility_now, utility_later = constant_func(interest_rate)
        pdf = stats.norm.pdf(z_score, 0, 1)
        expected_utility_now += utility_now * pdf
        expected_utility_later += utility_later * pdf

    return (expected_utility_now, expected_utility_later)


def breakeven_interest_rate(initial_x_risk=0.002, budget=1, exogenous_spending=None, movement_growth_rate=None,
                            exogenous_spending_target=None):
    '''
    Calculate the interest rate at which an actor is indifferent between giving
    now or giving later, using a model defined by one of the functions above.
    '''

    # Customize these bits
    period_utility = lambda t: 1
    convergence_value = lambda x_risk: 1 / x_risk
    f = permanent_reduction_binary_with_exogenous_target

    # Do not change anything below this line
    assert(
        (exogenous_spending is not None)
        ^ (movement_growth_rate is not None)
        ^ (exogenous_spending_target is not None)
    )

    custom_param = exogenous_spending or movement_growth_rate or exogenous_spending_target or 0

    def minimand(interest_rate):
        utility_now, utility_later = f(
            period_utility, convergence_value, initial_x_risk, custom_param,
            interest_rate=interest_rate, budget=budget
        )
        return (utility_now - utility_later)**2

    # for some reason this gives the wrong answer for some initial guesses, but
    # gets it right if the initial guess is close enough to 0. maybe step sizes
    # are too big?
    opt = optimize.minimize_scalar(minimand)
    res = f(
        period_utility, convergence_value, initial_x_risk, custom_param,
        interest_rate=opt.x, budget=budget,
    )
    print("Breakeven rate {:.3f}% gives utility {} (relative difference {:.0e})".format(100 * opt.x, res, (res[0] - res[1])/res[0]))


def optimal_leverage(initial_x_risk=0.002, budget=1, exogenous_spending=None, movement_growth_rate=None,
                     exogenous_spending_target=None):
    '''Find how much leverage maximizes the expected utility of giving later.'''

    # Customize these bits
    period_utility = lambda t: 1
    convergence_value = lambda x_risk: 1 / x_risk
    f = temporary_reduction_binary
    risky_mean = 0.02
    risky_sigma = 0.18

    # Do not change anything below this line
    assert(
        (exogenous_spending is not None)
        ^ (movement_growth_rate is not None)
        ^ (exogenous_spending_target is not None)
    )

    custom_param = exogenous_spending or movement_growth_rate or exogenous_spending_target or 0

    def minimand(leverage):
        stdev = leverage * risky_sigma
        mean = leverage * risky_mean - stdev**2/2
        expected_utility_now, expected_utility_later = expected_utility_of_giving(
            lambda interest_rate: f(
                period_utility, convergence_value, initial_x_risk, custom_param,
                interest_rate=interest_rate, budget=budget
            ),
            mean, stdev
        )
        print("{:.2f} -> {}".format(leverage, expected_utility_later))
        return -expected_utility_later

    opt = optimize.minimize_scalar(minimand)
    print("Optimal leverage {:.2f} gives return {:.2f}%, stdev {:.2f}%, E[utility] {}".format(
        opt.x,
        100 * opt.x * risky_mean,
        100 * opt.x * risky_sigma,
        -opt.fun,
    ))


def expected_utility_of_investment(mu, sigma, initial_x_risk=0.002, budget=0.01, periods_til_later=1):
    '''
    Calculate the expected utility of an investment under the
    temporary-reduction model, where x-risk reduction only lasts for a single
    period.

    mu and sigma are the parameters of the logarithm of the investment return
    minus the risk-free rate.
    '''
    return integrate.quad(
        lambda r: (
            (1 - initial_x_risk / (1 + budget*(1 + r)**periods_til_later))
            * stats.norm.pdf(r, mu, sigma)
        ),
        # < -100% returns are well-defined because they result in increasing x-risk
        -100, 100
    )[0]


# breakeven_interest_rate(budget=1, exogenous_spending=100)
optimal_leverage(budget=1, exogenous_spending=100)
