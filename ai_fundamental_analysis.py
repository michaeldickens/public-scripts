"""

ai_fundamental_analysis.py
--------------------------

Author: Michael Dickens <michael@mdickens.me>
Created: 2022-11-07

Estimate the fundamental value of AI stocks given certain assumptions about the Singularity.

Three possible outcomes:

1. mundane: singularity never happens; exponential growth continues
2. terminal: singularity happens and money becomes meaningless, either due to extinction or post-scarcity utopia
3. capture: singularity happens, and shareholders in AI stock capture the economic products of the AI

"""

import math
import sympy as s

def analyze_fundamentals():
    use_variables = True
    if use_variables:
        rra = 1
        ai_alloc, discount, years_to_singularity, years_to_reversion, p_capture, p_mundane, market_return, market_ai_alloc, starting_wealth = \
            s.symbols('lambda, delta, Y_s, Y_r, p_t, p_m, r, lambda_m, W
            _0')
        actual_years_to_reversion = 14
        ai_post_singularity = s.Float('1e32')
        market_post_singularity = ai_post_singularity * market_ai_alloc
        return_given_overpriced_ai = (1 - ai_alloc) * market_return  # 0% return on AI stocks
    else:
        ai_alloc = s.symbols('lambda')
        discount = 0.02
        rra = 1
        years_to_singularity = 20
        years_to_reversion = s.symbols('Y_r', negative=False)
        actual_years_to_reversion = 14
        p_capture = 0.1
        p_mundane = 0.4
        market_ai_alloc = 0.1
        ai_post_singularity = s.Float('1e32')
        market_post_singularity = ai_post_singularity * market_ai_alloc
        starting_wealth = s.Float('3e6')
        market_return = 0.05
        return_given_overpriced_ai = (1 - ai_alloc) * market_return  # 0% return on AI stocks

    def utility_general(starting_wealth, r):
        if rra == 1:
            return (
                (discount * s.log(starting_wealth * discount) + r - discount) / discount**2
            )
        else:
            return (
                starting_wealth**(1 - rra) / (1 - rra) * ((r * rra - r + discount) / rra)**(-rra)
                - (1 / (discount * (1 - rra)))
            )

    def apply_discount(x, years):
        return x / (1 + discount)**years

    def grow_wealth(wealth, r, years):
        return wealth * s.exp(years * (r - discount) / rra)

    def utility_until_time(starting_wealth, r, years):
        full_utility = utility_general(starting_wealth, r)
        later_utility = apply_discount(utility_general(grow_wealth(starting_wealth, r, years), r), years)
        return full_utility - later_utility

    utility_given_thesis = (
        utility_until_time(starting_wealth, market_return, years_to_singularity)
        +
        apply_discount(
            utility_general(ai_alloc * ai_post_singularity + (1 - ai_alloc) * market_post_singularity, 0),
            years_to_singularity
        )
    )

    # WARNING: If years_to_singularity is a SymPy variable and is less than
    # years_to_reversion then this will be wrong. Can deal with this by
    # branching in JavaScript and having two different equations.
    if not isinstance(years_to_singularity, int) or \
       actual_years_to_reversion <= years_to_singularity:
        utility_given_terminal = (
            utility_until_time(starting_wealth, return_given_overpriced_ai, years_to_reversion)
            +
            apply_discount(
                utility_until_time(grow_wealth(starting_wealth, return_given_overpriced_ai, years_to_reversion), market_return, years_to_singularity),
                years_to_reversion
            )
        )
    else:
        utility_given_terminal = (
            utility_until_time(starting_wealth, return_given_overpriced_ai, years_to_singularity)
        )

    utility_given_mundane = (
        utility_until_time(starting_wealth, return_given_overpriced_ai, years_to_reversion)
        +
        apply_discount(
            utility_general(grow_wealth(starting_wealth, return_given_overpriced_ai, years_to_reversion), market_return),
            years_to_reversion
        )
    )

    s.init_printing()
    utility = p_capture * utility_given_thesis + p_mundane * utility_given_mundane + (1 - p_capture - p_mundane) * utility_given_terminal

    deriv = s.diff(utility, ai_alloc)
    alloc_solution = s.solvers.solve(deriv.evalf(subs={years_to_reversion: actual_years_to_reversion}), ai_alloc, domain=s.S.Reals)

    # Find the value of years_to_reversion such that utility is maximized at
    # ai_alloc == market_ai_alloc. Do this by taking the derivative of utility wrt
    # ai_alloc and then solving for years_to_reversion.
    #
    # Note: s.solvers.solve doesn't work, but s.solveset does. idk why
    if use_variables:
        s.pprint(alloc_solution)
        print(alloc_solution)
        print(alloc_solution[0].evalf(subs={discount: 0.02, years_to_singularity: 20, years_to_reversion: 14, p_capture: 0.1, p_mundane: 0.4, market_return: 0.05, market_ai_alloc: 0.1, starting_wealth: 3e6}))

    else:
        reversion_solution = s.solveset(deriv.evalf(subs={ai_alloc: market_ai_alloc}), years_to_reversion, domain=s.S.Reals)

        print("Optimal allocation {:.1f}% gives utility {}".format(
            100 * alloc_solution[0],
            utility.evalf(subs={ai_alloc: alloc_solution[0], years_to_reversion: actual_years_to_reversion})
        ))

        print("Equilibrium years to reversion: {}".format(reversion_solution))


analyze_fundamentals()
