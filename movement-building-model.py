'''

Originally written by abergal.

'''

from scipy import optimize

market_rate = 1.05

salary = 100
susceptible_per_longtermist_per_year = 0.01
dollars_to_convert = 10
starting_longtermists = 100
starting_susceptible = 1000
starting_money = 0

# Invest
def invest(starting_longtermists, starting_susceptible, starting_money, t):
    money = starting_money
    longtermists = starting_longtermists
    susceptible = starting_susceptible

    for i in range(t):
        money = (money + salary) * market_rate
        susceptible += longtermists * susceptible_per_longtermist_per_year

    return longtermists, susceptible, money

# Don't invest
def spend(starting_longtermists, starting_susceptible, starting_money, t):
    money = starting_money
    longtermists = starting_longtermists
    susceptible = starting_susceptible

    for i in range(t):
        money += salary

        new_longtermists = min(money / dollars_to_convert, susceptible)
        longtermists += new_longtermists
        susceptible -= new_longtermists
        susceptible += longtermists * susceptible_per_longtermist_per_year

        money -= new_longtermists * dollars_to_convert

    return longtermists, susceptible, money

def spend_prop(starting_longtermists, starting_susceptible, starting_money, t, prop_spending):
    money = starting_money
    longtermists = starting_longtermists
    susceptible = starting_susceptible

    for i in range(t):
        # growth in money
        money = (money + salary * (1 - prop_spending)) * market_rate

        # growth in number of longtermists
        new_longtermists = min(money * prop_spending / dollars_to_convert, susceptible)
        longtermists += new_longtermists
        susceptible -= new_longtermists
        susceptible += longtermists * susceptible_per_longtermist_per_year

    return longtermists, susceptible, money

def run_strategies(starting_longtermists, starting_susceptible, starting_money):
    # Spend only
    l, _, _ = spend(starting_longtermists, starting_susceptible, starting_money, 100)
    print("Spend for 100:\t\t\t\t\t{}".format(l))

    # Invest 50, Spend 50
    l, s, m = invest(starting_longtermists, starting_susceptible, starting_money, 50)
    l, _, _ = spend(l, s, m, 50)
    print("Invest for 50, spend for 50:\t\t\t{}".format(l))

    # Invest 90, Spend 10
    l, s, m = invest(starting_longtermists, starting_susceptible, starting_money, 90)
    l, _, _ = spend(l, s, m, 10)
    print("Invest for 90, spend for 10:\t\t\t{}".format(l))

    # Invest 10, Spend 90
    l, s, m = invest(starting_longtermists, starting_susceptible, starting_money, 10)
    l, _, _ = spend(l, s, m, 90)
    print("Invest for 10, spend for 90:\t\t\t{}".format(l))

    # Invest 10, Spend 10, Invest 10, Spend 70
    l, s, m = invest(starting_longtermists, starting_susceptible, starting_money, 10)
    l, s, m = spend(l, s, m, 10)
    l, s, m = invest(l, s, m, 10)
    l, s, m = spend(l, s, m, 70)
    print("Invest 10, spend 10, invest 10, spend 70:\t{}\n".format(l))

def run_prop_strategies(starting_longtermists, starting_susceptible, starting_money):
    print("Spend 1%: {}".format(spend_prop(starting_longtermists, starting_susceptible, starting_money, 100, 0.01)[0]))
    print("Spend 10%: {}".format(spend_prop(starting_longtermists, starting_susceptible, starting_money, 100, 0.1)[0]))
    print("Spend 50%: {}".format(spend_prop(starting_longtermists, starting_susceptible, starting_money, 100, 0.5)[0]))
    print("Spend 75%: {}".format(spend_prop(starting_longtermists, starting_susceptible, starting_money, 100, 0.75)[0]))
    print("Spend 100%: {}".format(spend_prop(starting_longtermists, starting_susceptible, starting_money, 100, 1)[0]))

    bounds_constraint = optimize.LinearConstraint([1], lb=[0], ub=[1])
    opt = optimize.minimize(lambda prop: -spend_prop(starting_longtermists, starting_susceptible, starting_money, 100, prop)[0], [0.1], constraints=bounds_constraint)
    print("Optimal spending: {:.1f}% -> {}".format(
        opt.x[0] * 100, spend_prop(starting_longtermists, starting_susceptible, starting_money, 100, opt.x[0])[0]))

def run_model():
    print("{} market rate, {} salary, {} susceptible per longtermist per year, {} dollars to convert,\n{} starting longtermists, {} starting susceptible, {} starting money:\n".format
          (market_rate, salary, susceptible_per_longtermist_per_year, dollars_to_convert, starting_longtermists, starting_susceptible, starting_money))
    # run_strategies(starting_longtermists, starting_susceptible, starting_money)
    run_prop_strategies(starting_longtermists, starting_susceptible, starting_money)

# Default
run_model()

# Make it more expensive to convert people
# dollars_to_convert = 100
# run_model()
