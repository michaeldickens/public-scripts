"""

ai_now_later_discrete.py
------------------------

Author: Michael Dickens
Created: 2025-09-12

"""

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def assert_sums_to_1(probs):
    assert sum(probs.values()) > 0.999 and sum(probs.values()) < 1.001


def spending_schedule(years_to_AGI, takeoff_years):
    """
    temporary (?) simplifications:

    - ignore takeoff_years
    - all money goes to x-risk
    - the best spending schedule (given a worldview) is to equally distribute across all years
    - money takes zero time to deploy once it's spent
    - we don't learn anything over time about intervention cost-effectiveness
    - ignore investment returns
    - money is allocated to worldviews proportional to credence
    """
    spending_per_year = [0 for _ in range(100)]
    for k in years_to_AGI:
        weight = years_to_AGI[k]
        spending = 1 / k
        for i in range(k):
            spending_per_year[i] += spending * weight

    return spending_per_year


years_to_AGI = {
    5: 0.2,
    10: 0.25,
    15: 0.20,
    20: 0.05,
    30: 0.10,
    50: 0.10,
    100: 0.10,
}

takeoff_years = {
    0.1: 0.3,
    0.5: 0.1,
    1: 0.1,
    2: 0.1,
    3: 0.1,
    4: 0.1,
    6: 0.1,
    10: 0.05,
    20: 0.05,
}

assert_sums_to_1(years_to_AGI)
assert_sums_to_1(takeoff_years)

spending = spending_schedule(years_to_AGI, takeoff_years)


# fit spending to a power law
def func_power_law(x, c0, c, m):
    return c0 + c * x**m


X = list(range(100))
current_year = 2025
popt, pcov = curve_fit(func_power_law, X, spending)
print(f"Optimal fit: {popt[0]:.2f} + {popt[1]:.2f} x^{popt[2]:.2f}")
print("\nFirst 20 years:")
money_left = 1
for i in range(20):
    money_left -= spending[i]
    print(f"\t{current_year + i:4d}: {spending[i] * 100:.1f}%  ->  {money_left * 100:.0f}%")

plt.figure(figsize=(10, 6))
plt.plot(spending)
plt.plot(X, func_power_law(X, *popt))
plt.title("Spending Schedule")
plt.xlabel("Years from Now")
plt.ylabel("Relative Spending")
plt.grid()
plt.show()
