"""

ai_now_later_discrete.py
------------------------

Author: Michael Dickens
Created: 2025-09-12

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, PercentFormatter
from scipy.optimize import curve_fit


def assert_sums_to_1(probs):
    assert sum(probs.values()) > 0.999 and sum(probs.values()) < 1.001


def spending_schedule(years_until_point_of_no_return):
    """
    temporary (?) simplifications:

    - all money goes to x-risk
    - the best spending schedule (given a worldview) is to equally distribute across all years
    - money takes zero time to deploy once it's spent
    - we don't learn anything over time about intervention cost-effectiveness
    - ignore investment returns
    - money is allocated to worldviews proportional to credence
    """
    spending_per_year = [0 for _ in range(100)]
    for k in years_until_point_of_no_return:
        weight = years_until_point_of_no_return[k]
        spending = 1 / k
        for i in range(k):
            spending_per_year[i] += spending * weight

    return spending_per_year


years_until_point_of_no_return = {
    2: 0.1,
    3: 0.1,
    5: 0.3,
    10: 0.25,
    15: 0.05,
    20: 0.05,
    50: 0.15,
}

assert_sums_to_1(years_until_point_of_no_return)

spending = spending_schedule(years_until_point_of_no_return)


# Fit spending to a power law plus a lognormal bump: the lognormal captures the
# near-term peak, the power law captures the long tail.
def func_hybrid(x, c, m, a, mu, s):
    return c * x**m + a * np.exp(-((np.log(x) - mu) ** 2) / (2 * s**2)) / x


def func_pareto(x, c, m, a, mu, s):
    return c * x**m


def func_lognormal(x, c, m, a, mu, s):
    return a * np.exp(-((np.log(x) - mu) ** 2) / (2 * s**2)) / x

"""Customizable parameters"""
func_to_fit = func_lognormal
x_max = 10

X = np.arange(100) + 0.5
y = np.array(spending)
current_year = 2026
popt, pcov = curve_fit(
    func_to_fit,
    X,
    y,
    p0=[0.05, -1, 0.2, 1, 0.7],
    bounds=([0, -10, 0, -3, 0.05], [10, 0, 10, 6, 5]),
    maxfev=20000,
)
c, m, a, mu, s = popt
r_squared = 1 - np.sum((y - func_to_fit(X, *popt)) ** 2) / np.sum((y - y.mean()) ** 2)
if func_to_fit == func_hybrid:
    print(
        f"Best fit: {c:.3f} x^{m:.2f} + {a:.3f} exp(-(ln x - {mu:.2f})^2 / (2 * {s:.2f}^2)) / x"
    )
elif func_to_fit == func_pareto:
    print(
        f"Best fit: {c:.3f} x^{m:.2f}"
    )
elif func_to_fit == func_lognormal:
    print(
        f"Best fit: {a} * exp(-(ln x - {mu:.2f})^2 / (2 * {s:.2f}^2)) / x"
    )

if func_to_fit != func_pareto:
    print(f"\tlognormal peak median: {np.exp(mu):.1f} years")
print(f"\tR^2 = {r_squared:.3f}")

# Smoothed schedule: each year's amount comes from the fitted curve. The fit
# doesn't sum to exactly 100% over the horizon, so rescale it to spend the
# whole budget.
fitted = func_to_fit(X, *popt)
fit_scale = 1 / fitted.sum()
smoothed = fitted * fit_scale

print("\nFirst 20 years:\n\t      original\t\tsmoothed")
money_left = 1
smoothed_left = 1
for i in range(20):
    money_left -= spending[i]
    smoothed_left -= smoothed[i]
    print(
        f"\t{current_year + i:4d}: {spending[i] * 100:4.1f}%  ->  {money_left * 100:3.0f}%"
        f"\t{smoothed[i] * 100:4.1f}%  ->  {smoothed_left * 100:3.0f}%"
    )

FIT_LABELS = {
    func_hybrid: "Power law + lognormal fit",
    func_pareto: "Power law fit",
    func_lognormal: "Lognormal fit",
}
# Start at the first fitted point; the power-law term blows up as x -> 0.
x_fine = np.linspace(X[0], x_max, 1000)


# Graphing stuff, mainly written by Claude Opus 5.5
def plot_schedule(schedule, schedule_label, curve_scale, title, filename=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stairs(schedule, np.arange(101), fill=True, color="#2563eb", alpha=0.25)
    ax.stairs(schedule, np.arange(101), color="#2563eb", linewidth=1.5, label=schedule_label)
    ax.plot(
        x_fine,
        curve_scale * func_to_fit(x_fine, *popt),
        color="#dc2626",
        linewidth=2,
        label=f"{FIT_LABELS[func_to_fit]} (R² = {r_squared:.2f})",
    )
    ax.xaxis.set_major_locator(MultipleLocator(max(1, x_max // 10)))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, None)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Years from now")
    ax.set_ylabel("Share of starting budget spent per year")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=11)
    unspent = 1 - schedule[: int(x_max)].sum()
    if unspent > 0.0005:
        ax.text(
            0.98,
            0.6,
            f"Unspent after year {x_max}: {unspent * 100:.1f}%",
            transform=ax.transAxes,
            ha="right",
            fontsize=11,
            color="#374151",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#d1d5db"),
        )
    fig.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")


plot_schedule(y, "Spending schedule", 1, "Spending Schedule")
plot_schedule(
    smoothed, "Smoothed spending schedule", fit_scale, "Spending Schedule (Smoothed)",
    filename="images/ai_now_later_discrete.png"
)
plt.show()
