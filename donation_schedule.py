"""AI safety donation schedule calculator with uncertain AI timelines
==================================================================

Created 2026-03-21.

Primarily written by Claude Opus 4.6 and 4.7. Docstring by Michael Dickens.

The basic concept: I want to donate an equal amount of money every year from
now until the singularity. But I don't know when the singularity will happen.
How much should I donate each year (in terms of % of starting wealth)?

This script calculates the answer given either a Pareto or a log-normal
distribution over AI timelines.

The reason for donating an equal amount each year is that it's simple. As a
qualitative justification, there's a tradeoff between early donations having
compounding effects and late donations coming in at the right time/being better
informed. If you're unsure about which side of the tradeoff matters more, then
you should distribute donations over time.

The analytic solution for the annual donation amount is given by

    D(T) = integral from T to infinity of 1/t * f(t) dt

where f(t) is the probability density of the singularity occurring at time t.
Claude used this to derive the analytic solutions for Pareto and log-normal
distributions.

To customize the inputs (your timeline beliefs and distribution shape
parameters), edit the CONFIGURATION block below.

See also the Claude chats that generated this script:
https://claude.ai/share/5830f229-4dad-4427-aee8-30937746d468
https://claude.ai/share/a258d624-f4e9-456f-ba5d-89abc7844a63

"""

# ============================================================
# CONFIGURATION -- edit the values below, then run the script.
# ============================================================

# How many years to plan over. The script computes a donation amount for
# each year from 1 to YEARS.
YEARS = 30

# Your best guess for the median time (in years from now) until the
# singularity. "Median" means: you think there's a 50% chance the
# singularity happens before this year, and 50% chance after.
# Example: if you think there's a 50/50 chance of singularity by 2033,
# and it's currently 2026, set this to 7.
MEDIAN_TIMELINE = 7

# ---- Pareto distribution settings ----
# Pareto is a "heavy-tailed" distribution: it puts a lot of probability
# on early years but also has a long tail allowing for very late years.

# The earliest year the singularity could possibly happen, in your view.
# Pareto assigns zero probability to anything before this year.
PARETO_MIN_YEAR = 3

# Controls how heavy the tail is. Lower alpha = heavier tail (more weight
# on far-future outcomes). Typical values: 1.0 (very heavy) to 3.0 (light).
# Must be > 1 for the mean to exist.
PARETO_ALPHA = 1.5

# ---- Log-normal distribution settings ----
# Log-normal is symmetric on a log scale: equally likely to be 2x the
# median as 1/2x the median.

# Controls the spread. Larger sigma = more uncertainty.
# sigma=1 means roughly: 68% chance the singularity is between
# median/e and median*e (i.e., between ~2.6 and ~19 years for median=7).
# Typical values: 0.5 (confident) to 2 (very uncertain).
LOGNORMAL_SIGMA = 1.0

# ============================================================
# End of configuration. You shouldn't need to edit below here.
# ============================================================


import numpy as np
from scipy import stats


def donation_schedule(dist="pareto", years=30, **p):
    t = np.arange(1, years + 1, dtype=float)
    if dist == "pareto":
        a, tm = p["alpha"], p["t_min"]
        d = np.where(t >= tm, a * tm**a / ((a + 1) * t ** (a + 1)), a / ((a + 1) * tm))
    elif dist == "lognormal":
        mu, s = p["mu"], p["sigma"]
        d = np.exp(-mu + s**2 / 2) * stats.norm.sf((np.log(t) - (mu - s**2)) / s)
    return t.astype(int), d / d.sum()


def mixture_schedule(components, years=30):
    # Each component is (dist_name, params_dict, weight). Weights are
    # renormalized to sum to 1. Returns the per-year donation schedule
    # where each component's normalized schedule contributes its weight
    # share of the total budget. This is averaging the schedules, not
    # computing the schedule of the mixture distribution -- the former
    # is what "split the budget across worldviews" literally means.
    weights = np.array([w for _, _, w in components], dtype=float)
    weights /= weights.sum()
    t = None
    mixed = np.zeros(years)
    for (dist, params, _), w in zip(components, weights):
        t, planned = donation_schedule(dist, years, **params)
        mixed += w * planned
    return t, mixed


def _per_year_pmf(dist_obj, t):
    return dist_obj.cdf(t + 0.5) - dist_obj.cdf(t - 0.5)


def _draw_panel(ax, t, planned, pmf, title, color):
    ax.bar(t, planned * 100, color=color, alpha=0.85, width=0.8, label="Donation")
    if pmf is not None:
        ax.plot(t, pmf * 100, color="black", linewidth=1.2, label="Singularity probability")
    ax.set(xlabel="Year", ylabel="% per year")
    ax.set_title(title, fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=11)


def _mixture_pmf(components, t):
    weights = np.array([w for _, _, w in components], dtype=float)
    weights /= weights.sum()
    pmf = np.zeros_like(t, dtype=float)
    for (_, dist_obj, _), w in zip(components, weights):
        pmf += w * _per_year_pmf(dist_obj, t)
    return pmf


def plot_single(title, t, planned, pmf, filename, color="#2563eb"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _draw_panel(ax, t, planned, pmf, title, color)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_combined(scenarios, colors, filename):
    import matplotlib.pyplot as plt

    n = len(scenarios)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, (title, t, planned, pmf), color in zip(axes, scenarios, colors):
        _draw_panel(ax, t, planned, pmf, title, color)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}")


if __name__ == "__main__":
    if MEDIAN_TIMELINE <= PARETO_MIN_YEAR:
        raise ValueError(
            f"MEDIAN_TIMELINE ({MEDIAN_TIMELINE}) must be greater than "
            f"PARETO_MIN_YEAR ({PARETO_MIN_YEAR}). The median can't be earlier "
            f"than the earliest possible year."
        )
    if PARETO_ALPHA <= 1:
        raise ValueError(
            f"PARETO_ALPHA ({PARETO_ALPHA}) must be greater than 1, otherwise "
            f"the distribution has infinite mean and the math breaks."
        )

    cases = [
        # Put other cases here, if desired
        (
            f"Pareto(median=7, min=1, α=1)",
            "pareto",
            dict(alpha=1, t_min=1, median=7),
        ),
        (
            f"Pareto(median={MEDIAN_TIMELINE}, min={PARETO_MIN_YEAR}, α={PARETO_ALPHA})",
            "pareto",
            dict(alpha=PARETO_ALPHA, t_min=PARETO_MIN_YEAR, median=MEDIAN_TIMELINE),
        ),
        (
            f"LogNormal(median={MEDIAN_TIMELINE}, σ={LOGNORMAL_SIGMA})",
            "lognormal",
            dict(mu=np.log(MEDIAN_TIMELINE), sigma=LOGNORMAL_SIGMA),
        ),
    ]

    scenarios = []
    mixture_components = []
    for title, dist, params in cases:
        if dist == "pareto":
            scale = (params["median"] - params["t_min"]) / (
                2 ** (1 / params["alpha"]) - 1
            )
            loc = params["t_min"] - scale
            d = stats.pareto(b=params["alpha"], loc=loc, scale=scale)
        elif dist == "lognormal":
            d = stats.lognorm(s=params["sigma"], scale=np.exp(params["mu"]))

        # sanity-check that median matches MEDIAN_TIMELINE
        mean, median = d.mean(), d.median()
        print(f"\n=== {title} (mean {mean:.1f}, median {median:.1f}) ===")

        t, planned = donation_schedule(dist, YEARS, **params)
        scenarios.append((title, dist, t, planned, _per_year_pmf(d, t)))
        mixture_components.append((dist, params, d))

    # Equal-weight mixture across all configured distributions.
    weights = [1.0 / len(mixture_components)] * len(mixture_components)
    mix_for_schedule = [(dist, params, w) for (dist, params, _), w in zip(mixture_components, weights)]
    mix_for_pmf = [(dist, d, w) for (dist, _, d), w in zip(mixture_components, weights)]
    t_mix, planned_mix = mixture_schedule(mix_for_schedule, YEARS)
    pmf_mix = _mixture_pmf(mix_for_pmf, t_mix)
    scenarios.append((f"Mixture (equal weight, {len(mixture_components)} dists)", "mixture", t_mix, planned_mix, pmf_mix))

    # Side-by-side table
    titles = [s[0] for s in scenarios]
    years = scenarios[0][2]
    print("\n" + " " * 8 + "  ".join(f"{tt:>30s}" for tt in titles))
    for i, y in enumerate(years):
        row = f"Year {y:2d}  " + "  ".join(f"{s[3][i] * 100:29.2f}%" for s in scenarios)
        print(row)

    colors = [
        "#2563eb",  # blue
        "#dc2626",  # red
        "#059669",  # green
        "#d97706",  # amber
        "#7c3aed",  # violet
        "#0891b2",  # cyan
        "#db2777",  # pink
        "#65a30d",  # lime
        "#ea580c",  # orange
        "#4f46e5",  # indigo
        "#0d9488",  # teal
        "#a16207",  # yellow-brown
    ]
    for (title, dist, t, planned, pmf), color in zip(scenarios, colors):
        plot_single(title, t, planned, pmf, f"images/donation_schedule_{dist}.png", color)
    plot_combined(
        [(s[0], s[2], s[3], s[4]) for s in scenarios],
        colors,
        "images/donation_schedule_combined.png",
    )
