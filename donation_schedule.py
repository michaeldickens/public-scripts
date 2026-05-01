"""
AI safety donation schedule calculator with uncertain AI timelines
==================================================================

Created 2026-03-21.

Code originally written by Claude Opus 4.6. Docs by Michael Dickens.

The basic concept: I want to donate an equal amount of money every year from now until the singularity. But I don't know when the singularity will happen. How much should I donate each year (in terms of % of starting wealth)?

This script calculates the answer given either a Pareto or a log-normal distribution over AI timelines.

The reason for donating an equal amount each year is that it's simple. As a qualitative justification, there's a tradeoff between early donations having compounding effects and late donations coming in at the right time/being better informed. If you're unsure about which side of the tradeoff matters more, then you should distribute donations over time.

The analytic solution for the annual donation amount is given by

    d(t) = integral from t to infinity of 1/T * f(T) dT

where f(T) is the probability density of the singularity occurring at time T. Claude used this to derive the analytic solutions for Pareto and log-normal distributions.

I had Claude write a second script to calculate results using Monte Carlo as a sanity check that the formulas are correct.

"Planned" is the donation amount when planning all years in advance. "Conditional" is the amount to donate in year t conditioning on the fact that you know you've survived t-1 years (which changes your beliefs about timelines). I'm not sure the math is right on "Conditional" (it's conceptually wonky).

See also the Claude chat that generated this script:
https://claude.ai/share/5830f229-4dad-4427-aee8-30937746d468
"""

import numpy as np
from scipy import stats


def donation_schedule(dist="pareto", years=30, **p):
    t = np.arange(1, years + 1, dtype=float)
    if dist == "pareto":
        a, tm = p["alpha"], p["t_min"]
        d = np.where(t >= tm, a * tm**a / ((a + 1) * t ** (a + 1)), a / ((a + 1) * tm))
        surv = np.where(t >= tm, (tm / t) ** a, 1.0)
    elif dist == "lognormal":
        mu, s = p["mu"], p["sigma"]
        d = np.exp(-mu + s**2 / 2) * stats.norm.sf((np.log(t) - (mu - s**2)) / s)
        surv = stats.norm.sf((np.log(t) - mu) / s)
    cond = d / np.maximum(surv, 1e-15)
    return t.astype(int), d / d.sum(), cond / cond.sum()


def plot(title, t, planned, cond, dist_obj, filename="donation_schedule.png"):
    import matplotlib.pyplot as plt

    # Discretize PDF to per-year probability mass on the same x-grid as the bars,
    # so % probability mass per year is directly comparable to % of budget per year.
    edges_lo, edges_hi = t - 0.5, t + 0.5
    pmf = dist_obj.cdf(edges_hi) - dist_obj.cdf(edges_lo)
    # Conditional pmf: P(singularity in year t | survived to t-0.5)
    surv_lo = np.maximum(dist_obj.sf(edges_lo), 1e-15)
    cond_pmf = pmf / surv_lo

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, data, overlay, label, color in [
        (a1, planned, pmf, "Planned (ex ante)", "#2563eb"),
        (a2, cond, cond_pmf, "Conditional (updating on survival)", "#dc2626"),
    ]:
        ax.bar(t, data * 100, color=color, alpha=0.85, width=0.8, label="Donation")
        ax.plot(t, overlay * 100, color="black", linewidth=1,
                marker="o", markersize=0, label="Singularity probability")
        ax.set(xlabel="Year", ylabel="% per year")
        ax.set_title(label, fontsize=14)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=11)
    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"  Saved {filename}")


if __name__ == "__main__":
    median_timeline = 7
    alpha = 1.5
    t_min = 3
    sigma = 1
    cases = [
        (f"Pareto(median={median_timeline}, min={t_min}, α={alpha})", "pareto", dict(alpha=alpha, t_min=t_min, median=median_timeline)),
        (f"LogNormal(median={median_timeline}, σ={sigma})", "lognormal", dict(mu=np.log(median_timeline), sigma=sigma)),
    ]
    for title, dist, params in cases:
        if dist == "pareto":
            scale = (params["median"] - params["t_min"]) / (2**(1/params["alpha"]) - 1)
            loc = params["t_min"] - scale
            d = stats.pareto(b=params["alpha"], loc=loc, scale=scale)
        elif dist == "lognormal":
            d = stats.lognorm(s=params["sigma"], scale=np.exp(params["mu"]))

        # explicitly calculate median just to make sure it matches the
        # median_timeline parameter
        mean, median = d.mean(), d.median()
        print(f"\n=== {title} (mean {mean:.1f}, median {median:.1f}) ===")

        t, planned, cond = donation_schedule(dist, 30, **params)
        for y, p, c in zip(t, planned, cond):
            print(f"  Year {y:2d}:  planned {p*100:6.2f}%   conditional {c*100:6.2f}%")
        plot(title, t, planned, cond, d, f"images/donation_schedule_{dist}.png")
