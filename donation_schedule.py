"""
AI safety donation schedule calculator with uncertain AI timelines
==================================================================

Created 2026-03-21.

Code originally written by Claude Opus 4.6 and 4.7. Docs by Michael Dickens.

The basic concept: I want to donate an equal amount of money every year from now until the singularity. But I don't know when the singularity will happen. How much should I donate each year (in terms of % of starting wealth)?

This script calculates the answer given either a Pareto or a log-normal distribution over AI timelines.

The reason for donating an equal amount each year is that it's simple. As a qualitative justification, there's a tradeoff between early donations having compounding effects and late donations coming in at the right time/being better informed. If you're unsure about which side of the tradeoff matters more, then you should distribute donations over time.

The analytic solution for the annual donation amount is given by

    d(t) = integral from t to infinity of 1/T * f(T) dT

where f(T) is the probability density of the singularity occurring at time T. Claude used this to derive the analytic solutions for Pareto and log-normal distributions.

I had Claude write a second script to calculate results using Monte Carlo as a sanity check that the formulas are correct.

"Planned" is the donation amount when planning all years in advance. It's normalized to sum to 100% and represents fraction of budget allocated to year t under a single ex-ante plan.

"Conditional" is the amount to donate in year t conditioning on the fact that you've survived to year t (which changes your beliefs about timelines). It's computed by Monte Carlo: simulate T, then under the strategy "donate W_t / (T - t + 1) each year" (i.e. spread remaining wealth over remaining years if T were known), average the year-t donation over trials with T >= t. Each entry is "expected fraction of starting wealth donated in year t given survival to t" -- it does NOT sum to 100%.

See also the Claude chat that generated this script:
https://claude.ai/share/5830f229-4dad-4427-aee8-30937746d468
"""

import numpy as np
from scipy import stats


def donation_schedule(dist_obj, dist="pareto", years=30, n_mc=200_000, seed=0, **p):
    t = np.arange(1, years + 1, dtype=float)
    if dist == "pareto":
        a, tm = p["alpha"], p["t_min"]
        planned = np.where(t >= tm, a * tm**a / ((a + 1) * t ** (a + 1)),
                           a / ((a + 1) * tm))
    elif dist == "lognormal":
        mu, s = p["mu"], p["sigma"]
        planned = np.exp(-mu + s**2 / 2) * stats.norm.sf(
            (np.log(t) - (mu - s**2)) / s)

    # MC for conditional: each trial donates W/(T-t+1) per year, where W is
    # remaining wealth. Averaging year-t donations over trials with T >= t
    # gives the expected donation in year t given survival to t.
    rng = np.random.default_rng(seed)
    T_samples = np.ceil(dist_obj.rvs(size=n_mc, random_state=rng)).astype(int)
    T_samples = np.clip(T_samples, 1, None)
    don_sum = np.zeros(years)
    counts = np.zeros(years)
    for T in T_samples:
        W = 1.0
        for tt in range(1, min(T, years) + 1):
            d = W / (T - tt + 1)
            don_sum[tt - 1] += d
            counts[tt - 1] += 1
            W -= d
    cond = don_sum / np.maximum(counts, 1)
    return t.astype(int), planned / planned.sum(), cond


def plot(title, t, planned, cond, dist_obj, filename="donation_schedule.png"):
    import matplotlib.pyplot as plt

    edges_lo, edges_hi = t - 0.5, t + 0.5
    pmf = dist_obj.cdf(edges_hi) - dist_obj.cdf(edges_lo)
    surv_lo = np.maximum(dist_obj.sf(edges_lo), 1e-15)
    cond_pmf = pmf / surv_lo  # hazard given survival to t-0.5

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    panels = [
        (a1, planned, pmf, "Planned (ex ante)", "#2563eb",
        "% of budget", "Singularity probability"),
        (a2, cond, cond_pmf, "Conditional (updating on survival)", "#dc2626",
        "% of starting wealth (given survival)", "Singularity probability (given survival)"),
    ]
    for ax, data, overlay, label, color, ylabel, line_label in panels:
        ax.bar(t, data * 100, color=color, alpha=0.85, width=0.8, label="Donation")
        ax.plot(t, overlay * 100, color="black", linewidth=1,
                marker="o", markersize=0, label=line_label)
        ax.set(xlabel="Year", ylabel=ylabel)
        ax.set_title(label, fontsize=13)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=11)
    fig.suptitle(title, fontsize=15, fontweight="bold")
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

        t, planned, cond = donation_schedule(d, dist, 30, **params)
        for y, p, c in zip(t, planned, cond):
            print(f"  Year {y:2d}:  planned {p*100:6.2f}%   conditional {c*100:6.2f}%")
        plot(title, t, planned, cond, d, f"images/donation_schedule_{dist}.png")
