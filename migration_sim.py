"""
Value vs Growth Stock Simulation
=================================

Created 2026-02-23.

Primarily written by Claude Opus 4.6.

Setup:
- 100000 stocks with P/F ratios uniformly distributed on [5, 25]
- Cheapest half (by P/F) = Value; most expensive half = Growth
- Expected log F growth = 0.05 + 0.003 * (P/F)
- Individual stock log F growth ~ Normal(expected, 5% std dev)

ALL returns and growth rates are in log space:
  - "7% growth" means F1 = F0 * exp(0.07)
  - log_total_return = log(P1/P0) = log(F1/F0) + log(PF1/PF0)
                     = log_fund_return + log_revaluation
  - This decomposition is exact and additive (no cross terms, no Jensen's)

Scenario A: Price changes proportionally to fundamentals (P/F unchanged)
Scenario B: Value stocks all go up by value expected mean; growth by growth expected mean

After each scenario, reconstitute portfolios by re-ranking on new P/F,
then decompose returns into fundamentals vs revaluation.
"""

import numpy as np
import statsmodels.api as sm

# np.random.seed(42)

N = 100000
HALF = N // 2

# ---------------------------------------------------------------------------
# Generate initial stocks
# ---------------------------------------------------------------------------
# P/F ratios: evenly spaced on [5, 25]
pf_initial = np.linspace(5, 25, N)
log_pf_initial = np.log(pf_initial)

# Normalize: every stock starts with F=1, so P = P/F
F0 = np.ones(N)
P0 = pf_initial * F0

# Expected log fundamental growth for each stock
# expected_log_f_growth = 0.04 + 0.003 * pf_initial
expected_log_f_growth = 0.04 + 0.000 * pf_initial

# Actual log fundamental growth: add noise
noise = np.random.normal(0, 0.05, N)
log_f_growth = expected_log_f_growth + noise

# New fundamentals: F1 = F0 * exp(log_f_growth)
F1 = F0 * np.exp(log_f_growth)

# ---------------------------------------------------------------------------
# Initial portfolios (before any price changes)
# ---------------------------------------------------------------------------
initial_order = np.argsort(pf_initial)
value_mask_initial = np.zeros(N, dtype=bool)
growth_mask_initial = np.zeros(N, dtype=bool)
value_mask_initial[initial_order[:HALF]] = True
growth_mask_initial[initial_order[HALF:]] = True

value_expected_log_growth = expected_log_f_growth[value_mask_initial].mean()
growth_expected_log_growth = expected_log_f_growth[growth_mask_initial].mean()

print("=" * 80)
print("INITIAL PORTFOLIO CHARACTERISTICS")
print("=" * 80)
print(f"  R² of ΔF against P/F = {sm.OLS(log_f_growth, sm.add_constant(log_pf_initial)).fit().rsquared:.3f}")
print(f"  Value  stocks: avg P/F = {pf_initial[value_mask_initial].mean():.2f}, "
      f"avg expected log F growth = {value_expected_log_growth*100:.2f}%")
print(f"  Growth stocks: avg P/F = {pf_initial[growth_mask_initial].mean():.2f}, "
      f"avg expected log F growth = {growth_expected_log_growth*100:.2f}%")
print(f"  Value  stocks: avg actual log F growth = {log_f_growth[value_mask_initial].mean()*100:.2f}%")
print(f"  Growth stocks: avg actual log F growth = {log_f_growth[growth_mask_initial].mean()*100:.2f}%")
print()


def analyze_scenario(scenario_name, P1):
    """
    Given new prices P1, reconstitute portfolios and decompose returns.

    Notation:
      P0, F0          = initial price and fundamentals
      P- = P1, F- = F1 = price and fundamentals just BEFORE reconstitution
      P+, F+          = price and fundamentals of the RECONSTITUTED portfolio
                        (same stocks get new weights/membership after re-ranking)

    Note: P- and F- are per-stock values (every stock has a P1 and F1).
    P+ and F+ refer to the portfolio-average P/F after reconstitution, i.e.
    we re-sort stocks by P-/F- and take the cheapest/most expensive halves.
    Each stock's P and F don't change — only which stocks are in which portfolio.

    Three-term decomposition (all in log space, exact and additive):

      total return = fundamentals + revaluation + reconstitution

      fundamentals    = log(F-/F0)           = how much fundamentals grew
      revaluation     = log(P+/F+) - log(P0/F0)  = change in P/F between
                        initial and reconstituted portfolio
      reconstitution  = log(P-/F-) - log(P+/F+)  = P/F difference between
                        pre-recon and post-recon portfolio (the "migration" cost)

    Check: fund + reval + recon
         = log(F-/F0) + [log(P+/F+) - log(P0/F0)] + [log(P-/F-) - log(P+/F+)]
         = log(F-/F0) + log(P-/F-) - log(P0/F0)
         = log(F-) - log(F0) + log(P-) - log(F-) - log(P0) + log(F0)
         = log(P-/P0)
         = total return  ✓
    """
    print("=" * 80)
    print(f"SCENARIO {scenario_name}")
    print("=" * 80)

    # P- and F- (pre-reconstitution, i.e. end-of-period before re-ranking)
    # P1 and F1 are per-stock; P- = P1, F- = F1
    pf_pre = P1 / F1                          # P-/F- per stock
    log_pf_pre = np.log(pf_pre)

    # --- Reconstitute: re-rank by P-/F- ---
    new_order = np.argsort(pf_pre)
    value_mask_new = np.zeros(N, dtype=bool)
    growth_mask_new = np.zeros(N, dtype=bool)
    value_mask_new[new_order[:HALF]] = True
    growth_mask_new[new_order[HALF:]] = True

    # Per-stock log returns (these don't depend on portfolio membership)
    log_total_return = np.log(P1 / P0)        # log(P-/P0)
    log_fund = log_f_growth                    # log(F-/F0)

    # For the reconstituted portfolios
    results = {}
    for label, mask_new, mask_init in [
        ("Value",  value_mask_new,  value_mask_initial),
        ("Growth", growth_mask_new, growth_mask_initial),
    ]:
        # --- Portfolio-level P/F at each stage ---
        # Initial portfolio (t=0): avg log(P0/F0) over initial members
        avg_log_pf_0 = log_pf_initial[mask_init].mean()

        # Pre-reconstitution (P-/F-): avg over INITIAL members
        # (this is what the portfolio looks like just before we re-rank)
        avg_log_pf_pre = log_pf_pre[mask_init].mean()

        # Post-reconstitution (P+/F+): avg over NEW members
        # (same prices/fundamentals, but different stocks in the portfolio)
        avg_log_pf_post = log_pf_pre[mask_new].mean()

        # --- Three-term decomposition (portfolio averages) ---
        #
        # Total return is of the INITIAL portfolio (old members, held for one period).
        # We decompose using the reconstituted portfolio as an intermediate:
        #
        #   fundamentals    = <log(F-/F0)>_init
        #                   = how much fundamentals grew for old members
        #   revaluation     = <log(P-/F-)>_new - <log(P0/F0)>_init
        #                   = reconstituted portfolio's P/F at t=1 vs initial at t=0
        #   reconstitution  = <log(P-/F-)>_init - <log(P-/F-)>_new
        #                   = old members' P/F at t=1 minus new members' P/F at t=1
        #                   (positive when old members ended up more expensive)
        #
        # Check: fund + reval + recon
        #   = <log(F-/F0)>_init
        #     + [<log(P-/F-)>_new - <log(P0/F0)>_init]
        #     + [<log(P-/F-)>_init - <log(P-/F-)>_new]
        #   = <log(F-/F0)>_init + <log(P-/F-)>_init - <log(P0/F0)>_init
        #   = <log(P-/P0)>_init = total  ✓

        # Fundamentals: old members
        avg_log_fund_init = log_fund[mask_init].mean()

        # P/F of initial portfolio at start
        avg_log_pf_start_init = log_pf_initial[mask_init].mean()

        # P/F of old members at t=1 (pre-reconstitution)
        avg_log_pf_end_init = log_pf_pre[mask_init].mean()

        # P/F of new members at t=1 (post-reconstitution)
        avg_log_pf_end_new = log_pf_pre[mask_new].mean()

        # Three terms
        fundamentals = avg_log_fund_init
        revaluation = avg_log_pf_end_new - avg_log_pf_start_init
        reconstitution = avg_log_pf_end_init - avg_log_pf_end_new

        total = log_total_return[mask_init].mean()
        check = fundamentals + revaluation + reconstitution

        # P/F summary
        pf_init_start = np.exp(avg_log_pf_start_init)    # initial portfolio at t=0
        pf_init_end = np.exp(avg_log_pf_end_init)        # old members at t=1
        pf_new_end = np.exp(avg_log_pf_end_new)          # new members at t=1 (= recon portfolio)

        print(f"\n  {label}:")
        print(f"    P/F of initial portfolio at t=0          = {pf_init_start:.2f}")
        print(f"    P/F of initial portfolio at t=1          = {pf_init_end:.2f}")
        print(f"    P/F of reconstituted portfolio at t=1    = {pf_new_end:.2f}")
        print(f"    ---")
        print(f"    Log total return                      = {total*100:+.2f}%")
        print(f"    Fundamentals    log(F-/F0)_init       = {fundamentals*100:+.2f}%")
        print(f"    Revaluation     log(P+/F+)-log(P0/F0) = {revaluation*100:+.2f}%")
        print(f"    Reconstitution  log(P-/F-)-log(P+/F+) = {reconstitution*100:+.2f}%")
        print(f"    Check (sum of 3)                      = {check*100:+.2f}%")

        results[label.lower()] = {
            "total": total,
            "fund": fundamentals,
            "reval": revaluation,
            "recon": reconstitution,
        }

    # Value factor = Value - Growth
    v, g = results["value"], results["growth"]
    print(f"\n  Value Factor (Value - Growth):")
    for key, name in [("total", "Log total return"),
                      ("fund", "Fundamentals"),
                      ("reval", "Revaluation"),
                      ("recon", "Reconstitution")]:
        print(f"    {name:38s} = {(v[key] - g[key])*100:+.2f}%")

    # How many stocks switched?
    stayed_value = (value_mask_initial & value_mask_new).sum()
    stayed_growth = (growth_mask_initial & growth_mask_new).sum()
    switched = N - stayed_value - stayed_growth
    print(f"\n  Portfolio turnover: {switched} stocks switched categories "
          f"({switched/N*100:.1f}% of universe)")

    return {
        "value_total": v["total"], "growth_total": g["total"],
        "value_fund": v["fund"], "growth_fund": g["fund"],
        "value_reval": v["reval"], "growth_reval": g["reval"],
        "value_recon": v["recon"], "growth_recon": g["recon"],
    }


# ===========================================================================
# SCENARIO A: Price changes proportionally to fundamentals (P/F stays same)
# ===========================================================================

# MD: If P/F stays the same, that's equivalent to the market predicting 100%
# fundamentals growth persistence, which there won't be. The benefits don't
# show up this year, but next year, the growth yield should be much closer to
# zero. (Exactly zero, if there's complete mean reversion in fundamentals
# growth.)
P1_A = P0 * np.exp(log_f_growth)

results_A = analyze_scenario("A: Price tracks fundamentals (P/F unchanged)", P1_A)

print()

# ===========================================================================
# SCENARIO B: Value stocks up by expected mean; Growth by expected mean
# ===========================================================================
P1_B = np.empty(N)
P1_B[value_mask_initial] = P0[value_mask_initial] * np.exp(value_expected_log_growth)
P1_B[growth_mask_initial] = P0[growth_mask_initial] * np.exp(growth_expected_log_growth)

results_B = analyze_scenario(
    f"B: Value +{value_expected_log_growth*100:.2f}%, "
    f"Growth +{growth_expected_log_growth*100:.2f}% "
    f"(flat log returns within group)", P1_B)

print()

# ===========================================================================
# SCENARIO C: All prices move uniformly
# ===========================================================================
P1_C = P0.copy()

results_C = analyze_scenario("C: All prices unchanged", P1_C)

print()

# ===========================================================================
# SCENARIO D: Market re-prices each stock to the P/F "appropriate" for its
# realized fundamental growth.
# ===========================================================================
k = 10
P1_D = P0 * np.exp(log_f_growth + k * (log_f_growth - expected_log_f_growth))

results_D = analyze_scenario("D: Market re-prices to P/F implied by realized F growth", P1_D)

print()

# ===========================================================================
# SUMMARY TABLE
# ===========================================================================
print("=" * 80)
print("SUMMARY TABLE (all figures are log returns)")
print("=" * 80)
print()
all_results = [results_A, results_B, results_C, results_D]
labels1 = ["Scenario A", " Scenario B", "Scenario C", "Scenario D"]
labels2 = [" Fixed P/F", "Bucket-Wise", "  Fixed ΔP", " Re-Rating"]
header1 = f"{'':30s} | " + " | ".join(f"{l:>12s}" for l in labels1)
header2 = f"{'':30s} | " + " | ".join(f"{l:>12s}" for l in labels2)
print(header1)
print(header2)
print("-" * len(header1))

rows = [
    ("Value: Total Return",       [r["value_total"] for r in all_results]),
    ("Value: Fundamentals",        [r["value_fund"] for r in all_results]),
    ("Value: Revaluation",         [r["value_reval"] for r in all_results]),
    ("Value: Reconstitution",      [r["value_recon"] for r in all_results]),
    ("", None),
    ("Growth: Total Return",       [r["growth_total"] for r in all_results]),
    ("Growth: Fundamentals",       [r["growth_fund"] for r in all_results]),
    ("Growth: Revaluation",        [r["growth_reval"] for r in all_results]),
    ("Growth: Reconstitution",     [r["growth_recon"] for r in all_results]),
    ("", None),
    ("Factor: Total Return",       [r["value_total"] - r["growth_total"] for r in all_results]),
    ("Factor: Fundamentals",       [r["value_fund"] - r["growth_fund"] for r in all_results]),
    ("Factor: Revaluation",        [r["value_reval"] - r["growth_reval"] for r in all_results]),
    ("Factor: Reconstitution",     [r["value_recon"] - r["growth_recon"] for r in all_results]),
]

for label, vals in rows:
    if vals is None:
        print()
    else:
        nums = " | ".join(f"{v*100:+11.2f}%" for v in vals)
        print(f"  {label:28s} | {nums}")
