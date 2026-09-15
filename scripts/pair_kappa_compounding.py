#!/usr/bin/env python3
"""When is a low pair-level kappa evidence of an ill-defined event? (2026-08-13)

RLC-Audit flags a contract at the label stage when z and y are nominally the same
construct measured twice and do not agree.  MT-Bench is flagged at kappa(z,y)=0.074
while this paper's own contract reports a pair-level human-judge kappa of 0.266, and
a reviewer asked -- fairly -- where the line is.

The line should not be a bare kappa cutoff, because the pair-level event is an XOR of
two side labels and XOR compounds disagreement: two sides that each agree at kappa
0.9 can produce a pair-level kappa far below 0.9 purely mechanically.  The right
question is therefore not "is pair kappa low?" but "is pair kappa low *given* the side
agreement and the prevalence?"

This script derives that expectation by simulation.  For a grid of side-level
agreement and side refusal prevalence, it simulates two annotators labelling both
sides independently, forms each annotator's pair-level XOR, and records the resulting
pair-level kappa.  A contract is then well-defined at the label stage when its
observed pair kappa is consistent with the value its own side agreement predicts, and
flagged when it falls far below that prediction.

  python3 pair_kappa_compounding.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
N_SIM = 4000
N_ITEMS = 450
RNG = np.random.default_rng(0)

# Observed anchors.  Side kappas and pair kappa for this paper's primary contract
# come from the resolved human sample; the MT-Bench row is the RouterBench slice.
OBSERVED = {
    "primary contract (human vs judge)": {"side_kappas": [0.930, 0.715],
                                          "pair_kappa": 0.266, "n": 109},
    "RouterBench MT-Bench (judge vs judge)": {"side_kappas": [0.44, 0.50],
                                              "pair_kappa": 0.074, "n": 80},
}


def simulate_pair_kappa(side_kappa: float, prevalence: float, n: int,
                        n_sim: int, rng) -> np.ndarray:
    """Pair-level XOR kappa implied by two sides each agreeing at `side_kappa`.

    Annotator A draws the truth; annotator B copies it with a per-item flip rate
    chosen so that the side-level kappa matches the target.  Both sides are drawn
    independently, then each annotator's pair label is the XOR of their two sides.
    """
    p = prevalence
    p_e = p * p + (1 - p) * (1 - p)
    # kappa = (p_o - p_e) / (1 - p_e)  =>  p_o = kappa (1 - p_e) + p_e
    p_o = side_kappa * (1 - p_e) + p_e
    flip = np.clip(1.0 - p_o, 0.0, 1.0)

    out = np.empty(n_sim)
    for s in range(n_sim):
        pair_a = np.zeros(n, dtype=int)
        pair_b = np.zeros(n, dtype=int)
        for _ in range(2):                       # two sides of the pair
            truth = (rng.random(n) < p).astype(int)
            other = np.where(rng.random(n) < flip, 1 - truth, truth)
            pair_a ^= truth
            pair_b ^= other
        if len(set(pair_a)) < 2 or len(set(pair_b)) < 2:
            out[s] = np.nan
            continue
        out[s] = cohen_kappa_score(pair_a, pair_b)
    return out[~np.isnan(out)]


def simulate_correlated(side_kappa: float, prevalence: float, n: int, rho: float,
                        n_sim: int, rng) -> np.ndarray:
    """Same simulation, but a fraction `rho` of items are ambiguous for both sides.

    Independence is the wrong null for this construct: a prompt whose paired outputs
    leave a reader without enough context confuses that reader on *both* models, so
    the two side errors co-occur.  Here `rho` of items carry a shared ambiguity flag,
    and on those items the second annotator's label is drawn independently of the
    first on both sides; elsewhere the independent per-side flip rate applies.  The
    per-item flip rate is rescaled so the marginal side kappa still matches.
    """
    p = prevalence
    p_e = p * p + (1 - p) * (1 - p)
    p_o = side_kappa * (1 - p_e) + p_e
    # On ambiguous items agreement falls to chance (p_e); solve for the flip rate
    # on the remaining items so the marginal side agreement is still p_o.
    if rho >= 1.0:
        flip_clean = 0.0
    else:
        p_o_clean = (p_o - rho * p_e) / (1.0 - rho)
        flip_clean = float(np.clip(1.0 - p_o_clean, 0.0, 1.0))

    out = np.empty(n_sim)
    for s in range(n_sim):
        amb = rng.random(n) < rho
        pair_a = np.zeros(n, dtype=int)
        pair_b = np.zeros(n, dtype=int)
        for _ in range(2):
            truth = (rng.random(n) < p).astype(int)
            indep = (rng.random(n) < p).astype(int)
            noisy = np.where(rng.random(n) < flip_clean, 1 - truth, truth)
            other = np.where(amb, indep, noisy)
            pair_a ^= truth
            pair_b ^= other
        if len(set(pair_a)) < 2 or len(set(pair_b)) < 2:
            out[s] = np.nan
            continue
        out[s] = cohen_kappa_score(pair_a, pair_b)
    return out[~np.isnan(out)]


def main():
    prevalence = 0.45          # side-level refusal rate on XSTest for this pair
    print(f"simulated pair-level XOR kappa, side prevalence {prevalence:.2f}, "
          f"n={N_ITEMS}, {N_SIM} draws\n")
    print(f"{'side kappa':>11} {'pair kappa mean':>16} {'5th':>7} {'95th':>7} {'shrinkage':>10}")
    grid = {}
    for sk in [0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        v = simulate_pair_kappa(sk, prevalence, N_ITEMS, N_SIM // 4, RNG)
        m = float(v.mean())
        grid[sk] = {"mean": m, "p5": float(np.percentile(v, 5)),
                    "p95": float(np.percentile(v, 95)), "ratio": m / sk}
        print(f"{sk:>11.2f} {m:>16.3f} {np.percentile(v,5):>7.3f} "
              f"{np.percentile(v,95):>7.3f} {m/sk:>10.2f}")

    print("\nobserved contracts against the value their own side agreement predicts:")
    rows = {}
    for name, o in OBSERVED.items():
        sk = float(np.mean(o["side_kappas"]))
        v = simulate_pair_kappa(sk, prevalence, o["n"], N_SIM // 4, RNG)
        lo, hi = np.percentile(v, [5, 95])
        inside = lo <= o["pair_kappa"] <= hi
        print(f"  {name}")
        print(f"    mean side kappa {sk:.3f} -> predicted pair kappa "
              f"{v.mean():.3f} [{lo:.3f}, {hi:.3f}]  (n={o['n']})")
        print(f"    observed pair kappa {o['pair_kappa']:.3f} -> "
              f"{'consistent with compounding' if inside else 'BELOW the prediction'}")
        rows[name] = {"mean_side_kappa": sk, "predicted_mean": float(v.mean()),
                      "predicted_p5": float(lo), "predicted_p95": float(hi),
                      "observed_pair_kappa": o["pair_kappa"], "n": o["n"],
                      "consistent": bool(inside)}

    # How much shared item ambiguity is needed to reach the observed value?
    print("\nshared-ambiguity sweep for the primary contract "
          "(side kappa 0.823, n=109): what fraction of items must be ambiguous "
          "for both sides to reproduce pair kappa 0.266?")
    sweep = {}
    target = OBSERVED["primary contract (human vs judge)"]["pair_kappa"]
    best = None
    for rho in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        v = simulate_correlated(0.823, prevalence, 109, rho, N_SIM // 8, RNG)
        lo, hi = np.percentile(v, [5, 95])
        inside = lo <= target <= hi
        sweep[rho] = {"mean": float(v.mean()), "p5": float(lo), "p95": float(hi),
                      "covers_observed": bool(inside)}
        mark = "  <- covers 0.266" if inside else ""
        print(f"  rho={rho:.1f}  pair kappa {v.mean():.3f} [{lo:.3f}, {hi:.3f}]{mark}")
        if inside and best is None:
            best = rho
    print(f"\nsmallest shared-ambiguity fraction consistent with the observation: "
          f"{best if best is not None else 'none in sweep'}")

    out = {"prevalence": prevalence, "n_items": N_ITEMS, "grid": grid, "observed": rows,
           "ambiguity_sweep": sweep, "min_rho_covering_observed": best}
    p = ROOT / "analysis_results" / "pair_kappa_compounding.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\n[wrote] {p}")


if __name__ == "__main__":
    main()
