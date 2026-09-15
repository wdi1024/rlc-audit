#!/usr/bin/env python3
"""Tie sensitivity of the binary marker cue's routed count (2026-08-20).

A reviewer noticed that the marker-alone routed count is reported as 21/55 in
Table `tab:agcr-ablation` and 20/55 in Table `tab:marker-strip`.  Neither is a
typo.  The marker score is binary, so on the clean run 133 items share its top
value while the budget holds only 55 slots: which 55 enter the queue is decided
by the sort's tie-break, not by the score.  `analyze_rlc_composite_router.py`
calls `np.argsort(-score)` (introsort, unstable) and `marker_strip_symmetry.py`
calls `np.argsort(-score, kind="stable")`, and the two orderings disagree.

The fix is to stop reporting a single draw.  This script reports the routed
count as an expectation over random tie-breaks with a 95% interval, and checks
that every continuous score in those tables has a unique B-th value and is
therefore not exposed to the same ambiguity.

  python3 marker_tie_sensitivity.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
N_DRAWS = 20_000
SEED = 0

spec = importlib.util.spec_from_file_location(
    "agcr", HERE / "analyze_rlc_composite_router.py")
agcr = importlib.util.module_from_spec(spec)
sys.modules["agcr"] = agcr
spec.loader.exec_module(agcr)


def load_clean_run() -> dict[str, np.ndarray]:
    data_dir = ROOT / "results" / "disagree_routing"
    spec0 = agcr.PAIR_SPECS[0]
    rec_a = agcr.load_records(data_dir, spec0.phase_a, spec0.model_a)
    rec_b = agcr.load_records(data_dir, spec0.phase_b, spec0.model_b)
    ja = agcr.load_final_judge(data_dir, spec0.phase_a, spec0.model_a, spec0.judge_tag)
    jb = agcr.load_final_judge(data_dir, spec0.phase_b, spec0.model_b, spec0.judge_tag)
    pa = agcr.load_prompts(data_dir, spec0.phase_a)
    pb = agcr.load_prompts(data_dir, spec0.phase_b)

    ids = sorted(set(rec_a) & set(rec_b) & set(ja) & set(jb) & set(pa) & set(pb))
    ids = [i for i in ids if pa[i] == pb[i]
           and agcr.final_text(rec_a[i]).strip() and agcr.final_text(rec_b[i]).strip()]

    fin_a = [agcr.final_text(rec_a[i]) for i in ids]
    fin_b = [agcr.final_text(rec_b[i]) for i in ids]
    raw_a = [agcr.raw_text(rec_a[i]) for i in ids]
    raw_b = [agcr.raw_text(rec_b[i]) for i in ids]

    marker = np.array([int(agcr.is_kw(a) != agcr.is_kw(b))
                       for a, b in zip(fin_a, fin_b)], dtype=float)
    return {
        "y": np.array([int(ja[i] != jb[i]) for i in ids]),
        "marker": marker,
        "raw prefix TF-IDF": agcr.tfidf_distance(raw_a, raw_b, prefix=50),
        "final TF-IDF": agcr.tfidf_distance(fin_a, fin_b),
        "composite (lambda=0.7)": 0.7 * agcr.zscore(agcr.tfidf_distance(fin_a, fin_b))
        + 0.3 * marker,
    }


def tie_distribution(y: np.ndarray, score: np.ndarray, budget: int) -> dict:
    """Routed count over random tie-breaks, by permuting before a stable sort."""
    rng = np.random.default_rng(SEED)
    draws = np.empty(N_DRAWS, dtype=int)
    for k in range(N_DRAWS):
        perm = rng.permutation(len(score))
        draws[k] = int(y[perm[np.argsort(-score[perm], kind="stable")][:budget]].sum())
    return {
        "expected": float(draws.mean()),
        "ci95": [int(np.percentile(draws, 2.5)), int(np.percentile(draws, 97.5))],
        "range": [int(draws.min()), int(draws.max())],
        "n_draws": N_DRAWS,
    }


def tie_exposure(score: np.ndarray, budget: int) -> dict:
    """How many items sit exactly at the B-th value: 1 means the cut is unique."""
    threshold = np.sort(-score)[budget - 1]
    return {
        "distinct_values": int(len(np.unique(score))),
        "items_at_cut": int((score == -threshold).sum()),
        "items_at_top_value": int((score == score.max()).sum()),
    }


def main() -> None:
    d = load_clean_run()
    y, marker, budget = d["y"], d["marker"], 55
    n = len(y)

    tied_at_top = int((marker == marker.max()).sum())
    pos_among_tied = int(y[marker == marker.max()].sum())
    dist = tie_distribution(y, marker, budget)
    hypergeom = budget * pos_among_tied / tied_at_top

    print(f"clean run n={n}, budget B={budget}, semantic positives={int(y.sum())}\n")
    print(f"marker cue is binary: {tied_at_top} items tie at its top value, "
          f"{pos_among_tied} of them semantic positives")
    print(f"  routed count over {N_DRAWS} random tie-breaks: "
          f"expected {dist['expected']:.2f}, 95% CI {dist['ci95']}, "
          f"range {dist['range']}")
    print(f"  hypergeometric expectation {budget}*{pos_among_tied}/{tied_at_top} "
          f"= {hypergeom:.2f}")
    print(f"  the two published draws: unstable sort "
          f"{int(y[np.argsort(-marker)[:budget]].sum())}, stable sort "
          f"{int(y[np.argsort(-marker, kind='stable')[:budget]].sum())}\n")

    print("tie exposure of the continuous scores (items_at_cut == 1 means determinate):")
    exposure = {}
    for name in ("raw prefix TF-IDF", "final TF-IDF", "composite (lambda=0.7)"):
        exposure[name] = tie_exposure(d[name], budget)
        e = exposure[name]
        print(f"  {name:24s} distinct={e['distinct_values']}/{n}  "
              f"items_at_cut={e['items_at_cut']}")

    out = {
        "n": n, "budget": budget, "semantic_positives": int(y.sum()),
        "marker": {
            "tied_at_top_value": tied_at_top,
            "semantic_positives_among_tied": pos_among_tied,
            "hypergeometric_expectation": hypergeom,
            **dist,
        },
        "continuous_score_tie_exposure": exposure,
    }
    path = ROOT / "analysis_results" / "marker_tie_sensitivity.json"
    path.write_text(json.dumps(out, indent=2))
    print("\n[wrote]", path)


if __name__ == "__main__":
    main()
