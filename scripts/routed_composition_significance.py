#!/usr/bin/env python3
"""Significance of the routed-composition improvement (reviewer request, 2026-08-13).

The paper reports that at budget B the audit-guided contract revision raises the
number of semantic disagreements in the routed queue from 11/55 (matched
raw-prefix baseline) to 20/55 (final-span) and 28/55 (composite witness).  Those
are counts, and the paper has so far reported no test for them: the paired AUC
delta's CI includes zero, so the composition claim is load-bearing and needs its
own uncertainty statement.

Two tests, both on the same 548 clean-run items:

  paired bootstrap  resample items with replacement, recompute BOTH top-B queues,
                    and take the difference in semantic-disagreement counts.  This
                    respects the pairing (the two scores rank the same items) and
                    gives a CI on the improvement.

  permutation       shuffle the semantic labels y across items, recompute both
                    queues, and record the count difference.  This is the null in
                    which neither score carries semantic information, and gives a
                    p-value for the observed gap.

Scores are rebuilt with the analysis script's own functions, so they are the same
numbers that appear in the paper rather than a reimplementation.

  python3 routed_composition_significance.py
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
N_BOOT = 10000
N_PERM = 10000
RNG = np.random.default_rng(0)

spec = importlib.util.spec_from_file_location(
    "agcr", HERE / "analyze_rlc_composite_router.py")
agcr = importlib.util.module_from_spec(spec)
import sys
sys.modules["agcr"] = agcr        # dataclass 정의가 sys.modules 조회를 하므로 먼저 등록
spec.loader.exec_module(agcr)


def topb_count(score: np.ndarray, y: np.ndarray, budget: int) -> int:
    """Semantic disagreements inside the top-`budget` of `score` (ties by index)."""
    idx = np.argsort(-score, kind="stable")[:budget]
    return int(y[idx].sum())


def main():
    data_dir = ROOT / "results" / "disagree_routing"
    pair = agcr.PAIR_SPECS[0] if hasattr(agcr, "PAIR_SPECS") else None
    if pair is None:
        raise SystemExit("PAIR_SPECS not found in analyze_rlc_composite_router.py")

    # Rebuild the primary clean-run pair exactly as the paper does.
    rec_a = agcr.load_records(data_dir, pair.phase_a, pair.model_a)
    rec_b = agcr.load_records(data_dir, pair.phase_b, pair.model_b)
    ja = agcr.load_final_judge(data_dir, pair.phase_a, pair.model_a, pair.judge_tag)
    jb = agcr.load_final_judge(data_dir, pair.phase_b, pair.model_b, pair.judge_tag)
    pa = agcr.load_prompts(data_dir, pair.phase_a)
    pb = agcr.load_prompts(data_dir, pair.phase_b)
    ids = sorted(set(rec_a) & set(rec_b) & set(ja) & set(jb) & set(pa) & set(pb))
    ids = [i for i in ids if pa[i] == pb[i]
           and agcr.final_text(rec_a[i]).strip() and agcr.final_text(rec_b[i]).strip()]

    raw_a = [agcr.raw_text(rec_a[i]) for i in ids]
    raw_b = [agcr.raw_text(rec_b[i]) for i in ids]
    fin_a = [agcr.final_text(rec_a[i]) for i in ids]
    fin_b = [agcr.final_text(rec_b[i]) for i in ids]
    sem_a = np.array([int(ja[i]) for i in ids])
    sem_b = np.array([int(jb[i]) for i in ids])
    y = (sem_a != sem_b).astype(int)
    marker = np.array([int(agcr.is_kw(a) != agcr.is_kw(b))
                       for a, b in zip(fin_a, fin_b)], dtype=float)

    baseline = agcr.tfidf_distance(raw_a, raw_b, prefix=50)
    final_tfidf = agcr.tfidf_distance(fin_a, fin_b)
    composite = 0.7 * agcr.zscore(final_tfidf) + 0.3 * marker
    B = max(1, int(round(len(ids) * 0.10)))

    obs = {name: topb_count(s, y, B) for name, s in
           (("baseline", baseline), ("final_span", final_tfidf), ("composite", composite))}
    print(f"n={len(ids)}  budget B={B}  semantic positives={int(y.sum())}")
    print(f"observed routed semantic disagreements: "
          f"baseline {obs['baseline']}/{B}, final-span {obs['final_span']}/{B}, "
          f"composite {obs['composite']}/{B}\n")

    out = {"n": len(ids), "budget": B, "observed": obs, "tests": {}}
    for name, score in (("final_span", final_tfidf), ("composite", composite)):
        diff = obs[name] - obs["baseline"]

        boots = np.empty(N_BOOT)
        n = len(ids)
        for k in range(N_BOOT):
            sel = RNG.integers(0, n, n)
            boots[k] = (topb_count(score[sel], y[sel], B)
                        - topb_count(baseline[sel], y[sel], B))
        lo, hi = np.percentile(boots, [2.5, 97.5])

        perms = np.empty(N_PERM)
        for k in range(N_PERM):
            yp = RNG.permutation(y)
            perms[k] = topb_count(score, yp, B) - topb_count(baseline, yp, B)
        p = float((np.abs(perms) >= abs(diff)).mean())

        print(f"{name} vs baseline: +{diff} cases  "
              f"bootstrap 95% CI [{lo:+.1f}, {hi:+.1f}]  permutation p={p:.4f}")
        out["tests"][name] = {"diff": int(diff), "ci_lo": float(lo), "ci_hi": float(hi),
                              "perm_p": p, "n_boot": N_BOOT, "n_perm": N_PERM}

    p_out = ROOT / "analysis_results" / "routed_composition_significance.json"
    json.dump(out, open(p_out, "w"), indent=2)
    print(f"\n[wrote] {p_out}")


if __name__ == "__main__":
    main()
