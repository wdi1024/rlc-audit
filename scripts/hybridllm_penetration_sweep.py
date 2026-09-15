#!/usr/bin/env python3
"""When does a length asymmetry reach the operating point? (reviewer request, 2026-08-13)

The pooled MixInstruct audit found the deployed BARTScore target more length-coupled
than the construct in ranking, but the coupling did not reach the routed queue: the
target's most decisive 10% contradicts the independent verdict less often than the
population does.  A reviewer asked the useful follow-up -- is that true of every model
pair, or does the asymmetry penetrate the queue under some conditions?

For each model pair with enough directional comparisons we compute, at a 10% budget:
the queue's disagreement with the construct against the pair's own base rate, how much
of the queue a length-ranked queue reproduces, and the length coupling of the target
and the construct.  Penetration means a queue that is worse than base rate, or one
that a length ranking largely reproduces.

  python3 hybridllm_penetration_sweep.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
MIN_N = 400
BUDGET = 0.10


def main():
    ds = load_dataset("llm-blender/mix-instruct", split="test")
    per = defaultdict(lambda: {"gap": [], "y": [], "len": []})
    for row in ds:
        cand = {c["model"]: c for c in row["candidates"]}
        bs = {m: c["scores"]["bartscore"] for m, c in cand.items()
              if c["scores"].get("bartscore") is not None}
        try:
            cmp = json.loads(row["cmp_results"]) if row["cmp_results"] else {}
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(cmp, dict):
            continue
        for key, verdict in cmp.items():
            if "," not in key:
                continue
            a, b = key.split(",", 1)
            if a not in bs or b not in bs:
                continue
            if verdict == "A is better":
                y = 1
            elif verdict == "B is better":
                y = 0
            else:
                continue
            d = per[f"{a} vs {b}"]
            d["gap"].append(bs[a] - bs[b])
            d["y"].append(y)
            d["len"].append(len(cand[a]["text"] or "") - len(cand[b]["text"] or ""))

    rows = []
    for name, d in per.items():
        gap = np.array(d["gap"]); y = np.array(d["y"]); ln = np.array(d["len"])
        if len(y) < MIN_N or len(set(y)) < 2:
            continue
        z = (gap > 0).astype(int)
        base = float((z != y).mean())
        B = max(1, int(round(BUDGET * len(y))))
        q = np.argsort(-np.abs(gap), kind="stable")[:B]
        lq = np.argsort(-np.abs(ln), kind="stable")[:B]
        rows.append({
            "pair": name, "n": int(len(y)), "base_disagreement": base,
            "queue_disagreement": float((z[q] != y[q]).mean()),
            "penetration": float((z[q] != y[q]).mean()) - base,
            "overlap_with_length_queue": len(set(q.tolist()) & set(lq.tolist())) / B,
            "auc_len_proxy": float(roc_auc_score(z, ln)) if len(set(z)) > 1 else float("nan"),
            "auc_len_construct": float(roc_auc_score(y, ln)),
        })

    rows.sort(key=lambda r: -r["penetration"])
    print(f"{'pair':46} {'n':>6} {'base':>6} {'queue':>6} {'penetration':>12} "
          f"{'len overlap':>12}")
    for r in rows[:12]:
        print(f"{r['pair'][:44]:46} {r['n']:>6} {r['base_disagreement']:>6.3f} "
              f"{r['queue_disagreement']:>6.3f} {r['penetration']:>+12.3f} "
              f"{r['overlap_with_length_queue']:>12.3f}")
    pen = np.array([r["penetration"] for r in rows])
    worse = [r for r in rows if r["penetration"] > 0]
    print(f"\n{len(rows)} pairs with n>={MIN_N}; queue is worse than base rate in "
          f"{len(worse)} of them")
    print(f"penetration: median {np.median(pen):+.3f}, range [{pen.min():+.3f}, {pen.max():+.3f}]")
    if worse:
        w = max(worse, key=lambda r: r["penetration"])
        print(f"worst pair: {w['pair']} (+{w['penetration']:.3f}, length overlap "
              f"{w['overlap_with_length_queue']:.2f}, length AUC proxy "
              f"{w['auc_len_proxy']:.3f} vs construct {w['auc_len_construct']:.3f})")
    lam = np.array([r["auc_len_proxy"] - r["auc_len_construct"] for r in rows])
    if len(rows) > 3:
        c = float(np.corrcoef(lam, pen)[0, 1])
        print(f"correlation between a pair's length asymmetry and its penetration: r={c:+.3f}")
    out = {"min_n": MIN_N, "budget": BUDGET, "n_pairs": len(rows),
           "n_pairs_worse_than_base": len(worse),
           "penetration_median": float(np.median(pen)),
           "penetration_range": [float(pen.min()), float(pen.max())],
           "corr_asymmetry_penetration": float(np.corrcoef(lam, pen)[0, 1]) if len(rows) > 3 else None,
           "pairs": rows}
    p = ROOT / "analysis_results" / "hybridllm_penetration_sweep.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\n[wrote] {p}")


if __name__ == "__main__":
    main()
