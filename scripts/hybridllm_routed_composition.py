#!/usr/bin/env python3
"""What does a deployed routing target actually escalate? (reviewer request, 2026-08-13)

The MixInstruct audits so far are ranking statistics: agreement, kappa, and how well
length predicts each label.  This paper's own frame says a routing claim is about the
operating point, so the deployed target deserves the same treatment: build the queue a
router using the BARTScore gap would act on, and count what is in it.

A router acts most decisively where the gap is largest, so the queue is the top-B
comparisons by |gap| at a 10% budget.  Inside that queue we ask two things:

  does the independent adequacy verdict contradict the target more or less often
  than it does overall?

  and are the queue's cases the ones with large length differences?

If the target were a clean measurement of adequacy, its most confident decisions
would be its most reliable ones.  If it is length-coupled, the confident queue is
where length differences are largest, and confidence buys agreement with length
instead of agreement with the construct.

  python3 hybridllm_routed_composition.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
BUDGETS = [0.01, 0.05, 0.10, 0.25, 0.50]
N_BOOT = 2000
RNG = np.random.default_rng(0)


def main():
    ds = load_dataset("llm-blender/mix-instruct", split="test")
    gaps, ys, lens, pairs = [], [], [], []
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
            gaps.append(bs[a] - bs[b])
            ys.append(y)
            lens.append(len(cand[a]["text"] or "") - len(cand[b]["text"] or ""))
            pairs.append(f"{a} vs {b}")

    gap = np.array(gaps)
    y = np.array(ys)
    ln = np.array(lens)
    z = (gap > 0).astype(int)              # what the deployed target decides
    conf = np.abs(gap)                     # how decisively it decides
    n = len(y)
    overall_wrong = float((z != y).mean())
    overall_len = float(np.abs(ln).mean())
    print(f"n={n}  overall disagreement with the construct {overall_wrong:.3f}  "
          f"mean |length difference| {overall_len:.0f} chars\n")

    print(f"{'budget':>8} {'B':>8} {'disagree in queue':>18} {'mean |len diff|':>16} "
          f"{'vs overall':>12}")
    rows = {}
    order = np.argsort(-conf, kind="stable")
    for frac in BUDGETS:
        B = max(1, int(round(frac * n)))
        idx = order[:B]
        wrong = float((z[idx] != y[idx]).mean())
        mlen = float(np.abs(ln[idx]).mean())
        # bootstrap the queue's disagreement rate
        bs_vals = [float((z[idx][s] != y[idx][s]).mean())
                   for s in (RNG.integers(0, B, B) for _ in range(500))]
        lo, hi = np.percentile(bs_vals, [2.5, 97.5])
        print(f"{frac:>8.2f} {B:>8} {wrong:>10.3f} [{lo:.3f},{hi:.3f}] "
              f"{mlen:>16.0f} {mlen / overall_len:>11.2f}x")
        rows[frac] = {"B": B, "disagreement": wrong, "ci": [float(lo), float(hi)],
                      "mean_abs_len_diff": mlen, "len_ratio": mlen / overall_len}

    # The counterfactual a router cares about: at the same budget, how would a queue
    # ranked by length alone compare?  If the target's confident queue looks like the
    # length queue, confidence is buying length agreement.
    B = max(1, int(round(0.10 * n)))
    len_idx = np.argsort(-np.abs(ln), kind="stable")[:B]
    tgt_idx = order[:B]
    overlap = len(set(len_idx.tolist()) & set(tgt_idx.tolist())) / B
    print(f"\nat a 10% budget the target's queue and a length-ranked queue share "
          f"{overlap*100:.1f}% of their examples")
    print(f"  length-ranked queue disagrees with the construct on "
          f"{float((z[len_idx] != y[len_idx]).mean()):.3f} of its cases")

    out = {"n": int(n), "overall_disagreement": overall_wrong,
           "overall_mean_abs_len_diff": overall_len, "budgets": rows,
           "queue_overlap_with_length_ranking_at_10pct": overlap,
           "length_queue_disagreement": float((z[len_idx] != y[len_idx]).mean())}
    p = ROOT / "analysis_results" / "hybridllm_routed_composition.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\n[wrote] {p}")


if __name__ == "__main__":
    main()
