#!/usr/bin/env python3
"""Are the cached PRM rewards a working score, or is our scoring broken? (2026-08-21)

`audit_processbench_containment.py` produced a deployed-PRM score whose AUC against
final-answer correctness is 0.49-0.59 at every span and under every prefix
aggregation. That should be impossible on this benchmark: in ProcessBench a process
error and a wrong final answer travel together (kappa 0.97 on gsm8k, 0.81 on math,
and zero error-free-but-wrong solutions in any split), so a model that finds errors
must predict correctness. Either our scoring is wrong or the checkpoint is not doing
what its card says.

One test decides it, and it needs no GPU because the per-step rewards are cached.
ProcessBench's own task is to name the first erroneous step. Run that task on the
cached rewards and compare with the published numbers for this checkpoint:

  prediction   the first step whose reward falls below a threshold, else -1
  truth        the released `label`
  metric       accuracy on erroneous solutions, accuracy on correct ones, and their
               harmonic mean, which is the benchmark's F1

If the F1 lands anywhere near the published figures, the rewards are right and the
flat correctness curve is a real property we have to explain. If error localisation
is also at chance, our scoring is broken and every ProcessBench number is void.

We also report the score's resolution, because a score quantised to a handful of
distinct values cannot rank anything -- the paper's own precondition, applied to a
score we produced.

  python3 diagnose_processbench_rewards.py --cache pb_rewards.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SPLITS = ["gsm8k", "math", "olympiadbench", "omnimath"]
# Published ProcessBench F1 for Qwen2.5-Math-PRM-7B, from the model card, used only
# as an order-of-magnitude reference: the question is chance-vs-not, not the decimal.
REFERENCE_F1 = {"gsm8k": 82.4, "math": 77.6, "olympiadbench": 67.5, "omnimath": 66.3}


def first_below(r, thr):
    idx = np.nonzero(np.asarray(r) < thr)[0]
    return int(idx[0]) if len(idx) else -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="pb_rewards.json from the audit run")
    ap.add_argument("--data-dir", default=".", help="directory holding pb_<split>.json")
    a = ap.parse_args()

    cache = json.loads(Path(a.cache).read_text())
    print(f"cached solutions: {len(cache)}\n")

    allv = np.concatenate([np.asarray(v, dtype=np.float64) for v in cache.values()])
    print("=== score resolution (the paper's own precondition, applied to our score) ===")
    print(f"  per-step rewards: n={len(allv)}, distinct={len(np.unique(allv))}, "
          f"min={allv.min():.4f}, median={np.median(allv):.4f}, max={allv.max():.4f}")
    for q in (0.01, 0.1, 0.5, 0.9, 0.99):
        print(f"    q{q:<5} {np.quantile(allv, q):.4f}")
    if len(np.unique(allv)) < 50:
        print("  WARNING: the reward takes very few distinct values; it cannot rank.")

    print("\n=== ProcessBench's own task: locate the first erroneous step ===")
    print(f"  {'split':14s} {'thr':>5s} {'n':>5s} {'acc_err':>8s} {'acc_ok':>7s} "
          f"{'F1':>6s}   published")
    for split in SPLITS:
        p = Path(a.data_dir) / f"pb_{split}.json"
        if not p.exists():
            print(f"  {split}: {p} missing, skipped")
            continue
        rows = [(r, cache[r["id"]]) for r in json.loads(p.read_text()) if r["id"] in cache]
        if not rows:
            print(f"  {split}: nothing cached")
            continue
        best = None
        for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            pred = np.array([first_below(rw, thr) for _, rw in rows])
            truth = np.array([r["label"] for r, _ in rows])
            err = truth != -1
            # the benchmark scores an erroneous case correct only on the exact index
            acc_err = float((pred[err] == truth[err]).mean()) if err.any() else float("nan")
            acc_ok = float((pred[~err] == -1).mean()) if (~err).any() else float("nan")
            f1 = (0.0 if (acc_err + acc_ok) == 0 else
                  2 * acc_err * acc_ok / (acc_err + acc_ok)) * 100
            if best is None or f1 > best[3]:
                best = (thr, acc_err, acc_ok, f1)
        thr, acc_err, acc_ok, f1 = best
        ref = REFERENCE_F1.get(split)
        verdict = "  <-- at chance" if f1 < 15 else ""
        print(f"  {split:14s} {thr:5.1f} {len(rows):5d} {acc_err:8.3f} {acc_ok:7.3f} "
              f"{f1:6.1f}   ~{ref}{verdict}")

    print("\nreading:")
    print("  The threshold above is chosen to maximise F1, which favours the rewards;")
    print("  a best-case F1 near zero means the cached rewards do not localise errors")
    print("  at all, and the scoring pipeline -- not the checkpoint -- is the problem.")


if __name__ == "__main__":
    main()
