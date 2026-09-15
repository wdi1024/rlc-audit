#!/usr/bin/env python3
"""What do Math-Shepherd's step labels actually measure? (2026-08-21)

An earlier draft of this paper claimed Math-Shepherd "declares a prefix span it does
not have", on the grounds that its step labels are estimated by completing the
solution. That argument is wrong and we withdraw it. A process reward model's
estimand is the value function V(prefix) = E[correct | prefix], which is measurable
with respect to the prefix; estimating it by rollout is Monte Carlo integration, not
a span violation. Taken seriously the old argument would make every value-function
method a mis-declaration.

The real problem is one field to the left in the contract. The estimand is

    P(a completion from this prefix reaches the right answer)

and consumers read the label as

    "step k is correct"

which is a different quantity. A step can be wrong and recoverable, or locally
correct and on a doomed path. This script measures how far apart the two are, using
the fact that Math-Shepherd's hard labels are positive when *some* rollout succeeded
while the solution as written may still fail:

  recoverable-but-not-recovered   steps labelled + inside solutions whose own final
                                  answer is wrong. Under the "step k is correct"
                                  reading these are mislabelled; under the value
                                  reading they are exactly right.
  doomed-prefix positives         the same, restricted to prefixes where every
                                  remaining step is also labelled +, i.e. the label
                                  sequence never warns that the solution fails.

A large rate makes the construct mismatch a measured quantity rather than an
argument from the generation protocol.

  python3 mathshepherd_estimand_vs_construct.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SHARDS = sorted(Path("/tmp/mshep").glob("*.parquet"))
GOLD = Path("/tmp/gsm8k_gold.json")
OUT = (Path(__file__).resolve().parent.parent / "analysis_results"
       / "mathshepherd_estimand_vs_construct.json")

STEP = re.compile(r"Step\s+(\d+):\s*(.*?)\s*([+\-])\s*$")
ANSWER = re.compile(r"answer is:?\s*(.*)$", re.I)


def norm_num(s):
    s = re.sub(r"[^\d\.\-/]", "", str(s)).rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return s


def norm_text(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def main() -> None:
    gold = json.loads(GOLD.read_text())
    rows, n_seen = [], 0
    for shard in SHARDS:
        df = pd.read_parquet(shard, columns=["label", "task"])
        for lab in df[df["task"] == "GSM8K"]["label"]:
            n_seen += 1
            if "Step 1:" not in lab:
                continue
            problem = lab.split("Step 1:")[0].strip()
            key = norm_text(problem)
            if key not in gold:
                continue
            steps = []
            ok = True
            for line in lab[len(problem):].split("\n"):
                m = STEP.match(line.strip())
                if not m:
                    ok = False
                    break
                steps.append((m.group(2), 1 if m.group(3) == "+" else 0))
            if not ok or len(steps) < 3:
                continue
            m = ANSWER.search(steps[-1][0])
            if not m:
                continue
            rows.append((steps, int(norm_num(m.group(1)) == gold[key])))
        if len(rows) >= 20000:
            break

    print(f"solutions parsed: {len(rows)} (of {n_seen} GSM8K rows seen)")
    wrong = [s for s, y in rows if y == 0]
    print(f"  final answer wrong: {len(wrong)} ({len(wrong)/len(rows):.3f})")

    # Every non-final step, split by whether the solution it belongs to succeeded.
    pos_in_failed = tot_in_failed = 0
    all_pos_prefix_failed = 0
    for steps, y in rows:
        if y:
            continue
        body = steps[:-1]                      # exclude the answer-stating step
        tot_in_failed += len(body)
        pos_in_failed += sum(v for _, v in body)
        if body and all(v for _, v in body):
            all_pos_prefix_failed += 1

    n_failed = len(wrong)
    print("\n=== steps inside solutions whose final answer is WRONG ===")
    print(f"  non-final steps: {tot_in_failed}")
    print(f"  labelled '+'   : {pos_in_failed} ({pos_in_failed/max(1,tot_in_failed):.3f})")
    print(f"  failed solutions whose every non-final step is '+': "
          f"{all_pos_prefix_failed} ({all_pos_prefix_failed/max(1,n_failed):.3f})")

    print("\nreading:")
    print("  Under the reading consumers apply -- '+' means this step is correct -- every")
    print("  one of those positives inside a failed solution is a mislabelled step, and the")
    print("  all-positive failures are solutions the label sequence never flags at all.")
    print("  Under the estimand Math-Shepherd actually reports -- the probability that some")
    print("  completion from this prefix succeeds -- the same labels are correct: the prefix")
    print("  was recoverable and this particular continuation did not recover it.")
    print("  The two readings disagree on this many steps, which is the size of the construct")
    print("  mismatch. It is not a span violation: the value function is prefix-measurable.")

    OUT.write_text(json.dumps({
        "solutions": len(rows), "failed_solutions": n_failed,
        "nonfinal_steps_in_failed": tot_in_failed,
        "positive_steps_in_failed": pos_in_failed,
        "positive_rate_in_failed": pos_in_failed / max(1, tot_in_failed),
        "all_positive_failed_solutions": all_pos_prefix_failed,
        "all_positive_failed_rate": all_pos_prefix_failed / max(1, n_failed)}, indent=2))
    print("\n[wrote]", OUT)


if __name__ == "__main__":
    main()
