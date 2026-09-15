#!/usr/bin/env python3
"""The containment control on a deployed score, in the field. (2026-08-21)

Every contract the control has run on so far is one we built, and the reason is
a release gap: it needs per-example generations, the operative score, and a proxy
rule recomputable over an arbitrary span, and no sampled routing paper publishes
all three. PRM800K \\citep{lightman2024verify} does.

Its released `scored-test-samples.jsonl` carries, for every sampled solution to a
MATH test problem: the solution's steps as text, the process reward model's
per-step rating distribution, the released solution-level score, and independent
sympy-graded final-answer correctness. That is enough to instantiate a contract
where nothing is ours except the choice of proxy rule:

  score        the deployed PRM's prefix score over steps 1..k, aggregated as the
               product of P(rating=+1), the aggregation that reproduces the
               released prm_score (0.92 agreement; audit_prm800k_scored.py)
  score span   the first k steps -- the span at which a pruning PRM acts
  proxy        the gold answer string appears within the scored prefix: a cheap
               "is this partial solution on track" rule, computable by anyone
  construct    is_correct, the graded final answer
  disjoint     the same gold-answer rule read over steps k+1..L only

This is the configuration the paper argues is deployed: a score reading a partial
span while the construct depends on the whole solution. If the reported gap here
is containment, the disjoint control removes it. If it survives, a deployed score
has a proxy--construct mismatch that shared evidence does not explain -- the field
case the paper otherwise lacks.

AUCs are computed within problem and averaged over problems, matching
audit_prm800k_scored.py, because correctness varies within a problem and pooling
across problems would confound difficulty.

  python3 audit_prm800k_containment.py [--problems 500]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "prm800k_audit" / "scored-test-samples.jsonl"
OUT = ROOT / "analysis_results" / "prm800k_containment.json"
SPAN_FRACS = [0.25, 0.50]
N_BOOT = 2000
MIN_POS = 10


def norm(s: str) -> str:
    """Loose normalisation so the proxy rule is a plausible cheap one, not a strict grader."""
    s = s.lower().replace("\\!", "").replace("\\,", "").replace("$", "")
    s = re.sub(r"\\(left|right|dfrac|tfrac)", "", s)
    s = re.sub(r"\s+", "", s)
    return s


def load(max_problems: int):
    per = defaultdict(list)
    with DATA.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            steps, probs = r.get("steps"), r.get("rating_probs")
            gt = r.get("ground_truth_answer")
            if not steps or not probs or gt is None or len(steps) != len(probs):
                continue
            pid = r.get("unique_id")
            if pid not in per and len(per) >= max_problems:
                continue
            per[pid].append((steps, probs, bool(r.get("is_correct")), gt))
    return per


def per_problem_aucs(groups, xfn, yfn):
    """AUC within each problem, for problems where the label has both classes.
    Computed once; the bootstrap then resamples problems over these values, which is
    the correct resampling unit for a mean-of-within-problem statistic and avoids
    recomputing every AUC on every draw."""
    out = []
    for g in groups:
        y = np.array([yfn(s) for s in g])
        if len(set(y.tolist())) < 2:
            continue
        x = np.array([xfn(s) for s in g], dtype=float)
        out.append(roc_auc_score(y, x))
    return np.array(out, dtype=float)


def boot_gap_from(a_z: np.ndarray, a_y: np.ndarray, seed: int, orient: bool = True):
    """Bootstrap the orientation-robust gap between two mean-within-problem AUCs.
    The two arrays need not be the same length: a problem can be scoreable against
    one label and degenerate against the other, so each is resampled at its own size."""
    if len(a_z) == 0 or len(a_y) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(N_BOOT):
        z = a_z[rng.integers(0, len(a_z), len(a_z))].mean()
        y = a_y[rng.integers(0, len(a_y), len(a_y))].mean()
        d.append((abs(z - .5) - abs(y - .5)) if orient else (z - y))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", type=int, default=500)
    a = ap.parse_args()

    print(f"reading {DATA.name} ...", flush=True)
    per = load(a.problems)
    print(f"problems {len(per)}, samples {sum(len(v) for v in per.values())}")

    results = {}
    for frac in SPAN_FRACS:
        groups = []
        for pid, samples in per.items():
            g = []
            for steps, probs, correct, gt in samples:
                L = len(steps)
                k = max(1, int(round(frac * L)))
                if k >= L:                      # need a non-empty complement
                    continue
                pre = norm(" ".join(steps[:k]))
                rest = norm(" ".join(steps[k:]))
                gtn = norm(str(gt))
                s = float(np.prod([p.get("1", 0.0) for p in probs[:k]]))
                g.append({"s": s, "y": int(correct),
                          "z": int(gtn in pre), "zc": int(gtn in rest)})
            if len(g) >= 2:
                groups.append(g)

        n_samp = sum(len(g) for g in groups)
        ypos = sum(x["y"] for g in groups for x in g)
        zpos = sum(x["z"] for g in groups for x in g)
        zcpos = sum(x["zc"] for g in groups for x in g)
        print(f"\n=== span fraction {frac:.2f}: {len(groups)} problems, {n_samp} samples")
        print(f"    construct positives {ypos} ({ypos/max(1,n_samp):.3f}); "
              f"proxy in-prefix {zpos} ({zpos/max(1,n_samp):.3f}); "
              f"proxy off-span {zcpos} ({zcpos/max(1,n_samp):.3f})")

        f_s = lambda x: x["s"]
        A_y = per_problem_aucs(groups, f_s, lambda x: x["y"])
        A_z = per_problem_aucs(groups, f_s, lambda x: x["z"])
        A_zc = per_problem_aucs(groups, f_s, lambda x: x["zc"])
        ny, nz, nzc = len(A_y), len(A_z), len(A_zc)
        auc_y = float(A_y.mean()) if ny else float("nan")
        auc_z = float(A_z.mean()) if nz else float("nan")
        auc_zc = float(A_zc.mean()) if nzc else float("nan")
        d_auc = abs(auc_z - .5) - abs(auc_y - .5)
        d_dis = abs(auc_zc - .5) - abs(auc_y - .5)
        lo_a, hi_a = boot_gap_from(A_z, A_y, 1)
        lo_d, hi_d = boot_gap_from(A_zc, A_y, 2)

        print(f"    AUC(s,y)={auc_y:.3f} (n={ny})  AUC(s,z)={auc_z:.3f} (n={nz})  "
              f"AUC(s,z^c)={auc_zc:.3f} (n={nzc})")
        print(f"    dAUC={d_auc:+.3f} [{lo_a:+.3f},{hi_a:+.3f}]   "
              f"dDis={d_dis:+.3f} [{lo_d:+.3f},{hi_d:+.3f}]")

        underpowered = ypos < MIN_POS or (n_samp - ypos) < MIN_POS
        survives = (lo_d is not None and lo_d > 0) and not underpowered
        verdict = ("MISMATCH" if survives and d_dis >= 0.15 else
                   "CAUTION" if survives and d_dis >= 0.10 else
                   "ALIGNED")
        print(f"    verdict on the disjoint control: {verdict}")
        results[str(frac)] = {
            "problems": len(groups), "samples": n_samp,
            "y_positives": ypos, "z_positives": zpos, "zc_positives": zcpos,
            "auc_y": auc_y, "auc_z": auc_z, "auc_zc": auc_zc,
            "delta_auc_abs": d_auc, "delta_auc_ci": [lo_a, hi_a],
            "delta_dis_abs": d_dis, "delta_dis_ci": [lo_d, hi_d],
            "underpowered": underpowered, "verdict": verdict}

    print("\nreading:")
    hits = [f for f, r in results.items() if r["verdict"] in ("MISMATCH", "CAUTION")]
    if hits:
        print(f"  at span fraction(s) {', '.join(hits)} a deployed PRM's prefix score tracks a")
        print("  cheap on-track proxy better than it tracks final correctness, and the disjoint")
        print("  control does not explain the gap away. This is the field case the paper lacks.")
    else:
        print("  the gap does not survive the disjoint control at either span, so on this")
        print("  contract the deployed score is not caught by the audit. Report it as a clear.")

    OUT.write_text(json.dumps({"spans": results, "min_positives": MIN_POS,
                               "n_boot": N_BOOT}, indent=2))
    print("\n[wrote]", OUT)


if __name__ == "__main__":
    main()
