#!/usr/bin/env python3
"""Does the field verdict hold across proxy rules, or did we pick a lucky one? (2026-08-21)

`audit_prm800k_containment.py` runs the containment control on a deployed score --
OpenAI's process reward model, scored over the prefix of a released solution -- and
the contract clears. That result rests on one proxy rule we chose (the gold answer
appearing inside the scored prefix), and a single choice is not an audit of the
deployed system, it is an audit of our rule. A reader is entitled to ask whether a
different cheap label would have caught something.

So we run a family. Each is a rule over text, computable on any span, of the kind
someone validating a step-level scorer might actually reach for:

  gold_in_prefix   the gold answer string appears in the scored prefix
  boxed            the prefix already contains a \\boxed{...} final answer
  backtrack        the prefix contains backtracking or hedging markers
                   ("wait", "hmm", "actually", "let me try", "on second thought"),
                   the surface signature of a solution going badly
  arith_density    the prefix is arithmetic-heavy (digits and operators above the
                   median), a content-blind style feature
  length           the prefix is longer than the median, the crudest surface proxy

For each, the construct is the released sympy-graded correctness and the score is
the deployed PRM prefix score, so only the proxy varies. A contract that clears
under every rule is much stronger evidence that the deployed score is sound than
one that clears under ours; a contract that fails under any is the field case the
paper otherwise lacks, and we would report it as such.

The parsed subset is cached, because reading the 2.1 GB dump dominates runtime.

  python3 audit_prm800k_proxy_family.py [--problems 300]
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "prm800k_audit" / "scored-test-samples.jsonl"
CACHE = Path("/tmp/prm800k_parsed.pkl")
OUT = ROOT / "analysis_results" / "prm800k_proxy_family.json"
SPAN_FRACS = [0.25, 0.50]
N_BOOT = 2000
MIN_POS = 10

BACKTRACK = re.compile(r"\b(wait|hmm|actually|oops|let me try|on second thought|"
                       r"that'?s not right|i made a mistake|scratch that)\b", re.I)
BOXED = re.compile(r"\\boxed\s*\{")


def norm(s: str) -> str:
    s = s.lower().replace("\\!", "").replace("\\,", "").replace("$", "")
    s = re.sub(r"\\(left|right|dfrac|tfrac)", "", s)
    return re.sub(r"\s+", "", s)


def arith_density(s: str) -> float:
    if not s:
        return 0.0
    return sum(c.isdigit() or c in "+-*/=^" for c in s) / len(s)


def load(max_problems: int):
    if CACHE.exists():
        print(f"loading cached parse {CACHE}", flush=True)
        with CACHE.open("rb") as f:
            per = pickle.load(f)
        if len(per) >= max_problems:
            return {k: v for k, v in list(per.items())[:max_problems]}
    print(f"parsing {DATA.name} (cache miss) ...", flush=True)
    per = defaultdict(list)
    with DATA.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            steps, probs, gt = r.get("steps"), r.get("rating_probs"), r.get("ground_truth_answer")
            if not steps or not probs or gt is None or len(steps) != len(probs):
                continue
            pid = r.get("unique_id")
            if pid not in per and len(per) >= max_problems:
                continue
            per[pid].append((steps, [p.get("1", 0.0) for p in probs],
                             bool(r.get("is_correct")), gt))
    per = dict(per)
    with CACHE.open("wb") as f:
        pickle.dump(per, f, protocol=4)
    return per


def per_problem_aucs(groups, key):
    out = []
    for g in groups:
        y = np.array([x[key] for x in g])
        if len(set(y.tolist())) < 2:
            continue
        out.append(roc_auc_score(y, np.array([x["s"] for x in g], dtype=float)))
    return np.array(out, dtype=float)


def boot(a_z, a_y, seed):
    if len(a_z) == 0 or len(a_y) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    d = [abs(a_z[rng.integers(0, len(a_z), len(a_z))].mean() - .5)
         - abs(a_y[rng.integers(0, len(a_y), len(a_y))].mean() - .5) for _ in range(N_BOOT)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", type=int, default=300)
    a = ap.parse_args()
    per = load(a.problems)
    print(f"problems {len(per)}, samples {sum(len(v) for v in per.values())}")

    results = {}
    for frac in SPAN_FRACS:
        # one pass builds every proxy at this span, so the rules are compared on
        # exactly the same rows
        groups, dens = [], []
        for pid, samples in per.items():
            g = []
            for steps, ps, correct, gt in samples:
                L = len(steps)
                k = max(1, int(round(frac * L)))
                if k >= L:
                    continue
                pre_raw, rest_raw = " ".join(steps[:k]), " ".join(steps[k:])
                pre, rest, gtn = norm(pre_raw), norm(rest_raw), norm(str(gt))
                g.append({"s": float(np.prod(ps[:k])), "y": int(correct),
                          "gold_in_prefix": int(gtn in pre), "gold_in_prefix_c": int(gtn in rest),
                          "boxed": int(bool(BOXED.search(pre_raw))),
                          "boxed_c": int(bool(BOXED.search(rest_raw))),
                          "backtrack": int(bool(BACKTRACK.search(pre_raw))),
                          "backtrack_c": int(bool(BACKTRACK.search(rest_raw))),
                          "_ad": arith_density(pre_raw), "_ad_c": arith_density(rest_raw),
                          "_len": len(pre_raw), "_len_c": len(rest_raw)})
            if len(g) >= 2:
                groups.append(g)
                dens.extend(g)
        if not groups:
            continue
        for base in ("_ad", "_len"):
            med = float(np.median([x[base] for x in dens]))
            med_c = float(np.median([x[base + "_c"] for x in dens]))
            name = "arith_density" if base == "_ad" else "length"
            for g in groups:
                for x in g:
                    x[name] = int(x[base] > med)
                    x[name + "_c"] = int(x[base + "_c"] > med_c)

        A_y = per_problem_aucs(groups, "y")
        auc_y = float(A_y.mean())
        n_samp = sum(len(g) for g in groups)
        print(f"\n=== span {frac:.2f}: {len(groups)} problems, {n_samp} samples, "
              f"AUC(s,y)={auc_y:.3f}")
        print(f"{'proxy':16s} {'prev':>6s} {'AUC(s,z)':>9s} {'AUC(s,z^c)':>11s} "
              f"{'dDis':>8s}  95% CI          verdict")
        row = {}
        for i, name in enumerate(("gold_in_prefix", "boxed", "backtrack",
                                  "arith_density", "length")):
            prev = float(np.mean([x[name] for g in groups for x in g]))
            A_z = per_problem_aucs(groups, name)
            A_zc = per_problem_aucs(groups, name + "_c")
            if len(A_z) == 0 or len(A_zc) == 0:
                print(f"{name:16s} {prev:6.3f}   degenerate"); continue
            auc_z, auc_zc = float(A_z.mean()), float(A_zc.mean())
            d_dis = abs(auc_zc - .5) - abs(auc_y - .5)
            lo, hi = boot(A_zc, A_y, 100 + i)
            # the power precondition binds on the disjoint proxy too: a rule that
            # fires a handful of times cannot support a verdict either way
            zcpos = sum(x[name + "_c"] for g in groups for x in g)
            n_all = sum(len(g) for g in groups)
            blocked = zcpos < MIN_POS or (n_all - zcpos) < MIN_POS
            surv = (lo is not None and lo > 0) and not blocked
            verdict = ("MISMATCH" if surv and d_dis >= 0.15 else
                       "CAUTION" if surv and d_dis >= 0.10 else
                       "UNDECIDABLE" if blocked else "ALIGNED")
            print(f"{name:16s} {prev:6.3f} {auc_z:9.3f} {auc_zc:11.3f} {d_dis:+8.3f}  "
                  f"[{lo:+.3f},{hi:+.3f}]  {verdict}")
            row[name] = {"prevalence": prev, "auc_z": auc_z, "auc_zc": auc_zc,
                         "delta_dis_abs": d_dis, "ci": [lo, hi], "verdict": verdict}
        results[str(frac)] = {"problems": len(groups), "samples": n_samp,
                              "auc_y": auc_y, "proxies": row}

    flagged = [(f, n) for f, r in results.items()
               for n, v in r["proxies"].items() if v["verdict"] != "ALIGNED"]
    print("\nreading:")
    if flagged:
        print("  a deployed score fails the containment control under these proxy rules:")
        for f, n in flagged:
            print(f"    span {f}, proxy {n}")
        print("  this is a field contract exhibiting the failure.")
    else:
        print("  the deployed score clears under every proxy rule we tried, at both spans.")
        print("  The clear is therefore a property of the score rather than of our choice of")
        print("  rule, which is the stronger version of the negative result.")

    OUT.write_text(json.dumps({"spans": results, "n_boot": N_BOOT}, indent=2))
    print("\n[wrote]", OUT)


if __name__ == "__main__":
    main()
