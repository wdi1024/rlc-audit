#!/usr/bin/env python3
"""The verdict rule, with the branch the old one was missing. (2026-08-21)

A reviewer showed that our central dichotomy is incomplete. We said a surviving gap
has two causes calling for opposite repairs -- containment (fix the contract) or a
proxy--construct mismatch (fix the label) -- and treated spillover as a third case
handled by kappa. There is a fourth, and it is the one our own results land on:

    the score does not rank the construct at all.

Delta_|.| = |AUC(s,z^c) - 0.5| - |AUC(s,y) - 0.5| is large whenever the second term
is near zero, so a score that ranks nothing produces a large gap for a reason that
has nothing to do with the proxy. Prescribing "fix the label" there yields a system
validated on whatever the proxy happens to measure, which is the failure this paper
exists to name. The prescription was self-contradictory on exactly the rows carrying
our surviving verdicts.

The revised rule therefore asks, before any magnitude band:

  1. preconditions          power, non-degenerate complement, score resolution
  2. is the construct arm oriented?   bootstrap CI on AUC(s,y) excluding 1/2
        no, and the proxy IS ranked   -> SCORE FAILURE   (no label fix helps)
        no, and neither is ranked     -> NULL
  3. does the gap beat the contract's own permutation null?   (not a fixed cutoff)
  4. is the off-span proxy a restatement of the construct?    kappa(z^c, y)
        high                          -> proxy and construct have merged; ALIGNED
  5. magnitude band                   MISMATCH / CAUTION / ALIGNED

Steps 2 and 4 are new. Step 4 moves kappa(z^c,y) into the decision, where Section 3
already said the spillover distinction lived but the flowchart never used it.

  python3 verdict_rule_v2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
AR = ROOT / "analysis_results"
OUT = AR / "verdict_rule_v2.json"
MIN_POS = 10
KAPPA_MERGED = 0.40      # above this, z^c is a restatement of y rather than a rival
MAX_CHANCE = 0.55        # equivalence bound: "no better than chance" must be shown, not assumed
MISMATCH_BAND, CAUTION_BAND = 0.15, 0.10


def auc_ci(auc, n, n_pos):
    """Hanley--McNeil standard error; the construct arm needs an interval to be
    called oriented at all."""
    n_neg = n - n_pos
    if n_pos < 2 or n_neg < 2:
        return None, None
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc * auc)
           + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    se = var ** 0.5
    return auc - 1.96 * se, auc + 1.96 * se


def main() -> None:
    rows = json.loads((AR / "complement_span_proxy_control.json").read_text())["rows"]
    null = json.loads((AR / "disjoint_null_distribution.json").read_text())["contracts"]

    print(f"{'contract':28s} {'AUC(s,y)':>9s} {'oriented':>9s} {'AUC(s,zc)':>10s} "
          f"{'d_dis':>7s} {'>null':>6s} {'k(zc,y)':>8s}  old -> new")
    out, tally = {}, {}
    for r in rows:
        name = r["contract"]
        n, ypos, auc_y = r["n"], r["y_positives"], r["auc_y"]
        comp = r["comp"]
        lo, hi = auc_ci(auc_y, n, ypos)
        oriented = lo is not None and not (lo <= 0.5 <= hi)
        blocked = (ypos < MIN_POS or n - ypos < MIN_POS or r["complement_degenerate"]
                   or comp.get("degenerate"))
        d = comp.get("delta_abs")
        k = comp.get("kappa_vs_y")
        auc_zc = comp.get("auc")
        nl = null.get(name)
        beats_null = bool(nl and d is not None and d > nl["null_p95"])

        # the old verdict, as the paper currently reports it
        old = ("MISMATCH" if (not blocked and d is not None and d >= MISMATCH_BAND
                              and comp["ci_abs"][0] and comp["ci_abs"][0] > 0)
               else "CAUTION" if (not blocked and d is not None and d >= CAUTION_BAND)
               else "UNDECIDABLE" if blocked else "ALIGNED")

        # Three fixes over the first revision, each forced by a reviewer and each
        # costing us a verdict we had claimed:
        #
        #  1. A score that ranks the construct *backwards* is not aligned. The
        #     orientation-robust gap treats "predicts well" and "predicts exactly
        #     backwards" identically, so four GSM8K rows were counted as clears
        #     while anti-ranking correctness. They get their own category.
        #  2. "The score ranks the construct no better than chance" was read off an
        #     interval covering 1/2, which accepts the null. JailbreakBench's
        #     interval is [0.390, 0.626] and cannot exclude AUC 0.62. We now require
        #     an equivalence bound: the whole interval must sit below MAX_CHANCE.
        #  3. Anything that fails both tests is undecided, not clear.
        reversed_score = hi is not None and hi < 0.5
        near_chance = lo is not None and hi is not None and hi <= MAX_CHANCE and lo >= 1 - MAX_CHANCE
        proxy_ranked = auc_zc is not None and abs(auc_zc - 0.5) >= 0.10

        # Five names, not nine. "Reversed" and "at chance" are the same finding for a
        # reader -- the score is not forward evidence about the construct -- so they
        # are one verdict with a stated reason, and the three ways a precondition can
        # fail are one verdict with a stated reason.
        reason = None
        if blocked:
            new, reason = "UNDECIDABLE", "precondition"
        elif reversed_score:
            new, reason = "SCORE NOT EVIDENCE", "ranks the construct backwards"
        elif near_chance:
            new = "SCORE NOT EVIDENCE" if proxy_ranked else "UNDECIDABLE"
            reason = "at chance, proxy ranked" if proxy_ranked else "at chance, nothing ranked"
        elif not oriented:
            new, reason = "UNDECIDABLE", "interval too wide to decide"
        elif k is not None and k >= KAPPA_MERGED:
            new = "ALIGNED"                       # z^c has become a measure of y
        elif not beats_null:
            new = "ALIGNED"
        elif d >= MISMATCH_BAND:
            new = "MISMATCH"
        elif d >= CAUTION_BAND:
            new = "CAUTION"
        else:
            new = "ALIGNED"

        mark = "  <-- CHANGED" if new != old else ""
        print(f"{name:28s} {auc_y:9.3f} {str(oriented):>9s} "
              f"{(auc_zc if auc_zc is not None else float('nan')):10.3f} "
              f"{(d if d is not None else float('nan')):7.3f} {str(beats_null):>6s} "
              f"{(k if k is not None else float('nan')):8.3f}  {old} -> {new}{mark}")
        out[name] = {"reason": reason, "auc_y": auc_y, "auc_y_ci": [lo, hi], "oriented": oriented,
                     "auc_zc": auc_zc, "delta_dis": d, "kappa_zc_y": k,
                     "beats_null": beats_null, "verdict_old": old, "verdict_new": new}
        tally[new] = tally.get(new, 0) + 1

    print("\nnew verdict tally:", dict(sorted(tally.items())))
    print("\nreading:")
    sf = [(k, v["reason"]) for k, v in out.items() if v["verdict_new"] == "SCORE NOT EVIDENCE"]
    print(f"  SCORE NOT EVIDENCE on {len(sf)}:")
    for k, why in sf:
        print(f"    {k:26s} {why}")
    print("  These are the rows the old rule called MISMATCH. The score ranks an off-span")
    print("  proxy while failing to rank the construct, so the honest reading is that the")
    print("  score is not evidence for the claim, and no repair to the label makes it so.")
    OUT.write_text(json.dumps(out, indent=2))
    print("\n[wrote]", OUT)


if __name__ == "__main__":
    main()
