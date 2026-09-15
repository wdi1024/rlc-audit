#!/usr/bin/env python3
"""Is the orientation check a significance test, or a coin flip at 0.500? (2026-08-21)

The audit's second step declares a score *reversed* with respect to the construct
when AUC(s,y) < 0.5, and then reports the orientation-robust gap instead of the raw
one.  A reviewer points out that as written this fires on 0.499 exactly as hard as
on 0.290, and that the rows it fires on are printed without intervals.  Both are
fair: a threshold applied to a point estimate is not a test, and every row the check
withdraws is a row we are asking readers to accept was withdrawn for cause.

This script puts a bootstrap interval on AUC(s,y) for every verdict-bearing row and
splits the reversed set in two:

  significant   the 95\% interval lies wholly below 0.5.  The score really does rank
                the construct backwards, and the orientation branch is doing work.
  marginal      the interval covers 0.5.  The score is uninformative about the
                construct, and calling it "reversed" over-describes the evidence.

The distinction matters for what the paper claims.  A significantly reversed score
is a positive finding about the contract; a marginal one only says the construct is
not ranked, which the NULL branch already covers.  Anything landing in the second
bin should be reported as such.

  python3 orientation_significance.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "results" / "disagree_routing"
JUDGE = "anthropic_claude-haiku-4-5-20251001"
N_BOOT = 4000

spec = importlib.util.spec_from_file_location(
    "agcr", HERE / "analyze_rlc_composite_router.py")
agcr = importlib.util.module_from_spec(spec)
sys.modules["agcr"] = agcr
spec.loader.exec_module(agcr)

REFUSAL = [
    ("phase3_xstest_full", "XSTest 450 (primary run)"),
    ("phase4_advbench", "AdvBench 520"),
    ("phase4_simplesafety", "SimpleSafety 100"),
    ("phase5_xstest100_512tok", "XSTest 100 / 512tok"),
    ("phase5_advbench100_512tok", "AdvBench 100 / 512tok"),
    ("phase8_orbench_hard1k", "OR-Bench hard 1k"),
    ("phase15_jailbreakbench", "JailbreakBench"),
]
CORRECTNESS = [("phase14_gsm8k_2k", "GSM8K"), ("phase14_hotpot_2k", "HotpotQA")]
SPANS = [80, 120, 160, 200, 300, 600]


def auc_ci(y, s, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(N_BOOT):
        sel = rng.integers(0, len(s), len(s))
        if len(set(y[sel])) < 2:
            continue
        out.append(roc_auc_score(y[sel], s[sel]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def classify(auc, lo, hi):
    if hi < 0.5:
        return "reversed (significant)"
    if lo > 0.5:
        return "forward"
    return "unranked (covers 0.5)"


def main() -> None:
    rows = []

    for phase, label in REFUSAL:
        try:
            def judge(model):
                r = json.loads(
                    (DATA / f"{phase}_judge_{model}_{JUDGE}.json").read_text())["records"]
                return {x["id"]: int(bool(x["refusal_judge"])) for x in r}
            ta = {r["id"]: r for r in json.loads(
                (DATA / f"{phase}_traces_qwen3.5-2b.json").read_text())["records"]}
            tb = {r["id"]: r for r in json.loads(
                (DATA / f"{phase}_traces_gemma-4-e2b.json").read_text())["records"]}
            ja, jb = judge("qwen3.5-2b"), judge("gemma-4-e2b")
        except (FileNotFoundError, KeyError):
            continue
        ids = sorted(set(ta) & set(tb) & set(ja) & set(jb))
        a = [ta[i].get("trace") or "" for i in ids]
        b = [tb[i].get("trace") or "" for i in ids]
        y = np.array([int(ja[i] != jb[i]) for i in ids])
        s = agcr.tfidf_distance(a, b, prefix=50)
        auc = float(roc_auc_score(y, s))
        lo, hi = auc_ci(y, s, abs(hash(phase)) % (2**31))
        rows.append({"contract": label, "n": len(ids), "y_positives": int(y.sum()),
                     "auc_y": auc, "ci": [lo, hi], "class": classify(auc, lo, hi)})

    def cos(a, b):
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit(a + b)
        xa, xb = v.transform(a), v.transform(b)
        num = np.asarray(xa.multiply(xb).sum(1)).ravel()
        na = np.sqrt(np.asarray(xa.multiply(xa).sum(1)).ravel())
        nb = np.sqrt(np.asarray(xb.multiply(xb).sum(1)).ravel())
        return 1 - num / np.maximum(na * nb, 1e-9)

    for phase, dom in CORRECTNESS:
        ta = {r["id"]: r for r in json.loads(
            (DATA / f"{phase}_traces_qwen3.5-2b.json").read_text())["records"]}
        tb = {r["id"]: r for r in json.loads(
            (DATA / f"{phase}_traces_gemma-4-e2b.json").read_text())["records"]}
        ja = {r["id"]: r for r in json.loads(
            (DATA / f"{phase}_correctness_qwen3.5-2b_{JUDGE}.json").read_text())["records"]}
        jb = {r["id"]: r for r in json.loads(
            (DATA / f"{phase}_correctness_gemma-4-e2b_{JUDGE}.json").read_text())["records"]}
        ids = sorted(set(ta) & set(tb) & set(ja) & set(jb))
        a = [ta[i].get("trace") or "" for i in ids]
        b = [tb[i].get("trace") or "" for i in ids]
        y = np.array([int(bool(ja[i].get("correct_judge")) != bool(jb[i].get("correct_judge")))
                      for i in ids])
        for span in SPANS:
            s = cos([t[:span] for t in a], [t[:span] for t in b])
            auc = float(roc_auc_score(y, s))
            lo, hi = auc_ci(y, s, abs(hash(phase + str(span))) % (2**31))
            rows.append({"contract": f"{dom}, {span}-char span", "n": len(ids),
                         "y_positives": int(y.sum()), "auc_y": auc, "ci": [lo, hi],
                         "class": classify(auc, lo, hi)})

    print(f"{'contract':28s} {'n':>5s} {'y+':>5s} {'AUC(s,y)':>9s}  {'95% CI':>18s}  class")
    for r in rows:
        print(f"{r['contract'][:28]:28s} {r['n']:5d} {r['y_positives']:5d} {r['auc_y']:9.3f}  "
              f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]  {r['class']}")

    below = [r for r in rows if r["auc_y"] < 0.5]
    sig = [r for r in below if r["class"] == "reversed (significant)"]
    marg = [r for r in below if r["class"] != "reversed (significant)"]
    print(f"\nrows with a point estimate below 0.5: {len(below)}")
    print(f"  significantly reversed (interval wholly below 0.5): {len(sig)}")
    for r in sig:
        print(f"      {r['contract']}  {r['auc_y']:.3f} [{r['ci'][0]:.3f},{r['ci'][1]:.3f}]")
    print(f"  interval covers 0.5, so unranked and not reversed: {len(marg)}")
    for r in marg:
        print(f"      {r['contract']}  {r['auc_y']:.3f} [{r['ci'][0]:.3f},{r['ci'][1]:.3f}]")

    print("\nreading:")
    if marg:
        print("  the orientation branch as written fires on rows the data cannot distinguish from")
        print("  chance. It should require the interval to lie below 0.5; the remaining rows")
        print("  belong under NULL, which already describes a score that ranks nothing.")
    else:
        print("  every row the orientation branch fires on is significantly reversed, so the")
        print("  point-estimate threshold and the interval test agree on this corpus.")

    out = {"n_boot": N_BOOT, "rows": rows,
           "n_below_half": len(below), "n_significant": len(sig), "n_marginal": len(marg)}
    path = ROOT / "analysis_results" / "orientation_significance.json"
    path.write_text(json.dumps(out, indent=2))
    print("\n[wrote]", path)


if __name__ == "__main__":
    main()
