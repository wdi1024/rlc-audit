#!/usr/bin/env python3
"""What is Delta_dis under the null, when the prefix is not independent of the rest?
(2026-08-21)

Proposition 1 pins a disjoint proxy to AUC = 1/2 under x_<b independent of x_>=b.
That assumption cannot hold for autoregressive text --- the paper says so in
Section 2 and demonstrates the opposite in Section 4.2 --- so the null value of
Delta_dis is not 0 and the constants 0.15 and 0.10 have been applied to a
statistic whose null is unmeasured. A reviewer is right that this leaves the
theory decorative. This script measures the null instead of assuming it.

The null we want is not "no relationship at all". Shuffling the off-span proxy
across the whole sample would destroy its relationship with the construct too, and
a score that predicts the construct will predict anything correlated with it; that
null is too generous and would make every contract look significant. The question
is narrower:

    is the score associated with the off-span proxy BEYOND what the construct
    already explains?

So we permute z^c within strata of y. That holds P(z^c | y) fixed by construction,
preserves the marginal of every variable, and destroys only the extra association
between the score and the off-span text. Recomputing the orientation-robust gap on
each permutation gives the null distribution of Delta_dis for that contract, and
its upper tail is an empirical cutoff to compare against the paper's constants.

What this null does NOT absorb is spillover --- prefix-to-continuation dependence
that does not run through the construct --- because permutation destroys that as
well. Delta_dis above this null therefore means "extra association, from a
proxy--construct mismatch or from spillover", which is the distinction Section 3.1
already hands to kappa(z^c, y). We report both so the reader can separate them.

  python3 disjoint_null_distribution.py
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
OUT = ROOT / "analysis_results" / "disjoint_null_distribution.json"
JUDGE = "anthropic_claude-haiku-4-5-20251001"
N_PERM = 2000
REFUSAL_SPAN = 50
MIN_POSITIVES = 10
CORRECTNESS_SPANS = [50, 80, 120, 160, 200, 300, 600]

spec = importlib.util.spec_from_file_location(
    "agcr", HERE / "analyze_rlc_composite_router.py")
agcr = importlib.util.module_from_spec(spec)
sys.modules["agcr"] = agcr
spec.loader.exec_module(agcr)

REFUSAL_SETTINGS = [
    ("phase3_xstest_full", "XSTest 450 (primary run)"),
    ("phase4_advbench", "AdvBench 520"),
    ("phase4_simplesafety", "SimpleSafety 100"),
    ("phase5_xstest100_512tok", "XSTest 100 / 512tok"),
    ("phase5_advbench100_512tok", "AdvBench 100 / 512tok"),
    ("phase8_orbench_hard1k", "OR-Bench hard 1k"),
    ("phase15_jailbreakbench", "JailbreakBench"),
]
CORRECTNESS = [("phase14_gsm8k_2k", "GSM8K"), ("phase14_hotpot_2k", "HotpotQA")]


def cos_span(a, b):
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit(a + b)
    xa, xb = v.transform(a), v.transform(b)
    num = np.asarray(xa.multiply(xb).sum(1)).ravel()
    na = np.sqrt(np.asarray(xa.multiply(xa).sum(1)).ravel())
    nb = np.sqrt(np.asarray(xb.multiply(xb).sum(1)).ravel())
    return 1 - num / np.maximum(na * nb, 1e-9)


def contracts():
    """Yield (label, family, s, z_comp, y) for every contract the control runs on."""
    for phase, label in REFUSAL_SETTINGS:
        try:
            def judge(model):
                rows = json.loads(
                    (DATA / f"{phase}_judge_{model}_{JUDGE}.json").read_text())["records"]
                return {r["id"]: int(bool(r["refusal_judge"])) for r in rows}
            ta = {r["id"]: r for r in json.loads(
                (DATA / f"{phase}_traces_qwen3.5-2b.json").read_text())["records"]}
            tb = {r["id"]: r for r in json.loads(
                (DATA / f"{phase}_traces_gemma-4-e2b.json").read_text())["records"]}
            ja, jb = judge("qwen3.5-2b"), judge("gemma-4-e2b")
        except (FileNotFoundError, KeyError):
            continue
        ids = sorted(set(ta) & set(tb) & set(ja) & set(jb))
        a_txt = [ta[i].get("trace") or "" for i in ids]
        b_txt = [tb[i].get("trace") or "" for i in ids]
        y = np.array([int(ja[i] != jb[i]) for i in ids])
        sp = REFUSAL_SPAN
        z_comp = np.array([int(agcr.is_kw(a[sp:]) != agcr.is_kw(b[sp:]))
                           for a, b in zip(a_txt, b_txt)])
        yield label, "refusal", agcr.tfidf_distance(a_txt, b_txt, prefix=sp), z_comp, y

    for phase, dom in CORRECTNESS:
        try:
            ta = {r["id"]: r for r in json.loads(
                (DATA / f"{phase}_traces_qwen3.5-2b.json").read_text())["records"]}
            tb = {r["id"]: r for r in json.loads(
                (DATA / f"{phase}_traces_gemma-4-e2b.json").read_text())["records"]}
            ja = {r["id"]: r for r in json.loads(
                (DATA / f"{phase}_correctness_qwen3.5-2b_{JUDGE}.json").read_text())["records"]}
            jb = {r["id"]: r for r in json.loads(
                (DATA / f"{phase}_correctness_gemma-4-e2b_{JUDGE}.json").read_text())["records"]}
        except FileNotFoundError:
            continue
        ids = sorted(set(ta) & set(tb) & set(ja) & set(jb))
        a_txt = [ta[i].get("trace") or "" for i in ids]
        b_txt = [tb[i].get("trace") or "" for i in ids]
        gold = {i: (ja[i].get("gold_answer") or "").strip().lower() for i in ids}
        y = np.array([int(bool(ja[i].get("correct_judge")) != bool(jb[i].get("correct_judge")))
                      for i in ids])
        for span in CORRECTNESS_SPANS:
            pa, pb = [t[:span] for t in a_txt], [t[:span] for t in b_txt]
            ca, cb = [t[span:] for t in a_txt], [t[span:] for t in b_txt]
            z_comp = np.array([int((gold[i] in a.lower()) != (gold[i] in b.lower()))
                               for i, a, b in zip(ids, ca, cb)])
            yield (f"{dom}, {span}-char span", "correctness",
                   cos_span(pa, pb), z_comp, y)


def stratified_null(s, z, y, seed):
    """Permute z within strata of y. AUC(s,y) is invariant under this, so the null
    of the gap is a shift of the null of |AUC(s,z)-0.5|."""
    rng = np.random.default_rng(seed)
    a_y = roc_auc_score(y, s)
    base = abs(a_y - 0.5)
    idx_by_y = {v: np.flatnonzero(y == v) for v in np.unique(y)}
    zp = z.copy()
    out = []
    for _ in range(N_PERM):
        for v, idx in idx_by_y.items():
            zp[idx] = z[rng.permutation(idx)]
        if len(set(zp.tolist())) < 2:
            continue
        out.append(abs(roc_auc_score(zp, s) - 0.5) - base)
    return np.array(out), float(abs(roc_auc_score(z, s) - 0.5) - base)


def main() -> None:
    print(f"stratified permutation null, {N_PERM} draws per contract\n")
    print(f"{'contract':30s} {'n':>5s} {'obs':>7s} {'null mean':>9s} "
          f"{'null p95':>8s} {'p':>7s}  reading")
    rows = {}
    for label, family, s, z, y in contracts():
        n = len(s)
        if len(set(z.tolist())) < 2 or len(set(y.tolist())) < 2:
            print(f"{label:30s} {n:5d}   degenerate proxy or construct")
            continue
        if int(y.sum()) < MIN_POSITIVES or n - int(y.sum()) < MIN_POSITIVES:
            print(f"{label:30s} {n:5d}   excluded on power (y+={int(y.sum())})")
            continue
        null, obs = stratified_null(s, z, y, abs(hash(label)) % (2**31))
        if len(null) < N_PERM // 4:
            print(f"{label:30s} {n:5d}   null degenerate")
            continue
        p95 = float(np.percentile(null, 95))
        p = (1 + int((null >= obs).sum())) / (len(null) + 1)
        verdict = ("above null" if obs > p95 else "within null")
        print(f"{label:30s} {n:5d} {obs:+7.3f} {null.mean():+9.3f} {p95:+8.3f} "
              f"{p:7.4f}  {verdict}")
        rows[label] = {"family": family, "n": n, "observed": obs,
                       "null_mean": float(null.mean()), "null_p95": p95,
                       "null_p99": float(np.percentile(null, 99)), "p_perm": p}

    if rows:
        p95s = np.array([r["null_p95"] for r in rows.values()])
        print(f"\nempirical 95th percentile of the null across {len(rows)} contracts: "
              f"median {np.median(p95s):+.3f}, range {p95s.min():+.3f} to {p95s.max():+.3f}")
        print("the paper applies fixed constants 0.10 (CAUTION) and 0.15 (MISMATCH).")
        loose = [k for k, r in rows.items() if r["null_p95"] > 0.10]
        print(f"contracts whose own null reaches past 0.10: {len(loose)}"
              + (f" -- {', '.join(loose)}" if loose else ""))
        above = [k for k, r in rows.items() if r["observed"] > r["null_p95"]]
        print(f"contracts whose observed gap exceeds their own null p95: "
              f"{len(above)}" + (f" -- {', '.join(above)}" if above else ""))

    print("\nreading:")
    print("  A fixed cutoff is defensible only if it sits above the null for every")
    print("  contract it is applied to. Where the null p95 is small the constants are")
    print("  conservative; where it approaches them the verdict was resting on an")
    print("  unmeasured assumption, and the per-contract null should be used instead.")
    OUT.write_text(json.dumps({"n_perm": N_PERM, "contracts": rows}, indent=2))
    print("\n[wrote]", OUT)


if __name__ == "__main__":
    main()
