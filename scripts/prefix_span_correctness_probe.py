#!/usr/bin/env python3
"""Does the mechanism reproduce outside refusal? (2026-08-14)

The paper's headline contract is refusal routing, which invites the reading that
representation-label coupling is a quirk of refusal templates.  The audit says the
failure is span-specific, not domain-specific: it should appear wherever a proxy is
read from a partial span while the construct depends on the complete answer.

Reasoning and multi-hop QA are the sharpest test, because correctness depends on the
last step.  We reuse the cached HotpotQA and GSM8K runs, hold the construct at judged
correctness disagreement, and vary only the span the score and proxy read.  The same
contracts read over the full trace already clear (Appendix E), so any gap that opens
at a prefix is attributable to the span.

This is a probe: proxy positives at short prefixes are few, and the intervals are
correspondingly wide.  It is reported as motivation for a larger run, not as a result.

  python3 prefix_span_correctness_probe.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "disagree_routing"
JUDGE = "anthropic_claude-haiku-4-5-20251001"
DEFAULT_TASKS = [("phase11_gsm8k", "GSM8K reasoning"), ("phase10_hotpot", "HotpotQA multi-hop")]
SPANS = [50, 120, 200, 300, 600, None]
N_BOOT = 2000
RNG = np.random.default_rng(0)


def load(phase, model, judge=JUDGE):
    tr = {r["id"]: r for r in json.loads((DATA / f"{phase}_traces_{model}.json").read_text())["records"]}
    jd = {r["id"]: r for r in json.loads(
        (DATA / f"{phase}_correctness_{model}_{judge}.json").read_text())["records"]}
    return tr, jd


def cosine_distance(a, b):
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit(a + b)
    Xa, Xb = v.transform(a), v.transform(b)
    num = np.asarray(Xa.multiply(Xb).sum(1)).ravel()
    na = np.sqrt(np.asarray(Xa.multiply(Xa).sum(1)).ravel())
    nb = np.sqrt(np.asarray(Xb.multiply(Xb).sum(1)).ravel())
    return 1 - num / np.maximum(na * nb, 1e-9)


def boot_gap(z, y, s):
    d = []
    n = len(s)
    for _ in range(N_BOOT):
        sel = RNG.integers(0, n, n)
        if len(set(z[sel])) < 2 or len(set(y[sel])) < 2:
            continue
        d.append(roc_auc_score(z[sel], s[sel]) - roc_auc_score(y[sel], s[sel]))
    if not d:
        return float("nan"), float("nan")
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default=JUDGE,
                    help="correctness-judge suffix in the label filenames")
    ap.add_argument("--phases", default="",
                    help="comma-separated phase names; defaults to the cached runs")
    args = ap.parse_args()
    tasks = ([(p, p) for p in args.phases.split(",") if p.strip()]
             if args.phases else DEFAULT_TASKS)
    out = {}
    for phase, label in tasks:
        ta, ja = load(phase, "qwen3.5-2b", args.judge)
        tb, jb = load(phase, "gemma-4-e2b", args.judge)
        ids = sorted(set(ta) & set(tb) & set(ja) & set(jb))
        A = [ta[i].get("trace") or "" for i in ids]
        B = [tb[i].get("trace") or "" for i in ids]
        gold = {i: (ja[i].get("gold_answer") or "").strip() for i in ids}
        y = np.array([int(bool(ja[i].get("correct_judge")) != bool(jb[i].get("correct_judge")))
                      for i in ids])
        print(f"\n=== {label}: n={len(ids)}, construct positives {int(y.sum())} "
              f"(base rate {y.mean():.3f})")
        rows = []
        for P in SPANS:
            pa = [a[:P] if P else a for a in A]
            pb = [b[:P] if P else b for b in B]
            z = np.array([int((gold[i].lower() in a.lower()) != (gold[i].lower() in b.lower()))
                          for i, a, b in zip(ids, pa, pb)])
            name = f"prefix-{P}" if P else "full trace"
            if len(set(z)) < 2:
                print(f"  {name:12} proxy positives {int(z.sum())}: not instantiable")
                rows.append({"span": name, "proxy_positives": int(z.sum()),
                             "verdict": "not instantiable"})
                continue
            s = cosine_distance(pa, pb)
            az, ay = float(roc_auc_score(z, s)), float(roc_auc_score(y, s))
            lo, hi = boot_gap(z, y, s)
            Bq = max(1, round(len(ids) * 0.10))
            idx = np.argsort(-s, kind="stable")[:Bq]
            print(f"  {name:12} n+={int(z.sum()):>3}  AUC(s,z)={az:.3f}  AUC(s,y)={ay:.3f}  "
                  f"gap={az-ay:+.3f} [{lo:+.3f},{hi:+.3f}]  "
                  f"kappa={cohen_kappa_score(z,y):+.3f}  sem@{Bq}={int(y[idx].sum())}/{Bq}")
            rows.append({"span": name, "proxy_positives": int(z.sum()), "auc_proxy": az,
                         "auc_semantic": ay, "gap": az - ay, "gap_ci": [lo, hi],
                         "kappa": float(cohen_kappa_score(z, y)),
                         "routed_semantic": int(y[idx].sum()), "budget": Bq})
        out[label] = {"n": len(ids), "construct_positives": int(y.sum()), "rows": rows}
    suffix = "" if args.judge == JUDGE else f"_{args.judge}"
    p = ROOT / "analysis_results" / f"prefix_span_correctness_probe{suffix}.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\n[wrote] {p}")


if __name__ == "__main__":
    main()
