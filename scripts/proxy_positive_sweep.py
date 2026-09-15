#!/usr/bin/env python3
"""Does the diagnosis depend on having only a handful of proxy positives? (2026-08-13)

The headline proxy AUC is computed against a keyword-disagreement label with very few
positives (13 on the primary run, 6 on the clean run at a 50-character prefix), which
is a fair statistical objection: an AUC estimated from that many positives is coarse,
and a 26-way tie structure drives the AP.

The label's positive count is a function of two contract fields we can vary without
touching anything else: the span the keyword is read from, and the keyword list.
Widening either produces more positives.  If the proxy-versus-semantic gap is an
artifact of thin labels, it should shrink as the label thickens; if it is a property
of the contract, it should persist.

  python3 proxy_positive_sweep.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
N_BOOT = 2000
RNG = np.random.default_rng(0)

spec = importlib.util.spec_from_file_location("agcr", HERE / "analyze_rlc_composite_router.py")
agcr = importlib.util.module_from_spec(spec)
sys.modules["agcr"] = agcr
spec.loader.exec_module(agcr)

# The paper's list, plus a wider one drawn from standard refusal-detection vocabularies.
WIDE_KW = agcr.REFUSAL_KW + [
    "i can not", "i am not able", "i am unable", "as an ai", "i do not provide",
    "i don't provide", "i will not", "i'm afraid", "against my", "cannot assist",
    "can't assist", "not something i can", "i must refuse", "i apologize",
    "i'm sorry", "not able to help", "cannot help with", "decline to",
]
SPANS = [50, 80, 120, 200, 400, None]     # None = full visible text


def kw_hit(text: str, kws: list[str], span: int | None) -> bool:
    t = (text or "").strip().lower()
    if span is not None:
        t = t[:span]
    return any(k in t for k in kws)


def boot_gap(y_z, y_y, s) -> tuple[float, float]:
    d = []
    n = len(s)
    for _ in range(N_BOOT):
        sel = RNG.integers(0, n, n)
        if len(set(y_z[sel])) < 2 or len(set(y_y[sel])) < 2:
            continue
        d.append(roc_auc_score(y_z[sel], s[sel]) - roc_auc_score(y_y[sel], s[sel]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    data_dir = ROOT / "results" / "disagree_routing"
    pair = agcr.PAIR_SPECS[0]
    rec_a = agcr.load_records(data_dir, pair.phase_a, pair.model_a)
    rec_b = agcr.load_records(data_dir, pair.phase_b, pair.model_b)
    ja = agcr.load_final_judge(data_dir, pair.phase_a, pair.model_a, pair.judge_tag)
    jb = agcr.load_final_judge(data_dir, pair.phase_b, pair.model_b, pair.judge_tag)
    pa = agcr.load_prompts(data_dir, pair.phase_a)
    pb = agcr.load_prompts(data_dir, pair.phase_b)
    ids = sorted(set(rec_a) & set(rec_b) & set(ja) & set(jb) & set(pa) & set(pb))
    ids = [i for i in ids if pa[i] == pb[i]
           and agcr.final_text(rec_a[i]).strip() and agcr.final_text(rec_b[i]).strip()]
    raw_a = [agcr.raw_text(rec_a[i]) for i in ids]
    raw_b = [agcr.raw_text(rec_b[i]) for i in ids]
    y = np.array([int(int(ja[i]) != int(jb[i])) for i in ids])
    s = agcr.tfidf_distance(raw_a, raw_b, prefix=50)   # the score is held fixed
    auc_y = float(roc_auc_score(y, s))
    print(f"n={len(ids)}  semantic positives={int(y.sum())}  AUC(s,y)={auc_y:.3f} (fixed)\n")

    print(f"{'keyword list':>12} {'span':>6} {'proxy n+':>9} {'AUC(s,z)':>9} "
          f"{'gap':>7} {'gap 95% CI':>20}")
    rows = []
    for name, kws in (("paper", agcr.REFUSAL_KW), ("wide", WIDE_KW)):
        for span in SPANS:
            z = np.array([int(kw_hit(a, kws, span) != kw_hit(b, kws, span))
                          for a, b in zip(raw_a, raw_b)])
            if len(set(z)) < 2:
                print(f"{name:>12} {str(span):>6} {int(z.sum()):>9}   (no variation)")
                continue
            auc_z = float(roc_auc_score(z, s))
            lo, hi = boot_gap(z, y, s)
            print(f"{name:>12} {str(span):>6} {int(z.sum()):>9} {auc_z:>9.3f} "
                  f"{auc_z - auc_y:>+7.3f} {f'[{lo:+.3f}, {hi:+.3f}]':>20}")
            rows.append({"keywords": name, "span": span, "proxy_positives": int(z.sum()),
                         "auc_proxy": auc_z, "gap": auc_z - auc_y,
                         "gap_ci": [lo, hi]})

    out = {"n": len(ids), "semantic_positives": int(y.sum()), "auc_semantic": auc_y,
           "rows": rows, "wide_keyword_count": len(WIDE_KW)}
    p = ROOT / "analysis_results" / "proxy_positive_sweep.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\n[wrote] {p}")


if __name__ == "__main__":
    main()
