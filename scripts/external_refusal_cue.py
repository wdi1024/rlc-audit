#!/usr/bin/env python3
"""Rebuild the repair with a cue we neither built nor trained (reviewer request, 2026-08-13).

The audited composite absorbs a refusal-marker keyword, which invites the reading that
the repair is a better keyword rather than a better span.  Two earlier checks argue
otherwise -- stripping the marker leaves the judge's decision at kappa 0.961, and a
keyword-masked reader still recovers that decision at AUC 0.98 -- but the masked
reader is fitted to our own judge labels, so it cannot answer the circularity
objection on its own.

This replaces the marker with the output of a public refusal classifier that was
trained by someone else on other data.  If the composite reproduces its routed
composition with that cue, the repair is a property of the span rather than of our
keyword list or of any feature derived from our labels.

  python3 external_refusal_cue.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = "protectai/distilroberta-base-rejection-v1"
BATCH = 16
BUDGET = 55

spec = importlib.util.spec_from_file_location("agcr", HERE / "analyze_rlc_composite_router.py")
agcr = importlib.util.module_from_spec(spec)
sys.modules["agcr"] = agcr
spec.loader.exec_module(agcr)


def rejection_scores(texts: list[str], tok, mdl, idx: int) -> np.ndarray:
    out = np.zeros(len(texts))
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            chunk = [t[:2000] for t in texts[i:i + BATCH]]
            enc = tok(chunk, truncation=True, max_length=512, padding=True,
                      return_tensors="pt")
            out[i:i + len(chunk)] = torch.softmax(mdl(**enc).logits, -1)[:, idx].numpy()
    return out


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
    rej = next(k for k, v in mdl.config.id2label.items()
               if "reject" in v.lower() or "refus" in v.lower())

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
    fin_a = [agcr.final_text(rec_a[i]) for i in ids]
    fin_b = [agcr.final_text(rec_b[i]) for i in ids]
    A = np.array([int(ja[i]) for i in ids])
    B = np.array([int(jb[i]) for i in ids])
    y = (A != B).astype(int)

    sa = rejection_scores(fin_a, tok, mdl, rej)
    sb = rejection_scores(fin_b, tok, mdl, rej)
    print(f"instrument check: the classifier recovers the judged side decision at "
          f"AUC {roc_auc_score(A, sa):.3f} and {roc_auc_score(B, sb):.3f}")

    cue = np.abs(sa - sb)
    fin = agcr.tfidf_distance(fin_a, fin_b)
    marker = np.array([int(agcr.is_kw(a) != agcr.is_kw(b))
                       for a, b in zip(fin_a, fin_b)], dtype=float)

    def topb(sc):
        return int(y[np.argsort(-sc, kind="stable")[:BUDGET]].sum())

    rows = {}
    for name, sc in (("final TF-IDF only", fin),
                     ("composite with marker cue", 0.7 * agcr.zscore(fin) + 0.3 * marker),
                     ("composite with external classifier cue",
                      0.7 * agcr.zscore(fin) + 0.3 * agcr.zscore(cue)),
                     ("external classifier cue alone", cue)):
        rows[name] = {"auc_semantic": float(roc_auc_score(y, sc)),
                      "ap_semantic": float(average_precision_score(y, sc)),
                      "routed_semantic": topb(sc)}
        print(f"  {name:40} AUC {rows[name]['auc_semantic']:.3f}  "
              f"AP {rows[name]['ap_semantic']:.3f}  "
              f"sem@{BUDGET} {rows[name]['routed_semantic']}/{BUDGET}")
    corr = float(np.corrcoef(cue, marker)[0, 1])
    print(f"  external cue versus marker cue: r={corr:+.3f}")

    out = {"model": MODEL, "n": len(ids), "budget": BUDGET,
           "auc_side_a": float(roc_auc_score(A, sa)),
           "auc_side_b": float(roc_auc_score(B, sb)),
           "rows": rows, "corr_external_marker": corr}
    p = ROOT / "analysis_results" / "external_refusal_cue.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\n[wrote] {p}")


if __name__ == "__main__":
    main()
