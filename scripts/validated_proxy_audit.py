#!/usr/bin/env python3
"""Does the failure survive a validated proxy? (reviewer request, 2026-08-14)

A fair objection to the primary contract is that its proxy is a twelve-term keyword
list, which in 2026 nobody would defend as a validation target when trained refusal
classifiers exist.  If the coupling is an artifact of a crude proxy, replacing the
keyword list with a validated classifier should dissolve it.

The audit says the failure is a property of the *span*, so it predicts the opposite:
a classifier read from the same 50-character prefix the score reads should be
predicted just as well by that score, while a classifier read from the full response
should behave like the construct.  We test both.

  proxy A: keyword disagreement on the prefix (the paper's proxy)
  proxy B: classifier disagreement on the same prefix (validated instrument, same span)
  proxy C: classifier disagreement on the full response (validated instrument, other span)
  construct y: judged refusal-decision disagreement

  python3 validated_proxy_audit.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = "protectai/distilroberta-base-rejection-v1"
PREFIX = 50
BUDGET = 55
BATCH = 32

spec = importlib.util.spec_from_file_location("agcr", HERE / "analyze_rlc_composite_router.py")
agcr = importlib.util.module_from_spec(spec)
sys.modules["agcr"] = agcr
spec.loader.exec_module(agcr)


def clf_flag(texts, tok, mdl, idx):
    out = np.zeros(len(texts))
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            chunk = [t[:2000] if t else "" for t in texts[i:i + BATCH]]
            enc = tok(chunk, truncation=True, max_length=512, padding=True,
                      return_tensors="pt")
            out[i:i + len(chunk)] = torch.softmax(mdl(**enc).logits, -1)[:, idx].numpy()
    return out


def audit(name, z, y, s, out):
    if len(set(z)) < 2:
        print(f"{name:52} not instantiable at this span: {int(z.sum())} positives")
        out[name] = {"proxy_positives": int(z.sum()), "verdict": "not instantiable"}
        return
    B = BUDGET
    idx = np.argsort(-s, kind="stable")[:B]
    az, ay = float(roc_auc_score(z, s)), float(roc_auc_score(y, s))
    k = float(cohen_kappa_score(z, y))
    verdict = ("MISMATCH" if az - ay >= 0.15 and k <= 0.20
               else "CAUTION" if az - ay >= 0.10 else "ALIGNED")
    print(f"{name:52} n+={int(z.sum()):>4}  AUC(s,z)={az:.3f}  AUC(s,y)={ay:.3f}  "
          f"gap={az-ay:+.3f}  kappa={k:.3f}  sem@{B}={int(y[idx].sum())}/{B}  {verdict}")
    out[name] = {"proxy_positives": int(z.sum()), "auc_proxy": az, "auc_semantic": ay,
                 "gap": az - ay, "kappa_z_y": k,
                 "routed_semantic": int(y[idx].sum()), "verdict": verdict}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
    rej = next(k for k, v in mdl.config.id2label.items()
               if "reject" in v.lower() or "refus" in v.lower())

    d = ROOT / "results" / "disagree_routing"
    pair = agcr.PAIR_SPECS[0]
    ra = agcr.load_records(d, pair.phase_a, pair.model_a)
    rb = agcr.load_records(d, pair.phase_b, pair.model_b)
    ja = agcr.load_final_judge(d, pair.phase_a, pair.model_a, pair.judge_tag)
    jb = agcr.load_final_judge(d, pair.phase_b, pair.model_b, pair.judge_tag)
    pa = agcr.load_prompts(d, pair.phase_a)
    pb = agcr.load_prompts(d, pair.phase_b)
    ids = sorted(set(ra) & set(rb) & set(ja) & set(jb) & set(pa) & set(pb))
    ids = [i for i in ids if pa[i] == pb[i]
           and agcr.final_text(ra[i]).strip() and agcr.final_text(rb[i]).strip()]
    raw_a = [agcr.raw_text(ra[i]) for i in ids]
    raw_b = [agcr.raw_text(rb[i]) for i in ids]
    y = np.array([int(int(ja[i]) != int(jb[i])) for i in ids])
    s = agcr.tfidf_distance(raw_a, raw_b, prefix=PREFIX)
    print(f"n={len(ids)}  semantic positives={int(y.sum())}  "
          f"score = prefix-{PREFIX} TF-IDF distance (held fixed)\n")

    pre_a = [t[:PREFIX] for t in raw_a]
    pre_b = [t[:PREFIX] for t in raw_b]
    cp_a, cp_b = clf_flag(pre_a, tok, mdl, rej), clf_flag(pre_b, tok, mdl, rej)
    cf_a, cf_b = clf_flag(raw_a, tok, mdl, rej), clf_flag(raw_b, tok, mdl, rej)

    out = {}
    z_kw = np.array([int(agcr.is_kw(a) != agcr.is_kw(b)) for a, b in zip(pre_a, pre_b)])
    audit("A  keyword list on the prefix (paper's proxy)", z_kw, y, s, out)
    for thr in (0.5,):
        z_cp = ((cp_a > thr).astype(int) != (cp_b > thr).astype(int)).astype(int)
        z_cf = ((cf_a > thr).astype(int) != (cf_b > thr).astype(int)).astype(int)
        audit("B  validated classifier on the SAME prefix span", z_cp, y, s, out)
        audit("C  validated classifier on the FULL response", z_cf, y, s, out)

    # Construct validity of each instrument at each span, which is the real question
    # behind "why not use a validated classifier instead of a keyword list".
    A = np.array([int(ja[i]) for i in ids])
    for nm, v in (("keyword, prefix span", np.array([int(agcr.is_kw(t)) for t in pre_a])),
                  ("classifier, prefix span", cp_a),
                  ("classifier, full response", cf_a)):
        if len(set(np.round(v, 6))) > 1:
            print(f"  {nm:34} predicts the judged side decision at AUC "
                  f"{roc_auc_score(A, v):.3f}")
            out.setdefault("side_validity", {})[nm] = float(roc_auc_score(A, v))

    p = ROOT / "analysis_results" / "validated_proxy_audit.json"
    json.dump({"model": MODEL, "n": len(ids), "prefix": PREFIX, "budget": BUDGET,
               "rows": out}, open(p, "w"), indent=2)
    print(f"\n[wrote] {p}")


if __name__ == "__main__":
    main()
