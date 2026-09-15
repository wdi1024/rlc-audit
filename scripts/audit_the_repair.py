#!/usr/bin/env python3
"""Audit the repaired contract with the paper's own instrument (2026-08-18).

A reviewer observed that the paper proposes a repaired score and then never runs
RLC-Audit on it: Table `tab:mechanism-action` reports the composite's semantic
AUC but leaves its proxy column as a dash, so the one number the audit exists to
produce -- Delta_AUC = AUC(s,z) - AUC(s,y) -- is missing for the contract the
paper recommends.  That is a real hole, and it has two halves.

  1. Against the ORIGINAL artifact-bearing proxy (prefix-50 keyword
     disagreement), the repaired score should be *less* coupled than the raw
     prefix score.  This is the claim "less-coupled contract" and it is testable.

  2. Against the repaired score's OWN cheap indicator (final-span marker
     disagreement), Delta_AUC is meaningless, because that indicator is a
     component of the score.  Reporting it makes the methodological point that
     absorbing a proxy into a score disqualifies that proxy as the contract's
     validation label.

Both are computed here on the clean intervention run (n=548).

  python3 audit_the_repair.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

spec = importlib.util.spec_from_file_location(
    "agcr", HERE / "analyze_rlc_composite_router.py")
agcr = importlib.util.module_from_spec(spec)
sys.modules["agcr"] = agcr
spec.loader.exec_module(agcr)


def audit(y: np.ndarray, z: np.ndarray, s: np.ndarray) -> dict:
    return {
        "auc_z": float(roc_auc_score(z, s)),
        "auc_y": float(roc_auc_score(y, s)),
        "d_auc": float(roc_auc_score(z, s) - roc_auc_score(y, s)),
        "ap_z": float(average_precision_score(z, s)),
        "ap_y": float(average_precision_score(y, s)),
        "d_ap": float(average_precision_score(z, s) - average_precision_score(y, s)),
        "kappa_zy": float(cohen_kappa_score(z, y)),
        "z_pos": int(z.sum()), "y_pos": int(y.sum()), "n": int(len(y)),
    }


def verdict(a: dict) -> str:
    if a["d_auc"] >= 0.15 and a["kappa_zy"] <= 0.20:
        return "MISMATCH"
    if a["d_auc"] >= 0.10 or a["d_ap"] >= 0.10:
        return "CAUTION"
    return "ALIGNED"


def main():
    data_dir = ROOT / "results" / "disagree_routing"
    spec0 = agcr.PAIR_SPECS[0]                     # clean Qwen/Gemma run
    rec_a = agcr.load_records(data_dir, spec0.phase_a, spec0.model_a)
    rec_b = agcr.load_records(data_dir, spec0.phase_b, spec0.model_b)
    ja = agcr.load_final_judge(data_dir, spec0.phase_a, spec0.model_a, spec0.judge_tag)
    jb = agcr.load_final_judge(data_dir, spec0.phase_b, spec0.model_b, spec0.judge_tag)
    pa = agcr.load_prompts(data_dir, spec0.phase_a)
    pb = agcr.load_prompts(data_dir, spec0.phase_b)

    ids = sorted(set(rec_a) & set(rec_b) & set(ja) & set(jb) & set(pa) & set(pb))
    ids = [i for i in ids if pa[i] == pb[i]
           and agcr.final_text(rec_a[i]).strip() and agcr.final_text(rec_b[i]).strip()]

    raw_a = [agcr.raw_text(rec_a[i]) for i in ids]
    raw_b = [agcr.raw_text(rec_b[i]) for i in ids]
    fin_a = [agcr.final_text(rec_a[i]) for i in ids]
    fin_b = [agcr.final_text(rec_b[i]) for i in ids]

    y = np.array([int(ja[i] != jb[i]) for i in ids])
    # the original artifact-bearing proxy: keyword disagreement on the first 50 chars
    z_prefix = np.array([int(agcr.is_kw(a[:50]) != agcr.is_kw(b[:50]))
                         for a, b in zip(raw_a, raw_b)])
    # the repaired score's own cheap indicator, and a component of that score
    z_marker = np.array([int(agcr.is_kw(a) != agcr.is_kw(b))
                         for a, b in zip(fin_a, fin_b)], dtype=int)

    s_raw = agcr.tfidf_distance(raw_a, raw_b, prefix=50)
    s_final = agcr.tfidf_distance(fin_a, fin_b)
    s_comp = 0.7 * agcr.zscore(s_final) + 0.3 * z_marker.astype(float)

    out = {"n": len(ids), "scores": {}}
    for name, s in (("raw_prefix50_tfidf", s_raw),
                    ("final_tfidf", s_final),
                    ("composite_0.7_0.3", s_comp)):
        row = {"vs_prefix_keyword_proxy": audit(y, z_prefix, s),
               "vs_final_marker_indicator": audit(y, z_marker, s)}
        row["vs_prefix_keyword_proxy"]["verdict"] = verdict(row["vs_prefix_keyword_proxy"])
        row["vs_final_marker_indicator"]["verdict"] = verdict(row["vs_final_marker_indicator"])
        out["scores"][name] = row
        p, m = row["vs_prefix_keyword_proxy"], row["vs_final_marker_indicator"]
        print(f"{name:20s}  vs prefix-kw: AUC(z)={p['auc_z']:.3f} AUC(y)={p['auc_y']:.3f} "
              f"D={p['d_auc']:+.3f} kappa={p['kappa_zy']:.3f} -> {p['verdict']}")
        print(f"{'':20s}  vs final-marker: AUC(z)={m['auc_z']:.3f} AUC(y)={m['auc_y']:.3f} "
              f"D={m['d_auc']:+.3f} kappa={m['kappa_zy']:.3f} -> {m['verdict']}")

    p = ROOT / "analysis_results" / "audit_the_repair.json"
    p.write_text(json.dumps(out, indent=2))
    print("\n[wrote]", p)


if __name__ == "__main__":
    main()
