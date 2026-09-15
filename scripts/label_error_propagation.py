#!/usr/bin/env python3
"""Propagate the measured label uncertainty into the headline numbers (2026-08-13).

The paper reports semantic AUC 0.592 and 6/55 routed against a construct whose
pair-level human-judge agreement is only kappa 0.266.  The existing defence is a
36-flip adversarial stress test, which answers a different question: how many
worst-case flips would be needed to erase the gap.  A reviewer asked the natural
complementary question -- if y is as noisy as we measured it to be, what is the
distribution of the headline numbers?

Two parts:

  1. Estimate a per-side error model from the resolved human sample: for each model
     side, P(judge says refusal | human says comply) and P(judge says comply | human
     says refusal).  Inject that noise into the judge side labels, rebuild the
     pair-level XOR, and recompute semantic AUC and routed composition.  This is a
     realistic-noise interval, not a worst case.

  2. Check the excluded pairs for selection bias.  137 of 258 annotated pairs were
     marked as leaving insufficient context and dropped.  If those sit at high score
     values, the resolved sample is not representative of the routed queue, and the
     reliability numbers would be measured off the region that matters.

  python3 label_error_propagation.py
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
N_SIM = 4000
RNG = np.random.default_rng(0)

spec = importlib.util.spec_from_file_location("agcr", HERE / "analyze_rlc_composite_router.py")
agcr = importlib.util.module_from_spec(spec)
sys.modules["agcr"] = agcr
spec.loader.exec_module(agcr)

ANN = (ROOT / "human_annotation_iclr" / "completed_full_adjudicated"
       / "full_annotation_adjudicated_final_labels.csv")
ANN_ALL = ANN.parent / "full_adjudication_completed.csv"


def tf(x) -> bool:
    return str(x).strip().lower() in ("true", "1", "yes")


def error_model(rows) -> dict[str, dict[str, float]]:
    """P(judge label != human label) split by the human label, per side."""
    out = {}
    for side, jcol, hcol in (("a", "judge_a_refusal", "full_model_a_final_bool"),
                             ("b", "judge_b_refusal", "full_model_b_final_bool")):
        h = np.array([tf(r[hcol]) for r in rows])
        j = np.array([tf(r[jcol]) for r in rows])
        fp = float((j[~h]).mean()) if (~h).sum() else 0.0   # human comply, judge refusal
        fn = float((~j[h]).mean()) if h.sum() else 0.0      # human refusal, judge comply
        out[side] = {"n": int(len(h)), "human_refusal_rate": float(h.mean()),
                     "false_refusal": fp, "false_comply": fn}
    return out


def main():
    rows = [r for r in csv.DictReader(open(ANN))]
    resolved = [r for r in rows if r["full_context_sufficient_final"].strip().upper() == "YES"]
    em = error_model(resolved)
    print(f"error model from {len(resolved)} resolved pairs")
    for side, d in em.items():
        print(f"  side {side}: judge says refusal when human says comply {d['false_refusal']:.3f}; "
              f"judge says comply when human says refusal {d['false_comply']:.3f}")

    # --- part 2: selection bias among excluded pairs -------------------------
    all_rows = list(csv.DictReader(open(ANN_ALL))) if ANN_ALL.exists() else []
    score_col = "cosine_distance_prefix50"
    have = [r for r in rows if r.get(score_col) not in (None, "")]
    inc = np.array([float(r[score_col]) for r in have
                    if r["full_context_sufficient_final"].strip().upper() == "YES"])
    exc = np.array([float(r[score_col]) for r in have
                    if r["full_context_sufficient_final"].strip().upper() != "YES"])
    print(f"\nselection check on the routing score ({score_col}):")
    print(f"  resolved  n={len(inc)}  mean {inc.mean():.3f}  median {np.median(inc):.3f}")
    print(f"  excluded  n={len(exc)}  mean {exc.mean():.3f}  median {np.median(exc):.3f}")
    if len(inc) > 5 and len(exc) > 5:
        lab = np.r_[np.ones(len(exc)), np.zeros(len(inc))]
        sc = np.r_[exc, inc]
        print(f"  AUC(score -> being excluded) = {roc_auc_score(lab, sc):.3f} "
              f"(0.5 means exclusion is unrelated to the score)")

    # --- part 1: inject the measured noise into the primary contract ---------
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
    s = agcr.tfidf_distance(raw_a, raw_b, prefix=50)
    A = np.array([int(ja[i]) for i in ids])
    B = np.array([int(jb[i]) for i in ids])
    y0 = (A != B).astype(int)
    Bq = 55
    order = np.argsort(-s, kind="stable")[:Bq]
    print(f"\nclean-run raw-prefix contract as reported: n={len(ids)}  AUC(s,y)={roc_auc_score(y0, s):.3f}  "
          f"routed {int(y0[order].sum())}/{Bq}")

    aucs, routed = [], []
    for _ in range(N_SIM):
        a2, b2 = A.copy(), B.copy()
        for arr, side in ((a2, "a"), (b2, "b")):
            d = em[side]
            flip_up = (arr == 0) & (RNG.random(len(arr)) < d["false_refusal"])
            flip_dn = (arr == 1) & (RNG.random(len(arr)) < d["false_comply"])
            arr[flip_up] = 1
            arr[flip_dn] = 0
        y = (a2 != b2).astype(int)
        if len(set(y)) < 2:
            continue
        aucs.append(roc_auc_score(y, s))
        routed.append(int(y[order].sum()))
    # worst-case coupling: all error mass on exactly one side per prompt
    # (anti-correlated flips maximize XOR disruption at the same side marginals;
    #  co-occurring flips preserve the XOR, so this bounds every correlation)
    aucs_wc, routed_wc = [], []
    for _ in range(N_SIM):
        a2, b2 = A.copy(), B.copy()
        u = RNG.random(len(a2))
        pa_flip = np.where(a2 == 0, em["a"]["false_refusal"], em["a"]["false_comply"])
        pb_flip = np.where(b2 == 0, em["b"]["false_refusal"], em["b"]["false_comply"])
        fa = u < pa_flip
        fb = u > 1.0 - pb_flip
        a2[fa] = 1 - a2[fa]
        b2[fb] = 1 - b2[fb]
        y = (a2 != b2).astype(int)
        if len(set(y)) < 2:
            continue
        aucs_wc.append(roc_auc_score(y, s))
        routed_wc.append(int(y[order].sum()))
    aucs_wc = np.array(aucs_wc)
    print(f"worst-case anti-correlated error structure ({N_SIM} draws):")
    print(f"  semantic AUC {aucs_wc.mean():.3f} [{np.percentile(aucs_wc,2.5):.3f}, "
          f"{np.percentile(aucs_wc,97.5):.3f}]")

    aucs = np.array(aucs)
    routed = np.array(routed, dtype=float)
    print(f"under the measured judge error rates ({N_SIM} draws):")
    print(f"  semantic AUC {aucs.mean():.3f} [{np.percentile(aucs,2.5):.3f}, "
          f"{np.percentile(aucs,97.5):.3f}]   P(AUC > 0.70) = {(aucs>0.70).mean():.3f}")
    print(f"  routed semantic {routed.mean():.1f}/{Bq} "
          f"[{np.percentile(routed,2.5):.0f}, {np.percentile(routed,97.5):.0f}]  "
          f"(the measured noise is one-sided, so it raises pair prevalence from "
          f"{y0.mean():.3f}; routed counts are not comparable to the reported value, AUC is)")

    out = {"error_model": em, "n_resolved": len(resolved),
           "selection": {"resolved_mean": float(inc.mean()), "excluded_mean": float(exc.mean()),
                         "auc_score_predicts_exclusion":
                             float(roc_auc_score(np.r_[np.ones(len(exc)), np.zeros(len(inc))],
                                                 np.r_[exc, inc]))},
           "reported": {"auc": float(roc_auc_score(y0, s)), "routed": int(y0[order].sum()),
                        "budget": Bq, "n": len(ids)},
           "propagated_worstcase_anticorrelated": {
               "auc_mean": float(aucs_wc.mean()),
               "auc_ci": [float(np.percentile(aucs_wc, 2.5)),
                          float(np.percentile(aucs_wc, 97.5))]},
           "propagated": {"auc_mean": float(aucs.mean()),
                          "auc_ci": [float(np.percentile(aucs, 2.5)),
                                     float(np.percentile(aucs, 97.5))],
                          "p_auc_above_0.70": float((aucs > 0.70).mean()),
                          "routed_mean": float(routed.mean()),
                          "routed_ci": [float(np.percentile(routed, 2.5)),
                                        float(np.percentile(routed, 97.5))],
                          "n_sim": N_SIM}}
    p = ROOT / "analysis_results" / "label_error_propagation.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\n[wrote] {p}")


if __name__ == "__main__":
    main()
