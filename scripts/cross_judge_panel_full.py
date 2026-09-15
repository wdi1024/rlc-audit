#!/usr/bin/env python3
"""Cross-judge re-adjudication of the full ten-pair panel (reviewer request, 2026-08-13).

The panel's semantic labels come from GPT-4o-mini, a cost choice at ten pairs times
450 prompts, while the primary contract is labelled by Claude Haiku.  The paper
defended this with a cross-judge check on three pairs; a reviewer asked for all ten.

Side labels are shared across pairs, so the full check needs one re-adjudication per
(phase, model) rather than per pair.  Six of the eight combinations were already
labelled by the primary judge; this script consumes those plus the two that were
missing and reports, for every panel pair, the semantic label agreement between the
two judges and whether the final-span score's verdict changes when the labels do.

  python3 cross_judge_panel_full.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DIRS = [ROOT / "results" / "disagree_routing", ROOT / "data"]
J_MINI = "openai_gpt-4o-mini"
J_HAIKU = "anthropic_claude-haiku-4-5-20251001"

spec = importlib.util.spec_from_file_location("agcr", HERE / "analyze_rlc_composite_router.py")
agcr = importlib.util.module_from_spec(spec)
sys.modules["agcr"] = agcr
spec.loader.exec_module(agcr)


def find(name: str) -> Path | None:
    for d in DIRS:
        if (d / name).exists():
            return d / name
    return None


def load_judge(phase: str, model: str, tag: str) -> dict[str, bool] | None:
    p = find(f"{phase}_judge_final_{model}_{tag}.json")
    if p is None:
        return None
    recs = json.loads(p.read_text())["records"]
    return {r["id"]: bool(r["refusal_judge"]) for r in recs
            if r.get("refusal_judge") is not None and not r.get("error")}


def load_traces(phase: str, model: str) -> dict[str, dict]:
    p = find(f"{phase}_traces_{model}.json")
    return {r["id"]: r for r in json.loads(p.read_text())["records"]}


def main():
    rows = []
    for k, spec_ in enumerate(agcr.PAIR_SPECS[1:10], start=1):
        ja_m = load_judge(spec_.phase_a, spec_.model_a, J_MINI)
        jb_m = load_judge(spec_.phase_b, spec_.model_b, J_MINI)
        ja_h = load_judge(spec_.phase_a, spec_.model_a, J_HAIKU)
        jb_h = load_judge(spec_.phase_b, spec_.model_b, J_HAIKU)
        if not all([ja_m, jb_m, ja_h, jb_h]):
            missing = [n for n, v in (("a-mini", ja_m), ("b-mini", jb_m),
                                      ("a-haiku", ja_h), ("b-haiku", jb_h)) if not v]
            print(f"{spec_.model_a}/{spec_.model_b}: missing {missing}")
            continue
        ra = load_traces(spec_.phase_a, spec_.model_a)
        rb = load_traces(spec_.phase_b, spec_.model_b)
        ids = sorted(set(ja_m) & set(jb_m) & set(ja_h) & set(jb_h) & set(ra) & set(rb))
        ids = [i for i in ids
               if agcr.final_text(ra[i]).strip() and agcr.final_text(rb[i]).strip()]
        if len(ids) < 50:
            print(f"{spec_.model_a}/{spec_.model_b}: only {len(ids)} joined ids")
            continue
        fa = [agcr.final_text(ra[i]) for i in ids]
        fb = [agcr.final_text(rb[i]) for i in ids]
        s = agcr.tfidf_distance(fa, fb)
        y_m = np.array([int(ja_m[i] != jb_m[i]) for i in ids])
        y_h = np.array([int(ja_h[i] != jb_h[i]) for i in ids])
        B = max(1, int(round(len(ids) * 0.10)))
        idx = np.argsort(-s, kind="stable")[:B]

        def auc(y):
            return float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")

        z = np.array([int(agcr.is_kw(a) != agcr.is_kw(b)) for a, b in zip(fa, fb)])
        rows.append({
            "pair": f"{spec_.model_a}/{spec_.model_b}", "n": len(ids),
            "pair_label_agreement": float((y_m == y_h).mean()),
            "pair_label_kappa": float(cohen_kappa_score(y_m, y_h))
            if len(set(y_m)) > 1 and len(set(y_h)) > 1 else float("nan"),
            "n_pos_mini": int(y_m.sum()), "n_pos_haiku": int(y_h.sum()),
            "auc_sem_mini": auc(y_m), "auc_sem_haiku": auc(y_h),
            "auc_proxy": float(roc_auc_score(z, s)) if len(set(z)) > 1 else float("nan"),
            "gap_mini": (float(roc_auc_score(z, s)) - auc(y_m)) if len(set(z)) > 1 else float("nan"),
            "gap_haiku": (float(roc_auc_score(z, s)) - auc(y_h)) if len(set(z)) > 1 else float("nan"),
            "routed_mini": int(y_m[idx].sum()), "routed_haiku": int(y_h[idx].sum()),
            "budget": B,
        })

    print(f"\n{'pair':40} {'n':>4} {'label agr':>10} {'kappa':>7} "
          f"{'AUC sem mini':>13} {'AUC sem haiku':>14} {'gap mini':>9} {'gap haiku':>10}")
    for r in rows:
        print(f"{r['pair'][:38]:40} {r['n']:>4} {r['pair_label_agreement']:>10.3f} "
              f"{r['pair_label_kappa']:>7.3f} {r['auc_sem_mini']:>13.3f} "
              f"{r['auc_sem_haiku']:>14.3f} {r['gap_mini']:>+9.3f} {r['gap_haiku']:>+10.3f}")

    gm = np.array([r["gap_mini"] for r in rows])
    gh = np.array([r["gap_haiku"] for r in rows])
    print(f"\npanel proxy-minus-semantic gap: under GPT-4o-mini labels "
          f"[{gm.min():+.3f}, {gm.max():+.3f}], under the primary judge "
          f"[{gh.min():+.3f}, {gh.max():+.3f}]")
    print(f"pairs whose gap sign changes between judges: "
          f"{int(((gm > 0) != (gh > 0)).sum())}/{len(rows)}")
    print(f"mean pair-label agreement {np.mean([r['pair_label_agreement'] for r in rows]):.3f}, "
          f"mean kappa {np.nanmean([r['pair_label_kappa'] for r in rows]):.3f}")

    out = {"judges": [J_MINI, J_HAIKU], "n_pairs": len(rows), "pairs": rows,
           "gap_range_mini": [float(gm.min()), float(gm.max())],
           "gap_range_haiku": [float(gh.min()), float(gh.max())],
           "sign_changes": int(((gm > 0) != (gh > 0)).sum())}
    p = ROOT / "analysis_results" / "cross_judge_panel_full.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\n[wrote] {p}")


if __name__ == "__main__":
    main()
