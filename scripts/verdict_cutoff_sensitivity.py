#!/usr/bin/env python3
"""Are the verdict bins arbitrary? (reviewer request, 2026-08-13)

RLC-Audit calls a contract MISMATCH when the proxy-minus-semantic AUC gap is at least
0.15 and kappa(z,y) is at most 0.20.  Those two numbers are choices, and a reviewer is
right to ask how much of the paper rests on them.  This sweeps both thresholds over a
grid and reports, for every audited contract, the region in which its verdict is
stable.  A finding that survives only in a narrow corner of the grid would be a
finding about the corner.

Contracts are read from the cached runs and analysis outputs, so the grid is computed
from the same numbers the tables report.

  python3 verdict_cutoff_sensitivity.py
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
GAPS = np.round(np.arange(0.05, 0.41, 0.05), 2)
KAPPAS = np.round(np.arange(0.05, 0.51, 0.05), 2)

spec = importlib.util.spec_from_file_location("agcr", HERE / "analyze_rlc_composite_router.py")
agcr = importlib.util.module_from_spec(spec)
sys.modules["agcr"] = agcr
spec.loader.exec_module(agcr)


def find(name):
    for d in DIRS:
        if (d / name).exists():
            return d / name
    return None


def load_judge(phase, model, tag):
    p = find(f"{phase}_judge_final_{model}_{tag}.json")
    if p is None:
        return None
    return {r["id"]: bool(r["refusal_judge"])
            for r in json.loads(p.read_text())["records"]
            if r.get("refusal_judge") is not None and not r.get("error")}


def load_traces(phase, model):
    p = find(f"{phase}_traces_{model}.json")
    return {r["id"]: r for r in json.loads(p.read_text())["records"]}


def contract(spec_, tag, prefix=None, label=""):
    ja = load_judge(spec_.phase_a, spec_.model_a, tag)
    jb = load_judge(spec_.phase_b, spec_.model_b, tag)
    if not ja or not jb:
        return None
    ra = load_traces(spec_.phase_a, spec_.model_a)
    rb = load_traces(spec_.phase_b, spec_.model_b)
    ids = sorted(set(ja) & set(jb) & set(ra) & set(rb))
    txt = agcr.raw_text if prefix else agcr.final_text
    ids = [i for i in ids if txt(ra[i]).strip() and txt(rb[i]).strip()]
    if len(ids) < 50:
        return None
    a = [txt(ra[i]) for i in ids]
    b = [txt(rb[i]) for i in ids]
    s = agcr.tfidf_distance(a, b, prefix=prefix) if prefix else agcr.tfidf_distance(a, b)
    y = np.array([int(ja[i] != jb[i]) for i in ids])
    kwa = [t[:prefix] if prefix else t for t in a]
    kwb = [t[:prefix] if prefix else t for t in b]
    z = np.array([int(agcr.is_kw(x) != agcr.is_kw(w)) for x, w in zip(kwa, kwb)])
    if len(set(z)) < 2 or len(set(y)) < 2:
        return None
    return {"label": label, "n": len(ids),
            "gap": float(roc_auc_score(z, s) - roc_auc_score(y, s)),
            "kappa": float(cohen_kappa_score(z, y))}


def main():
    rows = []
    p0 = agcr.PAIR_SPECS[0]
    c = contract(p0, p0.judge_tag, prefix=50, label="primary raw-prefix (clean run)")
    if c:
        rows.append(c)
    for sp in agcr.PAIR_SPECS[1:10]:
        c = contract(sp, "anthropic_claude-haiku-4-5-20251001",
                     label=f"panel {sp.model_a}/{sp.model_b}")
        if c:
            rows.append(c)
    # correctness controls, from the cached decomposition reports
    for f, name in (("hotpot_decomposition_report.json", "HotpotQA correctness"),
                    ("gsm8k_decomposition_report.json", "GSM8K correctness")):
        p = ROOT / "analysis_results" / f
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        cells = d.get("cells", {})
        if "TF-IDF_x_keyword" in cells and "TF-IDF_x_judge" in cells:
            rows.append({"label": name, "n": d.get("n_common", 0),
                         "gap": cells["TF-IDF_x_keyword"]["auc"] - cells["TF-IDF_x_judge"]["auc"],
                         "kappa": float(d.get("rlc_audit", {}).get("kappa_surface_semantic",
                                                                   np.nan))})
    print(f"{'contract':44} {'n':>5} {'gap':>7} {'kappa':>7}")
    for r in rows:
        print(f"{r['label'][:42]:44} {r['n']:>5} {r['gap']:>+7.3f} {r['kappa']:>7.3f}")

    print("\nfraction of the grid in which each contract is MISMATCH "
          f"(gap thresholds {GAPS[0]}-{GAPS[-1]}, kappa thresholds {KAPPAS[0]}-{KAPPAS[-1]}):")
    grid = {}
    for r in rows:
        hits = sum(1 for g in GAPS for k in KAPPAS
                   if r["gap"] >= g and (np.isnan(r["kappa"]) or r["kappa"] <= k))
        frac = hits / (len(GAPS) * len(KAPPAS))
        grid[r["label"]] = frac
        print(f"  {r['label'][:42]:44} {frac:>6.2f}")

    prim = grid.get("primary raw-prefix (clean run)", float("nan"))
    others = [v for k, v in grid.items() if k != "primary raw-prefix (clean run)"]
    print(f"\nprimary contract is MISMATCH over {prim:.2f} of the grid; "
          f"the highest non-primary contract reaches {max(others):.2f}")
    out = {"gap_thresholds": GAPS.tolist(), "kappa_thresholds": KAPPAS.tolist(),
           "contracts": rows, "mismatch_fraction": grid}
    p = ROOT / "analysis_results" / "verdict_cutoff_sensitivity.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\n[wrote] {p}")


if __name__ == "__main__":
    main()
