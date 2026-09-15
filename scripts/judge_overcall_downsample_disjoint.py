#!/usr/bin/env python3
"""The over-call correction, re-run on the deciding control. (2026-09-04)

Review round 2, item 8: Section 4.2's down-sampling defence was computed on the
same-span orientation-robust gap, while the control the paper declares deciding
is Delta_dis. This script re-runs the same three arms on Delta_dis:

    Delta_dis = |AUC(s, z_c) - 0.5| - |AUC(s, y_ds) - 0.5|

with z_c the off-span keyword disagreement (identical to the pipeline's), which
does not involve y and is therefore fixed across arms. Loading, ids, y, s and
the down-sampling scheme mirror judge_overcall_downsample.py exactly; the random
arm reports the mean over 1,000 draws with a percentile interval over draws.
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
DATA = ROOT / "results" / "disagree_routing"
JUDGE = "anthropic_claude-haiku-4-5-20251001"
HUMAN_RATE = 0.028
N_DRAWS = 1000
SPAN = 50
OUT = ROOT / "analysis_results" / "judge_overcall_downsample_disjoint.json"

spec = importlib.util.spec_from_file_location(
    "agcr", HERE / "analyze_rlc_composite_router.py")
agcr = importlib.util.module_from_spec(spec)
sys.modules["agcr"] = agcr
spec.loader.exec_module(agcr)


def main() -> None:
    phase = "phase3_xstest_full"

    def judge(model):
        rows = json.loads((DATA / f"{phase}_judge_{model}_{JUDGE}.json").read_text())["records"]
        return {r["id"]: int(bool(r["refusal_judge"])) for r in rows}

    ta = {r["id"]: r for r in json.loads(
        (DATA / f"{phase}_traces_qwen3.5-2b.json").read_text())["records"]}
    tb = {r["id"]: r for r in json.loads(
        (DATA / f"{phase}_traces_gemma-4-e2b.json").read_text())["records"]}
    ja, jb = judge("qwen3.5-2b"), judge("gemma-4-e2b")
    ids = sorted(set(ta) & set(tb) & set(ja) & set(jb))
    a_txt = [ta[i].get("trace") or "" for i in ids]
    b_txt = [tb[i].get("trace") or "" for i in ids]
    y = np.array([int(ja[i] != jb[i]) for i in ids])
    z_c = np.array([int(agcr.is_kw(a[SPAN:]) != agcr.is_kw(b[SPAN:]))
                    for a, b in zip(a_txt, b_txt)])
    s = agcr.tfidf_distance(a_txt, b_txt, prefix=SPAN)

    auc_zc = roc_auc_score(z_c, s)
    n = len(ids)
    pos = np.flatnonzero(y == 1)
    keep_n = max(1, int(round(HUMAN_RATE * n)))
    # referee check against the recorded run
    print(f"n={n}, judge positives {len(pos)}, keep {keep_n}, "
          f"AUC(s,y)={roc_auc_score(y, s):.4f}, AUC(s,z_c)={auc_zc:.4f}")

    def d_dis(y_ds):
        return abs(auc_zc - 0.5) - abs(roc_auc_score(y_ds, s) - 0.5)

    order = np.argsort(-s[pos], kind="stable")   # positives, highest score first
    out = {"n": n, "judge_positives": int(len(pos)), "kept_positives": keep_n,
           "auc_zc": float(auc_zc),
           "before": {"auc_y": float(roc_auc_score(y, s)),
                      "delta_dis": float(d_dis(y))}}

    # adversarial: remove the positives the score ranks highest
    y_adv = y.copy(); y_adv[pos[order[:len(pos) - keep_n]]] = 0
    # favourable: remove the positives the score ranks lowest
    y_fav = y.copy(); y_fav[pos[order[-(len(pos) - keep_n):]]] = 0
    for tag, y_ds in (("adversarial", y_adv), ("favourable", y_fav)):
        out[tag] = {"auc_y": float(roc_auc_score(y_ds, s)),
                    "delta_dis": float(d_dis(y_ds))}
        print(f"{tag:12s} AUC(s,y)={out[tag]['auc_y']:.4f} "
              f"Delta_dis={out[tag]['delta_dis']:+.4f}")

    rng = np.random.default_rng(20260904)
    vals, aucs = [], []
    for _ in range(N_DRAWS):
        keep = rng.choice(pos, size=keep_n, replace=False)
        y_r = np.zeros_like(y); y_r[keep] = 1
        aucs.append(roc_auc_score(y_r, s))
        vals.append(d_dis(y_r))
    vals, aucs = np.array(vals), np.array(aucs)
    out["random"] = {"auc_y_mean": float(aucs.mean()),
                     "delta_dis_mean": float(vals.mean()),
                     "delta_dis_p2.5": float(np.percentile(vals, 2.5)),
                     "delta_dis_p97.5": float(np.percentile(vals, 97.5))}
    print(f"{'random':12s} AUC(s,y) mean={aucs.mean():.4f} "
          f"Delta_dis mean={vals.mean():+.4f} "
          f"[{np.percentile(vals, 2.5):+.4f},{np.percentile(vals, 97.5):+.4f}] over draws")
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
