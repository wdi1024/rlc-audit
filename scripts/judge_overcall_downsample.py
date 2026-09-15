#!/usr/bin/env python3
"""What if the judge's over-call is corrected? (2026-08-20)

Section 5.4 measures an over-call: the judge fires the pair-level disagreement event
on 15.6% of resolved prompts where human adjudication fires on 2.8%.  The paper's
existing robustness check injects measured side-level error into y, which *raises*
prevalence and so answers a different question.  It does not answer the one a
reviewer will ask: if y has too many positives, what happens to the diagnosis when
the surplus is removed?

This script removes it.  Judge positives are down-sampled until the pair-level
prevalence matches human adjudication, and the diagnosis is recomputed.  Which
positives are removed is not identified by the data, so we bracket it rather than
guess, taking the two extremes and the neutral case:

  random      removal independent of the score (1,000 draws)
  adversarial remove the positives the score ranks *highest* -- the ones a router
              would have caught -- which is the worst case for us
  favourable  remove the positives the score ranks lowest

The adversarial arm is the number that matters: if the diagnosis survives it, the
over-call cannot explain the result.  Reported for the primary contract's headline
quantities, semantic AUC and the routed count at the stated budget.

  python3 judge_overcall_downsample.py
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
BUDGET = 55
HUMAN_RATE = 0.028          # pair-level disagreement rate under human adjudication
N_DRAWS = 1000

spec = importlib.util.spec_from_file_location(
    "agcr", HERE / "analyze_rlc_composite_router.py")
agcr = importlib.util.module_from_spec(spec)
sys.modules["agcr"] = agcr
spec.loader.exec_module(agcr)


def routed(y: np.ndarray, s: np.ndarray, b: int = BUDGET) -> int:
    return int(y[np.argsort(-s, kind="stable")[:b]].sum())


def main() -> None:
    def plain_judge(phase, model):
        rows = json.loads((DATA / f"{phase}_judge_{model}_{JUDGE}.json").read_text())["records"]
        return {r["id"]: int(bool(r["refusal_judge"])) for r in rows}

    phase = "phase3_xstest_full"
    ta = {r["id"]: r for r in json.loads(
        (DATA / f"{phase}_traces_qwen3.5-2b.json").read_text())["records"]}
    tb = {r["id"]: r for r in json.loads(
        (DATA / f"{phase}_traces_gemma-4-e2b.json").read_text())["records"]}
    ja, jb = plain_judge(phase, "qwen3.5-2b"), plain_judge(phase, "gemma-4-e2b")
    ids = sorted(set(ta) & set(tb) & set(ja) & set(jb))

    a_txt = [ta[i].get("trace") or "" for i in ids]
    b_txt = [tb[i].get("trace") or "" for i in ids]
    y = np.array([int(ja[i] != jb[i]) for i in ids])
    z = np.array([int(agcr.is_kw(a[:50]) != agcr.is_kw(b[:50]))
                  for a, b in zip(a_txt, b_txt)])
    s = agcr.tfidf_distance(a_txt, b_txt, prefix=50)

    n = len(ids)
    pos = np.flatnonzero(y == 1)
    keep_n = max(1, int(round(HUMAN_RATE * n)))
    print(f"primary run n={n}: judge fires on {len(pos)} pairs ({len(pos)/n:.3f}); "
          f"human adjudication implies {keep_n} ({HUMAN_RATE:.3f})")
    print(f"so the correction removes {len(pos)-keep_n} of {len(pos)} positives\n")

    auc_z = float(roc_auc_score(z, s))
    base_auc = float(roc_auc_score(y, s))
    base_routed = routed(y, s)
    print(f"before correction: AUC(s,y)={base_auc:.3f}, gap vs proxy "
          f"{auc_z-base_auc:+.3f}, routed {base_routed}/{BUDGET}")

    order = np.argsort(-s[pos], kind="stable")      # highest-scoring positives first
    arms = {}

    # adversarial: keep the positives the score ranks lowest
    keep_adv = pos[order[-keep_n:]]
    # favourable: keep the ones it ranks highest
    keep_fav = pos[order[:keep_n]]
    for name, keep in (("adversarial", keep_adv), ("favourable", keep_fav)):
        yy = np.zeros_like(y); yy[keep] = 1
        a = float(roc_auc_score(yy, s))
        arms[name] = {"auc": a, "gap": auc_z - a,
                      "gap_abs": abs(auc_z - 0.5) - abs(a - 0.5),
                      "routed": routed(yy, s)}

    rng = np.random.default_rng(0)
    aucs, routs = [], []
    for _ in range(N_DRAWS):
        keep = rng.choice(pos, size=keep_n, replace=False)
        yy = np.zeros_like(y); yy[keep] = 1
        aucs.append(roc_auc_score(yy, s)); routs.append(routed(yy, s))
    ma = float(np.mean(aucs))
    arms["random"] = {"auc": ma,
                      "auc_ci": [float(np.percentile(aucs, 2.5)),
                                 float(np.percentile(aucs, 97.5))],
                      "gap": auc_z - ma,
                      "gap_abs": abs(auc_z - 0.5) - abs(ma - 0.5),
                      "routed": float(np.mean(routs))}

    print(f"\n{'arm':12s} {'AUC(s,y)':>9s} {'gap':>8s} {'gap_abs':>9s} {'routed':>8s}")
    for k in ("random", "adversarial", "favourable"):
        a = arms[k]
        print(f"{k:12s} {a['auc']:9.3f} {a['gap']:+8.3f} {a['gap_abs']:+9.3f} "
              f"{a['routed']:8.1f}")
    print("\n  gap_abs is the orientation-robust gap. It is the number to read in the")
    print("  adversarial arm, where removing the top-ranked positives drives AUC(s,y) below")
    print("  chance and inflates the naive gap for the wrong reason.")

    lo = min(arms[k]["gap_abs"] for k in arms)
    print("\nreading:")
    if lo >= 0.15:
        print(f"  the orientation-robust gap stays at or above {lo:+.3f} under every correction,")
        print("  past the MISMATCH cutoff: the over-call cannot explain the diagnosis.")
    else:
        print(f"  the orientation-robust gap falls to {lo:+.3f} under the worst correction, below the")
        print("  cutoff, so the diagnosis depends on which positives the judge over-called.")

    out = {"n": n, "judge_positives": int(len(pos)), "human_rate": HUMAN_RATE,
           "kept_positives": keep_n, "auc_proxy": auc_z,
           "before": {"auc": base_auc, "gap": auc_z - base_auc, "routed": base_routed},
           "arms": arms}
    path = ROOT / "analysis_results" / "judge_overcall_downsample.json"
    path.write_text(json.dumps(out, indent=2))
    print("\n[wrote]", path)


if __name__ == "__main__":
    main()
