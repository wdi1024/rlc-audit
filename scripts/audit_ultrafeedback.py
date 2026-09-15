#!/usr/bin/env python3
"""A fourth external audit: UltraFeedback's selection target (2026-08-14).

UltraFeedback ships, per instruction, several completions with two separately
produced quality signals: a scalar `overall_score`, and four fine-grained axis
ratings (helpfulness, honesty, instruction-following, truthfulness) elicited with
their own rationales.  Downstream work selects and trains on the scalar while
claiming the fine-grained property -- response quality -- which is the
score--proxy--construct shape.

  proxy z    = which completion has the higher overall_score
  construct y = which completion is better on truthfulness (and on the fine-grained mean)

We audit the L2 link and the length coupling of each side, as for Hybrid LLM.

  python3 audit_ultrafeedback.py [--max-rows 4000]
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import cohen_kappa_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent


def rating(c, axis):
    a = (c.get("annotations") or {}).get(axis) or {}
    r = a.get("Rating")
    try:
        return float(r)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=4000)
    args = ap.parse_args()

    ds = load_dataset("openbmb/UltraFeedback", split="train", streaming=True)
    gap, y_true, y_fine, dlen = [], [], [], []
    n_rows = 0
    for row in itertools.islice(ds, args.max_rows):
        comps = row.get("completions") or []
        n_rows += 1
        for a, b in itertools.combinations(comps, 2):
            try:
                oa, ob = float(a["overall_score"]), float(b["overall_score"])
            except (TypeError, ValueError, KeyError):
                continue
            ta, tb = rating(a, "truthfulness"), rating(b, "truthfulness")
            fa, fb = a.get("fine-grained_score"), b.get("fine-grained_score")
            if None in (ta, tb) or fa is None or fb is None:
                continue
            if oa == ob or ta == tb:
                continue          # ties carry no direction on either side
            gap.append(oa - ob)
            y_true.append(int(ta > tb))
            y_fine.append(int(float(fa) > float(fb)))
            dlen.append(len(a.get("response") or "") - len(b.get("response") or ""))

    gap = np.array(gap); yt = np.array(y_true); yf = np.array(y_fine); ln = np.array(dlen)
    z = (gap > 0).astype(int)
    print(f"instructions read: {n_rows}   directional comparisons: {len(z)}")
    for name, y in (("truthfulness", yt), ("fine-grained mean", yf)):
        if len(set(y)) < 2:
            continue
        agree = float((z == y).mean())
        k = float(cohen_kappa_score(z, y))
        auc = float(roc_auc_score(y, gap))
        print(f"\nconstruct = {name}")
        print(f"  agreement {agree:.3f}   kappa(z,y) {k:.3f}   AUC(overall gap, y) {auc:.3f}")
        print(f"  the selection target disagrees with the construct on "
              f"{(1-agree)*100:.1f}% of directional pairs")
        print(f"  length ranks the target at AUC {roc_auc_score(z, ln):.3f} and the "
              f"construct at {roc_auc_score(y, ln):.3f}")

    out = {"n_instructions": n_rows, "n_pairs": int(len(z)),
           "truthfulness": {"agreement": float((z == yt).mean()),
                            "kappa": float(cohen_kappa_score(z, yt)),
                            "auc_gap": float(roc_auc_score(yt, gap)),
                            "auc_len_proxy": float(roc_auc_score(z, ln)),
                            "auc_len_construct": float(roc_auc_score(yt, ln))},
           "fine_grained": {"agreement": float((z == yf).mean()),
                            "kappa": float(cohen_kappa_score(z, yf)),
                            "auc_gap": float(roc_auc_score(yf, gap)),
                            "auc_len_construct": float(roc_auc_score(yf, ln))}}
    p = ROOT / "analysis_results" / "ultrafeedback_audit.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\n[wrote] {p}")


if __name__ == "__main__":
    main()
