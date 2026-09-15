#!/usr/bin/env python3
"""Equivalence-band (TOST) reading of the construct-orientation step. (2026-09-04)

Review A-2: the paper's SCORE FAILURE rows were read off a 95% CI covering 1/2,
which accepts the null. The equivalence bound already exists in this repository:
verdict_rule_v2.py (2026-08-21) declares MAX_CHANCE = 0.55 with the comment
"'no better than chance' must be shown, not assumed" and cites JailbreakBench's
interval as the case that cannot be shown. This script computes the proper TOST
for every decided contract, from the same per-example vectors the pipeline uses:

  - reproduce the paper's 95% bootstrap CI on AUC(s,y) first (referee check),
  - TOST at alpha=0.05 with margin delta=0.05: the 90% bootstrap CI must sit
    inside [0.45, 0.55],
  - also report the stricter 95%-CI-inclusion reading.

No thresholds are chosen from these results: the band is the pre-existing
MAX_CHANCE constant, symmetric about chance.
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
AR = ROOT / "analysis_results"
OUT = AR / "equivalence_band_tost.json"
BAND = (0.45, 0.55)
N_BOOT = 2000

spec = importlib.util.spec_from_file_location(
    "cspc", HERE / "complement_span_proxy_control.py")
cspc = importlib.util.module_from_spec(spec)
sys.modules["cspc"] = cspc
spec.loader.exec_module(cspc)

VECS: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_orig_make_row = cspc.make_row


def capture_make_row(label, family, span, s, z, z_perp, z_comp, y, empty_share, seed):
    key = label if family == "refusal" else f"{label}, {span}-char span"
    VECS[key] = (np.asarray(s, dtype=float), np.asarray(y, dtype=int))
    return _orig_make_row(label, family, span, s, z, z_perp, z_comp, y, empty_share, seed)


cspc.make_row = capture_make_row


def boot_auc_y(s, y, seed=20260904):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(N_BOOT):
        sel = rng.integers(0, len(s), len(s))
        if len(set(y[sel])) < 2:
            continue
        vals.append(roc_auc_score(y[sel], s[sel]))
    v = np.array(vals)
    return {
        "auc_y": float(roc_auc_score(y, s)),
        "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
        "ci90": [float(np.percentile(v, 5.0)), float(np.percentile(v, 95.0))],
    }


def main() -> None:
    rows = cspc.refusal_rows() + cspc.correctness_rows()
    out = {"band": BAND, "n_boot": N_BOOT, "contracts": {}}
    print(f"{'contract':28s} {'AUC(s,y)':>8s} {'95% CI':>17s} {'90% CI':>17s} "
          f"{'TOST':>5s} {'95%-in':>6s}")
    for r in rows:
        key = r["contract"] if r["family"] == "refusal" else \
            f"{r['contract']}, {r['span']}-char span"
        if key not in VECS:
            continue
        s, y = VECS[key]
        if len(set(y)) < 2:
            continue
        b = boot_auc_y(s, y)
        lo90, hi90 = b["ci90"]
        lo95, hi95 = b["ci95"]
        tost = BAND[0] <= lo90 and hi90 <= BAND[1]
        strict = BAND[0] <= lo95 and hi95 <= BAND[1]
        out["contracts"][key] = {**b, "tost_pass": bool(tost),
                                 "ci95_inside_band": bool(strict),
                                 "y_positives": int(y.sum()), "n": int(len(y))}
        print(f"{key:28s} {b['auc_y']:8.3f} [{lo95:.3f},{hi95:.3f}] "
              f"[{lo90:.3f},{hi90:.3f}] {str(tost):>5s} {str(strict):>6s}")
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
