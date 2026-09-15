#!/usr/bin/env python3
"""The magnitude bands swept on the statistic that decides. (2026-09-05)

Review round 6, item 1-2: Appendix H sweeps the MISMATCH thresholds over
Delta_AUC, the same-span gap, while the verdict function applies its bands to
Delta_dis. A sweep over the deciding statistic is what the claim needs.

For every contract that reaches the magnitude bands (preconditions passed,
construct arm oriented, gap above the contract's own permutation null), we sweep
the MISMATCH band over [0.05, 0.40] and the CAUTION band over [0.02, 0.20],
recording the verdict at every pair, and report for each contract the fraction of
the grid on which it takes each verdict. The kappa(z^c, y) merge threshold is
swept alongside, over [0.20, 0.60], because it is the other constant on the
stage-3 path.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
AR = ROOT / "analysis_results"
OUT = AR / "band_sweep_disjoint.json"

MISMATCH_GRID = [round(x, 3) for x in np.arange(0.05, 0.4001, 0.025)]
CAUTION_GRID = [round(x, 3) for x in np.arange(0.02, 0.2001, 0.02)]
KAPPA_GRID = [round(x, 2) for x in np.arange(0.20, 0.6001, 0.05)]


def main() -> None:
    v2 = json.loads((AR / "verdict_rule_v2.json").read_text())
    tost = json.loads((AR / "equivalence_band_tost.json").read_text())["contracts"]

    def tost_key(name):
        if name in tost:
            return name
        alt = f"{name}, {name.split(',')[-1].strip()}" if "," in name else None
        for k in tost:
            if k.startswith(name):
                return k
        return alt

    rows = []
    for name, r in v2.items():
        if r["reason"] == "precondition" or r["delta_dis"] is None:
            continue
        k = tost_key(name)
        t = tost.get(k)
        if t is None:
            continue
        lo90, hi90 = t["ci90"]
        at_chance = 0.45 <= lo90 and hi90 <= 0.55
        lo95, hi95 = t["ci95"]
        reversed_score = hi95 < 0.5
        oriented = not (lo95 <= 0.5 <= hi95)
        if at_chance or reversed_score or not oriented:
            continue                       # settled before the bands
        if not r["beats_null"]:
            continue                       # below its own null: never reaches a band
        rows.append({"contract": name, "delta_dis": r["delta_dis"],
                     "kappa_zc_y": r["kappa_zc_y"]})

    print(f"contracts reaching the magnitude bands: {len(rows)}")
    out = {"mismatch_grid": MISMATCH_GRID, "caution_grid": CAUTION_GRID,
           "kappa_grid": KAPPA_GRID, "contracts": {}}
    for row in rows:
        tally = Counter()
        for mm in MISMATCH_GRID:
            for cc in CAUTION_GRID:
                if cc > mm:
                    continue
                for kk in KAPPA_GRID:
                    if row["kappa_zc_y"] is not None and row["kappa_zc_y"] >= kk:
                        tally["ALIGNED"] += 1
                    elif row["delta_dis"] >= mm:
                        tally["MISMATCH"] += 1
                    elif row["delta_dis"] >= cc:
                        tally["CAUTION"] += 1
                    else:
                        tally["ALIGNED"] += 1
        total = sum(tally.values())
        shares = {k: v / total for k, v in tally.items()}
        out["contracts"][row["contract"]] = {
            "delta_dis": row["delta_dis"], "kappa_zc_y": row["kappa_zc_y"],
            "grid_points": total, "verdict_shares": shares}
        share_str = ", ".join(f"{k} {v:.2f}" for k, v in sorted(shares.items()))
        print(f"  {row['contract']:28s} d_dis {row['delta_dis']:+.3f}  {share_str}")

    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
