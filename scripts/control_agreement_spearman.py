#!/usr/bin/env python3
"""How strongly the two controls agree, on a named row set. (2026-09-04)

Review round 5, A6: Appendix N.1 quoted a Spearman correlation between the partial
and the disjoint control without saying how many rows it was computed on, and no
current row set reproduces the quoted value. This script fixes the set -- the rows
that carry both controls after the power and complement preconditions -- and
writes the correlation over exactly those rows, so the number in the text has a
reproducible denominator.
"""
from __future__ import annotations

import json
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
AR = ROOT / "analysis_results"
OUT = AR / "control_agreement_spearman.json"


def main() -> None:
    rows = json.loads((AR / "complement_span_proxy_control.json").read_text())["rows"]
    keep = [r for r in rows
            if not r["underpowered"] and not r["complement_degenerate"]
            and r["comp"].get("delta_abs") is not None and r["perp"].get("delta_abs") is not None]
    names = [r["contract"] if r["family"] == "refusal" else f"{r['contract']} ({r['span']}-char)"
             for r in keep]
    ext = [r["perp"]["delta_abs"] for r in keep]
    dis = [r["comp"]["delta_abs"] for r in keep]
    rho, p = spearmanr(ext, dis)
    out = {"row_set": "rows carrying both controls after the power and complement preconditions",
           "n": len(keep), "rows": names, "spearman_rho": float(rho), "p": float(p)}
    print(f"n={len(keep)}  Spearman(Delta_ext, Delta_dis) = {rho:+.3f}  p = {p:.2e}")
    for nm, e, d in zip(names, ext, dis):
        print(f"  {nm:28s} ext {e:+.3f}  dis {d:+.3f}")
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
