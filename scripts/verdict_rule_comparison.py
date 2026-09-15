#!/usr/bin/env python3
"""What changes if Delta_perp decides the verdict? (2026-08-20)

The paper reports Delta_perp but the verdict function does not use it, and a
reviewer objected that this is incoherent: we say most of a reported gap is
geometry, then judge on the gap. The proposed fix is to let Delta_perp decide
wherever z_perp is constructible and fall back to Delta_AUC with a
*geometry-unverified* tag where it is not.

Before adopting that we need to know what it costs, because it is not a free
relabelling: the clean run's raw-prefix contract has Delta_AUC = +0.334 and
Delta_perp = -0.033, so under the new rule it clears -- and that contract is the
baseline against which Section 5.3 measures the repair. This script applies both
rules to every contract we can instantiate the control on and prints the
disagreements, so the decision is made on the list rather than on the one case.

  python3 verdict_rule_comparison.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AR = ROOT / "analysis_results"

MISMATCH_GAP, MISMATCH_KAPPA, CAUTION_GAP = 0.15, 0.20, 0.10

# kappa(z,y) for the contracts where the paper reports one; None where it does not
# enter the comparison (a missing kappa can only block MISMATCH, never create it).
KAPPA = {
    "XSTest 450 (primary run)": 0.024,
    "AdvBench 520": 0.008,
    "SimpleSafety 100": 0.10,
    "XSTest 100 / 512tok": 0.05,
    "AdvBench 100 / 512tok": 0.04,
    "OR-Bench hard 1k": 0.00,
    "JailbreakBench": 0.02,
}


def verdict(gap: float, kappa: float | None) -> str:
    if gap >= MISMATCH_GAP and (kappa is None or kappa <= MISMATCH_KAPPA):
        return "MISMATCH"
    if gap >= CAUTION_GAP:
        return "CAUTION"
    return "ALIGNED"


def main() -> None:
    rows = json.loads((AR / "span_independent_all_contracts.json").read_text())["rows"]

    print(f"{'contract':30s} {'dAUC':>7s} {'dPerp':>7s}  {'current':>9s} -> {'Delta_perp rule':<16s} change")
    changed, same = [], 0
    for r in rows:
        k = KAPPA.get(r["contract"])
        v_now = verdict(r["delta_auc"], k)
        v_new = verdict(r["delta_perp"], k)
        flag = "" if v_now == v_new else "  <-- CHANGES"
        if v_now == v_new:
            same += 1
        else:
            changed.append((r["contract"], v_now, v_new, r["delta_auc"], r["delta_perp"]))
        print(f"{r['contract'][:30]:30s} {r['delta_auc']:+7.3f} {r['delta_perp']:+7.3f}  "
              f"{v_now:>9s} -> {v_new:<16s}{flag}")

    print(f"\n{same}/{len(rows)} verdicts unchanged; {len(changed)} change:")
    for c, a, b, ga, gp in changed:
        print(f"  {c}: {a} -> {b}   ({ga:+.3f} -> {gp:+.3f})")

    print("\nwhat this costs the paper's own narrative:")
    prim = next(r for r in rows if r["contract"].startswith("XSTest 450"))
    print(f"  primary contract stays {verdict(prim['delta_perp'], KAPPA['XSTest 450 (primary run)'])} "
          f"under the new rule (dPerp {prim['delta_perp']:+.3f}) -- the headline is safe.")
    print("  the clean run's raw-prefix baseline is not in this table (its z_perp gives")
    print("  dPerp = -0.033, so it would clear), which is the baseline Section 5.3 repairs.")
    print("  Consequence: under a Delta_perp-decides rule the repair narrative must be")
    print("  restated as improving routed composition on an already-ALIGNED contract, not")
    print("  as fixing a MISMATCH.")

    out = {"rule": {"mismatch_gap": MISMATCH_GAP, "mismatch_kappa": MISMATCH_KAPPA,
                    "caution_gap": CAUTION_GAP},
           "n_rows": len(rows), "n_unchanged": same,
           "changed": [{"contract": c, "from": a, "to": b,
                        "delta_auc": ga, "delta_perp": gp} for c, a, b, ga, gp in changed]}
    path = AR / "verdict_rule_comparison.json"
    path.write_text(json.dumps(out, indent=2))
    print("\n[wrote]", path)


if __name__ == "__main__":
    main()
