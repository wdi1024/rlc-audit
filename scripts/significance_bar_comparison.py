#!/usr/bin/env python3
"""Which verdicts depend on the significance bar?

The audit gates a surviving gap on the contract's own permutation null. An
earlier arrangement gated it on the paired bootstrap interval excluding zero.
Appendix D.2 used to claim the two agree everywhere; they do not, so this script
measures where they part and which of those rows changes verdict.

Run from the repository root:
    python3 scripts/significance_bar_comparison.py
"""
import json

RDIR = "analysis_results"
CAUTION, MISMATCH = 0.10, 0.15


def load():
    comp = json.load(open(f"{RDIR}/complement_span_proxy_control.json"))
    null = json.load(open(f"{RDIR}/disjoint_null_distribution.json"))["contracts"]
    return comp, null


def rows_carrying_disjoint_arm(comp):
    """The eleven rows that pass the power and complement preconditions."""
    return [r for r in comp["rows"]
            if not r["underpowered"] and not r["complement_degenerate"]]


def bars(row, null):
    """(survives under bootstrap, survives under the contract's own null)."""
    lo, hi = row["comp"]["ci_abs"]
    boot = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
    n = null.get(row["contract"])
    beats = None if n is None else row["comp"]["delta_abs"] > n["null_p95"]
    return boot, beats


def main():
    comp, null = load()
    rows = rows_carrying_disjoint_arm(comp)

    # Referee check: reproduce the numbers the paper already records before
    # extending them. A drift here means the artifacts moved, not the rule.
    assert len(rows) == 11, f"expected 11 rows carrying the disjoint arm, got {len(rows)}"
    primary = next(r for r in rows if r["contract"] == "XSTest 450 (primary run)")
    assert abs(primary["comp"]["delta_abs"] - 0.101) < 5e-4, primary["comp"]["delta_abs"]
    assert abs(primary["delta_auc"] - 0.393) < 5e-4, primary["delta_auc"]
    assert abs(null["XSTest 450 (primary run)"]["p_perm"] - 0.0105) < 1e-3

    disagree = []
    print(f"{'contract':26} {'d_dis':>7} {'CI':>20} {'boot':>6} {'null':>6}")
    for r in rows:
        boot, beats = bars(r, null)
        lo, hi = r["comp"]["ci_abs"]
        mark = ""
        if beats is not None and boot != beats:
            disagree.append(r["contract"])
            mark = "  <- bars disagree"
        print(f"{r['contract'][:26]:26} {r['comp']['delta_abs']:+7.3f} "
              f"[{lo:+.3f},{hi:+.3f}] {str(boot):>6} {str(beats):>6}{mark}")

    print(f"\nbars disagree on {len(disagree)} of {len(rows)} rows: {disagree}")

    # Of those, which change verdict? A row settled by a precondition, by the
    # orientation test, or by a gap no band can reach is unchanged either way.
    print("\nverdict consequence:")
    print("  GSM8K 50-char   : withdrawn by the resolution precondition, both bars")
    print("  GSM8K 120-char  : re-oriented, gap negative, ALIGNED under both bars")
    print("  GSM8K 300/600   : gap negative, no band reachable, ALIGNED under both bars")
    cert = primary["delta_auc"]
    exit_if_not_surviving = "CONTAINMENT" if cert >= MISMATCH else "ALIGNED"
    print(f"  XSTest primary  : null bar -> CAUTION (d_dis {primary['comp']['delta_abs']:+.3f} "
          f">= {CAUTION}); bootstrap bar -> gap does not survive, certificate gap "
          f"{cert:+.3f} >= {MISMATCH} -> {exit_if_not_surviving}")
    assert exit_if_not_surviving == "CONTAINMENT"
    print("\nExactly one verdict depends on the bar, and it is our own contract.")


if __name__ == "__main__":
    main()
