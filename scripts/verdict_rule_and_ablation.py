#!/usr/bin/env python3
"""Which control should decide the verdict, and does the machinery earn its keep? (2026-08-21)

Two objections, one script, because they share a computation.

FIRST -- the self-contradiction. Section 3.2 shows the extended proxy is a partial
control: it leaves containment in. The verdict function nonetheless computes from
the extended gap, so the paper is deciding with an instrument it has just shown to
be insufficient, and our own primary contract keeps a MISMATCH label that the
disjoint control does not support. This applies the verdict function under all
three rules and prints every row that moves.

SECOND -- the value-added ablation. A reviewer notes that the audit's verdicts look
like they track kappa(z,y) alone, and asks what the seven fields, two controls and
three preconditions buy over the much cheaper prescription "report AUC(s,y) and
kappa(z,y) beside your proxy AUC". We implement that cheap rule and count the
contracts on which the two disagree. If the answer is none, the honest conclusion
is that the contribution is the control and the vocabulary, not the verdict
function, and the paper should say so.

  python3 verdict_rule_and_ablation.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AR = ROOT / "analysis_results"

MISMATCH_GAP, MISMATCH_KAPPA, CAUTION_GAP = 0.15, 0.20, 0.10

# kappa(z,y) as reported for the contracts that have one
KAPPA = {
    "XSTest 450 (primary run)": 0.024, "AdvBench 520": 0.008,
    "SimpleSafety 100": 0.10, "XSTest 100 / 512tok": 0.05,
    "AdvBench 100 / 512tok": 0.04, "OR-Bench hard 1k": 0.00,
    "JailbreakBench": 0.02,
}


def verdict(gap, kappa, y_ranked: bool, z_ranked: bool) -> str:
    """The paper's verdict function, with the NULL exit made explicit."""
    if not y_ranked and not z_ranked:
        return "NULL"
    if gap is None:
        return "UNDECIDABLE"
    if gap >= MISMATCH_GAP and (kappa is None or kappa <= MISMATCH_KAPPA):
        return "MISMATCH"
    if gap >= CAUTION_GAP:
        return "CAUTION"
    return "ALIGNED"


def cheap_rule(auc_y_ranked: bool, kappa) -> str:
    """The prescription the audit has to beat: look at whether the score ranks the
    construct, and whether the proxy agrees with it. No spans, no controls."""
    if kappa is None:
        return "n/a"
    if not auc_y_ranked and kappa <= MISMATCH_KAPPA:
        return "MISMATCH"
    if kappa <= MISMATCH_KAPPA:
        return "CAUTION"
    return "ALIGNED"


def main() -> None:
    rows = json.loads((AR / "complement_span_proxy_control.json").read_text())["rows"]
    orient = {r["contract"]: r for r in
              json.loads((AR / "orientation_significance.json").read_text())["rows"]}

    print(f"{'contract':26s} {'AUCy ranked':>12s} {'dAUC':>7s} {'dExt':>7s} {'dDis':>7s}"
          f"  {'by dAUC':>10s} {'by dExt':>10s} {'by dDis':>10s}")
    changed, table = [], []
    for r in rows:
        c = r["contract"]
        o = orient.get(c)
        y_ranked = bool(o and o["class"] != "unranked (covers 0.5)")
        k = KAPPA.get(c)
        gaps = {"dAUC": r["delta_abs"],
                "dExt": r["perp"]["delta_abs"],
                "dDis": None if r["complement_degenerate"] else r["comp"]["delta_abs"]}
        z_ranked = r["auc_z"] > 0.5
        v = {name: verdict(g, k, y_ranked, z_ranked) for name, g in gaps.items()}
        usable = not r["underpowered"]
        table.append({"contract": c, "usable": usable, "y_ranked": y_ranked,
                      "gaps": gaps, "verdicts": v})
        if usable and v["dExt"] != v["dDis"]:
            changed.append((c, v["dExt"], v["dDis"]))
        g = lambda x: f"{x:+7.3f}" if x is not None else "     --"
        print(f"{c[:26]:26s} {str(y_ranked):>12s} {g(gaps['dAUC'])} {g(gaps['dExt'])} "
              f"{g(gaps['dDis'])}  {v['dAUC']:>10s} {v['dExt']:>10s} {v['dDis']:>10s}"
              + ("" if usable else "   [underpowered]"))

    print(f"\n=== switching the verdict from the partial to the disjoint control ===")
    if changed:
        for c, a, b in changed:
            print(f"  {c}: {a} -> {b}")
    else:
        print("  no usable contract changes verdict.")

    prim = next(t for t in table if t["contract"].startswith("XSTest 450"))
    print(f"\nprimary contract: dAUC {prim['gaps']['dAUC']:+.3f} -> "
          f"{prim['verdicts']['dAUC']};  dExt {prim['gaps']['dExt']:+.3f} -> "
          f"{prim['verdicts']['dExt']};  dDis {prim['gaps']['dDis']:+.3f} -> "
          f"{prim['verdicts']['dDis']}")

    print("\n=== ablation: does the machinery beat 'report AUC(s,y) and kappa'? ===")
    print(f"{'contract':26s} {'full (dDis)':>12s} {'cheap rule':>12s}  agree?")
    agree = dis = 0
    for t in table:
        k = KAPPA.get(t["contract"])
        if k is None or not t["usable"]:
            continue
        cheap = cheap_rule(t["y_ranked"], k)
        full = t["verdicts"]["dDis"]
        same = (cheap == full)
        agree += same; dis += (not same)
        print(f"{t['contract'][:26]:26s} {full:>12s} {cheap:>12s}  {'yes' if same else 'NO'}")
    print(f"\n  agree on {agree}, disagree on {dis}")
    if dis == 0:
        print("  On these contracts the cheap prescription reproduces every verdict. The")
        print("  contract's value is then the control and the vocabulary, not the verdict")
        print("  function, and the paper must say so rather than imply otherwise.")

    out = {"cutoffs": {"mismatch": MISMATCH_GAP, "caution": CAUTION_GAP,
                       "mismatch_kappa": MISMATCH_KAPPA},
           "rows": table, "changed_under_disjoint": changed,
           "ablation_agree": agree, "ablation_disagree": dis}
    p = AR / "verdict_rule_and_ablation.json"
    p.write_text(json.dumps(out, indent=2))
    print("\n[wrote]", p)


if __name__ == "__main__":
    main()
