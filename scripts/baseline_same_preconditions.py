#!/usr/bin/env python3
"""A baseline that reports AUC(s,y) and kappa(z,y) under the same preconditions,
equivalence test and intervals as the audit, on the twenty controlled contracts.

The baseline has no off-span arm. It flags a contract as 'not evidence' when the
construct arm is shown at chance, and as a 'label' flag when the construct arm
ranks and kappa(z,y) < 0.40; it abstains under the same power, complement and
resolution preconditions. The comparison asks what the disjoint control adds:
the split between SCORE FAILURE and CONTAINMENT, and the reading that survives
the partial control. Run from the repository root.
"""
import importlib.util, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parent.parent
def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
rc = _load("recount_without_preconditions")
OUT = ROOT / "analysis_results" / "baseline_same_preconditions.json"


def kappa(a, b):
    po = float((a == b).mean()); pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


def baseline(v, row):
    s, z, y = v["s"], v["z"], v["y"]
    if row["underpowered"]: return "UNDECIDABLE (power)"
    if row["complement_degenerate"]: return "UNDECIDABLE (complement)"
    vals, cnt = np.unique(np.round(s, 6), return_counts=True)
    if cnt.max() / len(s) > 0.5: return "UNDECIDABLE (resolution)"
    ci90 = rc.boot(s, y, v["seed"], q=(5, 95)); ci95 = rc.boot(s, y, v["seed"])
    if rc.BAND[0] <= ci90[0] and ci90[1] <= rc.BAND[1]: return "FLAG: not evidence"
    if ci95[0] <= 0.5 <= ci95[1]: return "UNDECIDABLE (equivalence)"
    return "FLAG: label" if kappa(z, y) < 0.40 else "NO FLAG"


def main():
    rows = {r["contract"]: r for r in rc.cspc.refusal_rows() + rc.cspc.correctness_rows()}
    ours = json.load(open(ROOT / "analysis_results" / "recount_without_preconditions.json"))["verdicts"]["all rules"]
    out = {}
    print(f"{'contract':26s} {'kappa(z,y)':>10s} {'baseline':>26s}  {'audit':>26s}")
    for c in rows:
        b = baseline(rc.VEC[c], rows[c]); k = kappa(rc.VEC[c]["z"], rc.VEC[c]["y"])
        out[c] = dict(kappa_zy=float(k), baseline=b, audit=ours[c])
        print(f"{c[:26]:26s} {k:10.3f} {b:>26s}  {ours[c]:>26s}")
    flag_b = {c for c, r in out.items() if r["baseline"].startswith("FLAG")}
    flag_a = {c for c, r in out.items() if r["audit"].split(" (")[0] in ("SCORE FAILURE", "CAUTION", "DIVERGENCE", "CONTAINMENT")}
    print("\nbaseline flags:", sorted(flag_b)); print("audit flags   :", sorted(flag_a))
    print("baseline flags but audit does not:", sorted(flag_b - flag_a)); print("audit flags but baseline does not:", sorted(flag_a - flag_b))
    OUT.write_text(json.dumps(out, indent=1)); print("wrote", OUT)


if __name__ == "__main__":
    main()
