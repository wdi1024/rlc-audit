#!/usr/bin/env python3
"""Verdicts with and without the post-hoc preconditions, and the off-span
proxy's ranking interval on the at-chance rows.

Appendix F says the counts can be recomputed without the three preconditions
added after the results were seen. This script does that recount from the
per-contract vectors, re-running the rule of Section 3 with each precondition
switched off, and it bootstraps AUC(s, z^c) on the rows whose construct arm is
at chance so the 'off-span proxy still ranked' threshold can be read against an
interval rather than a point. Run from the repository root.
"""
import importlib.util, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
cspc = _load("complement_span_proxy_control")
nulls = _load("disjoint_null_distribution")
OUT = ROOT / "analysis_results" / "recount_without_preconditions.json"
BAND, MISMATCH, CAUTION, KAPPA, RANKED = (0.45, 0.55), 0.15, 0.10, 0.40, 0.10

VEC = {}
_orig = cspc.make_row
def _cap(label, family, span, s, z, z_perp, z_comp, y, empty, seed):
    VEC[label] = dict(s=np.asarray(s, float), z=np.asarray(z, int), zc=np.asarray(z_comp, int), y=np.asarray(y, int), empty=empty, seed=seed)
    return _orig(label, family, span, s, z, z_perp, z_comp, y, empty, seed)
cspc.make_row = _cap


def boot(s, lab, seed, n=4000, q=(2.5, 97.5)):
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        sel = rng.integers(0, len(s), len(s))
        if len(set(lab[sel])) < 2: continue
        v.append(roc_auc_score(lab[sel], s[sel]))
    return [float(np.percentile(v, q[0])), float(np.percentile(v, q[1]))]


def verdict(v, row, use_power=True, use_complement=True, use_resolution=True):
    s, z, zc, y = v["s"], v["z"], v["zc"], v["y"]
    if use_power and row["underpowered"]: return "UNDECIDABLE (power)"
    if use_complement and row["complement_degenerate"]: return "UNDECIDABLE (complement)"
    vals, cnt = np.unique(np.round(s, 6), return_counts=True)
    if use_resolution and cnt.max() / len(s) > 0.5: return "UNDECIDABLE (resolution)"
    if len(set(zc)) < 2 or len(set(y)) < 2: return "UNDECIDABLE (degenerate arm)"
    ci90 = boot(s, y, v["seed"], q=(5, 95)); ci95 = boot(s, y, v["seed"])
    a_zc = roc_auc_score(zc, s)
    if BAND[0] <= ci90[0] and ci90[1] <= BAND[1]:
        return "SCORE FAILURE" if abs(a_zc - 0.5) >= RANKED else "CONTAINMENT"
    if ci95[0] <= 0.5 <= ci95[1]: return "UNDECIDABLE (equivalence)"
    kap = row["comp"]["kappa_vs_y"]
    if kap is not None and kap >= KAPPA: return "ALIGNED"
    null, obs = nulls.stratified_null(s, zc, y, v["seed"])
    if obs <= np.percentile(null, 95): return "CONTAINMENT" if row["delta_abs"] >= MISMATCH else "ALIGNED"
    return "MISMATCH" if obs >= MISMATCH else ("CAUTION" if obs >= CAUTION else "ALIGNED")


def main():
    rows = {r["contract"]: r for r in cspc.refusal_rows() + cspc.correctness_rows()}
    configs = {"all rules": {}, "no power": {"use_power": False}, "no complement": {"use_complement": False},
               "no resolution": {"use_resolution": False}, "none of the three": {"use_power": False, "use_complement": False, "use_resolution": False}}
    table = {}
    for name, kw in configs.items():
        table[name] = {c: verdict(VEC[c], rows[c], **kw) for c in rows}
    # referee check: the reported tally under all rules
    tally = {}
    for vv in table["all rules"].values():
        k = vv.split(" (")[0]; tally[k] = tally.get(k, 0) + 1
    print("all rules:", tally)
    assert tally.get("CAUTION") == 1 and tally.get("SCORE FAILURE") == 1 and tally.get("ALIGNED") == 5 and tally.get("UNDECIDABLE") == 13, tally
    for name in configs:
        t = {}
        for vv in table[name].values():
            k = vv.split(" (")[0]; t[k] = t.get(k, 0) + 1
        print(f"{name:18s} {t}")
    changed = {c: {n: table[n][c] for n in configs if table[n][c] != table["all rules"][c]} for c in rows}
    changed = {c: d for c, d in changed.items() if d}
    print("rows that change:", json.dumps(changed, indent=1))
    ranked = {}
    for c in ("HotpotQA, 50-char span", "OR-Bench hard 1k", "JailbreakBench", "HotpotQA, 80-char span"):
        v = VEC[c]; ranked[c] = dict(auc_zc=float(roc_auc_score(v["zc"], v["s"])), ci95=boot(v["s"], v["zc"], v["seed"]))
        print(f"AUC(s,z^c) {c:24s} {ranked[c]['auc_zc']:.3f} {ranked[c]['ci95']}")
    OUT.write_text(json.dumps({"verdicts": table, "changed": changed, "ranked_interval": ranked}, indent=1)); print("wrote", OUT)


if __name__ == "__main__":
    main()
