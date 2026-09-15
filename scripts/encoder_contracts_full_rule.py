#!/usr/bin/env python3
"""Encoder-scored prefix-50 contracts through the full verdict rule.

The boundary table reports that MiniLM/e5 prefix encoders reproduce the
surface/semantic split on Delta_AUC, the superseded rule. This script puts the
same instrument through the deciding rule -- disjoint control, per-contract
permutation null, equivalence band, magnitude bands -- so an encoder-scored
contract carries a verdict like the TF-IDF rows. Referee check: the TF-IDF row
of each setting is recomputed first and must match the recorded numbers.

Run from the repository root:
    python3 scripts/encoder_contracts_full_rule.py
"""
import importlib.util, json, os, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
cspc = _load("complement_span_proxy_control")
nulls = _load("disjoint_null_distribution")
agcr = cspc.agcr
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from sentence_transformers import SentenceTransformer

SETTINGS = [("phase3_xstest_full", "XSTest 450 (primary run)", {"auc_z": 0.985, "auc_y": 0.592, "dis": 0.101}),
            ("phase8_orbench_hard1k", "OR-Bench hard 1k", {"auc_z": 0.866, "auc_y": 0.517, "dis": 0.248})]
ENCODERS = [("MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", ""),
            ("e5-small-v2", "intfloat/e5-small-v2", "query: ")]
BAND, MISMATCH, CAUTION, KAPPA = (0.45, 0.55), 0.15, 0.10, 0.40
SP = cspc.REFUSAL_SPAN
OUT = ROOT / "analysis_results" / "encoder_contracts_full_rule.json"


def load_setting(phase):
    def judge(model):
        rows = json.loads((cspc.DATA / f"{phase}_judge_{model}_{cspc.JUDGE}.json").read_text())["records"]
        return {r["id"]: int(bool(r["refusal_judge"])) for r in rows}
    ta = {r["id"]: r for r in json.loads((cspc.DATA / f"{phase}_traces_qwen3.5-2b.json").read_text())["records"]}
    tb = {r["id"]: r for r in json.loads((cspc.DATA / f"{phase}_traces_gemma-4-e2b.json").read_text())["records"]}
    ja, jb = judge("qwen3.5-2b"), judge("gemma-4-e2b")
    ids = sorted(set(ta) & set(tb) & set(ja) & set(jb))
    a = [ta[i].get("trace") or "" for i in ids]; b = [tb[i].get("trace") or "" for i in ids]
    y = np.array([int(ja[i] != jb[i]) for i in ids])
    z = np.array([int(agcr.is_kw(x[:SP]) != agcr.is_kw(w[:SP])) for x, w in zip(a, b)])
    z_perp = np.array([int(agcr.is_kw(x) != agcr.is_kw(w)) for x, w in zip(a, b)])
    z_comp = np.array([int(agcr.is_kw(x[SP:]) != agcr.is_kw(w[SP:])) for x, w in zip(a, b)])
    empty = float(np.mean([len(t) <= SP for t in a + b]))
    return a, b, z, z_perp, z_comp, y, empty


def encoder_distance(enc, a, b, pre):
    ea = enc.encode([pre + t[:SP] for t in a], show_progress_bar=False, normalize_embeddings=True)
    eb = enc.encode([pre + t[:SP] for t in b], show_progress_bar=False, normalize_embeddings=True)
    return 1.0 - (ea * eb).sum(axis=1)


def boot_auc_y(s, y, seed=20260904, n=4000):
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        sel = rng.integers(0, len(s), len(s))
        if len(set(y[sel])) < 2: continue
        v.append(roc_auc_score(y[sel], s[sel]))
    v = np.array(v)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], [float(np.percentile(v, 5)), float(np.percentile(v, 95))]


def verdict(row, s, z_comp, y, seed):
    """The rule of Section 3, in order."""
    out = {}
    vals, counts = np.unique(np.round(s, 6), return_counts=True)
    out["modal_share"] = float(counts.max() / len(s))
    if row["underpowered"]: return {**out, "verdict": "UNDECIDABLE (power)"}
    if row["complement_degenerate"]: return {**out, "verdict": "UNDECIDABLE (complement)"}
    if out["modal_share"] > 0.5: return {**out, "verdict": "UNDECIDABLE (resolution)"}
    ci95, ci90 = boot_auc_y(s, y)
    out["auc_y_ci95"], out["auc_y_ci90"] = ci95, ci90
    reversed_ = ci95[1] < 0.5
    at_chance = BAND[0] <= ci90[0] and ci90[1] <= BAND[1]
    a_zc = row["comp"]["auc"]
    if at_chance:
        out["equivalence"] = "shown"
        return {**out, "verdict": "SCORE FAILURE" if abs(a_zc - 0.5) > 0.05 else "CONTAINMENT"}
    if ci95[0] <= 0.5 <= ci95[1]:
        out["equivalence"] = "interval covers 0.5 but band not shown"
        return {**out, "verdict": "UNDECIDABLE (equivalence)"}
    out["reoriented"] = bool(reversed_)
    kap = row["comp"]["kappa_vs_y"]
    if kap is not None and kap >= KAPPA: return {**out, "verdict": "ALIGNED (proxy merged with construct)"}
    null, obs = nulls.stratified_null(s, z_comp, y, seed)
    p95 = float(np.percentile(null, 95)); p = float((null >= obs).mean())
    out.update(null_p95=p95, p_perm=p, observed_dis=float(obs))
    if obs <= p95:
        return {**out, "verdict": "CONTAINMENT" if row["delta_abs"] >= MISMATCH else "ALIGNED"}
    if obs >= MISMATCH: return {**out, "verdict": "MISMATCH"}
    if obs >= CAUTION: return {**out, "verdict": "CAUTION"}
    return {**out, "verdict": "ALIGNED"}


def main():
    results = []
    for phase, label, ref in SETTINGS:
        a, b, z, z_perp, z_comp, y, empty = load_setting(phase)
        seed = abs(hash(phase)) % (2**31)
        # referee check: the TF-IDF row must reproduce the paper
        s0 = agcr.tfidf_distance(a, b, prefix=SP)
        r0 = cspc.make_row(label, "refusal", SP, s0, z, z_perp, z_comp, y, empty, seed)
        assert abs(r0["auc_z"] - ref["auc_z"]) < 2e-3 and abs(r0["auc_y"] - ref["auc_y"]) < 2e-3 \
            and abs(r0["comp"]["delta_abs"] - ref["dis"]) < 2e-3, (label, r0["auc_z"], r0["auc_y"], r0["comp"]["delta_abs"])
        print(f"[referee] {label}: TF-IDF row reproduced (AUC z {r0['auc_z']:.3f}, y {r0['auc_y']:.3f}, dis {r0['comp']['delta_abs']:+.3f})")
        for enc_label, model, pre in ENCODERS:
            enc = SentenceTransformer(model)
            s = encoder_distance(enc, a, b, pre)
            r = cspc.make_row(f"{label} / {enc_label} prefix-50", "refusal", SP, s, z, z_perp, z_comp, y, empty, seed)
            v = verdict(r, s, z_comp, y, seed)
            row = {k: r[k] for k in ("contract", "n", "y_positives", "z_positives", "z_comp_positives", "auc_z", "auc_y", "delta_auc")}
            row.update(auc_z_perp=r["perp"]["auc"], delta_ext=r["perp"]["delta_abs"], auc_zc=r["comp"]["auc"],
                       delta_dis=r["comp"]["delta_abs"], delta_dis_ci=r["comp"]["ci_abs"], kappa_zc_y=r["comp"]["kappa_vs_y"], **v)
            results.append(row)
            print(f"  {row['contract']:42s} AUC z {row['auc_z']:.3f}  y {row['auc_y']:.3f} {v.get('auc_y_ci95','')}  "
                  f"zc {row['auc_zc']:.3f}  dAUC {row['delta_auc']:+.3f}  dext {row['delta_ext']:+.3f}  "
                  f"ddis {row['delta_dis']:+.3f} [{row['delta_dis_ci'][0]:+.3f},{row['delta_dis_ci'][1]:+.3f}]  "
                  f"null95 {v.get('null_p95', float('nan')):+.3f} p {v.get('p_perm', float('nan')):.4f}  -> {v['verdict']}")
    OUT.write_text(json.dumps({"span": SP, "band": BAND, "rows": results}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
