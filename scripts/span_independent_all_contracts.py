#!/usr/bin/env python3
"""How much of each certificate was guaranteed? All contracts. (2026-08-20)

`span_independent_proxy_control.py` establishes the control on four contracts.
Four is too few to support the claim the paper makes from it -- that
Delta_AUC and Delta_perp order contracts differently, so the reported number
cannot be read alone -- so this script runs the same control over every refusal
setting and every correctness span the paper audits.

For each contract the score is fixed at the prefix the contract scores, the
construct is fixed at the adjudicated label, and only the *evaluation span of the
proxy rule* changes:

    z          the proxy as the contract defines it, on the span the score reads
    z_perp     the identical rule applied to the complete output

Delta_AUC = AUC(s,z) - AUC(s,y) is what a standard report shows.
Delta_perp = AUC(s,z_perp) - AUC(s,y) is what survives when the proxy can no
longer see the scored span.  The difference between them is the share of the
certificate that the contract's geometry supplied for free.

Rank correlation between the two columns is reported at the end, because that is
the quantity the paper's claim rests on: if they agreed, Delta_AUC would already
tell a reader what they need.

  python3 span_independent_all_contracts.py
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "results" / "disagree_routing"
JUDGE = "anthropic_claude-haiku-4-5-20251001"
N_BOOT = 2000

spec = importlib.util.spec_from_file_location(
    "agcr", HERE / "analyze_rlc_composite_router.py")
agcr = importlib.util.module_from_spec(spec)
sys.modules["agcr"] = agcr
spec.loader.exec_module(agcr)

# refusal settings that store a plain per-side refusal judgement
REFUSAL_SETTINGS = [
    ("phase3_xstest_full", "XSTest 450 (primary run)"),
    ("phase4_advbench", "AdvBench 520"),
    ("phase4_simplesafety", "SimpleSafety 100"),
    ("phase5_xstest100_512tok", "XSTest 100 / 512tok"),
    ("phase5_advbench100_512tok", "AdvBench 100 / 512tok"),
    ("phase8_orbench_hard1k", "OR-Bench hard 1k"),
    ("phase15_jailbreakbench", "JailbreakBench"),
]
CORRECTNESS = [("phase14_gsm8k_2k", "GSM8K"), ("phase14_hotpot_2k", "HotpotQA")]
CORRECTNESS_SPANS = [80, 120, 160, 200, 300, 600]


def boot_ci(z, y, s, seed):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(N_BOOT):
        sel = rng.integers(0, len(s), len(s))
        if len(set(z[sel])) < 2 or len(set(y[sel])) < 2:
            continue
        d.append(roc_auc_score(z[sel], s[sel]) - roc_auc_score(y[sel], s[sel]))
    if not d:
        return None, None
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def row(label, family, s, z, z_perp, y, seed):
    if len(set(z)) < 2 or len(set(z_perp)) < 2 or len(set(y)) < 2:
        return None
    a_z = float(roc_auc_score(z, s))
    a_p = float(roc_auc_score(z_perp, s))
    a_y = float(roc_auc_score(y, s))
    lo, hi = boot_ci(z_perp, y, s, seed)
    return {"contract": label, "family": family, "n": int(len(s)),
            "z_positives": int(z.sum()), "z_perp_positives": int(z_perp.sum()),
            "y_positives": int(y.sum()),
            "auc_z": a_z, "auc_z_perp": a_p, "auc_y": a_y,
            "delta_auc": a_z - a_y, "delta_perp": a_p - a_y,
            "delta_perp_ci": [lo, hi],
            "guaranteed_share": (1.0 - (a_p - a_y) / (a_z - a_y)) if (a_z - a_y) > 0 else None}


def refusal_rows() -> list[dict]:
    out = []
    for phase, label in REFUSAL_SETTINGS:
        try:
            def judge(model):
                rows = json.loads(
                    (DATA / f"{phase}_judge_{model}_{JUDGE}.json").read_text())["records"]
                return {r["id"]: int(bool(r["refusal_judge"])) for r in rows}

            ta = {r["id"]: r for r in json.loads(
                (DATA / f"{phase}_traces_qwen3.5-2b.json").read_text())["records"]}
            tb = {r["id"]: r for r in json.loads(
                (DATA / f"{phase}_traces_gemma-4-e2b.json").read_text())["records"]}
            ja, jb = judge("qwen3.5-2b"), judge("gemma-4-e2b")
        except (FileNotFoundError, KeyError) as exc:
            print(f"  {label}: skipped ({type(exc).__name__})", flush=True)
            continue
        ids = sorted(set(ta) & set(tb) & set(ja) & set(jb))
        a_txt = [ta[i].get("trace") or "" for i in ids]
        b_txt = [tb[i].get("trace") or "" for i in ids]
        y = np.array([int(ja[i] != jb[i]) for i in ids])
        z = np.array([int(agcr.is_kw(a[:50]) != agcr.is_kw(b[:50]))
                      for a, b in zip(a_txt, b_txt)])
        z_perp = np.array([int(agcr.is_kw(a) != agcr.is_kw(b))
                           for a, b in zip(a_txt, b_txt)])
        s = agcr.tfidf_distance(a_txt, b_txt, prefix=50)
        r = row(label, "refusal", s, z, z_perp, y, abs(hash(phase)) % (2**31))
        if r:
            out.append(r)
            print(f"  {label:28s} dAUC {r['delta_auc']:+.3f}  dPerp {r['delta_perp']:+.3f} "
                  f"[{r['delta_perp_ci'][0]:+.3f},{r['delta_perp_ci'][1]:+.3f}]", flush=True)
        else:
            print(f"  {label:28s} not instantiable (a label has one class)", flush=True)
    return out


def correctness_rows() -> list[dict]:
    def cos(a, b):
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit(a + b)
        xa, xb = v.transform(a), v.transform(b)
        num = np.asarray(xa.multiply(xb).sum(1)).ravel()
        na = np.sqrt(np.asarray(xa.multiply(xa).sum(1)).ravel())
        nb = np.sqrt(np.asarray(xb.multiply(xb).sum(1)).ravel())
        return 1 - num / np.maximum(na * nb, 1e-9)

    out = []
    for phase, dom in CORRECTNESS:
        ta = {r["id"]: r for r in json.loads(
            (DATA / f"{phase}_traces_qwen3.5-2b.json").read_text())["records"]}
        tb = {r["id"]: r for r in json.loads(
            (DATA / f"{phase}_traces_gemma-4-e2b.json").read_text())["records"]}
        ja = {r["id"]: r for r in json.loads(
            (DATA / f"{phase}_correctness_qwen3.5-2b_{JUDGE}.json").read_text())["records"]}
        jb = {r["id"]: r for r in json.loads(
            (DATA / f"{phase}_correctness_gemma-4-e2b_{JUDGE}.json").read_text())["records"]}
        ids = sorted(set(ta) & set(tb) & set(ja) & set(jb))
        a_txt = [ta[i].get("trace") or "" for i in ids]
        b_txt = [tb[i].get("trace") or "" for i in ids]
        gold = {i: (ja[i].get("gold_answer") or "").strip().lower() for i in ids}
        y = np.array([int(bool(ja[i].get("correct_judge")) != bool(jb[i].get("correct_judge")))
                      for i in ids])
        z_perp = np.array([int((gold[i] in a.lower()) != (gold[i] in b.lower()))
                           for i, a, b in zip(ids, a_txt, b_txt)])
        for span in CORRECTNESS_SPANS:
            pa = [t[:span] for t in a_txt]
            pb = [t[:span] for t in b_txt]
            z = np.array([int((gold[i] in a.lower()) != (gold[i] in b.lower()))
                          for i, a, b in zip(ids, pa, pb)])
            s = cos(pa, pb)
            r = row(f"{dom}, {span}-char span", "correctness", s, z, z_perp, y,
                    abs(hash(phase + str(span))) % (2**31))
            if r:
                out.append(r)
                print(f"  {r['contract']:28s} dAUC {r['delta_auc']:+.3f}  "
                      f"dPerp {r['delta_perp']:+.3f} "
                      f"[{r['delta_perp_ci'][0]:+.3f},{r['delta_perp_ci'][1]:+.3f}]", flush=True)
    return out


def main() -> None:
    print("=== refusal settings (prefix-50 score, keyword proxy) ===", flush=True)
    rows = refusal_rows()
    print("\n=== correctness contracts (prefix score, gold-answer proxy) ===", flush=True)
    rows += correctness_rows()

    d_auc = np.array([r["delta_auc"] for r in rows])
    d_perp = np.array([r["delta_perp"] for r in rows])
    from scipy.stats import spearmanr, pearsonr
    rho, p_rho = spearmanr(d_auc, d_perp)
    rp, p_rp = pearsonr(d_auc, d_perp)

    print(f"\n=== {len(rows)} contracts ===", flush=True)
    pos = [r for r in rows if r["delta_auc"] > 0]
    survive = [r for r in pos if r["delta_perp_ci"][0] > 0]
    print(f"contracts with a positive reported gap: {len(pos)}", flush=True)
    print(f"  of which the gap survives the control: {len(survive)} "
          f"({', '.join(r['contract'] for r in survive)})", flush=True)
    shares = [r["guaranteed_share"] for r in pos if r["guaranteed_share"] is not None]
    print(f"guaranteed share of the reported gap: median {np.median(shares):.2f}, "
          f"range {min(shares):.2f}--{max(shares):.2f}", flush=True)
    print(f"Spearman(dAUC, dPerp) = {rho:+.3f} (p={p_rho:.3f}); "
          f"Pearson = {rp:+.3f} (p={p_rp:.3f})", flush=True)

    out = {"n_contracts": len(rows), "spearman": float(rho), "spearman_p": float(p_rho),
           "pearson": float(rp), "pearson_p": float(p_rp),
           "n_positive_gap": len(pos), "n_surviving": len(survive),
           "guaranteed_share_median": float(np.median(shares)),
           "rows": rows}
    path = ROOT / "analysis_results" / "span_independent_all_contracts.json"
    path.write_text(json.dumps(out, indent=2))
    print("\n[wrote]", path, flush=True)


if __name__ == "__main__":
    main()
