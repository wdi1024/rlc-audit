#!/usr/bin/env python3
"""Is the truncation result definitional? (2026-08-20)

The sharpest objection to the span dose-response is that we designed the coupling
rather than found it.  In the truncated contract the proxy z is the gold-answer XOR
*within the scored span*, and the score s is a TF-IDF distance over *the same span*,
so z is a deterministic function of the very text s vectorises.  A high AUC(s, z) at
50 characters is then close to guaranteed, and "the gap closes as the span grows"
reduces to "a prefix proxy shrinks with its prefix".

The control this calls for is a proxy that does not move with the span.  We fix the
proxy at the full output -- z_full is the gold-answer XOR over the complete trace,
identical for every row -- and re-score it with each prefix score s_P.  Now only the
score is truncated, and the two targets z_full and y are both defined outside the
span, so neither can shrink to meet s_P.

  If AUC(s_P, z_full) still exceeds AUC(s_P, y) at short spans, the score prefers a
  surface-derived target over the construct even when the surface target is fixed at
  full span, and the coupling is not an artifact of co-truncation.

  If AUC(s_P, z_full) collapses to AUC(s_P, y), the objection is right about this
  experiment, and the same-span result must be reported as a property of contracts
  whose proxy and score share a span by design -- which is what FrugalGPT-style
  recipes do -- rather than as evidence of a general mechanism.

We report whichever occurs.  The refusal contract gets the same treatment, with
z_full the full-trace refusal-keyword XOR against the prefix-50 score.

  python3 span_independent_proxy_control.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "disagree_routing"
JUDGE = "anthropic_claude-haiku-4-5-20251001"
SPANS = [50, 80, 120, 160, 200, 300, 600, None]
N_BOOT = 1500
DOMAINS = [("phase14_gsm8k_2k", "GSM8K"), ("phase14_hotpot_2k", "HotpotQA")]


def load(phase: str, model: str):
    tr = {r["id"]: r for r in json.loads(
        (DATA / f"{phase}_traces_{model}.json").read_text())["records"]}
    jd = {r["id"]: r for r in json.loads(
        (DATA / f"{phase}_correctness_{model}_{JUDGE}.json").read_text())["records"]}
    return tr, jd


def cosine_distance(a: list[str], b: list[str]) -> np.ndarray:
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit(a + b)
    xa, xb = v.transform(a), v.transform(b)
    num = np.asarray(xa.multiply(xb).sum(1)).ravel()
    na = np.sqrt(np.asarray(xa.multiply(xa).sum(1)).ravel())
    nb = np.sqrt(np.asarray(xb.multiply(xb).sum(1)).ravel())
    return 1 - num / np.maximum(na * nb, 1e-9)


def degenerate(s: np.ndarray, z: np.ndarray) -> bool:
    modal = Counter(s.tolist()).most_common(1)[0][1] / len(s)
    return len(np.unique(s[z == 1])) <= 1 or modal > 0.50


def boot_gap(z, y, s, seed):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(N_BOOT):
        sel = rng.integers(0, len(s), len(s))
        if len(set(z[sel])) < 2 or len(set(y[sel])) < 2:
            continue
        d.append(roc_auc_score(z[sel], s[sel]) - roc_auc_score(y[sel], s[sel]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def boot_auc(lab, s, seed):
    rng = np.random.default_rng(seed)
    a = []
    for _ in range(N_BOOT):
        sel = rng.integers(0, len(s), len(s))
        if len(set(lab[sel])) < 2:
            continue
        a.append(roc_auc_score(lab[sel], s[sel]))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


# --------------------------------------------------------------- refusal ---
def refusal_control() -> dict:
    """The same control on the two refusal runs, prefix-50 score against a
    full-trace keyword proxy."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "agcr", Path(__file__).resolve().parent / "analyze_rlc_composite_router.py")
    agcr = importlib.util.module_from_spec(spec)
    import sys as _sys
    _sys.modules["agcr"] = agcr
    spec.loader.exec_module(agcr)

    out = {}

    def measure(label, ids, a_txt, b_txt, y):
        z_pre = np.array([int(agcr.is_kw(a[:50]) != agcr.is_kw(b[:50]))
                          for a, b in zip(a_txt, b_txt)])
        z_full = np.array([int(agcr.is_kw(a) != agcr.is_kw(b))
                           for a, b in zip(a_txt, b_txt)])
        s = agcr.tfidf_distance(a_txt, b_txt, prefix=50)
        az_p = float(roc_auc_score(z_pre, s))
        az_f = float(roc_auc_score(z_full, s))
        ay = float(roc_auc_score(y, s))
        lo, hi = boot_gap(z_full, y, s, 11)
        out[label] = {"n": len(ids), "y_positives": int(y.sum()),
                      "z_prefix_positives": int(z_pre.sum()),
                      "z_full_positives": int(z_full.sum()),
                      "auc_z_same_span": az_p, "auc_z_full": az_f, "auc_y": ay,
                      "gap_same_span": az_p - ay,
                      "gap_span_independent": az_f - ay,
                      "gap_span_independent_ci": [lo, hi]}
        print(f"  {label:26s} same-span {az_p:.3f} (gap {az_p-ay:+.3f})  "
              f"span-indep {az_f:.3f} (gap {az_f-ay:+.3f} [{lo:+.3f},{hi:+.3f}])  "
              f"y {ay:.3f}  z_full+={int(z_full.sum())}", flush=True)

    # primary run: untagged raw traces, plain refusal judge
    def plain_judge(phase, model):
        rows = json.loads((DATA / f"{phase}_judge_{model}_{JUDGE}.json").read_text())["records"]
        return {r["id"]: int(bool(r["refusal_judge"])) for r in rows}

    ta = {r["id"]: r for r in json.loads(
        (DATA / "phase3_xstest_full_traces_qwen3.5-2b.json").read_text())["records"]}
    tb = {r["id"]: r for r in json.loads(
        (DATA / "phase3_xstest_full_traces_gemma-4-e2b.json").read_text())["records"]}
    ja = plain_judge("phase3_xstest_full", "qwen3.5-2b")
    jb = plain_judge("phase3_xstest_full", "gemma-4-e2b")
    ids = sorted(set(ta) & set(tb) & set(ja) & set(jb))
    measure("primary run (phase-3)", ids,
            [ta[i].get("trace") or "" for i in ids],
            [tb[i].get("trace") or "" for i in ids],
            np.array([int(ja[i] != jb[i]) for i in ids]))

    # clean run: tagged generation, final-channel judge
    s0 = agcr.PAIR_SPECS[0]
    ra = agcr.load_records(DATA, s0.phase_a, s0.model_a)
    rb = agcr.load_records(DATA, s0.phase_b, s0.model_b)
    ja2 = agcr.load_final_judge(DATA, s0.phase_a, s0.model_a, s0.judge_tag)
    jb2 = agcr.load_final_judge(DATA, s0.phase_b, s0.model_b, s0.judge_tag)
    pa = agcr.load_prompts(DATA, s0.phase_a)
    pb = agcr.load_prompts(DATA, s0.phase_b)
    ids2 = sorted(set(ra) & set(rb) & set(ja2) & set(jb2) & set(pa) & set(pb))
    ids2 = [i for i in ids2 if pa[i] == pb[i]
            and agcr.final_text(ra[i]).strip() and agcr.final_text(rb[i]).strip()]
    measure("clean run (phase-13)", ids2,
            [agcr.raw_text(ra[i]) for i in ids2],
            [agcr.raw_text(rb[i]) for i in ids2],
            np.array([int(ja2[i] != jb2[i]) for i in ids2]))
    return out


def main() -> None:
    out = {}
    for phase, label in DOMAINS:
        ta, ja = load(phase, "qwen3.5-2b")
        tb, jb = load(phase, "gemma-4-e2b")
        ids = sorted(set(ta) & set(tb) & set(ja) & set(jb))
        a_txt = [ta[i].get("trace") or "" for i in ids]
        b_txt = [tb[i].get("trace") or "" for i in ids]
        gold = {i: (ja[i].get("gold_answer") or "").strip().lower() for i in ids}
        y = np.array([int(bool(ja[i].get("correct_judge")) != bool(jb[i].get("correct_judge")))
                      for i in ids])
        # the span-independent proxy: gold-answer XOR over the COMPLETE trace
        z_full = np.array([int((gold[i] in a.lower()) != (gold[i] in b.lower()))
                           for i, a, b in zip(ids, a_txt, b_txt)])

        print(f"\n=== {label}  n={len(ids)}  y+={int(y.sum())}  "
              f"z_full+={int(z_full.sum())}", flush=True)
        print(f"  {'span':>5} {'AUC(s,z_span)':>13} {'AUC(s,z_full)':>13} {'AUC(s,y)':>9} "
              f"{'gap_span':>9} {'gap_full':>9}  {'gap_full 95% CI':>20}", flush=True)

        rows = []
        for span in SPANS:
            pa = [t[:span] if span else t for t in a_txt]
            pb = [t[:span] if span else t for t in b_txt]
            z_span = np.array([int((gold[i] in a.lower()) != (gold[i] in b.lower()))
                               for i, a, b in zip(ids, pa, pb)])
            name = str(span) if span else "full"
            row = {"span": name, "span_chars": span,
                   "z_span_positives": int(z_span.sum()),
                   "z_full_positives": int(z_full.sum())}
            if len(set(z_span)) < 2:
                row["note"] = "same-span proxy has no variation"
            s = cosine_distance(pa, pb)
            if len(set(z_span)) >= 2 and degenerate(s, z_span):
                row["note"] = "score degenerate at this span"
                rows.append(row)
                print(f"  {name:>5} {'--':>13} {'--':>13} {'--':>9}  score degenerate", flush=True)
                continue
            a_span = float(roc_auc_score(z_span, s)) if len(set(z_span)) >= 2 else float("nan")
            a_full = float(roc_auc_score(z_full, s))
            a_y = float(roc_auc_score(y, s))
            lo, hi = boot_gap(z_full, y, s, abs(hash(phase + name)) % (2**31))
            row.update(auc_z_span=a_span, auc_z_full=a_full, auc_y=a_y,
                       gap_same_span=a_span - a_y, gap_span_independent=a_full - a_y,
                       gap_span_independent_ci=[lo, hi])
            rows.append(row)
            star = " *" if lo > 0 else ""
            print(f"  {name:>5} {a_span:13.3f} {a_full:13.3f} {a_y:9.3f} "
                  f"{a_span - a_y:+9.3f} {a_full - a_y:+9.3f}  [{lo:+.3f},{hi:+.3f}]{star}",
                  flush=True)
        out[label] = {"n": len(ids), "construct_positives": int(y.sum()),
                      "z_full_positives": int(z_full.sum()), "rows": rows}

    print("\n=== refusal contracts ===", flush=True)
    out["refusal"] = refusal_control()

    path = ROOT / "analysis_results" / "span_independent_proxy_control.json"
    path.write_text(json.dumps(out, indent=2))
    print("\n[wrote]", path, flush=True)


if __name__ == "__main__":
    main()
