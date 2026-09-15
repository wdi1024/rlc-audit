#!/usr/bin/env python3
"""Does the score have enough resolution for AUC to mean anything? (2026-08-20)

A reviewer flagged that the GSM8K prefix-50 gap CI in the span dose-response table
is impossibly narrow for eight proxy positives: [+0.335, +0.407], width 0.072, next
to a width of 0.315 on the same table's twelve-positive row.  The resampling code is
correct.  The score is not.

At 50 characters both models emit a fixed opening -- Qwen has five distinct 50-char
prefixes across 1,319 GSM8K items (842 of them identical), Gemma has six (1,309
identical) -- so the TF-IDF distance between them takes only twelve values on the
whole run, and all eight proxy positives land on a single one.  An AUC computed
there is not a ranking statistic: with every positive tied at one value v, AUC is
fixed at P(neg < v) + 0.5*P(neg = v) regardless of which positives the bootstrap
draws, which is exactly why its interval collapses.

That is a precondition failure, not a finding, and it belongs in the verdict
function alongside the other not-instantiable branches.  This script defines the
check and runs it over every contract the paper audits, so the paper can report
which rows pass it rather than asserting that they do.

Two quantities per contract:

  modal_mass          fraction of items sharing the single most common score value
  distinct_at_pos     number of distinct score values among the proxy positives

A contract is SCORE-DEGENERATE when distinct_at_pos == 1 (the score cannot order the
positives at all) or when modal_mass exceeds 0.5 (a majority of items are unranked).
Both are properties of the score and the span alone; neither consults the construct
label, so applying the check cannot bias the verdict it precedes.

  python3 audit_score_resolution.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "results" / "disagree_routing"
JUDGE = "anthropic_claude-haiku-4-5-20251001"

MODAL_MASS_LIMIT = 0.50
SPANS = [50, 80, 120, 160, 200, 250, 300, 400, 500, 600, 800, 1000, 1400, None]

spec = importlib.util.spec_from_file_location(
    "agcr", HERE / "analyze_rlc_composite_router.py")
agcr = importlib.util.module_from_spec(spec)
sys.modules["agcr"] = agcr
spec.loader.exec_module(agcr)


def resolution(score: np.ndarray, z: np.ndarray) -> dict:
    counts = Counter(score.tolist())
    modal_mass = counts.most_common(1)[0][1] / len(score)
    distinct_at_pos = int(len(np.unique(score[z == 1]))) if z.sum() else 0
    degenerate = distinct_at_pos <= 1 or modal_mass > MODAL_MASS_LIMIT
    return {
        "n": int(len(score)),
        "distinct_scores": int(len(np.unique(score))),
        "modal_mass": float(modal_mass),
        "proxy_positives": int(z.sum()),
        "distinct_scores_at_positives": distinct_at_pos,
        "score_degenerate": bool(degenerate),
    }


def cosine_distance(a: list[str], b: list[str]) -> np.ndarray:
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit(a + b)
    xa, xb = v.transform(a), v.transform(b)
    num = np.asarray(xa.multiply(xb).sum(1)).ravel()
    na = np.sqrt(np.asarray(xa.multiply(xa).sum(1)).ravel())
    nb = np.sqrt(np.asarray(xb.multiply(xb).sum(1)).ravel())
    return 1 - num / np.maximum(na * nb, 1e-9)


def correctness_contracts() -> dict:
    """The span dose-response rows: GSM8K and HotpotQA, score and proxy on one span."""
    out = {}
    for phase, label in (("phase14_gsm8k_2k", "GSM8K"), ("phase14_hotpot_2k", "HotpotQA")):
        def load(model):
            tr = {r["id"]: r for r in json.loads(
                (DATA / f"{phase}_traces_{model}.json").read_text())["records"]}
            jd = {r["id"]: r for r in json.loads(
                (DATA / f"{phase}_correctness_{model}_{JUDGE}.json").read_text())["records"]}
            return tr, jd

        ta, ja = load("qwen3.5-2b")
        tb, _ = load("gemma-4-e2b")
        ids = sorted(set(ta) & set(tb) & set(ja))
        a_txt = [ta[i].get("trace") or "" for i in ids]
        b_txt = [tb[i].get("trace") or "" for i in ids]
        gold = {i: (ja[i].get("gold_answer") or "").strip().lower() for i in ids}

        rows = {}
        for span in SPANS:
            pa = [t[:span] if span else t for t in a_txt]
            pb = [t[:span] if span else t for t in b_txt]
            z = np.array([int((gold[i] in a.lower()) != (gold[i] in b.lower()))
                          for i, a, b in zip(ids, pa, pb)])
            name = f"prefix-{span}" if span else "full trace"
            if len(set(z)) < 2:
                rows[name] = {"proxy_positives": int(z.sum()),
                              "verdict": "not instantiable (no proxy variation)"}
                continue
            r = resolution(cosine_distance(pa, pb), z)
            r["distinct_prefixes_a"] = len(set(pa))
            r["distinct_prefixes_b"] = len(set(pb))
            rows[name] = r
        out[label] = rows
    return out


def refusal_contracts() -> dict:
    """The refusal contracts, at the audited prefix span and over the full trace."""
    out = {}
    for spec_ in agcr.PAIR_SPECS:
        try:
            rec_a = agcr.load_records(DATA, spec_.phase_a, spec_.model_a)
            rec_b = agcr.load_records(DATA, spec_.phase_b, spec_.model_b)
            pa = agcr.load_prompts(DATA, spec_.phase_a)
            pb = agcr.load_prompts(DATA, spec_.phase_b)
        except Exception as exc:                       # cached run absent
            out[f"{spec_.model_a}/{spec_.model_b}"] = {"skipped": str(exc)[:80]}
            continue
        ids = sorted(set(rec_a) & set(rec_b) & set(pa) & set(pb))
        ids = [i for i in ids if pa[i] == pb[i]]
        if not ids:
            continue
        a_txt = [agcr.raw_text(rec_a[i]) for i in ids]
        b_txt = [agcr.raw_text(rec_b[i]) for i in ids]

        rows = {}
        for span in (50, None):
            qa = [t[:span] if span else t for t in a_txt]
            qb = [t[:span] if span else t for t in b_txt]
            z = np.array([int(agcr.is_kw(a) != agcr.is_kw(b)) for a, b in zip(qa, qb)])
            score = (agcr.tfidf_distance(a_txt, b_txt, prefix=span) if span
                     else agcr.tfidf_distance(a_txt, b_txt))
            name = f"prefix-{span}" if span else "full trace"
            if len(set(z)) < 2:
                rows[name] = {"proxy_positives": int(z.sum()),
                              "verdict": "not instantiable (no proxy variation)"}
                continue
            r = resolution(score, z)
            r["distinct_prefixes_a"] = len(set(qa))
            r["distinct_prefixes_b"] = len(set(qb))
            rows[name] = r
        out[f"{spec_.model_a}/{spec_.model_b} ({spec_.phase_a})"] = rows
    return out


def main() -> None:
    result = {"modal_mass_limit": MODAL_MASS_LIMIT,
              "correctness": correctness_contracts(),
              "refusal": refusal_contracts()}

    flagged = []
    for family, contracts in (("correctness", result["correctness"]),
                              ("refusal", result["refusal"])):
        print(f"\n===== {family} contracts")
        for name, rows in contracts.items():
            if "skipped" in rows:
                print(f"  {name}: skipped ({rows['skipped']})")
                continue
            print(f"  {name}")
            for span, r in rows.items():
                if "verdict" in r:
                    print(f"    {span:12s} z+={r['proxy_positives']:4d}  {r['verdict']}")
                    continue
                mark = "  <-- SCORE-DEGENERATE" if r["score_degenerate"] else ""
                if r["score_degenerate"]:
                    flagged.append((family, name, span))
                print(f"    {span:12s} z+={r['proxy_positives']:4d}  "
                      f"distinct={r['distinct_scores']:5d}/{r['n']:<5d} "
                      f"modal={r['modal_mass']:.3f}  "
                      f"distinct@pos={r['distinct_scores_at_positives']:4d}{mark}")

    result["flagged"] = [{"family": f, "contract": c, "span": s} for f, c, s in flagged]
    print(f"\n{len(flagged)} contract-span rows fail the resolution precondition:")
    for f, c, s in flagged:
        print(f"  - [{f}] {c} @ {s}")

    path = ROOT / "analysis_results" / "score_resolution_audit.json"
    path.write_text(json.dumps(result, indent=2))
    print("\n[wrote]", path)


if __name__ == "__main__":
    main()
