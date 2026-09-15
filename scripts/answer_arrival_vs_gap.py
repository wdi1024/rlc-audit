#!/usr/bin/env python3
"""Where does the construct's evidence arrive, and is that where the gap closes?

A reviewer's central objection to the span result is that it risks being a
restatement of "short prefixes carry less information".  The paper's defence has
been that the span at which the gap closes tracks *where the answer appears*, and
that this differs by domain in a predicted direction -- but the paper asserts that
ordering without ever measuring the answer's position.  This script measures it.

Two quantities, both computed per domain:

  arrival     the character offset at which the gold answer string first appears in
              a trace, over traces that contain it at all
  coverage    C(P), the fraction of items whose gold answer has appeared by offset P

The tautology charge is answered by re-parameterising.  Plotted against span P the
two domains give two different curves, which says only that they have different
length scales.  Plotted against C(P) -- the fraction of items whose evidence has
actually arrived -- they should collapse onto one curve, because the claim is that
the gap closes when the evidence arrives, not when the span gets long.  That
collapse is a prediction that could fail, and it is not implied by "longer spans
carry more information".

Substring matching follows the proxy's own rule so the two are commensurable; a
word-boundary variant is reported alongside it, because a bare GSM8K numeral can
match inside a restated problem or an intermediate result.

  python3 answer_arrival_vs_gap.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "disagree_routing"
JUDGE = "anthropic_claude-haiku-4-5-20251001"
SPANS = [50, 120, 200, 300, 600, None]
N_BOOT = 2000

DOMAINS = [
    ("phase14_gsm8k_2k", "GSM8K"),
    ("phase14_hotpot_2k", "HotpotQA"),
    ("phase11_gsm8k", "GSM8K (replication)"),
    ("phase10_hotpot", "HotpotQA (replication)"),
]


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


def first_offset(text: str, gold: str, word_boundary: bool) -> int | None:
    """Character offset of the gold answer's first appearance, or None."""
    if not gold:
        return None
    hay, needle = text.lower(), gold.lower()
    if word_boundary:
        m = re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", hay)
        return m.start() if m else None
    i = hay.find(needle)
    return i if i >= 0 else None


def boot_gap(z: np.ndarray, y: np.ndarray, s: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(N_BOOT):
        sel = rng.integers(0, len(s), len(s))
        if len(set(z[sel])) < 2 or len(set(y[sel])) < 2:
            continue
        d.append(roc_auc_score(z[sel], s[sel]) - roc_auc_score(y[sel], s[sel]))
    if not d:
        return float("nan"), float("nan")
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def score_is_degenerate(s: np.ndarray, z: np.ndarray) -> bool:
    """The precondition from audit_score_resolution.py, applied inline."""
    from collections import Counter
    modal = Counter(s.tolist()).most_common(1)[0][1] / len(s)
    return len(np.unique(s[z == 1])) <= 1 or modal > 0.50


def analyse(phase: str, label: str) -> dict | None:
    try:
        ta, ja = load(phase, "qwen3.5-2b")
        tb, jb = load(phase, "gemma-4-e2b")
    except FileNotFoundError as exc:
        print(f"  {label}: skipped ({exc.filename})")
        return None

    ids = sorted(set(ta) & set(tb) & set(ja) & set(jb))
    a_txt = [ta[i].get("trace") or "" for i in ids]
    b_txt = [tb[i].get("trace") or "" for i in ids]
    gold = {i: (ja[i].get("gold_answer") or "").strip() for i in ids}
    y = np.array([int(bool(ja[i].get("correct_judge")) != bool(jb[i].get("correct_judge")))
                  for i in ids])

    out: dict = {"phase": phase, "n": len(ids), "construct_positives": int(y.sum())}

    # ---- answer arrival -----------------------------------------------------
    for tag, wb in (("substring", False), ("word_boundary", True)):
        offs = []
        for i, a, b in zip(ids, a_txt, b_txt):
            for t in (a, b):
                o = first_offset(t, gold[i], wb)
                if o is not None:
                    offs.append(o)
        arr = np.array(offs, dtype=float)
        out[f"arrival_{tag}"] = {
            "n_traces_containing": int(len(arr)),
            "n_traces_total": int(2 * len(ids)),
            "median": float(np.median(arr)) if len(arr) else None,
            "q25": float(np.percentile(arr, 25)) if len(arr) else None,
            "q75": float(np.percentile(arr, 75)) if len(arr) else None,
        }

    # ---- coverage and gap, per span ----------------------------------------
    rows = []
    for span in SPANS:
        pa = [t[:span] if span else t for t in a_txt]
        pb = [t[:span] if span else t for t in b_txt]
        name = f"prefix-{span}" if span else "full trace"

        cov = float(np.mean([
            (first_offset(a, gold[i], True) is not None)
            or (first_offset(b, gold[i], True) is not None)
            for i, a, b in zip(ids, pa, pb)]))

        z = np.array([int((gold[i].lower() in a.lower()) != (gold[i].lower() in b.lower()))
                      for i, a, b in zip(ids, pa, pb)])
        row = {"span": name, "span_chars": span, "coverage": cov,
               "proxy_positives": int(z.sum())}
        if len(set(z)) < 2:
            row["verdict"] = "not instantiable (no proxy variation)"
            rows.append(row)
            continue
        s = cosine_distance(pa, pb)
        if score_is_degenerate(s, z):
            row["verdict"] = "not instantiable (score degenerate at this span)"
            rows.append(row)
            continue
        az, ay = float(roc_auc_score(z, s)), float(roc_auc_score(y, s))
        lo, hi = boot_gap(z, y, s, seed=abs(hash(phase + name)) % (2**31))
        row.update(auc_proxy=az, auc_semantic=ay, gap=az - ay, gap_ci=[lo, hi])
        rows.append(row)
    out["rows"] = rows

    # ---- where does the gap close? -----------------------------------------
    usable = [r for r in rows if "gap" in r]
    closes_at_span = None
    closes_at_coverage = None
    for prev, cur in zip(usable, usable[1:]):
        if prev["gap"] > 0 and cur["gap_ci"][0] <= 0:
            closes_at_span = cur["span_chars"]
            closes_at_coverage = cur["coverage"]
            break
    out["gap_closes_at_span"] = closes_at_span
    out["gap_closes_at_coverage"] = closes_at_coverage
    return out


def main() -> None:
    result = {}
    print("=== answer arrival and gap closure ===")
    for phase, label in DOMAINS:
        r = analyse(phase, label)
        if r is None:
            continue
        result[label] = r
        wb = r["arrival_word_boundary"]
        print(f"\n{label}  (n={r['n']}, construct+={r['construct_positives']})")
        print(f"  gold answer present in {wb['n_traces_containing']}/{wb['n_traces_total']} traces; "
              f"first appears at char median {wb['median']:.0f} "
              f"[q25 {wb['q25']:.0f}, q75 {wb['q75']:.0f}]")
        for row in r["rows"]:
            if "gap" not in row:
                print(f"    {row['span']:12s} coverage={row['coverage']:.3f}  "
                      f"z+={row['proxy_positives']:4d}  {row['verdict']}")
                continue
            print(f"    {row['span']:12s} coverage={row['coverage']:.3f}  "
                  f"z+={row['proxy_positives']:4d}  gap={row['gap']:+.3f} "
                  f"[{row['gap_ci'][0]:+.3f},{row['gap_ci'][1]:+.3f}]")
        print(f"  gap closes at span {r['gap_closes_at_span']} "
              f"(coverage {r['gap_closes_at_coverage']:.3f})"
              if r["gap_closes_at_span"] else "  gap does not close in the swept range")

    print("\n=== the re-parameterisation test ===")
    print("If the gap closes when evidence arrives, the closure coverage should agree")
    print("across domains even though the closure span does not.")
    for label, r in result.items():
        if r["gap_closes_at_span"]:
            print(f"  {label:24s} closes at span {r['gap_closes_at_span']:>4} chars, "
                  f"coverage {r['gap_closes_at_coverage']:.3f}, "
                  f"median arrival {r['arrival_word_boundary']['median']:.0f} chars")

    path = ROOT / "analysis_results" / "answer_arrival_vs_gap.json"
    path.write_text(json.dumps(result, indent=2))
    print("\n[wrote]", path)


if __name__ == "__main__":
    main()
