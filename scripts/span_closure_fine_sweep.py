#!/usr/bin/env python3
"""Localise where the gap closes, and test whether evidence arrival explains it.

The coarse span grid (50/120/200/300/600/full) brackets the closure point too
loosely to compare domains: on HotpotQA the gap is coupled at 50 and clear at 120,
which places closure anywhere in a range whose coverage runs from 0.17 to 0.38.
This script sweeps fourteen spans instead of six so that closure is localised, and
records the answer-arrival coverage at each one.

The test that matters is the re-parameterisation.  Against span, the two domains
must differ -- GSM8K answers arrive around character 1074, HotpotQA around 376, so
of course their curves sit at different scales, and a critic can call that
"longer prefixes carry more information".  Against coverage, the claim is that they
should agree, because the assertion is that the gap closes when the construct's
evidence arrives.  That is falsifiable and is what is reported here.

  python3 span_closure_fine_sweep.py
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
SPANS = [50, 80, 120, 160, 200, 250, 300, 400, 500, 600, 800, 1000, 1400, None]
N_BOOT = 1500
DOMAINS = [("phase14_gsm8k_2k", "GSM8K"), ("phase14_hotpot_2k", "HotpotQA"),
           ("phase11_gsm8k", "GSM8K-rep"), ("phase10_hotpot", "HotpotQA-rep")]


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


def first_offset(text: str, gold: str) -> int | None:
    if not gold:
        return None
    m = re.search(r"(?<!\w)" + re.escape(gold.lower()) + r"(?!\w)", text.lower())
    return m.start() if m else None


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


def interpolate_closure(rows: list[dict]) -> dict:
    """Closure = the first span at which the gap stops being significantly positive.

    Interpolating the point where the gap hits zero only works where it goes on to
    become negative, which happens on GSM8K and not on HotpotQA, where it settles
    just above zero instead.  Loss of significance applies to both, so it is the
    primary definition; the zero crossing is reported when it exists.
    """
    # Walk forward from the shortest usable span while the gap stays significantly
    # positive, and stop at the first span that clears.  Taking the *last* span with
    # a significant gap instead would be wrong: on HotpotQA the gap clears at 120 and
    # then drifts back to nominal significance at 800--1000 at a magnitude (+0.045)
    # an order below the short-span effect, which is late noise, not re-coupling.
    run = 0
    while run < len(rows) and rows[run]["gap_ci"][0] > 0:
        run += 1
    if run == 0 or run >= len(rows):
        return {}
    last, first_clear = rows[run - 1], rows[run]
    out = {
        "last_coupled_span": last["span_chars"],
        "first_clear_span": first_clear["span_chars"],
        "span": 0.5 * ((last["span_chars"] or 0) + (first_clear["span_chars"] or 0)),
        "coverage": 0.5 * (last["coverage"] + first_clear["coverage"]),
    }
    for prev, cur in zip(rows, rows[1:]):
        if prev["gap"] > 0 >= cur["gap"]:
            w = prev["gap"] / (prev["gap"] - cur["gap"])
            out["zero_crossing_span"] = ((prev["span_chars"] or 0)
                                         + w * ((cur["span_chars"] or 0) - (prev["span_chars"] or 0)))
            break
    return out


def main() -> None:
    result = {}
    for phase, label in DOMAINS:
        try:
            ta, ja = load(phase, "qwen3.5-2b")
            tb, jb = load(phase, "gemma-4-e2b")
        except FileNotFoundError as exc:
            print(f"\n=== {label}: skipped ({exc.filename})", flush=True)
            continue
        ids = sorted(set(ta) & set(tb) & set(ja) & set(jb))
        a_txt = [ta[i].get("trace") or "" for i in ids]
        b_txt = [tb[i].get("trace") or "" for i in ids]
        gold = {i: (ja[i].get("gold_answer") or "").strip() for i in ids}
        y = np.array([int(bool(ja[i].get("correct_judge")) != bool(jb[i].get("correct_judge")))
                      for i in ids])

        arrivals = [o for i, a, b in zip(ids, a_txt, b_txt) for t in (a, b)
                    if (o := first_offset(t, gold[i])) is not None]
        arr = np.array(arrivals, dtype=float)

        print(f"\n=== {label}  n={len(ids)}  construct+={int(y.sum())}", flush=True)
        print(f"    answer arrival: median {np.median(arr):.0f} chars "
              f"[q25 {np.percentile(arr,25):.0f}, q75 {np.percentile(arr,75):.0f}], "
              f"present in {len(arr)}/{2*len(ids)} traces", flush=True)

        rows = []
        for span in SPANS:
            pa = [t[:span] if span else t for t in a_txt]
            pb = [t[:span] if span else t for t in b_txt]
            cov = float(np.mean([first_offset(t, gold[i]) is not None
                                 for i, pair in zip(ids, zip(pa, pb)) for t in pair]))
            z = np.array([int((gold[i].lower() in a.lower()) != (gold[i].lower() in b.lower()))
                          for i, a, b in zip(ids, pa, pb)])
            name = str(span) if span else "full"
            row = {"span": name, "span_chars": span, "coverage": cov,
                   "proxy_positives": int(z.sum())}
            if len(set(z)) < 2:
                row["verdict"] = "no proxy variation"
            else:
                s = cosine_distance(pa, pb)
                if degenerate(s, z):
                    row["verdict"] = "score degenerate"
                else:
                    az, ay = float(roc_auc_score(z, s)), float(roc_auc_score(y, s))
                    lo, hi = boot_gap(z, y, s, abs(hash(phase + name)) % (2**31))
                    row.update(auc_proxy=az, auc_semantic=ay, gap=az - ay, gap_ci=[lo, hi])
            rows.append(row)
            if "gap" in row:
                mark = " *" if row["gap_ci"][0] > 0 else ""
                print(f"    {name:5s} cov={cov:.3f} z+={row['proxy_positives']:4d} "
                      f"gap={row['gap']:+.3f} [{row['gap_ci'][0]:+.3f},{row['gap_ci'][1]:+.3f}]{mark}",
                      flush=True)
            else:
                print(f"    {name:5s} cov={cov:.3f} z+={row['proxy_positives']:4d} "
                      f"{row['verdict']}", flush=True)

        usable = [r for r in rows if "gap" in r]
        closure = interpolate_closure(usable)
        result[label] = {
            "phase": phase, "n": len(ids), "construct_positives": int(y.sum()),
            "arrival_median": float(np.median(arr)),
            "arrival_q25": float(np.percentile(arr, 25)),
            "arrival_q75": float(np.percentile(arr, 75)),
            "rows": rows, "closure": closure,
        }
        if closure:
            print(f"    -> gap reaches zero at span ~{closure['span']:.0f} chars, "
                  f"coverage ~{closure['coverage']:.3f}", flush=True)

    print("\n=== re-parameterisation test ===", flush=True)
    for label, r in result.items():
        c = r["closure"]
        if c:
            print(f"  {label:10s} closure span ~{c['span']:6.0f} chars   "
                  f"closure coverage ~{c['coverage']:.3f}   "
                  f"median arrival {r['arrival_median']:.0f} chars", flush=True)
    print("\n  closure span as a fraction of median arrival offset:", flush=True)
    for label, r in result.items():
        c = r["closure"]
        if c:
            frac = c["span"] / r["arrival_median"]
            r["closure_over_arrival"] = float(frac)
            print(f"    {label:14s} {c['span']:6.0f} / {r['arrival_median']:6.0f} = {frac:.3f}", flush=True)
    fr = [r["closure_over_arrival"] for r in result.values() if "closure_over_arrival" in r]
    if len(fr) > 1:
        print(f"    -> range {min(fr):.3f}-{max(fr):.3f} across {len(fr)} domain-runs "
              f"(mean {sum(fr)/len(fr):.3f})", flush=True)
    cv = [r["closure"]["coverage"] for r in result.values() if r["closure"]]
    if len(cv) > 1:
        print(f"  closure coverage range {min(cv):.3f}-{max(cv):.3f} "
              f"({max(cv)/max(1e-9,min(cv)):.1f}x spread)", flush=True)

    path = ROOT / "analysis_results" / "span_closure_fine_sweep.json"
    path.write_text(json.dumps(result, indent=2))
    print("\n[wrote]", path, flush=True)


if __name__ == "__main__":
    main()
