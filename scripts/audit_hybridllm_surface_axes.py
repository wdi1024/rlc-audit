#!/usr/bin/env python3
"""How much of a deployed routing target is recoverable from surface form alone?

The first MixInstruct audit (audit_hybridllm_mixinstruct.py) showed that response
length ranks Hybrid LLM's BARTScore target better than it ranks the independently
elicited adequacy judgment.  A reviewer's fair objection is that the length effect
is small (AUC 0.609 vs 0.542), so this script asks the same question with the
surface channel widened: given only content-blind features of the two responses --
no tokens, no topic, no model identity -- how well can a classifier recover the
deployed proxy, and how well can the same classifier recover the construct?

Features are differences (A minus B) of content-blind statistics: character and
word count, mean word length, type-token ratio, sentence count, list/markdown
markers, digit and uppercase share, terminal punctuation, and self-repetition.
None reads what the response says.  Scoring is out-of-fold so the reported AUCs
are held-out, and the two targets share folds so the gap is paired.

The claim this supports is bounded: a routing target that a content-blind model
recovers substantially better than the construct it stands in for is carrying
surface variance the construct does not.  That is the L2 asymmetry, measured on
released artifacts.

  python3 audit_hybridllm_surface_axes.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
N_FOLDS = 5
N_BOOT = 2000
RNG = np.random.default_rng(0)
MIN_PAIR_N = 200

WORD = re.compile(r"[A-Za-z']+")
SENT = re.compile(r"[.!?]+")
LIST_MARK = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s")

FEATURES = ["chars", "words", "mean_word_len", "ttr", "sentences", "mean_sent_len",
            "list_markers", "digit_share", "upper_share", "ends_punct", "repetition"]


def surface(text: str) -> dict[str, float]:
    """Content-blind statistics: shape of the response, never its subject."""
    t = text or ""
    words = WORD.findall(t.lower())
    n_w = len(words)
    sents = [x for x in SENT.split(t) if x.strip()]
    # self-repetition: fraction of bigrams that are not unique
    bigrams = list(zip(words, words[1:]))
    rep = 1.0 - (len(set(bigrams)) / len(bigrams)) if bigrams else 0.0
    return {
        "chars": len(t),
        "words": float(n_w),
        "mean_word_len": float(np.mean([len(w) for w in words])) if n_w else 0.0,
        "ttr": len(set(words)) / n_w if n_w else 0.0,
        "sentences": float(len(sents)),
        "mean_sent_len": n_w / len(sents) if sents else 0.0,
        "list_markers": float(len(LIST_MARK.findall(t))),
        "digit_share": sum(c.isdigit() for c in t) / len(t) if t else 0.0,
        "upper_share": sum(c.isupper() for c in t) / len(t) if t else 0.0,
        "ends_punct": float(bool(t.strip()[-1:] in ".!?")),
        "repetition": rep,
    }


def oof_auc(X: np.ndarray, target: np.ndarray, folds) -> tuple[float, np.ndarray]:
    """Out-of-fold scores for one binary target, and its AUC."""
    pred = np.zeros(len(target))
    for tr, te in folds:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], target[tr])
        pred[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(target, pred)), pred


def main():
    ds = load_dataset("llm-blender/mix-instruct", split="test")
    print(f"MixInstruct test rows: {len(ds)}")

    rows: list[tuple[str, float, int, dict[str, float]]] = []
    for row in ds:
        cand = {c["model"]: c for c in row["candidates"]}
        bs = {m: c["scores"]["bartscore"] for m, c in cand.items()
              if c["scores"].get("bartscore") is not None}
        try:
            cmp = json.loads(row["cmp_results"]) if row["cmp_results"] else {}
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(cmp, dict):
            continue
        feat_cache: dict[str, dict[str, float]] = {}
        for key, verdict in cmp.items():
            if "," not in key:
                continue
            a, b = key.split(",", 1)
            if a not in bs or b not in bs:
                continue
            if verdict == "A is better":
                y = 1
            elif verdict == "B is better":
                y = 0
            else:
                continue
            for m in (a, b):
                if m not in feat_cache:
                    feat_cache[m] = surface(cand[m]["text"] or "")
            d = {k: feat_cache[a][k] - feat_cache[b][k] for k in FEATURES}
            rows.append((f"{a} vs {b}", bs[a] - bs[b], y, d))

    print(f"directional comparisons: {len(rows)}")
    pair_names = np.array([r[0] for r in rows])
    gap = np.array([r[1] for r in rows], dtype=float)
    y = np.array([r[2] for r in rows], dtype=int)
    z = (gap > 0).astype(int)                    # the deployed proxy's direction
    X = np.array([[r[3][k] for k in FEATURES] for r in rows], dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Shared folds so the two AUCs are paired on the same held-out rows.
    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                 random_state=0).split(X, z))
    auc_z, pred_z = oof_auc(X, z, folds)
    auc_y, pred_y = oof_auc(X, y, folds)
    print(f"\ncontent-blind surface model, out-of-fold:")
    print(f"  recovers the deployed proxy z at AUC {auc_z:.3f}")
    print(f"  recovers the construct     y at AUC {auc_y:.3f}")
    print(f"  asymmetry {auc_z - auc_y:+.3f}")

    # Paired bootstrap on the gap between the two held-out AUCs.
    diffs = np.empty(N_BOOT)
    n = len(y)
    for k in range(N_BOOT):
        sel = RNG.integers(0, n, n)
        if len(set(z[sel])) < 2 or len(set(y[sel])) < 2:
            diffs[k] = np.nan
            continue
        diffs[k] = (roc_auc_score(z[sel], pred_z[sel])
                    - roc_auc_score(y[sel], pred_y[sel]))
    d = diffs[~np.isnan(diffs)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"  paired bootstrap 95% CI on the asymmetry [{lo:+.3f}, {hi:+.3f}]")

    # Which axes carry it: single-feature AUCs on both targets.
    per_feat = {}
    for j, name in enumerate(FEATURES):
        v = X[:, j]
        if np.allclose(v, v[0]):
            continue
        per_feat[name] = {"auc_proxy": float(roc_auc_score(z, v)),
                          "auc_construct": float(roc_auc_score(y, v))}
    print("\nsingle-feature AUC (proxy / construct / gap):")
    for name, d_ in sorted(per_feat.items(),
                           key=lambda kv: -(kv[1]["auc_proxy"] - kv[1]["auc_construct"])):
        print(f"  {name:15} {d_['auc_proxy']:.3f} / {d_['auc_construct']:.3f} "
              f"/ {d_['auc_proxy'] - d_['auc_construct']:+.3f}")

    # Per-pair: is the asymmetry a pooling artifact or present within pairs?
    per_pair = {}
    for name in sorted(set(pair_names)):
        m = pair_names == name
        if m.sum() < MIN_PAIR_N or len(set(z[m])) < 2 or len(set(y[m])) < 2:
            continue
        per_pair[name] = {"n": int(m.sum()),
                          "auc_proxy": float(roc_auc_score(z[m], pred_z[m])),
                          "auc_construct": float(roc_auc_score(y[m], pred_y[m]))}
    gaps = np.array([v["auc_proxy"] - v["auc_construct"] for v in per_pair.values()])
    pos = int((gaps > 0).sum())
    print(f"\nper-pair (n>={MIN_PAIR_N}): {len(per_pair)} pairs, asymmetry positive in "
          f"{pos}/{len(per_pair)}; median {float(np.median(gaps)):+.3f}, "
          f"range [{float(gaps.min()):+.3f}, {float(gaps.max()):+.3f}]")

    out = {"n": int(n), "n_features": len(FEATURES), "features": FEATURES,
           "pooled": {"auc_surface_vs_proxy": auc_z, "auc_surface_vs_construct": auc_y,
                      "asymmetry": auc_z - auc_y,
                      "asymmetry_ci": [float(lo), float(hi)], "n_folds": N_FOLDS},
           "per_feature": per_feat,
           "per_pair": {"n_pairs": len(per_pair), "n_positive_gap": pos,
                        "median_gap": float(np.median(gaps)),
                        "min_gap": float(gaps.min()), "max_gap": float(gaps.max()),
                        "pairs": per_pair}}
    p = ROOT / "analysis_results" / "hybridllm_surface_axes.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\n[wrote] {p}")


if __name__ == "__main__":
    main()
