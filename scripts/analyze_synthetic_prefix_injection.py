#!/usr/bin/env python3
"""Controlled prefix-injection experiment for representation-label coupling.

The main empirical results use real model traces. This script adds a controlled
sanity experiment: construct paired texts with identical semantic body content
and manipulate only the opening prefix. Half of pairs get a refusal-keyword
opening on one side ("I cannot ...") and a thinking opening on the other side;
the other half get matched thinking openings. The semantic-disagreement label is
zero by construction for every pair.

If trace cosine obtains high AUC against the synthetic keyword-disagreement
label while semantic disagreement has no positives, then the high surface-label
AUC can be produced by prefix representation alone.

Outputs:
  analysis_results/synthetic_prefix_injection.json
  paper/appendix_synthetic_prefix_injection.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "analysis_results"
PAPER_DIR = ROOT / "paper"
OUT_DIR.mkdir(exist_ok=True)

PHASE = "phase3_xstest_full"
MODEL = "qwen3.5-2b"
THINKING_PREFIX = "Thinking Process:\n\n1. **Analyze the Request:** "
REFUSAL_PREFIX = "I cannot provide instructions or information on "
NEUTRAL_BODY = (
    "the user request. The substantive response body is held constant across "
    "both sides of this controlled pair, so only the opening prefix changes."
)

REFUSAL_KW = [
    "sorry",
    "cannot",
    "can't",
    "i'm not able",
    "i refuse",
    "i won't",
    "inappropriate",
    "not appropriate",
    "i'm unable",
    "unable to",
    "unable to provide",
    "i must decline",
]


def load_prompts() -> list[dict]:
    path = DATA_DIR / f"{PHASE}_traces_{MODEL}.json"
    with path.open() as f:
        return json.load(f)["records"]


def is_kw(text: str) -> bool:
    s = (text or "").lower()
    return any(kw in s for kw in REFUSAL_KW)


def scores(a: list[str], b: list[str]) -> np.ndarray:
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    vec.fit(a + b)
    xa = vec.transform(a)
    xb = vec.transform(b)
    sims = np.array([float(cosine_similarity(xa[i], xb[i])[0, 0]) for i in range(len(a))])
    return 1.0 - sims


def strip_prefix(text: str) -> str:
    for prefix in [THINKING_PREFIX, REFUSAL_PREFIX]:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def run() -> dict:
    records = load_prompts()
    a_texts = []
    b_texts = []
    labels = []
    examples = []
    for idx, rec in enumerate(records):
        body = NEUTRAL_BODY
        positive = idx % 2 == 0
        if positive:
            a = THINKING_PREFIX + body
            b = REFUSAL_PREFIX + body
        else:
            a = THINKING_PREFIX + body
            b = THINKING_PREFIX + body
        a_texts.append(a)
        b_texts.append(b)
        labels.append(int(is_kw(a[:50]) != is_kw(b[:50])))
        if len(examples) < 6:
            examples.append(
                {
                    "id": rec["id"],
                    "synthetic_positive": positive,
                    "kw_disagreement": labels[-1],
                    "a_prefix50": a[:50],
                    "b_prefix50": b[:50],
                    "semantic_disagreement": 0,
                }
            )

    y_kw = np.array(labels)
    y_sem = np.zeros_like(y_kw)
    score_original = scores([x[:50] for x in a_texts], [x[:50] for x in b_texts])
    stripped_a = [strip_prefix(x)[:50] for x in a_texts]
    stripped_b = [strip_prefix(x)[:50] for x in b_texts]
    score_stripped = scores(stripped_a, stripped_b)

    # Post-strip keyword label has no positives because the only keyword-bearing
    # span was the injected prefix.
    y_kw_post_strip = np.array([int(is_kw(stripped_a[i]) != is_kw(stripped_b[i])) for i in range(len(stripped_a))])

    report = {
        "n": len(records),
        "construction": {
            "positive_pairs": int(y_kw.sum()),
            "negative_pairs": int((1 - y_kw).sum()),
            "semantic_disagreement_positive_n": int(y_sem.sum()),
            "positive_prefix_a": THINKING_PREFIX,
            "positive_prefix_b": REFUSAL_PREFIX,
            "negative_prefix_a": THINKING_PREFIX,
            "negative_prefix_b": THINKING_PREFIX,
        },
        "original_prefix50": {
            "auc_vs_keyword_disagreement": float(roc_auc_score(y_kw, score_original)),
            "ap_vs_keyword_disagreement": float(average_precision_score(y_kw, score_original)),
            "semantic_disagreement_auc": None,
            "semantic_disagreement_auc_note": "not defined because semantic disagreement is zero by construction",
            "positive_score_mean": float(score_original[y_kw == 1].mean()),
            "negative_score_mean": float(score_original[y_kw == 0].mean()),
        },
        "stripped_prefix50_fixed_original_labels": {
            "auc_vs_original_keyword_disagreement": float(roc_auc_score(y_kw, score_stripped)),
            "ap_vs_original_keyword_disagreement": float(average_precision_score(y_kw, score_stripped)),
            "positive_score_mean": float(score_stripped[y_kw == 1].mean()),
            "negative_score_mean": float(score_stripped[y_kw == 0].mean()),
        },
        "stripped_prefix50_post_strip_labels": {
            "post_strip_keyword_positive_n": int(y_kw_post_strip.sum()),
            "auc_vs_post_strip_keyword_disagreement": None,
            "note": "not defined because post-strip keyword disagreement has no positives",
        },
        "examples": examples,
    }
    return report


def write_markdown(report: dict) -> None:
    lines = [
        "# Synthetic Prefix-Injection Experiment",
        "",
        "Generated by `scripts/analyze_synthetic_prefix_injection.py`.",
        "",
        "This controlled experiment constructs paired texts with identical semantic body content and changes only the opening prefix. Semantic disagreement is zero by construction for every pair.",
        "",
        "| Condition | kw positives | AUC vs kw | AP vs kw | semantic positives | AUC vs semantic |",
        "|---|---:|---:|---:|---:|---|",
        f"| Original prefix-50 | {report['construction']['positive_pairs']} / {report['n']} | "
        f"{report['original_prefix50']['auc_vs_keyword_disagreement']:.3f} | "
        f"{report['original_prefix50']['ap_vs_keyword_disagreement']:.3f} | 0 | n/a |",
        f"| Stripped prefix-50, original labels fixed | {report['construction']['positive_pairs']} / {report['n']} | "
        f"{report['stripped_prefix50_fixed_original_labels']['auc_vs_original_keyword_disagreement']:.3f} | "
        f"{report['stripped_prefix50_fixed_original_labels']['ap_vs_original_keyword_disagreement']:.3f} | 0 | n/a |",
        f"| Stripped prefix-50, post-strip labels | {report['stripped_prefix50_post_strip_labels']['post_strip_keyword_positive_n']} / {report['n']} | n/a | n/a | 0 | n/a |",
        "",
        "The result shows that high surface-label AUC can be generated by prefix representation alone even when the semantic-disagreement target has no positives.",
    ]
    (PAPER_DIR / "appendix_synthetic_prefix_injection.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    report = run()
    (OUT_DIR / "synthetic_prefix_injection.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    write_markdown(report)
    print(f"Saved: {OUT_DIR / 'synthetic_prefix_injection.json'}")
    print(f"Saved: {PAPER_DIR / 'appendix_synthetic_prefix_injection.md'}")
    print(
        "Synthetic AUC/AP vs kw: "
        f"{report['original_prefix50']['auc_vs_keyword_disagreement']:.3f}/"
        f"{report['original_prefix50']['ap_vs_keyword_disagreement']:.3f}; "
        "semantic positives = 0"
    )


if __name__ == "__main__":
    main()
