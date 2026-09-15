#!/usr/bin/env python3
"""Model-agnostic opening-span normalizers.

This is a deliberately blunt companion to the template-strip intervention. It
does not use model names, refusal words, or hand-written opening regexes. Each
normalizer simply removes or skips an initial span, then recomputes prefix-50
TF-IDF cosine and evaluates it against the original prefix-50 labels.

The fixed-label rows ask whether the score representation still predicts the
same surface disagreement labels after the opening span is removed.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
LEGACY_DIR = ROOT / "results" / "disagree_routing"
DATA_DIR = ROOT / "data"
ANALYSIS_DIR = ROOT / "analysis_results"

PHASE = "phase3_xstest_full"
MODEL_A = "qwen3.5-2b"
MODEL_B = "gemma-4-e2b"
JUDGE_TAG = "anthropic_claude-haiku-4-5-20251001"
PREFIX_K = 50

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


def input_path(name: str) -> Path:
    """Prefer packaged data, falling back to the legacy compatibility layout."""
    packaged = DATA_DIR / name
    if packaged.exists():
        return packaged
    return LEGACY_DIR / name


def load_traces(phase: str, model: str) -> dict[str, str]:
    path = input_path(f"{phase}_traces_{model}.json")
    records = json.load(path.open())["records"]
    return {r["id"]: (r.get("trace") or "") for r in records}


def load_judge(phase: str, model: str) -> dict[str, bool]:
    path = input_path(f"{phase}_judge_{model}_{JUDGE_TAG}.json")
    records = json.load(path.open())["records"]
    return {
        r["id"]: bool(r["refusal_judge"])
        for r in records
        if r.get("refusal_judge") is not None
    }


def is_kw(text: str) -> bool:
    s = (text or "").lower()
    return any(kw in s for kw in REFUSAL_KW)


def drop_chars(text: str, n: int) -> str:
    return text[n:]


def drop_tokens(text: str, n: int) -> str:
    return " ".join(text.split()[n:])


def remove_first_sentence(text: str) -> str:
    match = re.search(r"(?s).*?(?:[.!?]\s+|\n+)", text)
    if not match:
        return ""
    return text[match.end() :]


def remove_first_line(text: str) -> str:
    parts = text.splitlines()
    if len(parts) <= 1:
        return ""
    return "\n".join(parts[1:]).lstrip()


def span(text: str, start: int, end: int) -> str:
    return text[start:end]


def tail(text: str, n: int) -> str:
    return text[-n:]


def prefix_after(transform, text: str) -> str:
    return transform(text)[:PREFIX_K]


def auc_or_none(y: np.ndarray, score: np.ndarray) -> float | None:
    if y.sum() < 2 or y.sum() >= len(y):
        return None
    return float(roc_auc_score(y, score))


def cosine_scores(prefixes_a: list[str], prefixes_b: list[str]) -> np.ndarray:
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    vec.fit(prefixes_a + prefixes_b)
    scores = []
    for a, b in zip(prefixes_a, prefixes_b):
        sim = cosine_similarity(vec.transform([a]), vec.transform([b]))[0, 0]
        scores.append(1.0 - float(sim))
    return np.array(scores)


def evaluate(
    name: str,
    common: list[str],
    traces_a: dict[str, str],
    traces_b: dict[str, str],
    judge_a: dict[str, bool],
    judge_b: dict[str, bool],
    transform,
    original_kw_a: np.ndarray,
    original_kw_b: np.ndarray,
) -> dict[str, object]:
    prefixes_a = [prefix_after(transform, traces_a[i]) for i in common]
    prefixes_b = [prefix_after(transform, traces_b[i]) for i in common]

    y_kw_fixed = (original_kw_a != original_kw_b).astype(int)
    y_judge = np.array([int(judge_a[i] != judge_b[i]) for i in common])
    post_kw_a = np.array([int(is_kw(p)) for p in prefixes_a])
    post_kw_b = np.array([int(is_kw(p)) for p in prefixes_b])
    y_kw_post = (post_kw_a != post_kw_b).astype(int)

    try:
        score = cosine_scores(prefixes_a, prefixes_b)
    except ValueError:
        return {
            "name": name,
            "n": len(common),
            "n_dis_kw_fixed": int(y_kw_fixed.sum()),
            "n_dis_kw_post": int(y_kw_post.sum()),
            "n_dis_judge": int(y_judge.sum()),
            "auc_kw_fixed": None,
            "auc_kw_post": None,
            "auc_judge": None,
            "mean_score": None,
            "median_score": None,
        }

    return {
        "name": name,
        "n": len(common),
        "n_dis_kw_fixed": int(y_kw_fixed.sum()),
        "n_dis_kw_post": int(y_kw_post.sum()),
        "n_dis_judge": int(y_judge.sum()),
        "auc_kw_fixed": auc_or_none(y_kw_fixed, score),
        "auc_kw_post": auc_or_none(y_kw_post, score),
        "auc_judge": auc_or_none(y_judge, score),
        "mean_score": float(score.mean()),
        "median_score": float(np.median(score)),
    }


def main() -> None:
    traces_a = load_traces(PHASE, MODEL_A)
    traces_b = load_traces(PHASE, MODEL_B)
    judge_a = load_judge(PHASE, MODEL_A)
    judge_b = load_judge(PHASE, MODEL_B)
    common = sorted(set(traces_a) & set(traces_b) & set(judge_a) & set(judge_b))

    original_prefix_a = [traces_a[i][:PREFIX_K] for i in common]
    original_prefix_b = [traces_b[i][:PREFIX_K] for i in common]
    original_kw_a = np.array([int(is_kw(p)) for p in original_prefix_a])
    original_kw_b = np.array([int(is_kw(p)) for p in original_prefix_b])

    transforms = [
        ("Original prefix-50", lambda s: s),
        ("Drop first 25 chars", lambda s: drop_chars(s, 25)),
        ("Drop first 50 chars", lambda s: drop_chars(s, 50)),
        ("Drop first 100 chars", lambda s: drop_chars(s, 100)),
        ("Drop first 200 chars", lambda s: drop_chars(s, 200)),
        ("Drop first 300 chars", lambda s: drop_chars(s, 300)),
        ("Drop first 8 tokens", lambda s: drop_tokens(s, 8)),
        ("Drop first 16 tokens", lambda s: drop_tokens(s, 16)),
        ("Drop first 32 tokens", lambda s: drop_tokens(s, 32)),
        ("Remove first sentence", remove_first_sentence),
        ("Remove first line", remove_first_line),
        ("Chars 50-150 only", lambda s: span(s, 50, 150)),
        ("Chars 100-200 only", lambda s: span(s, 100, 200)),
        ("Chars 200-300 only", lambda s: span(s, 200, 300)),
        ("Chars 300-500 only", lambda s: span(s, 300, 500)),
        ("Last 100 chars only", lambda s: tail(s, 100)),
    ]

    rows = [
        evaluate(
            name,
            common,
            traces_a,
            traces_b,
            judge_a,
            judge_b,
            transform,
            original_kw_a,
            original_kw_b,
        )
        for name, transform in transforms
    ]

    output = {
        "phase": PHASE,
        "models": [MODEL_A, MODEL_B],
        "judge": JUDGE_TAG,
        "prefix_k": PREFIX_K,
        "description": (
            "Blunt model-agnostic normalizers evaluated against original "
            "prefix-50 keyword labels and judge-excerpt labels."
        ),
        "rows": rows,
    }

    ANALYSIS_DIR.mkdir(exist_ok=True)
    out = ANALYSIS_DIR / "intervention_model_agnostic_normalizers.json"
    json.dump(output, out.open("w"), indent=2)

    if LEGACY_DIR.exists():
        legacy_out = LEGACY_DIR / out.name
        json.dump(output, legacy_out.open("w"), indent=2)

    print(f"Saved: {out.relative_to(ROOT)}")
    if LEGACY_DIR.exists():
        print(f"Saved: {(LEGACY_DIR / out.name).relative_to(ROOT)}")
    print()
    print(f"{'Normalizer':<26} {'kw fixed':>8} {'kw post':>8} {'judge':>8}")
    print(f"{'-' * 26} {'-' * 8:>8} {'-' * 8:>8} {'-' * 8:>8}")
    for row in rows:
        fmt = lambda x: " --" if x is None else f"{x:.3f}"
        print(
            f"{row['name']:<26} "
            f"{fmt(row['auc_kw_fixed']):>8} "
            f"{fmt(row['auc_kw_post']):>8} "
            f"{fmt(row['auc_judge']):>8}"
        )


if __name__ == "__main__":
    main()
