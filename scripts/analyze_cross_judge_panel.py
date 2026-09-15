#!/usr/bin/env python3
"""Cross-judge consistency check for the tagged final-span panel.

Compares OpenAI gpt-4o-mini and Anthropic claude-haiku-4-5 final-channel
refusal judgments on three panel pairs (Qwen/Gemma, Qwen/Mistral,
Gemma/Mistral): per-model label agreement, pair-level semantic-disagreement
agreement, and final-span score AUC under each judge's labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/disagree_routing"
OUT = ROOT / "analysis_results/cross_judge_panel.json"

PHASE_QG = "phase_colab_qwen_gemma_big_xstest450_tagged768_thinkingoff_greedy"
PHASE_QM = "phase_colab_qwen_mistral_xstest450_tagged768_thinkingoff_greedy"
JUDGE_A = "openai_gpt-4o-mini"
JUDGE_B = "anthropic_claude-haiku-4-5-20251001"

PAIRS = {
    "Qwen/Gemma": ((PHASE_QG, "qwen3.5-9b"), (PHASE_QG, "gemma-2-9b-it")),
    "Qwen/Mistral": ((PHASE_QM, "qwen3.5-9b"), (PHASE_QM, "mistral-7b-instruct-v0.3")),
    "Gemma/Mistral": ((PHASE_QG, "gemma-2-9b-it"), (PHASE_QM, "mistral-7b-instruct-v0.3")),
}


def load_judge(phase: str, model: str, judge_tag: str) -> dict[str, bool]:
    path = DATA / f"{phase}_judge_final_{model}_{judge_tag}.json"
    rows = json.loads(path.read_text())["records"]
    return {
        r["id"]: bool(r["refusal_judge"])
        for r in rows
        if r.get("refusal_judge") is not None and not r.get("error")
    }


def load_final(phase: str, model: str) -> dict[str, str]:
    path = DATA / f"{phase}_traces_{model}.json"
    return {
        r["id"]: r.get("final") or ""
        for r in json.loads(path.read_text())["records"]
    }


def score_distance(texts_a: list[str], texts_b: list[str]) -> np.ndarray:
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    vec.fit(texts_a + texts_b)
    xa = vec.transform(texts_a)
    xb = vec.transform(texts_b)
    sims = np.array([float(cosine_similarity(xa[i], xb[i])[0, 0]) for i in range(len(texts_a))])
    return 1.0 - sims


def main() -> None:
    report: dict = {"judges": [JUDGE_A, JUDGE_B], "models": {}, "pairs": {}}

    for phase, model in {pm for pair in PAIRS.values() for pm in pair}:
        ja = load_judge(phase, model, JUDGE_A)
        jb = load_judge(phase, model, JUDGE_B)
        ids = sorted(set(ja) & set(jb))
        a = np.array([ja[i] for i in ids])
        b = np.array([jb[i] for i in ids])
        report["models"][model] = {
            "n": len(ids),
            "refusal_n_gpt4omini": int(a.sum()),
            "refusal_n_haiku": int(b.sum()),
            "raw_agreement": float((a == b).mean()),
            "kappa": float(cohen_kappa_score(a, b)),
        }

    for pair, ((ph_a, m_a), (ph_b, m_b)) in PAIRS.items():
        ja_a = load_judge(ph_a, m_a, JUDGE_A)
        jb_a = load_judge(ph_a, m_a, JUDGE_B)
        ja_b = load_judge(ph_b, m_b, JUDGE_A)
        jb_b = load_judge(ph_b, m_b, JUDGE_B)
        fin_a = load_final(ph_a, m_a)
        fin_b = load_final(ph_b, m_b)
        ids = sorted(set(ja_a) & set(jb_a) & set(ja_b) & set(jb_b))
        ids = [i for i in ids if fin_a.get(i, "").strip() and fin_b.get(i, "").strip()]

        y_a = np.array([ja_a[i] != ja_b[i] for i in ids])  # gpt-4o-mini semantic disagreement
        y_b = np.array([jb_a[i] != jb_b[i] for i in ids])  # haiku semantic disagreement
        s = score_distance([fin_a[i] for i in ids], [fin_b[i] for i in ids])

        report["pairs"][pair] = {
            "n": len(ids),
            "sem_disagreement_n_gpt4omini": int(y_a.sum()),
            "sem_disagreement_n_haiku": int(y_b.sum()),
            "pairlabel_raw_agreement": float((y_a == y_b).mean()),
            "pairlabel_kappa": float(cohen_kappa_score(y_a, y_b)),
            "final_auc_sem_gpt4omini": float(roc_auc_score(y_a, s)) if 0 < y_a.sum() < len(y_a) else None,
            "final_auc_sem_haiku": float(roc_auc_score(y_b, s)) if 0 < y_b.sum() < len(y_b) else None,
        }

    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
