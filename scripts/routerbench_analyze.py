#!/usr/bin/env python3
"""External RLC-Audit of RouterBench contracts.

Audit 1 (mtbench, n=80, pair claude-instant-v1 / mixtral-8x7b-chat):
  score s   = TF-IDF cosine distance between the pair's stored responses
              (full span and opening-200-chars span)
  proxy z   = stored GPT-4-judge score disagreement (binarized at 0.7, XOR)
  construct y = independent Haiku adequacy adjudication disagreement (XOR)
  Reports AUC(s->z), AUC(s->y), kappa(z,y), routed composition at B=10%.

Audit 2 (mmlu-professional-law, n=1534):
  score s   = TF-IDF distance between letter responses
  proxy z   = extracted answer-letter disagreement
  construct y = stored exact-match correctness disagreement (ground truth)

Also summarizes the GSM8K label-provenance finding (fractional labels with a
single stored response: the construct label is not a function of the stored
evidence, so no score-proxy-construct contract can be instantiated).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/routerbench_audit"
OUT = ROOT / "analysis_results/routerbench_audit.json"


def tfidf_distance(texts_a: list[str], texts_b: list[str], char_level: bool = False) -> np.ndarray:
    # char_level for letter-only MMLU responses: the default word tokenizer
    # drops single-character tokens, collapsing every distance to a constant.
    if char_level:
        vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), sublinear_tf=True)
    else:
        vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    vec.fit(texts_a + texts_b)
    xa, xb = vec.transform(texts_a), vec.transform(texts_b)
    return 1.0 - np.array([float(cosine_similarity(xa[i], xb[i])[0, 0]) for i in range(len(texts_a))])


def ranking(y: np.ndarray, s: np.ndarray) -> dict:
    if y.sum() in (0, len(y)):
        return {"n_pos": int(y.sum()), "auc": None, "ap": None}
    return {
        "n_pos": int(y.sum()),
        "auc": float(roc_auc_score(y, s)),
        "ap": float(average_precision_score(y, s)),
    }


def routed(y: np.ndarray, z: np.ndarray, s: np.ndarray, budget: int) -> dict:
    top = np.argsort(-s)[:budget]
    return {
        "budget": budget,
        "construct_disagreements": int(y[top].sum()),
        "proxy_disagreements": int(z[top].sum()),
    }


LETTER_RE = re.compile(r"[A-J]")


def extract_letter(resp: str) -> str | None:
    m = LETTER_RE.search(resp.strip().upper())
    return m.group(0) if m else None


def main() -> None:
    report: dict = {}

    # ---- Audit 1: mtbench ----
    mt = json.loads((DATA / "mtbench_pair.json").read_text())["records"]
    judg = {r["sample_id"]: r for r in json.loads((DATA / "mtbench_haiku_judgments.json").read_text())["records"]}
    rows = [r for r in mt if judg[r["sample_id"]]["adequate_a"] is not None
            and judg[r["sample_id"]]["adequate_b"] is not None]
    z = np.array([(r["score_a"] >= 0.7) != (r["score_b"] >= 0.7) for r in rows])
    z05 = np.array([(r["score_a"] >= 0.5) != (r["score_b"] >= 0.5) for r in rows])
    y = np.array([judg[r["sample_id"]]["adequate_a"] != judg[r["sample_id"]]["adequate_b"] for r in rows])
    resp_a = [r["response_a"] for r in rows]
    resp_b = [r["response_b"] for r in rows]
    s_full = tfidf_distance(resp_a, resp_b)
    s_open = tfidf_distance([t[:200] for t in resp_a], [t[:200] for t in resp_b])
    budget = max(1, round(0.1 * len(rows)))
    report["mtbench"] = {
        "n": len(rows),
        "proxy": "stored GPT-4 judge score >= 0.7, XOR",
        "construct": "haiku-4-5 adequacy adjudication, XOR",
        "kappa_proxy_construct": float(cohen_kappa_score(z, y)),
        "proxy_threshold_sensitivity_kappa_at_0.5": float(cohen_kappa_score(z05, y)),
        "full_span": {
            "proxy": ranking(z, s_full), "construct": ranking(y, s_full),
            "routed": routed(y, z, s_full, budget),
        },
        "opening_200": {
            "proxy": ranking(z, s_open), "construct": ranking(y, s_open),
            "routed": routed(y, z, s_open, budget),
        },
        "haiku_adequate_rate_a": float(np.mean([judg[r["sample_id"]]["adequate_a"] for r in rows])),
        "haiku_adequate_rate_b": float(np.mean([judg[r["sample_id"]]["adequate_b"] for r in rows])),
        "stored_good_rate_a": float(np.mean([r["score_a"] >= 0.7 for r in rows])),
        "stored_good_rate_b": float(np.mean([r["score_b"] >= 0.7 for r in rows])),
    }
    for side, key in (("a", "score_a"), ("b", "score_b")):
        stored = np.array([r[key] >= 0.7 for r in rows])
        haiku = np.array([judg[r["sample_id"]][f"adequate_{side}"] for r in rows])
        report["mtbench"][f"side_{side}_stored_vs_haiku"] = {
            "raw_agreement": float((stored == haiku).mean()),
            "kappa": float(cohen_kappa_score(stored, haiku)),
        }

    # ---- Audit 2: mmlu-professional-law ----
    law = json.loads((DATA / "mmlu_law_pair.json").read_text())["records"]
    rows = [r for r in law if extract_letter(r["response_a"]) and extract_letter(r["response_b"])]
    z = np.array([extract_letter(r["response_a"]) != extract_letter(r["response_b"]) for r in rows])
    y = np.array([bool(r["correct_a"]) != bool(r["correct_b"]) for r in rows])
    s = tfidf_distance([r["response_a"] for r in rows], [r["response_b"] for r in rows], char_level=True)
    budget = max(1, round(0.1 * len(rows)))
    report["mmlu_professional_law"] = {
        "n": len(rows),
        "n_dropped_no_letter": len(law) - len(rows),
        "proxy": "extracted answer-letter disagreement",
        "construct": "stored exact-match correctness disagreement",
        "kappa_proxy_construct": float(cohen_kappa_score(z, y)),
        "full_span": {
            "proxy": ranking(z, s), "construct": ranking(y, s),
            "routed": routed(y, z, s, budget),
        },
    }

    # ---- GSM8K provenance ----
    prov = json.loads((DATA / "gsm8k_label_provenance.json").read_text())
    report["gsm8k_provenance"] = {
        "n": prov["n"],
        "rows_with_any_fractional_label": prov["rows_with_any_fractional_label"],
        "example_histogram_claude_instant": prov["models"]["claude-instant-v1"]["label_histogram"],
        "finding": "stored labels are fractional aggregates while a single response string is stored per model; the construct label is not a function of the stored evidence, so no score-proxy-construct contract over the released responses can be instantiated or audited",
    }

    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
