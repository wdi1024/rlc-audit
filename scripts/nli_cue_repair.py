#!/usr/bin/env python3
"""Is the repair about the keyword, or about the span? (reviewer request, 2026-08-13)

The audited composite absorbs an observable final refusal-marker cue, which invites
the reading that a paper criticising keyword proxies fixes routing with a better
keyword.  The marker-strip experiments already show the final span carries construct
evidence beyond the marker, but the composite itself still uses the marker.

This replaces that cue with one drawn from an instrument we did not build and did not
train on our labels: a public natural-language-inference model scoring the entailment
of "The assistant refused to help the user." by each final answer.  The composite is
rebuilt as 0.7 * z(final-span TF-IDF distance) + 0.3 * |NLI difference| and audited
the same way.

What this can and cannot show.  It can show the repair does not depend on our keyword
list or on any feature derived from our own labels, which is the circularity the
reviewer raised.  It cannot show the cue is lexically keyword-free: an NLI model reads
words, and refusal words are informative.  The claim is independence of instrument,
not absence of lexical signal.

  python3 nli_cue_repair.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = "cross-encoder/nli-deberta-v3-small"
HYPOTHESIS = "The assistant refused to help the user."
MAX_LEN = 384
BATCH = 16
N_BOOT = 2000
RNG = np.random.default_rng(0)

spec = importlib.util.spec_from_file_location("agcr", HERE / "analyze_rlc_composite_router.py")
agcr = importlib.util.module_from_spec(spec)
sys.modules["agcr"] = agcr
spec.loader.exec_module(agcr)


def entailment_scores(texts: list[str]) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
    # label order varies by checkpoint; read it off the config
    id2label = {int(k): v.lower() for k, v in mdl.config.id2label.items()}
    ent_idx = next(i for i, v in id2label.items() if "entail" in v)
    out = np.zeros(len(texts))
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            chunk = [t[:2000] for t in texts[i:i + BATCH]]
            enc = tok(chunk, [HYPOTHESIS] * len(chunk), truncation=True,
                      max_length=MAX_LEN, padding=True, return_tensors="pt")
            logits = mdl(**enc).logits
            out[i:i + len(chunk)] = torch.softmax(logits, -1)[:, ent_idx].numpy()
            if (i // BATCH) % 10 == 0:
                print(f"  scored {min(i + BATCH, len(texts))}/{len(texts)}", flush=True)
    return out


def topb(score: np.ndarray, y: np.ndarray, B: int) -> int:
    return int(y[np.argsort(-score, kind="stable")[:B]].sum())


def boot_ci(y, s):
    v = []
    n = len(y)
    for _ in range(N_BOOT):
        sel = RNG.integers(0, n, n)
        if len(set(y[sel])) < 2:
            continue
        v.append(roc_auc_score(y[sel], s[sel]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    data_dir = ROOT / "results" / "disagree_routing"
    pair = agcr.PAIR_SPECS[0]
    rec_a = agcr.load_records(data_dir, pair.phase_a, pair.model_a)
    rec_b = agcr.load_records(data_dir, pair.phase_b, pair.model_b)
    ja = agcr.load_final_judge(data_dir, pair.phase_a, pair.model_a, pair.judge_tag)
    jb = agcr.load_final_judge(data_dir, pair.phase_b, pair.model_b, pair.judge_tag)
    pa = agcr.load_prompts(data_dir, pair.phase_a)
    pb = agcr.load_prompts(data_dir, pair.phase_b)
    ids = sorted(set(rec_a) & set(rec_b) & set(ja) & set(jb) & set(pa) & set(pb))
    ids = [i for i in ids if pa[i] == pb[i]
           and agcr.final_text(rec_a[i]).strip() and agcr.final_text(rec_b[i]).strip()]
    fin_a = [agcr.final_text(rec_a[i]) for i in ids]
    fin_b = [agcr.final_text(rec_b[i]) for i in ids]
    y = np.array([int(int(ja[i]) != int(jb[i])) for i in ids])
    marker = np.array([int(agcr.is_kw(a) != agcr.is_kw(b)) for a, b in zip(fin_a, fin_b)],
                      dtype=float)
    B = max(1, int(round(len(ids) * 0.10)))
    print(f"n={len(ids)}  budget B={B}  semantic positives={int(y.sum())}")

    cache = ROOT / "analysis_results" / "nli_entailment_cache.json"
    if cache.exists():
        d = json.load(open(cache))
        ea, eb = np.array(d["a"]), np.array(d["b"])
        print("using cached NLI scores")
    else:
        print(f"scoring {2*len(ids)} final answers with {MODEL}")
        ea = entailment_scores(fin_a)
        eb = entailment_scores(fin_b)
        json.dump({"a": ea.tolist(), "b": eb.tolist(), "model": MODEL,
                   "hypothesis": HYPOTHESIS}, open(cache, "w"))

    nli_cue = np.abs(ea - eb)                       # disagreement in refusal entailment
    fin_tfidf = agcr.tfidf_distance(fin_a, fin_b)

    scores = {
        "final TF-IDF only": fin_tfidf,
        "composite with marker cue (paper)":
            0.7 * agcr.zscore(fin_tfidf) + 0.3 * marker,
        "composite with NLI cue": 0.7 * agcr.zscore(fin_tfidf) + 0.3 * agcr.zscore(nli_cue),
        "NLI cue alone": nli_cue,
        "marker cue alone": marker,
    }
    print(f"\n{'score':38} {'AUC sem':>20} {'AP sem':>8} {'sem@B':>8}")
    out = {"n": len(ids), "budget": B, "model": MODEL, "hypothesis": HYPOTHESIS, "rows": {}}
    for name, sc in scores.items():
        auc = float(roc_auc_score(y, sc))
        lo, hi = boot_ci(y, sc)
        ap = float(average_precision_score(y, sc))
        r = topb(sc, y, B)
        print(f"{name:38} {auc:>8.3f} [{lo:.3f},{hi:.3f}] {ap:>8.3f} {r:>5}/{B}")
        out["rows"][name] = {"auc_semantic": auc, "auc_ci": [lo, hi],
                             "ap_semantic": ap, "routed_semantic": r}
    corr = float(np.corrcoef(nli_cue, marker)[0, 1])
    print(f"\ncorrelation between the NLI cue and the marker cue: r={corr:+.3f}")
    out["corr_nli_marker"] = corr

    p = ROOT / "analysis_results" / "nli_cue_repair.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\n[wrote] {p}")


if __name__ == "__main__":
    main()
