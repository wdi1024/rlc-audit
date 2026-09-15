#!/usr/bin/env python3
"""Symmetric span intervention on the revised contract (reviewer request, 2026-08-13).

The primary contract is diagnosed by stripping model-specific openings with labels
held fixed: proxy alignment collapses (AUC 0.985 -> 0.645) while the semantic side
does not move, which is what makes the opening span artifact-bearing.

The revised contract deserves the same test, and a reviewer asked for it directly:
the composite score absorbs an observable final refusal-marker cue, and the semantic
label is also adjudicated on the final channel, so "why is the revision not itself
coupling?" is a fair question.  This script runs the mirror-image intervention --
remove refusal-marker sentences from the final span, hold all labels fixed, and
re-score -- and reports what survives.

The asymmetry is the result to look for:

  opening strip   -> proxy alignment collapses, semantic alignment does not move
                     (the stripped evidence is artifact-bearing)
  marker strip    -> if semantic alignment survives, the final span carries
                     construct-bearing evidence beyond the marker itself; if it
                     collapses, the repair is the marker and the paper should say so

Either outcome is reportable.  What is not acceptable is asserting the distinction
without measuring it.

  python3 marker_strip_symmetry.py
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
N_BOOT = 2000
RNG = np.random.default_rng(0)

spec = importlib.util.spec_from_file_location("agcr", HERE / "analyze_rlc_composite_router.py")
agcr = importlib.util.module_from_spec(spec)
sys.modules["agcr"] = agcr
spec.loader.exec_module(agcr)

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def strip_marker_sentences(text: str) -> str:
    """Drop every sentence containing a refusal keyword, keep the rest verbatim."""
    keep = [s for s in SENT_SPLIT.split(text or "") if not agcr.is_kw(s)]
    return " ".join(keep).strip()


def boot_ci(y: np.ndarray, score: np.ndarray, fn) -> tuple[float, float, float]:
    obs = float(fn(y, score))
    vals = []
    n = len(y)
    for _ in range(N_BOOT):
        sel = RNG.integers(0, n, n)
        if len(set(y[sel])) < 2:
            continue
        vals.append(fn(y[sel], score[sel]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return obs, float(lo), float(hi)


def topb(score: np.ndarray, y: np.ndarray, B: int) -> int:
    return int(y[np.argsort(-score, kind="stable")[:B]].sum())


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
    z = np.array([int(agcr.is_kw(a) != agcr.is_kw(b)) for a, b in zip(fin_a, fin_b)])
    B = max(1, int(round(len(ids) * 0.10)))

    # Labels are held fixed; only the text the score may read changes.
    str_a = [strip_marker_sentences(t) for t in fin_a]
    str_b = [strip_marker_sentences(t) for t in fin_b]
    kept = np.mean([len(s) / max(1, len(f)) for s, f in zip(str_a + str_b, fin_a + fin_b)])
    empt = np.mean([len(s.strip()) == 0 for s in str_a + str_b])
    print(f"n={len(ids)}  budget B={B}  semantic positives={int(y.sum())}  "
          f"proxy positives={int(z.sum())}")
    print(f"marker strip keeps {kept*100:.1f}% of characters; "
          f"{empt*100:.1f}% of responses become empty\n")

    # Volume control: the marker strip also removes text, so a matched amount of
    # randomly chosen sentences is dropped instead.  If the semantic loss is about
    # missing marker evidence rather than missing text, this control should not
    # reproduce it.  Averaged over seeds so one unlucky draw is not the result.
    def strip_random_matched(text: str, target_keep: float, rng) -> str:
        sents = [s for s in SENT_SPLIT.split(text or "") if s.strip()]
        if not sents:
            return text or ""
        order = rng.permutation(len(sents))
        total = sum(len(s) for s in sents)
        drop, budget = set(), (1.0 - target_keep) * total
        for j in order:
            if budget <= 0:
                break
            drop.add(int(j))
            budget -= len(sents[j])
        return " ".join(s for k, s in enumerate(sents) if k not in drop).strip()

    ctrl_auc, ctrl_routed = [], []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        ca = [strip_random_matched(t, len(s) / max(1, len(t)), rng)
              for t, s in zip(fin_a, str_a)]
        cb = [strip_random_matched(t, len(s) / max(1, len(t)), rng)
              for t, s in zip(fin_b, str_b)]
        sc = agcr.tfidf_distance(ca, cb)
        ctrl_auc.append(float(roc_auc_score(y, sc)))
        ctrl_routed.append(topb(sc, y, B))
    print(f"volume control (random sentences removed to the same character budget, "
          f"20 seeds): semantic AUC {np.mean(ctrl_auc):.3f} "
          f"[{np.percentile(ctrl_auc,5):.3f},{np.percentile(ctrl_auc,95):.3f}], "
          f"routed {np.mean(ctrl_routed):.1f}/{B}\n")

    marker = z.astype(float)
    scores = {
        "final TF-IDF": agcr.tfidf_distance(fin_a, fin_b),
        "composite (final TF-IDF + marker)":
            0.7 * agcr.zscore(agcr.tfidf_distance(fin_a, fin_b)) + 0.3 * marker,
        "final TF-IDF, markers stripped": agcr.tfidf_distance(str_a, str_b),
        "composite, markers stripped":
            0.7 * agcr.zscore(agcr.tfidf_distance(str_a, str_b)) + 0.3 * marker,
        "marker only": marker,
    }

    out = {"n": len(ids), "budget": B, "chars_kept": float(kept),
           "semantic_positives": int(y.sum()), "rows": {}}
    print(f"{'score':38} {'AUC proxy':>10} {'AUC sem':>18} {'AP sem':>8} {'sem@B':>7}")
    for name, sc in scores.items():
        auc_z = float(roc_auc_score(z, sc)) if len(set(z)) > 1 else float("nan")
        auc_y, lo, hi = boot_ci(y, sc, roc_auc_score)
        ap_y = float(average_precision_score(y, sc))
        n_at_b = topb(sc, y, B)
        print(f"{name:38} {auc_z:>10.3f} {auc_y:>8.3f} [{lo:.3f},{hi:.3f}] "
              f"{ap_y:>8.3f} {n_at_b:>4}/{B}")
        out["rows"][name] = {"auc_proxy": auc_z, "auc_semantic": auc_y,
                            "auc_semantic_ci": [lo, hi], "ap_semantic": ap_y,
                            "routed_semantic": n_at_b}

    # The asymmetry, stated as the two deltas the audit cares about.
    d_prox = out["rows"]["final TF-IDF"]["auc_proxy"] - \
        out["rows"]["final TF-IDF, markers stripped"]["auc_proxy"]
    d_sem = out["rows"]["final TF-IDF"]["auc_semantic"] - \
        out["rows"]["final TF-IDF, markers stripped"]["auc_semantic"]
    print(f"\nmarker strip on the final span: proxy AUC {d_prox:+.3f}, "
          f"semantic AUC {d_sem:+.3f}")
    print("compare the primary contract's opening strip: proxy 0.985 -> 0.645 "
          "(-0.340) with the semantic side unchanged")
    out["strip_deltas"] = {"proxy_auc_drop": d_prox, "semantic_auc_drop": d_sem}
    out["volume_control"] = {"auc_semantic_mean": float(np.mean(ctrl_auc)),
                             "auc_semantic_p5": float(np.percentile(ctrl_auc, 5)),
                             "auc_semantic_p95": float(np.percentile(ctrl_auc, 95)),
                             "routed_mean": float(np.mean(ctrl_routed)), "n_seeds": 20}

    p = ROOT / "analysis_results" / "marker_strip_symmetry.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\n[wrote] {p}")


if __name__ == "__main__":
    main()
