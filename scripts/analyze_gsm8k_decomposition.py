#!/usr/bin/env python3
"""4-cell decomposition + diagnostic d on GSM8K (Phase 11).

Tests whether the (i)+(ii) framework predicts identifiability outside the
safety-boundary regime AND beyond multi-hop QA. Same decomposition as
analyze_hotpot_decomposition.py, but with numeric-correctness label.

Outputs: analysis_results/gsm8k_decomposition_report.json
"""
import json
import os
import re
import warnings

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

RDIR = "data"
ODIR = "analysis_results"


def has_answer_keyword(trace, gold):
    """Loose 'gold number in trace' check — analog of refusal-keyword."""
    if not trace or gold is None or gold == "":
        return False
    g = str(gold).strip()
    if not g:
        return False
    # Allow comma-stripped, decimal-stripped, sign-preserved match
    g_norm = g.replace(",", "").rstrip("0").rstrip(".") or g
    pattern = re.compile(r"(?<![\d\.])" + re.escape(g_norm) + r"(?![\d\.])")
    return bool(pattern.search(trace))


def load_traces(slm):
    return json.load(open(f"{RDIR}/phase11_gsm8k_traces_{slm}.json"))["records"]


def load_correct_judge(slm):
    path = f"{RDIR}/phase11_gsm8k_correctness_{slm}_anthropic_claude-haiku-4-5-20251001.json"
    data = json.load(open(path))
    return {r["id"]: r["correct_judge"] for r in data["records"] if r.get("correct_judge") is not None}


def auc_with_ci(labels, scores, n_boot=2000, seed=42):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(set(labels)) < 2:
        return None, None
    point = roc_auc_score(labels, scores)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        if len(set(labels[idx])) < 2:
            continue
        try:
            boots.append(roc_auc_score(labels[idx], scores[idx]))
        except Exception:
            pass
    if not boots:
        return float(point), None
    boots = np.array(boots)
    ci = np.percentile(boots, [2.5, 97.5])
    return float(point), [float(ci[0]), float(ci[1])]


def cohens_d(scores, labels):
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    s_a = s[y == 0]
    s_d = s[y == 1]
    if len(s_d) < 2 or len(s_a) < 2:
        return None
    n_a, n_d = len(s_a), len(s_d)
    var_a, var_d = s_a.var(ddof=1), s_d.var(ddof=1)
    pooled = np.sqrt((var_a * (n_a - 1) + var_d * (n_d - 1)) / (n_a + n_d - 2))
    if pooled == 0:
        return None
    return float((s_a.mean() - s_d.mean()) / pooled)


def main():
    print("=" * 78)
    print("GSM8K 4-cell decomposition + diagnostic d (Phase 11)")
    print("=" * 78)

    print("\n[1/5] Loading data...")
    qwen = load_traces("qwen3.5-2b")
    gemma = load_traces("gemma-4-e2b")
    qj = load_correct_judge("qwen3.5-2b")
    gj = load_correct_judge("gemma-4-e2b")
    meta = json.load(open(f"{RDIR}/phase11_gsm8k_meta.json"))["prompts"]
    gold_map = {p["id"]: p.get("answer", "") for p in meta}

    qm = {r["id"]: r for r in qwen}
    gm = {r["id"]: r for r in gemma}
    common = sorted(set(qm) & set(gm) & set(qj) & set(gj))
    print(f"  common ids: {len(common)}")

    qwen_traces = [qm[i].get("trace") or "" for i in common]
    gemma_traces = [gm[i].get("trace") or "" for i in common]

    q_kw = np.array([int(has_answer_keyword(qm[i].get("trace"), gold_map.get(i, ""))) for i in common])
    g_kw = np.array([int(has_answer_keyword(gm[i].get("trace"), gold_map.get(i, ""))) for i in common])
    q_jd = np.array([int(qj[i]) for i in common])
    g_jd = np.array([int(gj[i]) for i in common])

    y_kw = (q_kw != g_kw).astype(int)
    y_judge = (q_jd != g_jd).astype(int)

    print(f"\n[2/5] Per-model accuracy:")
    print(f"  Qwen kw (gold-num-in-trace): {q_kw.mean()*100:.1f}%, judge correctness: {q_jd.mean()*100:.1f}%")
    print(f"  Gemma kw (gold-num-in-trace): {g_kw.mean()*100:.1f}%, judge correctness: {g_jd.mean()*100:.1f}%")
    print(f"  Disagreements: kw={int(y_kw.sum())}/{len(common)} ({y_kw.mean()*100:.1f}%), "
          f"judge={int(y_judge.sum())}/{len(common)} ({y_judge.mean()*100:.1f}%)")

    print("\n[3/5] Computing TF-IDF + optional sentence similarity scores...")
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    vec.fit(qwen_traces + gemma_traces)
    XA = vec.transform(qwen_traces)
    XB = vec.transform(gemma_traces)
    tfidf_sims = np.array([float(cosine_similarity(a, b)[0, 0]) for a, b in zip(XA, XB)])

    score_sets = [("TF-IDF", tfidf_sims)]
    try:
        sent_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
        ea = sent_model.encode(qwen_traces, show_progress_bar=False, normalize_embeddings=True)
        eb = sent_model.encode(gemma_traces, show_progress_bar=False, normalize_embeddings=True)
        score_sets.append(("Sentence-L12", (ea * eb).sum(axis=1)))
    except Exception as exc:
        print(f"  Skipping Sentence-L12: {exc}")

    print("\n[4/5] 4-cell decomposition + diagnostic d:")
    print(f"\n  {'Cell':30s} {'AUC':>7s} {'95% CI':>22s} {'sep_d':>8s} {'n_dis':>6s}")
    cells = {}
    for repr_name, sims in score_sets:
        for label_name, y in [("keyword", y_kw), ("judge", y_judge)]:
            score = -sims
            auc, ci = auc_with_ci(y, score)
            ap = float(average_precision_score(y, score))
            top50 = int(y[np.argsort(-score)[:50]].sum())
            d = cohens_d(sims, y)
            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—"
            d_str = f"{d:.3f}" if d is not None else "—"
            cells[f"{repr_name}_x_{label_name}"] = dict(
                auc=float(auc) if auc else None,
                ap=ap,
                ci=ci,
                sep_d=d,
                n_disagree=int(y.sum()),
                top50_disagreements=top50,
            )
            print(f"  {repr_name+' x '+label_name:30s} {auc:>7.3f} {ci_str:>22s} {d_str:>8s} {int(y.sum()):>6d}")

    tfidf_keyword = cells["TF-IDF_x_keyword"]
    tfidf_judge = cells["TF-IDF_x_judge"]
    out = dict(
        n_common=len(common),
        qwen_kw_acc=float(q_kw.mean()),
        gemma_kw_acc=float(g_kw.mean()),
        qwen_judge_acc=float(q_jd.mean()),
        gemma_judge_acc=float(g_jd.mean()),
        kw_disagree=int(y_kw.sum()),
        judge_disagree=int(y_judge.sum()),
        kappa_surface_semantic=float(cohen_kappa_score(y_kw, y_judge)),
        tfidf_rlc_audit=dict(
            status="ALIGNED",
            surface_proxy="gold-number-in-trace disagreement",
            semantic_construct="LLM-judge numeric-correctness disagreement",
            score_span="full trace",
            auc_surface=tfidf_keyword["auc"],
            auc_semantic=tfidf_judge["auc"],
            ap_surface=tfidf_keyword["ap"],
            ap_semantic=tfidf_judge["ap"],
            top50_surface_disagreements=tfidf_keyword["top50_disagreements"],
            top50_semantic_disagreements=tfidf_judge["top50_disagreements"],
        ),
        cells=cells,
    )
    os.makedirs(ODIR, exist_ok=True)
    out_path = f"{ODIR}/gsm8k_decomposition_report.json"
    json.dump(out, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    # ─────────────────────────────────────────────────────────────────
    # Verdict
    # ─────────────────────────────────────────────────────────────────
    print("\n[5/5] " + "=" * 70)
    print("VERDICT — Does (i)+(ii) framework predict identifiability on GSM8K?")
    print("=" * 78)

    headline_name = "Sentence-L12_x_judge" if "Sentence-L12_x_judge" in cells else "TF-IDF_x_judge"
    sj = cells[headline_name]
    auc = sj["auc"]
    sepd = sj["sep_d"]
    n_dis = sj["n_disagree"]

    print(f"\nHeadline ({headline_name.replace('_x_', ' x ')} correctness):")
    print(f"  n_disagreement: {n_dis}/{len(common)} = {n_dis/len(common)*100:.1f}%")
    print(f"  AUC: {auc:.3f} (CI {sj['ci']})")
    print(f"  sep_d: {sepd:.3f}")
    print()

    cond_i = n_dis >= 30
    cond_ii = sepd is not None and sepd > 0.15
    print(f"Condition (i) sufficient power (n_dis ≥ 30): {'✓' if cond_i else '✗'}")
    print(f"Condition (ii) separation (sep_d > 0.15): {'✓' if cond_ii else '✗'}")
    print()
    if cond_i and cond_ii and auc >= 0.6:
        print("→ Verdict α: (i)+(ii) condition satisfied AND AUC identifiable.")
        print("  Framework GENERALIZES from safety-boundary to math-reasoning correctness.")
        print("  Strong evidence — Direction C (Cohen's d pre-flight) pivot now clearly viable.")
    elif cond_i and not cond_ii:
        print("→ Verdict β: enough disagreement events but separation fails.")
        print("  GSM8K agree/disagree traces are not separable in MiniLM-L12 space.")
        print("  → Second consistent task-generalization fail-prediction (after HotpotQA).")
        print("  Framework correctly fail-predicts. Honest scope statement strengthened.")
    elif cond_i and cond_ii and auc < 0.6:
        print("→ Verdict γ: separation positive but AUC below 0.6.")
        print("  Diagnostic d caveat (necessary but not sufficient) directly demonstrated.")
    else:
        print("→ Verdict undetermined: insufficient power or borderline.")


if __name__ == "__main__":
    main()
