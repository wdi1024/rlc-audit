#!/usr/bin/env python3
"""OR-Bench Sonnet judge sensitivity check (Tier B1).

Recomputes the headline (sentence × judge) AUC and diagnostic d on OR-Bench-
Hard-1K using Claude-Sonnet-4-6 instead of Claude-Haiku-4-5. Tests whether the
(ii) failure on OR-Bench is judge-invariant or judge-specific.
"""
import json
import warnings

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score

warnings.filterwarnings("ignore")

RDIR = "results/disagree_routing"


def load_judge(path):
    data = json.load(open(path))
    return {r["id"]: r["refusal_judge"] for r in data["records"] if r.get("refusal_judge") is not None}


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
    pooled = np.sqrt((s_a.var(ddof=1) * (len(s_a) - 1) + s_d.var(ddof=1) * (len(s_d) - 1)) / (len(s_a) + len(s_d) - 2))
    if pooled == 0:
        return None
    return float((s_a.mean() - s_d.mean()) / pooled)


def main():
    print("=" * 78)
    print("OR-Bench Sonnet judge sensitivity (Tier B1)")
    print("=" * 78)

    qwen = json.load(open(f"{RDIR}/phase8_orbench_hard1k_traces_qwen3.5-2b.json"))["records"]
    gemma = json.load(open(f"{RDIR}/phase8_orbench_hard1k_traces_gemma-4-e2b.json"))["records"]
    qm = {r["id"]: r for r in qwen}
    gm = {r["id"]: r for r in gemma}

    haiku_q = load_judge(f"{RDIR}/phase8_orbench_hard1k_judge_qwen3.5-2b_anthropic_claude-haiku-4-5-20251001.json")
    haiku_g = load_judge(f"{RDIR}/phase8_orbench_hard1k_judge_gemma-4-e2b_anthropic_claude-haiku-4-5-20251001.json")
    sonnet_q = load_judge(f"{RDIR}/phase8_orbench_hard1k_judge_qwen3.5-2b_anthropic_claude-sonnet-4-6.json")
    sonnet_g = load_judge(f"{RDIR}/phase8_orbench_hard1k_judge_gemma-4-e2b_anthropic_claude-sonnet-4-6.json")

    common = sorted(set(qm) & set(gm) & set(haiku_q) & set(haiku_g) & set(sonnet_q) & set(sonnet_g))
    print(f"\n[1/3] Common ids (across both SLMs and both judges): {len(common)}")

    # Cross-judge agreement
    h_q_arr = [haiku_q[i] for i in common]
    s_q_arr = [sonnet_q[i] for i in common]
    h_g_arr = [haiku_g[i] for i in common]
    s_g_arr = [sonnet_g[i] for i in common]
    kappa_q = cohen_kappa_score(h_q_arr, s_q_arr)
    kappa_g = cohen_kappa_score(h_g_arr, s_g_arr)
    print(f"  Haiku-Sonnet kappa: Qwen {kappa_q:.3f}, Gemma {kappa_g:.3f}")
    print(f"  Avg kappa: {(kappa_q + kappa_g)/2:.3f}")

    # Per-judge stats
    print(f"\n  Refusal rates:")
    print(f"    Qwen:  haiku {np.mean(h_q_arr)*100:.1f}%, sonnet {np.mean(s_q_arr)*100:.1f}%")
    print(f"    Gemma: haiku {np.mean(h_g_arr)*100:.1f}%, sonnet {np.mean(s_g_arr)*100:.1f}%")

    # Compute sentence similarity once
    print("\n[2/3] Computing sentence similarity (MiniLM-L12)...")
    sent_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
    qwen_traces = [qm[i].get("trace") or "" for i in common]
    gemma_traces = [gm[i].get("trace") or "" for i in common]
    ea = sent_model.encode(qwen_traces, show_progress_bar=False, normalize_embeddings=True)
    eb = sent_model.encode(gemma_traces, show_progress_bar=False, normalize_embeddings=True)
    sims = (ea * eb).sum(axis=1)

    # Per-judge headline analysis
    print("\n[3/3] Headline (Sentence × judge) AUC + sep_d under each judge:")
    print(f"\n  {'Judge':10s} {'n_dis':>6s} {'rate':>7s} {'AUC':>7s} {'95% CI':>22s} {'sep_d':>8s}")

    results = {}
    for judge_name, qj_arr, gj_arr in [
        ("haiku", h_q_arr, h_g_arr),
        ("sonnet", s_q_arr, s_g_arr),
    ]:
        qj = np.array(qj_arr, dtype=int)
        gj = np.array(gj_arr, dtype=int)
        y = (qj != gj).astype(int)
        n_dis = int(y.sum())
        rate = float(y.mean())
        auc, ci = auc_with_ci(y, -sims)
        d = cohens_d(sims, y)
        results[judge_name] = dict(
            n_dis=n_dis, rate=rate, auc=auc, ci=ci, sep_d=d,
        )
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        print(f"  {judge_name:10s} {n_dis:>6d} {rate*100:>6.1f}% {auc:>7.3f} {ci_str:>22s} {d:>8.3f}")

    # Save
    out = dict(
        kappa_qwen=float(kappa_q),
        kappa_gemma=float(kappa_g),
        avg_kappa=float((kappa_q + kappa_g) / 2),
        haiku_qwen_refusal=float(np.mean(h_q_arr)),
        haiku_gemma_refusal=float(np.mean(h_g_arr)),
        sonnet_qwen_refusal=float(np.mean(s_q_arr)),
        sonnet_gemma_refusal=float(np.mean(s_g_arr)),
        results_per_judge=results,
        n_common=len(common),
    )
    out_path = f"{RDIR}/orbench_sonnet_robustness.json"
    json.dump(out, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    # Verdict
    print("\n" + "=" * 78)
    print("VERDICT — Is OR-Bench (ii) failure judge-invariant?")
    print("=" * 78)
    h = results["haiku"]
    s = results["sonnet"]
    print(f"\n  Haiku:  AUC {h['auc']:.3f} [CI {h['ci'][0]:.3f}, {h['ci'][1]:.3f}], sep_d {h['sep_d']:.3f}")
    print(f"  Sonnet: AUC {s['auc']:.3f} [CI {s['ci'][0]:.3f}, {s['ci'][1]:.3f}], sep_d {s['sep_d']:.3f}")
    print()
    if abs(h['auc'] - s['auc']) < 0.05 and h['sep_d'] < 0.15 and s['sep_d'] < 0.15:
        print("→ Verdict: (ii) failure is judge-invariant. Both Haiku and Sonnet")
        print("  produce sep_d < 0.15 and AUC ≈ 0.5 → OR-Bench's (ii) failure is a")
        print("  structural property of the (benchmark × SLM-pair × encoder), not a")
        print("  judge artifact. Strong robustness evidence.")
    elif s['auc'] > 0.6 and s['sep_d'] > 0.15:
        print("→ Verdict: Sonnet recovers identifiability. (ii) failure may be")
        print("  judge-specific. Honest finding — paper claim should be qualified.")
    else:
        print("→ Verdict: mixed — both judges show some collapse but not identical.")


if __name__ == "__main__":
    main()
