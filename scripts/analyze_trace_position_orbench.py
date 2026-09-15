#!/usr/bin/env python3
"""Experiment I — Trace-position mechanism on OR-Bench.

Tests the F4b (saturated semantic separation) mechanistic hypothesis: if the
OR-Bench (ii)-failure is caused by the SLM-pair traces sharing similar
reasoning content with divergence concentrated near the *end* of the trace
(i.e., the final refusal-vs-comply commitment is a small fraction of the
sentence-embedding average), then truncating to the *last* N characters
should recover higher Sentence × judge AUC and higher Cohen's d.

We measure on both OR-Bench-Hard-1K (where the test matters) and XSTest (as
a control benchmark where the headline already holds, to verify position
itself isn't a confound).

Truncations:
  - first-N: first 50 / 100 / 200 / 500 / full chars
  - last-N:  last  50 / 100 / 200 / 500 chars

Predicted outcomes:
  - (α) Last-N AUC on OR-Bench recovers to ≥ 0.60 → F4b mechanism confirmed:
        the late-divergence-masked-by-averaging story holds, and last-N
        becomes a practical workaround for OR-Bench-style benchmarks.
  - (β) Last-N AUC on OR-Bench stays at ≈ 0.53 → mechanism refuted; the
        F4b failure is not about averaging but about a deeper similarity in
        agree/disagree class trace content.

Output: results/disagree_routing/trace_position_orbench.json
"""
import json
import warnings

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score

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
    n_a, n_d = len(s_a), len(s_d)
    var_a, var_d = s_a.var(ddof=1), s_d.var(ddof=1)
    pooled = np.sqrt((var_a * (n_a - 1) + var_d * (n_d - 1)) / (n_a + n_d - 2))
    if pooled == 0:
        return None
    return float((s_a.mean() - s_d.mean()) / pooled)


def truncate(traces, mode, n):
    """mode in {first, last, full}. n is char budget."""
    out = []
    for t in traces:
        t = t or ""
        if mode == "full":
            out.append(t)
        elif mode == "first":
            out.append(t[:n])
        elif mode == "last":
            out.append(t[-n:] if len(t) > n else t)
        else:
            raise ValueError(mode)
    return out


def run_benchmark(name, q_path, g_path, qj_path, gj_path, sent_model):
    print(f"\n{'='*78}\n{name}\n{'='*78}")
    qwen = json.load(open(q_path))["records"]
    gemma = json.load(open(g_path))["records"]
    qj = load_judge(qj_path)
    gj = load_judge(gj_path)
    qm = {r["id"]: r for r in qwen}
    gm = {r["id"]: r for r in gemma}
    common = sorted(set(qm) & set(gm) & set(qj) & set(gj))

    qwen_traces = [qm[i].get("trace") or "" for i in common]
    gemma_traces = [gm[i].get("trace") or "" for i in common]
    q_jd = np.array([int(qj[i]) for i in common])
    g_jd = np.array([int(gj[i]) for i in common])
    y = (q_jd != g_jd).astype(int)

    print(f"  n={len(common)}, judge disagree rate {y.mean()*100:.1f}% ({y.sum()})")
    qlens = [len(t) for t in qwen_traces]
    glens = [len(t) for t in gemma_traces]
    print(f"  trace lengths: Q mean {np.mean(qlens):.0f} / Gemma mean {np.mean(glens):.0f} chars")

    SCHEDULE = [
        ("first", 50), ("first", 100), ("first", 200), ("first", 500), ("full", None),
        ("last", 50), ("last", 100), ("last", 200), ("last", 500),
    ]
    rows = []
    print(f"\n  {'mode':>6s} {'N':>5s} {'mean_qlen':>10s} {'AUC':>7s} {'CI':>22s} {'sep_d':>8s}")
    for mode, n in SCHEDULE:
        qt = truncate(qwen_traces, mode, n)
        gt = truncate(gemma_traces, mode, n)
        ea = sent_model.encode(qt, show_progress_bar=False, normalize_embeddings=True)
        eb = sent_model.encode(gt, show_progress_bar=False, normalize_embeddings=True)
        sims = (ea * eb).sum(axis=1)
        auc, ci = auc_with_ci(y, -sims)
        d = cohens_d(sims, y)
        mean_q = float(np.mean([len(t) for t in qt]))

        rows.append(dict(
            mode=mode, n=n,
            mean_qlen_after=mean_q,
            auc=float(auc) if auc is not None else None,
            ci=ci,
            sep_d=d,
            mu_agree=float(sims[y == 0].mean()) if (y == 0).any() else None,
            mu_disagree=float(sims[y == 1].mean()) if (y == 1).any() else None,
        ))
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—"
        d_str = f"{d:.3f}" if d is not None else "—"
        n_str = f"{n}" if n else "full"
        print(f"  {mode:>6s} {n_str:>5s} {mean_q:>10.0f} {auc:>7.3f} {ci_str:>22s} {d_str:>8s}")
    return rows


def main():
    print("=" * 78)
    print("Experiment I — Trace-position mechanism on OR-Bench")
    print("=" * 78)

    print("\nLoading sentence encoder...")
    sent_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

    out = {}
    out["OR-Bench-Hard-1K"] = run_benchmark(
        "OR-Bench-Hard-1K (target benchmark, F4b failure)",
        f"{RDIR}/phase8_orbench_hard1k_traces_qwen3.5-2b.json",
        f"{RDIR}/phase8_orbench_hard1k_traces_gemma-4-e2b.json",
        f"{RDIR}/phase8_orbench_hard1k_judge_qwen3.5-2b_anthropic_claude-haiku-4-5-20251001.json",
        f"{RDIR}/phase8_orbench_hard1k_judge_gemma-4-e2b_anthropic_claude-haiku-4-5-20251001.json",
        sent_model,
    )
    out["XSTest"] = run_benchmark(
        "XSTest (control — headline already holds at full trace)",
        f"{RDIR}/phase3_xstest_full_traces_qwen3.5-2b.json",
        f"{RDIR}/phase3_xstest_full_traces_gemma-4-e2b.json",
        f"{RDIR}/phase3_xstest_full_judge_qwen3.5-2b_anthropic_claude-haiku-4-5-20251001.json",
        f"{RDIR}/phase3_xstest_full_judge_gemma-4-e2b_anthropic_claude-haiku-4-5-20251001.json",
        sent_model,
    )

    out_path = f"{RDIR}/trace_position_orbench.json"
    json.dump(out, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    # ─────────────────────────────────────────────────────────────────
    # Verdict
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("VERDICT — Does last-N recover OR-Bench's headline AUC?")
    print("=" * 78)

    or_full = next((r for r in out["OR-Bench-Hard-1K"] if r["mode"] == "full"), None)
    or_last_50 = next((r for r in out["OR-Bench-Hard-1K"] if r["mode"] == "last" and r["n"] == 50), None)
    or_last_100 = next((r for r in out["OR-Bench-Hard-1K"] if r["mode"] == "last" and r["n"] == 100), None)
    or_last_200 = next((r for r in out["OR-Bench-Hard-1K"] if r["mode"] == "last" and r["n"] == 200), None)

    or_full_auc = or_full["auc"] if or_full else None
    or_l50 = or_last_50["auc"] if or_last_50 else None
    or_l100 = or_last_100["auc"] if or_last_100 else None
    or_l200 = or_last_200["auc"] if or_last_200 else None

    if or_full_auc and or_l100:
        gap = or_l100 - or_full_auc
        print(f"\nOR-Bench full AUC:        {or_full_auc:.3f}")
        print(f"OR-Bench last-100 AUC:    {or_l100:.3f}")
        print(f"OR-Bench last-50 AUC:     {or_l50:.3f}" if or_l50 else "")
        print(f"OR-Bench last-200 AUC:    {or_l200:.3f}" if or_l200 else "")
        print(f"Last-100 lift over full:  {gap:+.3f}")
        if gap > 0.05:
            print("→ Verdict α: F4b 'late-divergence-masked' mechanism CONFIRMED.")
            print("  Last-N truncation recovers signal on OR-Bench, suggesting the late")
            print("  refusal/comply token is informative but averaged out at full trace.")
        elif gap < -0.05:
            print("→ Verdict (unusual): Last-N WORSE than full. Mechanism more complex.")
        else:
            print("→ Verdict β: F4b mechanism NOT confirmed.")
            print("  Last-N gives the same AUC as full trace — averaging is not the issue.")
            print("  The agree/disagree classes share similarity at all trace positions.")

    print("\n" + "=" * 78)
    print("PAPER-READY STATEMENT (§5.3.4 mechanism note / §5.4.2 truncation)")
    print("=" * 78)
    statement = f"""
Trace-position mechanism for the F4b failure. We hypothesized in §5.3.4 that
OR-Bench's (ii)-failure mechanism is "late-divergence masked by averaging":
because the prompts share a uniform linguistic register, both SLMs produce
similar reasoning content for the bulk of the trace and only diverge at the
final refusal-vs-comply commitment, which sentence-level cosine averages
over the full trace cannot pick up. We test this directly by truncating
OR-Bench traces to the last N characters and re-computing the headline.

  Truncation        AUC      sep_d
  ----------------  -------  ------
  First-50          {next(r['auc'] for r in out['OR-Bench-Hard-1K'] if r['mode']=='first' and r['n']==50):.3f}     {next(r['sep_d'] for r in out['OR-Bench-Hard-1K'] if r['mode']=='first' and r['n']==50):.3f}
  First-100         {next(r['auc'] for r in out['OR-Bench-Hard-1K'] if r['mode']=='first' and r['n']==100):.3f}     {next(r['sep_d'] for r in out['OR-Bench-Hard-1K'] if r['mode']=='first' and r['n']==100):.3f}
  First-200         {next(r['auc'] for r in out['OR-Bench-Hard-1K'] if r['mode']=='first' and r['n']==200):.3f}     {next(r['sep_d'] for r in out['OR-Bench-Hard-1K'] if r['mode']=='first' and r['n']==200):.3f}
  Full trace        {or_full_auc:.3f}     {or_full['sep_d']:.3f}
  Last-50           {or_l50:.3f}     {next(r['sep_d'] for r in out['OR-Bench-Hard-1K'] if r['mode']=='last' and r['n']==50):.3f}
  Last-100          {or_l100:.3f}     {next(r['sep_d'] for r in out['OR-Bench-Hard-1K'] if r['mode']=='last' and r['n']==100):.3f}
  Last-200          {or_l200:.3f}     {next(r['sep_d'] for r in out['OR-Bench-Hard-1K'] if r['mode']=='last' and r['n']==200):.3f}
"""
    print(statement)


if __name__ == "__main__":
    main()
