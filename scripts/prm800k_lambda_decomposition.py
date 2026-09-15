#!/usr/bin/env python3
"""How much of a PRM's reported AUC is about the question best-of-N asks? (2026-08-22)

A process reward model is deployed inside a single problem: given N sampled
solutions to *one* problem, return the one the PRM scores highest. The quantity
that governs that decision is the model's ability to order samples *within a
problem*. But PRM quality is reported as an AUC pooled over every sample in the
test set, and the Mann--Whitney form of AUC partitions its pairs exactly:

    A_pooled = lambda * A_within + (1 - lambda) * A_cross,
    lambda   = sum_q n+_q n-_q / (n+ n-)

Cross-problem pairs -- "does this step of problem 17 score above that step of
problem 249?" -- are a comparison best-of-N never makes. With Q problems of
comparable size lambda is about 1/Q, so on a 500-problem test set the reported
number answers the deployed question with weight near one five-hundredth.

We compute the decomposition exactly on OpenAI's released scored test samples
for "Let's Verify Step by Step" (Lightman et al., 2023): 815,632 scored samples
over 500 MATH problems, with per-step rating distributions, the released
solution score, an ORM comparator, and sympy-graded correctness.

The null is the same within-stratum permutation used elsewhere in this work:
permute correctness inside each problem with the scores held fixed. Because the
scores are fixed, their ranks are fixed, and the pooled AUC under that null has
an exact expectation,

    E[A_pooled | null] = (sum_q n+_q * mean_rank_q - n+(n+ +1)/2) / (n+ n-),

which we report alongside a simulated interval.

  python3 prm800k_lambda_decomposition.py [path/to/scored-test-samples.jsonl]
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    ROOT / "results" / "prm800k_audit" / "scored-test-samples.jsonl")
CACHE = Path("/tmp/prm800k_lambda_cache.pkl")
OUT = ROOT / "analysis_results" / "prm800k_lambda_decomposition.json"
SPANS = [0.25, 0.50, 0.75, 1.00]
N_PERM = 300
RNG = np.random.default_rng(0)


def parse():
    if CACHE.exists():
        print(f"[cache] {CACHE}")
        return pickle.load(open(CACHE, "rb"))
    d = defaultdict(lambda: defaultdict(list))
    n = 0
    with open(DATA) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ps = [float(x.get("1", x.get(1, 0.0))) for x in r["rating_probs"]]
            if not ps:
                continue
            ps = np.array(ps, dtype=float)
            L = len(ps)
            pid = r.get("unique_id") or r["problem"][:80]
            g = d[pid]
            g["y"].append(int(bool(r["is_correct"])))
            g["released"].append(float(r["prm_score"]))
            g["orm"].append(float(r["orm_score"]) if r.get("orm_score") is not None else np.nan)
            g["first"].append(ps[0])
            for fr in SPANS:
                k = max(1, int(np.ceil(fr * L)))
                g[f"s{fr}"].append(float(np.prod(ps[:k])))
            n += 1
            if n % 100000 == 0:
                print(f"  ...{n} rows", flush=True)
    out = {pid: {k: np.array(v, dtype=float) for k, v in g.items()} for pid, g in d.items()}
    pickle.dump(out, open(CACHE, "wb"))
    print(f"parsed {n} rows over {len(out)} problems -> {CACHE}")
    return out


def decompose(y, s, b):
    """Exact split of the pooled Mann--Whitney statistic into within- and
    cross-stratum terms. Per-stratum AUCs are computed from ranks, so the whole
    thing is O(n log n) rather than O(n^2) over 4e8 pairs."""
    ok = ~np.isnan(s)
    y, s, b = y[ok].astype(int), s[ok], b[ok]
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return None
    R = rankdata(s)
    pooled = (R[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)

    w_num = w_den = 0.0
    for t in np.unique(b):
        m = b == t
        yy, ss = y[m], s[m]
        p, q = int(yy.sum()), int((1 - yy).sum())
        if p == 0 or q == 0:
            continue
        r = rankdata(ss)
        a = (r[yy == 1].sum() - p * (p + 1) / 2) / (p * q)
        w_num += p * q * a
        w_den += p * q
    lam = w_den / (npos * nneg)
    a_within = w_num / w_den
    a_cross = ((pooled * npos * nneg) - w_num) / (npos * nneg - w_den)

    # exact null mean for the pooled statistic under within-problem permutation
    exp_pos_rank = sum(int(y[b == t].sum()) * R[b == t].mean() for t in np.unique(b))
    null_mean = (exp_pos_rank - npos * (npos + 1) / 2) / (npos * nneg)

    # simulated null interval: reassign which rows in each problem are positive
    idx_by = [np.where(b == t)[0] for t in np.unique(b)]
    cnt = [int(y[i].sum()) for i in idx_by]
    draws = []
    for _ in range(N_PERM):
        tot = 0.0
        for ix, c in zip(idx_by, cnt):
            if c:
                tot += R[RNG.choice(ix, size=c, replace=False)].sum()
        draws.append((tot - npos * (npos + 1) / 2) / (npos * nneg))
    draws = np.array(draws)
    return dict(
        n=int(len(y)), n_pos=npos, n_strata=int(len(np.unique(b))),
        lam=round(float(lam), 5),
        pooled=round(float(pooled), 4),
        within=round(float(a_within), 4),
        cross=round(float(a_cross), 4),
        recon=round(float(lam * a_within + (1 - lam) * a_cross), 4),
        pooled_null_mean=round(float(null_mean), 4),
        pooled_null_ci=[round(float(np.percentile(draws, 2.5)), 4),
                        round(float(np.percentile(draws, 97.5)), 4)],
    )


def main() -> None:
    data = parse()
    pids = sorted(data)
    b = np.concatenate([np.full(len(data[p]["y"]), i) for i, p in enumerate(pids)])
    y = np.concatenate([data[p]["y"] for p in pids])
    scores = {}
    for key, label in [("first", "step 1 only"), ("s0.25", "prefix 25%"),
                       ("s0.5", "prefix 50%"), ("s0.75", "prefix 75%"),
                       ("s1.0", "full product"), ("released", "released prm_score"),
                       ("orm", "ORM comparator")]:
        if all(key in data[p] for p in pids):
            scores[label] = np.concatenate([data[p][key] for p in pids])

    print(f"\n{len(y)} samples, {len(pids)} problems, base rate {y.mean():.3f}\n")
    print(f"{'score':22s} {'lambda':>8s} {'pooled':>7s} {'within':>7s} {'cross':>7s} "
          f"{'recon':>7s} {'null mean':>9s} {'null 95%':>18s}")
    res = {}
    for label, s in scores.items():
        r = decompose(y, s, b)
        if r is None:
            continue
        res[label] = r
        ci = f"[{r['pooled_null_ci'][0]:.3f},{r['pooled_null_ci'][1]:.3f}]"
        print(f"{label:22s} {r['lam']:8.5f} {r['pooled']:7.3f} {r['within']:7.3f} "
              f"{r['cross']:7.3f} {r['recon']:7.3f} {r['pooled_null_mean']:9.3f} {ci:>18s}")

    print("\nreading:")
    k = "released prm_score" if "released prm_score" in res else list(res)[0]
    r = res[k]
    print(f"  The deployed question is within-problem: best-of-N never compares a sample")
    print(f"  of one problem against a sample of another. The reported statistic gives")
    print(f"  that question weight lambda = {r['lam']:.5f} (about 1/{round(1/r['lam'])}).")
    print(f"  On '{k}': pooled {r['pooled']:.3f}, of which the within-problem term is")
    print(f"  {r['within']:.3f} and the cross-problem term -- never exercised at")
    print(f"  deployment -- is {r['cross']:.3f}.")
    print(f"  Under the exact within-problem null the pooled statistic sits at "
          f"{r['pooled_null_mean']:.3f}, not at 0.500.")

    OUT.write_text(json.dumps({"n_perm": N_PERM, "scores": res}, indent=2))
    print("\n[wrote]", OUT)


if __name__ == "__main__":
    main()
