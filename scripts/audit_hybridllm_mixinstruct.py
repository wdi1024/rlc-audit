#!/usr/bin/env python3
"""External RLC-Audit of a surveyed system's routing target: Hybrid LLM on MixInstruct.

Hybrid LLM (Ding et al., ICLR 2024) defines its routing target as the BARTScore
quality gap H(x) = q(S(x)) - q(L(x)) and trains/validates a router against it.
The semantic construct the router is sold as serving is "would the small model's
answer be good enough here?"  Those are different objects, and MixInstruct
releases both: per-candidate `bartscore` (the proxy the system uses) and
`cmp_results`, pairwise adequacy judgments elicited separately from ChatGPT.

This audits the L2 link of the score--proxy--construct contract on released
artifacts alone -- no generation, no new judge calls:

    z_i  = 1 if bartscore(A_i) > bartscore(B_i)          (the deployed proxy)
    y_i  = 1 if the independent pairwise judgment prefers A  (the construct)

and reports agreement, kappa, and how often a router optimised for z would send
the query to the model the independent judgment considers worse.

  python3 audit_hybridllm_mixinstruct.py
"""
from __future__ import annotations

import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import cohen_kappa_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
N_BOOT = 2000
RNG = np.random.default_rng(0)
MIN_PAIR_N = 200          # only report pairs with enough jointly labelled rows


def main():
    ds = load_dataset("llm-blender/mix-instruct", split="test")
    print(f"MixInstruct test rows: {len(ds)}")

    # collect per-pair (bartscore gap, independent verdict)
    per_pair: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
    verdicts = Counter()
    for row in ds:
        bs = {c["model"]: c["scores"]["bartscore"] for c in row["candidates"]
              if c["scores"].get("bartscore") is not None}
        txt = {c["model"]: c["text"] or "" for c in row["candidates"]}
        try:
            cmp = json.loads(row["cmp_results"]) if row["cmp_results"] else {}
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(cmp, dict):
            continue
        for key, verdict in cmp.items():
            verdicts[verdict] += 1
            if "," not in key:
                continue
            a, b = key.split(",", 1)
            if a not in bs or b not in bs or a not in txt or b not in txt:
                continue
            if verdict == "A is better":
                y = 1
            elif verdict == "B is better":
                y = 0
            else:
                continue                      # ties carry no direction
            per_pair[(a, b)].append((bs[a] - bs[b], y, len(txt[a]) - len(txt[b])))

    print("verdict distribution:", dict(verdicts))
    print(f"model pairs with directional judgments: {len(per_pair)}\n")

    rows = []
    pooled_gap, pooled_y, pooled_len = [], [], []
    for (a, b), items in sorted(per_pair.items(), key=lambda kv: -len(kv[1])):
        if len(items) < MIN_PAIR_N:
            continue
        gap = np.array([g for g, _, _ in items], dtype=float)
        y = np.array([v for _, v, _ in items], dtype=int)
        ln = np.array([d for _, _, d in items], dtype=float)
        z = (gap > 0).astype(int)             # what the deployed proxy says
        if len(set(y)) < 2 or len(set(z)) < 2:
            continue
        agree = float((z == y).mean())
        kappa = float(cohen_kappa_score(z, y))
        auc = float(roc_auc_score(y, gap))    # does the proxy gap rank the construct?
        rows.append({"pair": f"{a} vs {b}", "n": len(items),
                     "agreement": round(agree, 3), "kappa": round(kappa, 3),
                     "auc_gap_vs_construct": round(auc, 3),
                     "prevalence_y": round(float(y.mean()), 3)})
        pooled_gap.append(gap)
        pooled_y.append(y)
        pooled_len.append(ln)

    print(f"{'pair':52} {'n':>5} {'agree':>6} {'kappa':>7} {'AUC':>6}")
    for r in rows:
        print(f"{r['pair'][:50]:52} {r['n']:>5} {r['agreement']:>6.3f} "
              f"{r['kappa']:>7.3f} {r['auc_gap_vs_construct']:>6.3f}")

    gap = np.concatenate(pooled_gap)
    y = np.concatenate(pooled_y)
    z = (gap > 0).astype(int)
    k = float(cohen_kappa_score(z, y))
    auc = float(roc_auc_score(y, gap))
    idx = RNG.integers(0, len(y), size=(N_BOOT, len(y)))
    kb = np.array([cohen_kappa_score(z[i], y[i]) for i in idx[:200]])
    lo, hi = np.percentile(kb, [2.5, 97.5])
    print(f"\npooled  n={len(y)}  agreement={float((z==y).mean()):.3f}  "
          f"kappa={k:.3f} [{lo:.3f}, {hi:.3f}]  AUC(gap vs construct)={auc:.3f}")
    print(f"the proxy sends the query to the independently dispreferred model in "
          f"{float((z != y).mean())*100:.1f}% of directional cases")

    # Artifact check: is the deployed proxy more surface-coupled than the construct?
    lg = np.concatenate(pooled_len)
    r_proxy = float(np.corrcoef(gap, lg)[0, 1])
    r_constr = float(np.corrcoef(y.astype(float), lg)[0, 1])
    auc_len_z = float(roc_auc_score(z, lg))
    auc_len_y = float(roc_auc_score(y, lg))
    print(f"\nlength-difference coupling: proxy r={r_proxy:+.3f} vs construct r={r_constr:+.3f}; "
          f"length alone predicts the proxy at AUC {auc_len_z:.3f} and the construct at {auc_len_y:.3f}")

    out = {"n_pairs": len(rows),
           "artifact": {"r_proxy_length": r_proxy, "r_construct_length": r_constr,
                        "auc_length_vs_proxy": auc_len_z, "auc_length_vs_construct": auc_len_y},
           "pooled": {"n": int(len(y)), "kappa": k,
           "kappa_ci": [float(lo), float(hi)], "auc": auc,
           "disagreement_rate": float((z != y).mean())}, "pairs": rows,
           "verdict_distribution": dict(verdicts)}
    p = ROOT / "analysis_results" / "hybridllm_mixinstruct_audit.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\n[wrote] {p}")


if __name__ == "__main__":
    main()
