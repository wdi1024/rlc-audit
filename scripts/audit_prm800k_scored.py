#!/usr/bin/env python3
"""External RLC-Audit of OpenAI's PRM best-of-N selection contract (PRM800K).

"Let's Verify Step by Step" (Lightman et al., 2023) selects the returned
answer by argmax of a process reward model's solution score over up to 1,860
samples per MATH test problem.  The released scored-test-samples.jsonl keeps,
for every sample, the PRM's per-step rating distribution (rating_probs), the
solution-level prm_score, an ORM comparator score, and independent
final-answer correctness (is_correct, sympy-graded) -- so the L2 link
(proxy vs construct) and its span structure are auditable from released
artifacts alone: no generation, no judge calls, no GPU.

For sample i of problem q with steps 1..L:
    p_j(i)  = P(rating=+1) at step j                (per-step proxy)
    s_k(i)  = aggregate of p_1..p_k                 (prefix-span proxy)
    y(i)    = is_correct                            (the construct)
    z(q)    = argmax_i s_L(i)                       (the deployed decision)

Reports: aggregation validation (which aggregate reproduces the released
prm_score), AUC(s_k, y) across span fractions (pooled and within-problem),
deployed best-of-N selection harm across N, ORM-vs-PRM contrast, and
surface coupling (length vs proxy vs construct).

  python3 audit_prm800k_scored.py [path/to/scored-test-samples.jsonl]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    ROOT / "results" / "prm800k_audit" / "scored-test-samples.jsonl"
)
OUT = ROOT / "analysis_results" / "prm800k_audit.json"
N_BOOT = 2000
RNG = np.random.default_rng(0)
SPAN_FRACS = [0.25, 0.50, 0.75, 1.00]
BON_NS = [4, 16, 64, 256, 1024]


def stream_records(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    # per-problem compact arrays
    probs: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    n_rows = 0
    agg_check = {"prod": [], "min": [], "mean": [], "last": [], "released": []}

    for r in stream_records(DATA):
        pid = r.get("unique_id") or r["problem"][:80]
        # per-step P(+1); tolerate both str and int keys
        ps = []
        for d in r["rating_probs"]:
            ps.append(float(d.get("1", d.get(1, 0.0))))
        if not ps:
            continue
        ps = np.array(ps, dtype=float)
        L = len(ps)
        d = probs[pid]
        d["y"].append(int(bool(r["is_correct"])))
        d["s_full_released"].append(float(r["prm_score"]))
        d["orm"].append(float(r["orm_score"]) if r.get("orm_score") is not None else np.nan)
        d["first"].append(ps[0])
        for f in SPAN_FRACS:
            k = max(1, int(np.ceil(f * L)))
            d[f"s{f}"].append(float(np.prod(ps[:k])))
        d["len_chars"].append(sum(len(s) for s in r["steps"]))
        d["n_steps"].append(L)
        if n_rows < 20000:
            agg_check["prod"].append(float(np.prod(ps)))
            agg_check["min"].append(float(ps.min()))
            agg_check["mean"].append(float(ps.mean()))
            agg_check["last"].append(float(ps[-1]))
            agg_check["released"].append(float(r["prm_score"]))
        n_rows += 1
        if n_rows % 100000 == 0:
            print(f"  ...{n_rows} rows")

    print(f"rows: {n_rows}, problems: {len(probs)}")
    res: dict = {"rows": n_rows, "problems": len(probs)}

    # which aggregation reproduces the released prm_score?
    rel = np.array(agg_check["released"])
    res["agg_validation"] = {
        k: float(np.corrcoef(np.array(v), rel)[0, 1])
        for k, v in agg_check.items()
        if k != "released"
    }
    res["agg_validation_max_abs_diff"] = {
        k: float(np.max(np.abs(np.array(v) - rel)))
        for k, v in agg_check.items()
        if k != "released"
    }
    print("aggregation vs released prm_score:", res["agg_validation"],
          "| max|diff|:", res["agg_validation_max_abs_diff"])

    # convert to arrays
    groups = []
    for pid, d in probs.items():
        groups.append(
            dict(
                pid=pid,
                y=np.array(d["y"], dtype=int),
                first=np.array(d["first"]),
                spans={f: np.array(d[f"s{f}"]) for f in SPAN_FRACS},
                released=np.array(d["s_full_released"]),
                orm=np.array(d["orm"]),
                lens=np.array(d["len_chars"], dtype=float),
                nsteps=np.array(d["n_steps"], dtype=float),
            )
        )

    all_y = np.concatenate([g["y"] for g in groups])
    res["base_rate"] = float(all_y.mean())

    def within(vals_key_fn):
        aucs = []
        for g in groups:
            y = g["y"]
            if 0 < y.sum() < len(y):
                v = vals_key_fn(g)
                ok = ~np.isnan(v)
                if 0 < y[ok].sum() < ok.sum():
                    aucs.append(roc_auc_score(y[ok], v[ok]))
        return float(np.mean(aucs)), len(aucs)

    res["auc"] = {}
    span_fns = {"first": lambda g: g["first"]}
    for f in SPAN_FRACS:
        span_fns[str(f)] = (lambda ff: (lambda g: g["spans"][ff]))(f)
    span_fns["released_full"] = lambda g: g["released"]
    span_fns["orm"] = lambda g: g["orm"]
    for key, fn in span_fns.items():
        v = np.concatenate([fn(g) for g in groups])
        ok = ~np.isnan(v)
        pooled = roc_auc_score(all_y[ok], v[ok])
        w, wn = within(fn)
        res["auc"][key] = dict(pooled=float(pooled), within=w, within_n=wn)
        print(f"AUC {key}: pooled {pooled:.3f}, within {w:.3f} (n={wn})")

    # deployed best-of-N selection harm, subsampled pools
    res["selection"] = {}
    for N in BON_NS + ["full"]:
        harms, sel_accs, orm_sel_accs = [], [], []
        for _ in range(200 if N != "full" else 1):
            wrong = solvable = 0
            sel_ok_n = orm_ok_n = tot = 0
            for g in groups:
                y, s, o = g["y"], g["released"], g["orm"]
                if N != "full":
                    if len(y) < N:
                        continue
                    idx = RNG.choice(len(y), N, replace=False)
                    y, s, o = y[idx], s[idx], o[idx]
                pick = int(np.argmax(s))
                tot += 1
                sel_ok_n += y[pick]
                if not np.isnan(o).any():
                    orm_ok_n += y[int(np.argmax(o))]
                if y.sum() > 0:
                    solvable += 1
                    wrong += 1 - y[pick]
            if solvable:
                harms.append(wrong / solvable)
                sel_accs.append(sel_ok_n / tot)
                orm_sel_accs.append(orm_ok_n / tot)
        if harms:
            res["selection"][str(N)] = dict(
                harm_rate=float(np.mean(harms)),
                harm_sd=float(np.std(harms)),
                prm_select_acc=float(np.mean(sel_accs)),
                orm_select_acc=float(np.mean(orm_sel_accs)),
            )
            print(f"BoN N={N}: harm {np.mean(harms):.3f}, "
                  f"prm-sel acc {np.mean(sel_accs):.3f}, orm-sel acc {np.mean(orm_sel_accs):.3f}")

    # surface coupling
    all_s = np.concatenate([g["released"] for g in groups])
    all_l = np.concatenate([g["lens"] for g in groups])
    all_n = np.concatenate([g["nsteps"] for g in groups])
    res["surface"] = dict(
        r_len_proxy=float(np.corrcoef(all_l, all_s)[0, 1]),
        r_len_construct=float(np.corrcoef(all_l, all_y)[0, 1]),
        r_steps_proxy=float(np.corrcoef(all_n, all_s)[0, 1]),
        r_steps_construct=float(np.corrcoef(all_n, all_y)[0, 1]),
        auc_len_construct_within=within(lambda g: g["lens"])[0],
    )
    print("surface:", res["surface"])

    OUT.write_text(json.dumps(res, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
