#!/usr/bin/env python3
"""A deployed routing target, audited with the containment controls. (2026-08-21)

Every contract the containment control has run on so far is one we built, and the
paper says so. The reason is a release gap: the control needs per-example
generations, and the sampled routing papers do not publish them. RouteLLM
\\citep{ong2025routellm} is an exception we found outside that sample. Its released
`routellm/gpt4_dataset` carries, for 119{,}101 prompts, both model responses in full
and the GPT-4 judge score its routing target is defined from.

That makes one contract instantiable that is not ours:

  score span   a prefix of the two released responses
  score        prefix TF-IDF cosine distance between them, the configuration
               Section 3.1 of the paper finds deployed in test-time search
  proxy        a same-prefix surface rule (shared opening / length agreement)
  construct    RouteLLM's own routing target: the weak model's response judged
               inadequate, mixtral_score <= 3 on the released 1-5 scale
  pair         gpt-4-1106-preview vs mixtral-8x7b-instruct-v0.1
  budget       10% of the audited rows

Two caveats stated before the numbers. First, RouteLLM's own router scores the
*query*, not either response, so it cannot exhibit containment by construction; what
we audit here is a partial-span contract over its released artifacts and target, not
its router. Second, the construct is itself an automatic judge score, so this
inherits the caveat of Section 5.3 rather than escaping it.

  python3 audit_routellm_deployed.py [--n 20000]
"""
from __future__ import annotations

import argparse
import io
import json
import re
import ssl
import urllib.request
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path("/tmp/routellm_gpt4_dataset.parquet")
URL = ("https://huggingface.co/api/datasets/routellm/gpt4_dataset"
       "/parquet/default/train/0.parquet")
CTX = ssl.create_default_context(cafile=certifi.where())
UA = {"User-Agent": "Mozilla/5.0 (research audit)"}

SPAN = 200          # characters of each response the score reads
BUDGET_FRAC = 0.10
N_BOOT = 2000
MIN_POSITIVES = 10

# A same-span surface rule of the kind a cheap validator would use: do the two
# responses open in the same register? Implemented as agreement on three
# format-level features visible in the first SPAN characters.
LIST_RE = re.compile(r"(^|\n)\s*(\d+[.)]|[-*•])\s+")
CODE_RE = re.compile(r"```|\bdef\b|\bclass\b|;\s*$", re.M)


def surface_signature(text: str) -> tuple:
    return (bool(LIST_RE.search(text)), bool(CODE_RE.search(text)),
            len(text.split()) > 40)


def fetch() -> pd.DataFrame:
    if not CACHE.exists():
        print(f"downloading {URL} ...", flush=True)
        req = urllib.request.Request(URL, headers=UA)
        with urllib.request.urlopen(req, timeout=600, context=CTX) as r:
            CACHE.write_bytes(r.read())
    print(f"reading {CACHE} ({CACHE.stat().st_size/1e6:.0f} MB)", flush=True)
    return pd.read_parquet(CACHE,
                           columns=["prompt", "gpt4_response",
                                    "mixtral_response", "mixtral_score"])


def boot_ci(lab, y, s, seed, orient=False):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(N_BOOT):
        sel = rng.integers(0, len(s), len(s))
        if len(set(lab[sel])) < 2 or len(set(y[sel])) < 2:
            continue
        az, ay = roc_auc_score(lab[sel], s[sel]), roc_auc_score(y[sel], s[sel])
        out.append((abs(az - .5) - abs(ay - .5)) if orient else (az - ay))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()

    df = fetch()
    df = df.dropna(subset=["gpt4_response", "mixtral_response", "mixtral_score"])
    print(f"rows with both responses and a judge score: {len(df)}")
    if len(df) > a.n:
        df = df.sample(n=a.n, random_state=0).reset_index(drop=True)
        print(f"fixed-seed subsample: {len(df)}")

    A = df["gpt4_response"].astype(str).tolist()
    B = df["mixtral_response"].astype(str).tolist()
    score_raw = pd.to_numeric(df["mixtral_score"], errors="coerce")
    keep = score_raw.notna().values
    A = [x for x, k in zip(A, keep) if k]
    B = [x for x, k in zip(B, keep) if k]
    sc = score_raw[keep].values

    # construct: RouteLLM's routing target -- the weak model was judged inadequate
    y = (sc <= 3).astype(int)

    pa, pb = [x[:SPAN] for x in A], [x[:SPAN] for x in B]
    ca, cb = [x[SPAN:] for x in A], [x[SPAN:] for x in B]

    def cos(u, v):
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit(u + v)
        xu, xv = vec.transform(u), vec.transform(v)
        num = np.asarray(xu.multiply(xv).sum(1)).ravel()
        nu = np.sqrt(np.asarray(xu.multiply(xu).sum(1)).ravel())
        nv = np.sqrt(np.asarray(xv.multiply(xv).sum(1)).ravel())
        return 1 - num / np.maximum(nu * nv, 1e-9)

    s = cos(pa, pb)

    z = np.array([int(surface_signature(x) != surface_signature(w))
                  for x, w in zip(pa, pb)])
    z_ext = np.array([int(surface_signature(x) != surface_signature(w))
                      for x, w in zip(A, B)])
    z_dis = np.array([int(surface_signature(x) != surface_signature(w))
                      for x, w in zip(ca, cb)])
    empty = float(np.mean([len(x) <= SPAN for x in A + B]))

    n = len(s)
    print(f"\nn={n}   construct positives (mixtral judged inadequate) = {int(y.sum())} "
          f"({y.mean():.3f})")
    print(f"proxy positives: same-span {int(z.sum())}, extended {int(z_ext.sum())}, "
          f"disjoint {int(z_dis.sum())};  complement empty on {empty:.3f} of sides")

    if int(y.sum()) < MIN_POSITIVES or n - int(y.sum()) < MIN_POSITIVES:
        print("contract fails the power precondition; stopping.")
        return

    auc_y = float(roc_auc_score(y, s))
    rows = {}
    for name, lab in (("z (same span)", z), ("z+ (whole output)", z_ext),
                      ("z^c (off-span)", z_dis)):
        if len(set(lab)) < 2:
            rows[name] = None
            continue
        auc = float(roc_auc_score(lab, s))
        d_abs = abs(auc - .5) - abs(auc_y - .5)
        lo, hi = boot_ci(lab, y, s, abs(hash(name)) % (2**31), orient=True)
        rows[name] = {"auc": auc, "delta_abs": d_abs, "ci_abs": [lo, hi],
                      "prevalence": float(lab.mean())}

    print(f"\nAUC(s,y) against the deployed target = {auc_y:.3f}")
    print(f"{'target':22s} {'AUC(s,.)':>9s} {'prev':>7s} {'delta_abs':>10s}  95% CI")
    for k, v in rows.items():
        if v is None:
            print(f"{k:22s} {'degenerate':>9s}")
            continue
        print(f"{k:22s} {v['auc']:9.3f} {v['prevalence']:7.3f} {v['delta_abs']:+10.3f}  "
              f"[{v['ci_abs'][0]:+.3f},{v['ci_abs'][1]:+.3f}]")

    B_ = max(1, int(BUDGET_FRAC * n))
    routed = int(y[np.argsort(-s, kind="stable")[:B_]].sum())
    print(f"\nat a {BUDGET_FRAC:.0%} budget ({B_} slots) the prefix score routes "
          f"{routed} genuinely inadequate responses "
          f"({routed/B_:.3f} vs base rate {y.mean():.3f})")

    same = rows.get("z (same span)")
    disj = rows.get("z^c (off-span)")
    print("\nreading:")
    if same and same["ci_abs"][0] > 0 and disj and disj["ci_abs"][0] <= 0:
        print("  the same-span certificate is real and does not survive the disjoint control:")
        print("  containment on a deployed target's released artifacts, not only on ours.")
    elif same and same["ci_abs"][0] <= 0:
        print("  no same-span gap to explain: the prefix score tracks the deployed target about")
        print("  as well as it tracks a same-span surface rule, and the contract clears.")
    else:
        print("  report the three columns; the pattern is not one of the clean cases.")

    out = {"dataset": "routellm/gpt4_dataset", "n": n, "span": SPAN,
           "construct": "mixtral_score <= 3 (RouteLLM routing target)",
           "y_positives": int(y.sum()), "auc_y": auc_y,
           "budget_frac": BUDGET_FRAC, "routed_inadequate": routed,
           "empty_complement_share": empty, "targets": rows}
    p = ROOT / "analysis_results" / "routellm_deployed_audit.json"
    p.write_text(json.dumps(out, indent=2))
    print("\n[wrote]", p)


if __name__ == "__main__":
    main()
