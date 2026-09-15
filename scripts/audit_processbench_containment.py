#!/usr/bin/env python3
"""The containment control with a human off-span proxy. (2026-08-21)

PRM800K and Math-Shepherd both clear, but in both the disjoint proxy is a rule we
wrote -- a gold-answer string match, a length threshold. A reader can reasonably say
that a surface rule is a weak instrument off-span, and that the clear might be the
instrument rather than the score. ProcessBench \\citep{zheng2024processbench} removes
that objection, because it releases a *human* span-local label: the index of the
first erroneous step in each solution.

That label is the disjoint proxy the paper has wanted throughout. Split at step k:

  score        Qwen2.5-Math-PRM-7B's reward at the end of step k -- a released,
               widely used checkpoint applied exactly as its card specifies, so the
               score is the field's and not ours
  score span   steps 1..k
  proxy z      a human-annotated error occurs at or before step k (in-span)
  disjoint z^c a human-annotated error occurs after step k (off-span only)
  construct    final_answer_correct, released with the benchmark

z and z^c are mutually exclusive by construction and each is measurable on its own
side of the split, which is exactly the disjoint arm the control asks for. If the
PRM's prefix score tracks "an error is coming later" better than it tracks final
correctness, the gap survives a proxy the score cannot have read, and that is a
deployed contract exhibiting the failure. We also run the surface family from the
other two audits so the three artifacts are comparable.

Splits are audited separately: the four subsets differ in difficulty and pooling
would confound it. Each solution is one example, so AUC is pooled within split and
the bootstrap resamples examples.

This is the one audit in the paper that needs a GPU. On an A100 it is roughly an
hour for all four splits; --split gsm8k alone is a few minutes.

  pip install transformers torch datasets scikit-learn
  python3 audit_processbench_containment.py [--split all] [--max-examples 0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from transformers import AutoConfig, AutoModel, AutoTokenizer

MODEL = "Qwen/Qwen2.5-Math-PRM-7B"
SEP = "<extra_0>"
SPLITS = ["gsm8k", "math", "olympiadbench", "omnimath"]
SPAN_FRACS = [0.25, 0.50]
AGGS = ["prod", "min", "last"]
AGG = "prod"
N_BOOT = 2000
MIN_POS = 10
OUT = Path(__file__).resolve().parent.parent / "analysis_results" / "processbench_containment.json"


def step_rewards(logits, mask):
    """Per-step reward as the model card defines it: softmax over the two classes at
    each separator position, take P(correct)."""
    probs = F.softmax(logits, dim=-1) * mask.unsqueeze(-1)
    return [p[p != 0].view(-1, 2)[:, 1] for p in probs]


@torch.no_grad()
def score_solution(model, tok, device, problem, steps):
    """Return one reward per step. The separator marks each step boundary, so the
    reward at position i reads steps 1..i and nothing after."""
    msgs = [{"role": "system", "content": "Please reason step by step, and put your "
             "final answer within \\boxed{}."},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": SEP.join(steps) + SEP}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    ids = tok.encode(text, return_tensors="pt").to(device)
    # use_cache=False is load-bearing, not a tidy-up: the checkpoint's bundled
    # modelling code builds a KV cache via DynamicCache.from_legacy_cache, which
    # current transformers has removed. We score with a single forward pass and
    # never generate, so the cache is pure overhead anyway.
    out = model(input_ids=ids, use_cache=False)
    sep_id = tok.encode(SEP)[0]
    r = step_rewards(out[0], (ids == sep_id))[0]
    return r.float().cpu().numpy()


def aggregate(r, k, how):
    """Solution-level score from the per-step rewards of steps 1..k. Every option
    reads only the scored span, so the contract's span field is unchanged; they
    differ in how an early error that the model later writes past is carried
    forward. 'prod' matches the aggregation used in the PRM800K audit.

    'last' is included because it is the tempting choice and it is wrong: a PRM
    scores each step conditional on its prefix, so the final step of a solution
    that went astray early can still look locally fine.
    """
    v = np.asarray(r[:k], dtype=np.float64)
    if how == "last":
        return float(v[-1])
    if how == "min":
        return float(v.min())
    return float(np.prod(v))


def boot_gap(s, z, y, seed):
    """Orientation-robust gap between AUC(s,z) and AUC(s,y), resampling examples.
    Returns the interval and a one-sided bootstrap p-value for the gap exceeding
    zero, which the caller feeds to Benjamini--Hochberg across cells."""
    rng = np.random.default_rng(seed)
    n, d = len(s), []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        zb, yb = z[i], y[i]
        if len(set(zb)) < 2 or len(set(yb)) < 2:
            continue
        d.append(abs(roc_auc_score(zb, s[i]) - .5) - abs(roc_auc_score(yb, s[i]) - .5))
    if len(d) < N_BOOT // 4:
        return None, None, None
    d = np.array(d)
    p = (1 + int((d <= 0).sum())) / (len(d) + 1)      # bottoms out at 1/(B+1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(p)


def auc_ci(s, y, seed, n_boot=2000):
    """Bootstrap interval for a single AUC, used to ask whether the construct arm is
    oriented at all. A gap measured against a construct AUC that cannot be told from
    chance is a statement about the score's silence, not about a proxy--construct
    mismatch, and has to be labelled as such."""
    rng = np.random.default_rng(seed)
    n, a = len(s), []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(set(y[i].tolist())) < 2:
            continue
        a.append(roc_auc_score(y[i], s[i]))
    if not a:
        return None, None
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def audit(name, s, y, proxies, frac):
    if len(set(y.tolist())) < 2:
        print(f"  [{name} @{frac}] construct degenerate; NOT INSTANTIABLE")
        return {}
    auc_y = roc_auc_score(y, s)
    ylo, yhi = auc_ci(s, y, 900)
    n, ypos = len(s), int(y.sum())
    oriented = ylo is not None and not (ylo <= 0.5 <= yhi)
    flag = "" if oriented else "   <-- construct arm NOT oriented"
    print(f"\n  === {name} @ span {frac:.2f}: n={n}, y+={ypos}, "
          f"AUC(s,y)={auc_y:.3f} [{ylo:.3f},{yhi:.3f}]{flag}")
    print(f"  {'proxy':26s} {'prev':>6s} {'AUC(s,z^c)':>11s} {'dDis':>8s}  "
          f"95% CI            {'p':>7s}  verdict")
    rows = {}
    for i, (pname, zc) in enumerate(proxies.items()):
        pos = int(zc.sum())
        prev = pos / n
        blocked = (pos < MIN_POS or n - pos < MIN_POS
                   or ypos < MIN_POS or n - ypos < MIN_POS)
        if blocked:
            print(f"  {pname:26s} {prev:6.3f} {'--':>11s} {'--':>8s}  "
                  f"{'':25s} UNDECIDABLE (off-span positives {pos})")
            rows[pname] = {"prevalence": prev, "zc_positives": pos,
                           "verdict": "UNDECIDABLE"}
            continue
        auc_zc = roc_auc_score(zc, s)
        d_dis = abs(auc_zc - .5) - abs(auc_y - .5)
        lo, hi, p = boot_gap(s, zc, y, 300 + i)
        surv = lo is not None and lo > 0
        verdict = ("MISMATCH" if surv and d_dis >= 0.15 else
                   "CAUTION" if surv and d_dis >= 0.10 else "ALIGNED")
        print(f"  {pname:26s} {prev:6.3f} {auc_zc:11.3f} {d_dis:+8.3f}  "
              f"[{lo:+.3f},{hi:+.3f}]  {p:7.4f}  {verdict}")
        rows[pname] = {"prevalence": prev, "zc_positives": pos, "auc_zc": auc_zc,
                       "delta_dis_abs": d_dis, "ci": [lo, hi], "p_one_sided": p,
                       "verdict": verdict, "construct_oriented": oriented}
    return {"n": n, "y_positives": ypos, "auc_y": auc_y, "auc_y_ci": [ylo, yhi],
            "construct_oriented": oriented, "proxies": rows}


def load_model(name: str, device: str):
    """The checkpoint ships its own modelling code, which predates the transformers
    release Colab installs. Two incompatibilities have to be bridged rather than
    worked around, because pinning an old transformers breaks the CUDA build:

      1. `Qwen2Model.__init__` reads `config.pad_token_id`, which current
         transformers no longer materialises as a default. We fill it from the
         tokenizer before the model is constructed.
      2. `torch_dtype=` is deprecated in favour of `dtype=`, and bfloat16 needs
         Ampere or newer, so a T4 has to fall back to float16.
    """
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
    if getattr(cfg, "pad_token_id", None) is None:
        cfg.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        print(f"    patched config.pad_token_id = {cfg.pad_token_id}")
    cfg.use_cache = False
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dt = torch.bfloat16 if bf16 else (torch.float16 if device == "cuda" else torch.float32)
    print(f"    dtype {dt}")
    kw = dict(config=cfg, device_map=device, trust_remote_code=True)
    try:
        model = AutoModel.from_pretrained(name, dtype=dt, **kw)
    except TypeError:                     # older transformers wants torch_dtype
        model = AutoModel.from_pretrained(name, torch_dtype=dt, **kw)
    return tok, model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all")
    ap.add_argument("--max-examples", type=int, default=0, help="0 = all")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="", help="output JSON path; defaults to the repo's "
                    "analysis_results/, which does not exist on a Colab runtime")
    ap.add_argument("--agg", default=AGG, choices=AGGS,
                    help="prefix aggregation for the verdict table (default prod, "
                         "matching the PRM800K audit)")
    ap.add_argument("--cache", default="/content/pb_rewards.json",
                    help="per-step rewards are cached here; reruns that only change "
                         "the aggregation or the proxies then cost no GPU at all")
    a = ap.parse_args()
    splits = SPLITS if a.split == "all" else [a.split]

    cache_path = Path(a.cache) if a.cache else None
    cache = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text())
        print(f"    reward cache: {len(cache)} solutions from {cache_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = model = None
    need_gpu = True                      # decided per split once we know the cache

    results = {}
    for split in splits:
        ds = load_dataset("Qwen/ProcessBench", split=split)
        if a.max_examples:
            ds = ds.select(range(min(a.max_examples, len(ds))))
        print(f"\n### {split}: {len(ds)} examples")
        need_gpu = any(ex["id"] not in cache for ex in ds if len(ex["steps"]) >= 3)
        if need_gpu and model is None:
            if device == "cpu":
                print("WARNING: no GPU visible; a 7B forward pass per example will be very slow.")
            else:
                gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"    {torch.cuda.get_device_name(0)}, {gb:.0f} GB")
                if gb < 20:
                    print("    WARNING: a 7B checkpoint in half precision needs ~16 GB of "
                          "weights plus activations; expect OOM below ~24 GB.")
            print(f"loading {a.model} ...", flush=True)
            tok, model = load_model(a.model, device)
        elif not need_gpu:
            print("    fully cached; no model load needed")

        scored, n_eligible, n_failed, n_ragged = [], 0, 0, 0
        for j, ex in enumerate(ds):
            steps = ex["steps"]
            if len(steps) < 3:            # need a non-empty complement at both spans
                continue
            n_eligible += 1
            if ex["id"] in cache:
                r = np.array(cache[ex["id"]], dtype=np.float64)
                if len(r) == len(steps):
                    scored.append((r, steps, int(ex["final_answer_correct"]),
                                   int(ex["label"])))
                    continue
            try:
                r = score_solution(model, tok, device, ex["problem"], steps)
            except Exception:
                # An occasional long solution failing is tolerable. Nothing ever
                # working is a broken run, not a thin one, and must not be swallowed
                # into a verdict -- so the first failure before any success is fatal.
                if not scored:
                    print(f"    FATAL on the first eligible example ({ex['id']}); "
                          "not retrying 300 times to produce an empty audit.")
                    raise
                n_failed += 1
                if n_failed <= 5:
                    print(f"    skip {ex['id']}: scoring failed")
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue
            if len(r) != len(steps):
                n_ragged += 1
                continue
            cache[ex["id"]] = [float(x) for x in r]
            scored.append((r, steps, int(ex["final_answer_correct"]), int(ex["label"])))
            if (j + 1) % 100 == 0:
                print(f"    scored {len(scored)} of {j+1} seen", flush=True)
                if cache_path:
                    cache_path.write_text(json.dumps(cache))
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache))
        print(f"    eligible {n_eligible}, usable {len(scored)}, "
              f"failed {n_failed}, reward/step mismatch {n_ragged}")
        if not scored:
            raise SystemExit(f"no usable examples in {split}; refusing to report a verdict")

        # Validity check on our own scoring before any verdict is read off it. If the
        # score over the whole solution does not predict final correctness, the
        # pipeline is wrong and every gap below is noise. A rising curve is also the
        # PRM800K span shape, replicated on a different checkpoint and corpus.
        #
        # The aggregation is the thing being checked. A step-level PRM's reward at
        # step k answers "is step k right given the prefix", so reading the last
        # step alone misses an error committed earlier and locally recovered from;
        # the solution-level score a consumer forms is an aggregate over the prefix.
        # We check all three and let the curve say which one is a working score.
        curves = {}
        for agg in AGGS:
            curve = {}
            for frac in (0.25, 0.50, 0.75, 1.00):
                ss, yy = [], []
                for r, steps, correct, _ in scored:
                    k = max(1, min(len(steps), int(round(frac * len(steps)))))
                    ss.append(aggregate(r, k, agg))
                    yy.append(correct)
                ss, yy = np.array(ss), np.array(yy)
                if len(set(yy.tolist())) < 2:
                    continue
                lo, hi = auc_ci(ss, yy, 800)
                curve[frac] = [float(roc_auc_score(yy, ss)), lo, hi]
            curves[agg] = curve
            print(f"    span curve AUC(s,y) [{agg:4s}]: " + "  ".join(
                f"{f:.2f}={v[0]:.3f}[{v[1]:.3f},{v[2]:.3f}]" for f, v in curve.items()))
        full = {a: c[1.00][0] for a, c in curves.items() if 1.00 in c}
        best = max(full, key=full.get) if full else AGG
        print(f"    full-solution AUC(s,y) by aggregation: "
              + ", ".join(f"{a}={v:.3f}" for a, v in full.items()))
        if full and full[best] < 0.65:
            print("    WARNING: no aggregation makes the full-solution score predict final")
            print("    correctness, yet ProcessBench couples process error to a wrong answer")
            print("    (kappa 0.48-0.97). The scoring pipeline is suspect; do not read the")
            print("    verdicts below as measurements of the deployed PRM.")
        else:
            print(f"    validity check passes under '{best}'.")
        results[f"{split}@curve"] = {"span_curve_auc_y": curves, "best_agg": best,
                                     "full_span_auc_y": full}

        for frac in SPAN_FRACS:
            s, y, zc_h, texts_pre, texts_post = [], [], [], [], []
            for r, steps, correct, first_err in scored:
                L = len(steps)
                k = max(1, int(round(frac * L)))
                if k >= L:
                    continue
                s.append(aggregate(r, k, a.agg))
                y.append(correct)
                # ProcessBench label is the index of the first erroneous step, -1 if
                # the solution is fully correct. The off-span proxy is "the first
                # human-annotated error lies strictly after the scored span".
                zc_h.append(int(first_err >= k))
                texts_pre.append(" ".join(steps[:k]))
                texts_post.append(" ".join(steps[k:]))
            if not s:
                continue
            s, y = np.array(s), np.array(y)
            post_len = np.array([len(t) for t in texts_post])
            post_ad = np.array([sum(c.isdigit() or c in "+-*/=^" for c in t) / max(1, len(t))
                                for t in texts_post])
            proxies = {
                "human first-error off-span": np.array(zc_h),
                "length (off-span)": (post_len > np.median(post_len)).astype(int),
                "arith density (off-span)": (post_ad > np.median(post_ad)).astype(int),
            }
            results[f"{split}@{frac}"] = audit(split, s, y, proxies, frac)

    decided = [(k, p, r) for k, v in results.items()
               for p, r in v.get("proxies", {}).items()
               if r.get("verdict") in ("MISMATCH", "CAUTION", "ALIGNED")]

    # Benjamini--Hochberg over every decided cell. Twenty-four cells is enough that
    # a couple of intervals clearing zero is the expected yield of chance alone, and
    # the paper applies BH elsewhere; not applying it here would be a double standard.
    if decided:
        order = sorted(range(len(decided)), key=lambda i: decided[i][2]["p_one_sided"])
        m, crit = len(order), None
        for rank, i in enumerate(order, 1):
            if decided[i][2]["p_one_sided"] <= 0.05 * rank / m:
                crit = decided[i][2]["p_one_sided"]
        for _, _, r in decided:
            r["survives_bh"] = crit is not None and r["p_one_sided"] <= crit
            if not r["survives_bh"] and r["verdict"] in ("MISMATCH", "CAUTION"):
                r["verdict_bh"] = "ALIGNED (fails BH)"
            else:
                r["verdict_bh"] = r["verdict"]
        print(f"\nBenjamini--Hochberg over {m} decided cells at q=0.05: "
              f"{sum(r['survives_bh'] for _, _, r in decided)} survive")

    hits = [(k, p, r) for k, p, r in decided
            if r["verdict"] in ("MISMATCH", "CAUTION") and r.get("survives_bh")]
    human = [(k, r) for k, p, r in decided if p.startswith("human")]
    print("\nreading:")
    if not decided:
        print("  every cell was blocked by a precondition; there is no verdict here to read.")
    elif hits:
        print("  a deployed PRM's prefix score fails the containment control against a")
        print("  proxy it could not have read:")
        for k, p, r in hits:
            note = "" if r.get("construct_oriented") else "  [construct arm NOT oriented]"
            print(f"    {k}, proxy {p}, dDis {r['delta_dis_abs']:+.3f}{note}")
        if all(not r.get("construct_oriented") for _, _, r in hits):
            print("  Every hit sits on a cell whose construct AUC cannot be told from chance,")
            print("  so the gap says the score is silent about correctness at this span, not")
            print("  that a proxy outranks a construct the score actually tracks. Report it")
            print("  that way; it is a weaker claim than a mismatch on an oriented contract.")
        if all(not p.startswith("human") for _, p, _ in hits):
            print("  No hit comes from the human off-span label. The failures are surface")
            print("  rules only, which is the reading to publish.")
    else:
        print(f"  all {len(decided)} decided cells clear after BH correction.")
        if human:
            print(f"  {len(human)} of them use the human off-span label, so the clear is not an")
            print("  artifact of a blunt surface rule. Three artifacts, no field failure.")
        else:
            print("  NOTE: no cell used the human off-span label -- every one was blocked on")
            print("  power. The surface rules alone do not answer the objection this audit")
            print("  exists to answer; report it as undecided, not as a clear.")

    out = Path(a.out) if a.out else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": a.model, "cells": results, "n_boot": N_BOOT}, indent=2))
    print("\n[wrote]", out)


if __name__ == "__main__":
    main()
