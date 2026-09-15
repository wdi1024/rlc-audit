#!/usr/bin/env python3
"""A second deployed score in the field, chosen because its span is mis-declared. (2026-08-21)

PRM800K clears the containment control under every proxy rule we tried
(audit_prm800k_proxy_family.py). That is a clean negative, but it leaves the paper
without a field contract that fails, so we went looking for one where the contract's
score-span field is wrong on the face of the generation protocol.

Math-Shepherd \\citep{wang2024mathshepherd} is that case. Its per-step labels are the
training signal behind a large fraction of released process reward models, and they
are read by consumers as step-local quality: is this step, on its own, a good step.
They are not produced that way. Each label is estimated by completing the solution
from that step many times and asking how often the completion reaches the right
answer. The label is a functional of the *continuation*, not of the prefix it is
attached to.

The contract this instantiates:

  score        the deployed prefix score over steps 1..k -- the mean of the released
               Math-Shepherd step labels, the aggregation a consumer of this signal
               would form when deciding whether to keep a partial solution
  score span   steps 1..k, which is what the field's usage declares
  proxy        a cheap rule over the same prefix
  construct    final-answer correctness, graded against GSM8K gold rather than
               against the last Math-Shepherd label, so the construct is independent
               of the annotation that produced the score
  disjoint     the same proxy rule read over steps k+1..L only

The prediction is sharp and it is the paper's, not a hedge. If the declared span were
the real one, the prefix score would carry little about text it never saw, and
AUC(s, z^c) would sit near 1/2. If the label is really computed from the future, the
score has seen the continuation, so it should predict off-span content and the gap
should survive the disjoint control. We report whichever way it comes out.

  python3 audit_mathshepherd_containment.py [--problems 400]
"""
from __future__ import annotations

import argparse
import json
import re
import ssl
import urllib.request
from collections import defaultdict

import certifi
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
SHARDS = sorted(Path("/tmp/mshep").glob("*.parquet"))
GSM_CACHE = Path("/tmp/gsm8k_gold.json")
OUT = ROOT / "analysis_results" / "mathshepherd_containment.json"
SPAN_FRACS = [0.25, 0.50]
N_BOOT = 2000
MIN_POS = 10
MAX_EMPTY_COMPLEMENT = 0.15

STEP = re.compile(r"Step\s+(\d+):\s*(.*?)\s*([+\-ки])\s*$")
BACKTRACK = re.compile(r"\b(wait|hmm|actually|oops|let me try|on second thought|"
                       r"that'?s not right|i made a mistake|scratch that)\b", re.I)
ANSWER = re.compile(r"answer is:?\s*(.*)$", re.I)


def norm_num(s: str) -> str:
    s = re.sub(r"[^\d\.\-/]", "", str(s))
    s = s.rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return s


def norm_text(s: str) -> str:
    """Match key for joining a Math-Shepherd problem to its GSM8K gold answer. Folds
    case, whitespace and punctuation, because the two releases differ in quote
    characters and spacing on otherwise identical problem statements."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def arith_density(s: str) -> float:
    return sum(c.isdigit() or c in "+-*/=^" for c in s) / len(s) if s else 0.0


def gsm8k_gold() -> dict:
    """problem text -> gold final answer, so the construct is independent of the
    Math-Shepherd annotation that produced the score."""
    if GSM_CACHE.exists():
        return json.loads(GSM_CACHE.read_text())
    gold = {}
    api = "https://huggingface.co/api/datasets/openai/gsm8k/parquet/main"
    ctx = ssl.create_default_context(cafile=certifi.where())
    urls = json.loads(urllib.request.urlopen(api, timeout=60, context=ctx).read())
    for split in ("train", "test"):
        for i, u in enumerate(urls.get(split, [])):
            local = Path(f"/tmp/gsm8k_{split}_{i}.parquet")
            if not local.exists():
                with urllib.request.urlopen(u, timeout=120, context=ctx) as r:
                    local.write_bytes(r.read())
            df = pd.read_parquet(local)
            for q, a in zip(df["question"], df["answer"]):
                gold[norm_text(q)] = norm_num(a.split("####")[-1])
    GSM_CACHE.write_text(json.dumps(gold))
    return gold


def parse_row(inp: str, lab: str):
    """Return (problem, [(step_text, label)]) or None. The two fields are the same
    text with the placeholder replaced by the sign, so steps are aligned by index."""
    if "Step 1:" not in lab:
        return None
    problem = lab.split("Step 1:")[0].strip()
    steps = []
    for line in lab[len(problem):].split("\n"):
        m = STEP.match(line.strip())
        if not m:
            return None
        sign = m.group(3)
        if sign not in "+-":
            return None
        steps.append((m.group(2), 1 if sign == "+" else 0))
    # three steps is the shortest solution that leaves a non-empty complement at
    # both span fractions
    return (problem, steps) if len(steps) >= 3 else None


def per_problem_aucs(groups, key):
    out = []
    for g in groups:
        y = [x[key] for x in g]
        if len(set(y)) < 2:
            continue
        out.append(roc_auc_score(y, [x["s"] for x in g]))
    return np.array(out, dtype=float)


def boot(a_z, a_y, seed):
    if len(a_z) == 0 or len(a_y) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    d = [abs(a_z[rng.integers(0, len(a_z), len(a_z))].mean() - .5)
         - abs(a_y[rng.integers(0, len(a_y), len(a_y))].mean() - .5) for _ in range(N_BOOT)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", type=int, default=400)
    a = ap.parse_args()

    gold = gsm8k_gold()
    print(f"gsm8k gold answers: {len(gold)}")

    per, skipped, nogold = defaultdict(list), 0, 0
    for shard in SHARDS:
        df = pd.read_parquet(shard, columns=["input", "label", "task"])
        df = df[df["task"] == "GSM8K"]
        for inp, lab in zip(df["input"], df["label"]):
            p = parse_row(inp, lab)
            if p is None:
                skipped += 1
                continue
            problem, steps = p
            key = norm_text(problem)
            if key not in gold:
                nogold += 1
                continue
            if key not in per and len(per) >= a.problems:
                continue
            m = ANSWER.search(steps[-1][0])
            if not m:
                skipped += 1
                continue
            per[key].append((steps, int(norm_num(m.group(1)) == gold[key])))
        if len(per) >= a.problems and sum(len(v) for v in per.values()) > 20000:
            break

    per = {k: v for k, v in per.items() if len(v) >= 2}
    n_all = sum(len(v) for v in per.values())
    print(f"problems {len(per)}, solutions {n_all}  (unparsed {skipped}, no gold {nogold})")

    # how often does the released last-step label agree with independent grading?
    agree = np.mean([s[-1][1] == y for v in per.values() for s, y in v])
    print(f"last released step label agrees with GSM8K grading on {agree:.3f} of solutions")

    results = {}
    for frac in SPAN_FRACS:
        groups, dens, empty = [], [], 0
        for key, sols in per.items():
            g = []
            for steps, y in sols:
                L = len(steps)
                k = max(1, int(round(frac * L)))
                if k >= L:
                    continue
                pre = " ".join(t for t, _ in steps[:k])
                rest = " ".join(t for t, _ in steps[k:])
                empty += int(not rest.strip())
                g.append({"s": float(np.mean([v for _, v in steps[:k]])), "y": y,
                          "backtrack": int(bool(BACKTRACK.search(pre))),
                          "backtrack_c": int(bool(BACKTRACK.search(rest))),
                          "_ad": arith_density(pre), "_ad_c": arith_density(rest),
                          "_len": len(pre), "_len_c": len(rest)})
            if len(g) >= 2:
                groups.append(g)
                dens.extend(g)
        if not groups:
            continue
        for base, name in (("_ad", "arith_density"), ("_len", "length")):
            med = float(np.median([x[base] for x in dens]))
            med_c = float(np.median([x[base + "_c"] for x in dens]))
            for g in groups:
                for x in g:
                    x[name] = int(x[base] > med)
                    x[name + "_c"] = int(x[base + "_c"] > med_c)

        n_samp = sum(len(g) for g in groups)
        ypos = sum(x["y"] for g in groups for x in g)
        A_y = per_problem_aucs(groups, "y")
        auc_y = float(A_y.mean())
        frac_empty = empty / max(1, n_samp)
        print(f"\n=== span {frac:.2f}: {len(groups)} problems, {n_samp} solutions, "
              f"y+={ypos}, AUC(s,y)={auc_y:.3f}, empty complement {frac_empty:.3f}")
        print(f"{'proxy':16s} {'prev':>6s} {'AUC(s,z)':>9s} {'AUC(s,z^c)':>11s} "
              f"{'dDis':>8s}  95% CI            verdict")
        row = {}
        for i, name in enumerate(("backtrack", "arith_density", "length")):
            prev = float(np.mean([x[name] for g in groups for x in g]))
            A_z, A_zc = per_problem_aucs(groups, name), per_problem_aucs(groups, name + "_c")
            if len(A_z) == 0 or len(A_zc) == 0:
                print(f"{name:16s} {prev:6.3f}   NOT INSTANTIABLE (degenerate proxy)")
                row[name] = {"prevalence": prev, "verdict": "NOT INSTANTIABLE"}
                continue
            auc_z, auc_zc = float(A_z.mean()), float(A_zc.mean())
            d_dis = abs(auc_zc - .5) - abs(auc_y - .5)
            lo, hi = boot(A_zc, A_y, 200 + i)
            # the power precondition binds on the disjoint proxy as well as on the
            # construct: a rule that fires eight times cannot support a verdict
            zcpos = sum(x[name + "_c"] for g in groups for x in g)
            blocked = (ypos < MIN_POS or (n_samp - ypos) < MIN_POS
                       or zcpos < MIN_POS or (n_samp - zcpos) < MIN_POS
                       or frac_empty > MAX_EMPTY_COMPLEMENT)
            surv = (lo is not None and lo > 0) and not blocked
            verdict = ("MISMATCH" if surv and d_dis >= 0.15 else
                       "CAUTION" if surv and d_dis >= 0.10 else
                       "UNDECIDABLE" if blocked else "ALIGNED")
            print(f"{name:16s} {prev:6.3f} {auc_z:9.3f} {auc_zc:11.3f} {d_dis:+8.3f}  "
                  f"[{lo:+.3f},{hi:+.3f}]  {verdict}"
                  + (f"   (off-span positives {zcpos})" if blocked else ""))
            row[name] = {"prevalence": prev, "auc_z": auc_z, "auc_zc": auc_zc,
                         "zc_positives": int(zcpos),
                         "delta_dis_abs": d_dis, "ci": [lo, hi], "verdict": verdict}
        results[str(frac)] = {"problems": len(groups), "solutions": n_samp,
                              "y_positives": ypos, "auc_y": auc_y,
                              "empty_complement": frac_empty, "proxies": row}

    hits = [(f, n) for f, r in results.items()
            for n, v in r["proxies"].items() if v.get("verdict") in ("MISMATCH", "CAUTION")]
    print("\nreading:")
    if hits:
        print("  a deployed step-label score fails the containment control:")
        for f, n in hits:
            print(f"    span {f}, proxy {n}")
        print("  The score's declared span is steps 1..k, but it predicts text outside that")
        print("  span, which is what a label estimated by completion should do. This is a")
        print("  field contract whose score-span field is wrong, caught by the control.")
    else:
        print("  no cell reaches the caution band; the contract clears the disjoint control.")
    print("  off-span predictiveness AUC(s,z^c) above is the quantity to read regardless of")
    print("  verdict: it is how much of the continuation the 'prefix' score already knows.")

    OUT.write_text(json.dumps({"spans": results, "n_boot": N_BOOT,
                               "last_label_agreement": float(agree)}, indent=2))
    print("\n[wrote]", OUT)


if __name__ == "__main__":
    main()
