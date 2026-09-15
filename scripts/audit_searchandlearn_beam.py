#!/usr/bin/env python3
"""External RLC-Audit of a deployed partial-span verification contract:
HuggingFace search-and-learn PRM-guided beam search on MATH-500.

The search-and-learn pipeline (Beeching et al., "Scaling test-time compute")
prunes beams DURING generation by a process reward model's score on the
partial chain (RLHFlow/Llama3.1-8B-PRM-Deepseek-Data, agg_strategy=last),
then selects the returned answer by the same proxy (pred_naive = argmax).
The construct the proxy is sold as serving is final-answer correctness.
The released datasets retain, for every surviving beam, the full per-step
score trajectory plus the gold answer -- so the L2 link (proxy vs construct)
is auditable from released artifacts alone: no generation, no judge calls.

For each completion i of problem q:
    s_k(i) = PRM score after step k        (the deployed proxy, span = prefix)
    y(i)   = final-answer correctness      (the construct, graded vs gold)
and the deployed decision z(q) = argmax_i s_full(i)  (pred_naive@n).

Reports: grader validation vs the released --evals accuracies, pooled and
within-problem AUC(s_k, y) across span fractions, deployed-selection harm
(naive top-1 wrong while the pool contained a correct beam), and surface
coupling (token length vs proxy vs construct).

  python3 audit_searchandlearn_beam.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis_results" / "searchandlearn_beam_audit.json"
N_BOOT = 2000
RNG = np.random.default_rng(0)

BEAM_DS = "HuggingFaceH4/Llama-3.2-1B-Instruct-beam-search-completions"
DVTS_DS = "HuggingFaceH4/Llama-3.2-1B-Instruct-DVTS-completions"
BON_DS = "HuggingFaceH4/Llama-3.2-1B-Instruct-best-of-N-completions"
CFG = "HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-{n}--m-4--iters-40--look-0--seed-{seed}--agg_strategy-last"
BON_CFG = "HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-{n}--max_tokens-2048--bsz-8--seed-{seed}--agg_strategy-last"

ARMS = (
    [("beam", BEAM_DS, 16, s) for s in range(5)]
    + [("beam", BEAM_DS, 64, s) for s in range(5)]
    + [("beam", BEAM_DS, 256, s) for s in range(5)]
    + [("dvts", DVTS_DS, 256, s) for s in range(5)]
    + [("bon", BON_DS, 256, s) for s in range(5)]
)
SPAN_FRACS = [0.25, 0.50, 0.75, 1.00]  # plus "first step" handled separately


# ---------------------------------------------------------------- grading ---
# Hendrycks MATH-style answer normalisation (as used by Minerva / lm-eval),
# plus a numeric-equivalence fallback through sympy.

def _last_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        m = re.match(r"\\boxed\s+(\S+)", text[idx:])
        return m.group(1) if m else None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
        j += 1
    return None


def _fix_fracs(s: str) -> str:
    parts = s.split("\\frac")
    out = parts[0]
    for p in parts[1:]:
        if p.startswith("{"):
            out += "\\frac" + p
        elif len(p) >= 2:
            out += "\\frac{" + p[0] + "}{" + p[1] + "}" + p[2:]
        else:
            out += "\\frac" + p
    return out


def _fix_sqrt(s: str) -> str:
    return re.sub(r"\\sqrt(?!\{)(\w)", r"\\sqrt{\1}", s)


def _strip(s: str) -> str:
    s = s.strip()
    s = s.replace("\n", "").replace("\\!", "").replace("\\ ", "").replace(" ", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{([^{}]*)\}", r"\1", s)
    if s.startswith("0.") :
        s = s.replace("0.", ".", 1) if False else s
    # units like "\\text{ cm}" already removed; trailing period
    s = s.rstrip(".")
    s = _fix_sqrt(_fix_fracs(s))
    # 0.5 -> \frac{1}{2} style unification: normalise simple decimals
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    # strip one layer of surrounding braces
    while len(s) > 1 and s[0] == "{" and s[-1] == "}" and s.count("{") == 1:
        s = s[1:-1]
    return s


_NUM_CACHE: dict[str, object] = {}


class _ParseTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _ParseTimeout


def _to_number(s: str):
    """Try to evaluate a normalised answer string to a sympy number."""
    if s in _NUM_CACHE:
        return _NUM_CACHE[s]
    import signal

    import sympy

    # guard against pathological expressions (e.g. towers of huge powers) that
    # make sympify run for hours; equivalence then falls back to string match
    if len(s) > 200 or re.search(r"\*\*\s*\(?\s*-?\d{7,}", s.replace("^", "**")):
        _NUM_CACHE[s] = None
        return None

    val = None
    t = s
    t = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", t)
    t = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", t)
    t = t.replace("\\pi", "pi").replace("^", "**").replace("\\cdot", "*").replace("\\times", "*")
    if not re.fullmatch(r"[0-9a-zA-Z+\-*/().,\s_{}]*", t):
        _NUM_CACHE[s] = None
        return None
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(3)
    try:
        expr = sympy.sympify(t, evaluate=True)
        if expr.free_symbols:
            val = None
        else:
            val = expr
    except Exception:  # includes _ParseTimeout
        val = None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    _NUM_CACHE[s] = val
    return val


def _pred_answer(v: str | None) -> str | None:
    """pred_* fields are released as '\\boxed{...}' strings; unwrap before grading."""
    if v is None:
        return None
    return _last_boxed(v) if "\\boxed" in v else v


def math_equal(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    p, g = _strip(pred), _strip(gold)
    if p == g:
        return True
    pn, gn = _to_number(p), _to_number(g)
    if pn is not None and gn is not None:
        try:
            return bool(abs(float(pn) - float(gn)) < 1e-6)
        except Exception:
            import signal

            import sympy

            old = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(3)
            try:
                return bool(sympy.simplify(pn - gn) == 0)
            except Exception:
                return False
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)
    return False


# ---------------------------------------------------------------- analysis ---

def within_problem_auc(groups: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, int]:
    """Mean AUC(s, y) across problems whose pool contains both classes."""
    aucs = []
    for s, y in groups:
        if 0 < y.sum() < len(y):
            aucs.append(roc_auc_score(y, s))
    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs)


def audit_arm(kind: str, ds_name: str, n: int, seed: int) -> dict:
    cfg = BON_CFG if kind == "bon" else CFG
    ds = load_dataset(ds_name, data_dir=cfg.format(n=n, seed=seed), split="train")
    pred_key = f"pred_naive@{n}"
    rows = []
    for r in ds:
        gold = r["answer"]
        comps, scores, toks = r["completions"], r["scores"], r["completion_tokens"]
        if not isinstance(toks, list) or len(toks) != len(comps):
            # DVTS dumps store completion_tokens as a scalar -1; fall back to
            # character length for the surface-coupling stats
            toks = [len(c) for c in comps]
            len_unit = "chars"
        else:
            len_unit = "tokens"
        y = np.array([math_equal(_last_boxed(c), gold) for c in comps], dtype=int)
        s_full = np.array([sc[-1] if sc else np.nan for sc in scores], dtype=float)
        span = {}
        for f in SPAN_FRACS:
            span[f] = np.array(
                [sc[max(0, int(np.ceil(f * len(sc))) - 1)] if sc else np.nan for sc in scores],
                dtype=float,
            )
        span["first"] = np.array([sc[0] if sc else np.nan for sc in scores], dtype=float)
        rows.append(
            dict(
                y=y,
                s_full=s_full,
                span=span,
                toks=np.array(toks, dtype=float),
                nsteps=np.array([len(sc) for sc in scores], dtype=float),
                naive_ok=math_equal(_pred_answer(r.get(pred_key)), gold),
                weighted_ok=math_equal(_pred_answer(r.get(f"pred_weighted@{n}")), gold),
                maj_ok=math_equal(_pred_answer(r.get(f"pred_maj@{n}")), gold),
            )
        )

    res: dict = dict(kind=kind, n=n, seed=seed, problems=len(rows), len_unit=len_unit)

    # grader validation vs released --evals
    res["acc_naive"] = float(np.mean([r["naive_ok"] for r in rows]) * 100)
    res["acc_weighted"] = float(np.mean([r["weighted_ok"] for r in rows]) * 100)
    res["acc_maj"] = float(np.mean([r["maj_ok"] for r in rows]) * 100)

    # pooled + within-problem AUC per span fraction
    all_y = np.concatenate([r["y"] for r in rows])
    res["base_rate"] = float(all_y.mean())
    res["auc"] = {}
    for key in ["first"] + SPAN_FRACS:
        sv = np.concatenate([r["span"][key] for r in rows])
        ok = ~np.isnan(sv)
        pooled = roc_auc_score(all_y[ok], sv[ok]) if 0 < all_y[ok].sum() < ok.sum() else float("nan")
        wauc, wn = within_problem_auc([(r["span"][key], r["y"]) for r in rows])
        res["auc"][str(key)] = dict(pooled=float(pooled), within=wauc, within_n=wn)

    # deployed selection harm: naive top-1 wrong while pool had a correct beam
    solvable = [r for r in rows if r["y"].sum() > 0]
    harmed = [r for r in solvable if not r["naive_ok"]]
    res["selection"] = dict(
        solvable=len(solvable),
        naive_wrong_in_solvable=len(harmed),
        harm_rate=float(len(harmed) / len(solvable)) if solvable else float("nan"),
    )
    # bootstrap CI on harm rate (problem-level resample)
    if solvable:
        flags = np.array([not r["naive_ok"] for r in solvable], dtype=float)
        boots = [
            flags[RNG.integers(0, len(flags), len(flags))].mean() for _ in range(N_BOOT)
        ]
        res["selection"]["harm_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    # surface coupling: token length vs proxy vs construct
    all_s = np.concatenate([r["s_full"] for r in rows])
    all_t = np.concatenate([r["toks"] for r in rows])
    ok = ~np.isnan(all_s)
    res["surface"] = dict(
        r_len_proxy=float(np.corrcoef(all_t[ok], all_s[ok])[0, 1]),
        r_len_construct=float(np.corrcoef(all_t[ok], all_y[ok])[0, 1]),
        auc_len_construct_within=within_problem_auc([(r["toks"], r["y"]) for r in rows])[0],
        auc_len_proxytop_within=within_problem_auc(
            [(r["toks"], (r["s_full"] >= np.nanmedian(r["s_full"])).astype(int)) for r in rows]
        )[0],
    )
    return res


def main():
    out = {"arms": []}
    if OUT.exists():
        out = json.loads(OUT.read_text())
    done = {(a["kind"], a["n"], a["seed"]) for a in out["arms"]}
    for kind, ds_name, n, seed in ARMS:
        if (kind, n, seed) in done:
            print(f"SKIP (cached) {kind} n={n} seed={seed}")
            continue
        try:
            res = audit_arm(kind, ds_name, n, seed)
        except Exception as e:  # missing config etc.
            print(f"SKIP {kind} n={n} seed={seed}: {e}")
            continue
        out["arms"].append(res)
        OUT.parent.mkdir(exist_ok=True)
        tmp = OUT.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(out, indent=2))
        tmp.replace(OUT)
        a = res["auc"]
        print(
            f"{kind} n={n} seed={seed}: acc naive/wt/maj = "
            f"{res['acc_naive']:.1f}/{res['acc_weighted']:.1f}/{res['acc_maj']:.1f} | "
            f"base {res['base_rate']:.3f} | "
            f"AUC within first/25/50/75/full = "
            f"{a['first']['within']:.3f}/{a['0.25']['within']:.3f}/{a['0.5']['within']:.3f}/"
            f"{a['0.75']['within']:.3f}/{a['1.0']['within']:.3f} | "
            f"harm {res['selection']['harm_rate']:.3f}"
        )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
