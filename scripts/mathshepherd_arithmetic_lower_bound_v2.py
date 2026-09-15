#!/usr/bin/env python3
"""Arithmetic lower bound on Math-Shepherd label/construct disagreement, v2.

A manual read of 200 steps flagged by v1 (seed 20260905) found that 53 were
parser artefacts (fraction targets such as "= 1/5" read as "= 1", percent
targets, leading-dot decimals, chained equalities "= 40+60 = 100", tails of a
larger expression such as "8000 gallons x 3/4 = 6000", algebra) and 39 were
false only beyond the precision the model displayed ("2000/12 = 166.67").
This version closes those classes and treats an equation as false only when
the displayed target disagrees with the expression under both standard
precedence and left-to-right evaluation, beyond the rounding the display
allows. Sign errors ("3 - 5 = 8") stay false: the equation is false as written.

Run from the repository root; reads the same shards and gold file as v1 and
writes analysis_results/mathshepherd_arithmetic_lower_bound_v2.json.
"""
import json, random, re, sys
from fractions import Fraction
from pathlib import Path
import importlib.util
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("v1", ROOT / "scripts" / "mathshepherd_arithmetic_lower_bound.py")
v1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)
OUT = ROOT / "analysis_results" / "mathshepherd_arithmetic_lower_bound_v2.json"

NUM = r"(?:\d[\d,]*(?:\.\d+)?|\.\d+)"
FRAC = rf"{NUM}(?:\s*/\s*{NUM})?"                      # 1/5, 2/3
MIXED = rf"(?:{NUM}\s+)?{FRAC}"                        # 2 2/3
EXPR = rf"({NUM}(?:\s*[+\-*/x×]\s*{NUM})+)"
EQ = re.compile(rf"{EXPR}\s*=\s*({MIXED})\s*(%?)")
CHAIN_TAIL = re.compile(rf"\s*=\s*({MIXED})\s*(%?)")


def _num(s):
    return float(s.replace(",", ""))


def _target_value(tok):
    """'1/5' -> 0.2, '2 2/3' -> 2.667, '166.67' -> 166.67; also the decimals shown."""
    tok = tok.strip()
    m = re.fullmatch(rf"(?:({NUM})\s+)?({NUM})(?:\s*/\s*({NUM}))?", tok)
    whole, a, b = m.group(1), m.group(2), m.group(3)
    if b is not None:
        val = (_num(whole) if whole else 0.0) + _num(a) / _num(b)
        return val, None                                # exact fraction: no rounding slack
    val = _num(a)
    dec = len(a.split(".")[1]) if "." in a else 0
    return val, dec


def _tol(dec):
    """Rounding slack the displayed precision allows: half a unit in the last place."""
    return None if dec is None else 0.5 * 10 ** (-dec) + 1e-9


def _std(expr):
    e = expr.replace(",", "").replace("x", "*").replace("×", "*")
    try:
        return float(eval(e, {"__builtins__": {}}, {}))
    except Exception:
        return None


def _ltr(expr):
    return v1._eval_ltr(expr.replace(",", "").replace("x", "*").replace("×", "*"))


def _matches(val, tgt, dec, pct):
    if val is None:
        return True                                     # unevaluable: do not flag
    cands = [tgt]
    if pct:
        cands.append(tgt / 100.0)
    tol = _tol(dec)
    for t in cands:
        if tol is None:
            if abs(val - t) <= 1e-9 * max(1.0, abs(t)):
                return True
        else:
            if abs(val - t) <= max(tol, 1e-6 * abs(t)):
                return True
    return False


def equations(text):
    """Yield (expr, targets) with calculator annotations parsed once, chained
    targets collected, and tails of larger expressions skipped."""
    out = []
    for m in v1.CALC.finditer(text):
        inner = m.group(1)
        out.extend(_from(inner, inner_is_calc=True))
    stripped = v1.CALC.sub(" ", text)
    out.extend(_from(stripped, inner_is_calc=False))
    return out


def _from(s, inner_is_calc):
    res = []
    for m in EQ.finditer(s):
        pre = s[:m.start()].rstrip()
        if pre and (pre[-1] in "+-*/×=x" or pre[-1].isalpha() and pre.endswith((" x",))):
            continue                                    # tail of a larger expression
        if pre and re.search(r"[A-Za-z]\s*$", pre) and re.search(r"[A-Za-z][+\-*/=]", pre[-3:] + " "):
            continue
        # algebra: a letter adjacent to the expression on the left
        if pre and re.search(r"[A-Za-z][+\-*/]\s*$", pre + s[m.start():m.start()+1]):
            continue
        expr = m.group(1)
        targets = [(m.group(2), m.group(3))]
        rest = s[m.end():]
        while True:
            t = CHAIN_TAIL.match(rest)
            if not t:
                break
            targets.append((t.group(1), t.group(2)))
            rest = rest[t.end():]
        res.append((expr, targets))
    return res


def arithmetically_false(text):
    eqs = equations(text)
    if not eqs:
        return None
    for expr, targets in eqs:
        vs, vl = _std(expr), _ltr(expr)
        ok = False
        for tok, pct in targets:
            tgt, dec = _target_value(tok)
            if _matches(vs, tgt, dec, pct) or _matches(vl, tgt, dec, pct):
                ok = True; break
            tv = _std(tok) if re.search(r"[+\-*/]", tok) else None
            if tv is not None and vs is not None and abs(tv - vs) <= 1e-6 * max(1, abs(vs)):
                ok = True; break
        if not ok:
            return True
    return False


def load_rows():
    gold = json.loads(v1.GOLD.read_text())
    rows = []
    for shard in v1.SHARDS:
        df = pd.read_parquet(shard, columns=["label", "task"])
        for lab in df[df["task"] == "GSM8K"]["label"]:
            if "Step 1:" not in lab:
                continue
            problem = lab.split("Step 1:")[0].strip()
            key = v1.norm_text(problem)
            if key not in gold:
                continue
            steps, ok = [], True
            for line in lab[len(problem):].split("\n"):
                m = v1.STEP.match(line.strip())
                if not m:
                    ok = False; break
                steps.append((m.group(2), 1 if m.group(3) == "+" else 0))
            if not ok or len(steps) < 3:
                continue
            m = v1.ANSWER.search(steps[-1][0])
            if not m:
                continue
            rows.append((steps, int(v1.norm_num(m.group(1)) == gold[key])))
    return rows


def main():
    rows = load_rows()
    assert len(rows) == 127482, len(rows)               # referee check: same corpus as v1
    res = {}
    for name, y_val in (("failed", 0), ("succeeded_control", 1)):
        n_pos = n_cov = n_false = 0
        flagged = []
        for steps, y in rows:
            if y != y_val:
                continue
            for text, l in steps[:-1]:
                if l != 1:
                    continue
                n_pos += 1
                f = arithmetically_false(text)
                if f is None:
                    continue
                n_cov += 1
                if f:
                    n_false += 1; flagged.append(text)
        res[name] = dict(positive_steps=n_pos, with_equations=n_cov, arithmetically_false=n_false,
                         false_rate_of_covered=n_false / n_cov, coverage=n_cov / n_pos)
        res[name]["_flagged"] = flagged
    v1_failed = json.loads(v1.OUT.read_text())["failed"]
    print(f"v1 failed: {v1_failed['arithmetically_false']} / {v1_failed['with_equations']} "
          f"({v1_failed['false_rate_of_covered']:.4f})")
    for k in ("failed", "succeeded_control"):
        r = res[k]
        print(f"v2 {k:18}: {r['arithmetically_false']} / {r['with_equations']} "
              f"({r['false_rate_of_covered']:.4f}), coverage {r['coverage']:.3f}")
    # never-warned solutions containing a flagged step
    never = [s for s, y in rows if y == 0 and s[:-1] and all(l == 1 for _, l in s[:-1])]
    hit = sum(1 for s in never if any(arithmetically_false(t) for t, _ in s[:-1]))
    res["solution_level"] = dict(never_warned=len(never), with_false_step=hit, share=hit / len(never))
    print(f"never-warned: {hit} / {len(never)} ({hit/len(never):.4f})")
    random.seed(20260905)
    sample = random.sample(res["failed"]["_flagged"], 200)
    out_sample = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out_sample:
        with open(out_sample, "w", encoding="utf-8") as f:
            for i, t in enumerate(sample, 1):
                f.write(f"[{i:03d}] {t}\n      EQS: {equations(t)}\n")
        print("wrote", out_sample)
    for k in ("failed", "succeeded_control"):
        del res[k]["_flagged"]
    res["manual_precision_v1"] = dict(sample=200, seed=20260905, genuine=91, sign_error=12,
                                      false_as_written_other=5, rounding_only=39, parser_artifact=53)
    OUT.write_text(json.dumps(res, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
