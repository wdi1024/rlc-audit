#!/usr/bin/env python3
"""A measured lower bound to pair with the 38.9% upper bound. (2026-09-04)

Review A-3: 38.9% (positive-labeled non-final steps inside failing solutions) is
the set on which the value reading and the step-correctness reading CAN part, not
a measured disagreement rate -- a step that is correct on its own terms keeps its
positive label under both readings. The reviewer's proposed ProcessBench
cross-check is not runnable: ProcessBench annotates solutions from Qwen2/2.5 and
Llama-3 generators, none of which are Math-Shepherd's, so no solution carries
both label types.

What the released files do support is a deterministic lower bound: a step that
contains an arithmetically false equation ("a op b = c" with c wrong) is wrong
under the step-correctness reading no matter what comes later, while its positive
rollout label stays correct under the value reading whenever the mistake is
recoverable. Every such step is a measured disagreement between the two readings.

Design, fixed before results:
  - Parse solutions exactly as mathshepherd_estimand_vs_construct.py does
    (same STEP/ANSWER regexes, same gold matching); first reproduce that
    script's published counts as a referee check.
  - Equation extraction: strict numeric patterns only -- chains of
    number (+|-|*|/|x) number ... = number, commas and $ stripped, no letters
    inside the expression. Percent forms and word math are NOT parsed; missing
    them only loosens the bound.
  - A step is arithmetically false if it contains >= 1 parseable equation whose
    left side does not evaluate to its right side (rel. tol 1e-6).
  - Control: the same rate on positive steps inside SUCCEEDING solutions.
    Parser false positives push both numbers up equally; the finding is the gap.
  - Kill gate (pre-registered): if fewer than 30% of positive steps in failing
    solutions contain any parseable equation, report coverage and stop.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

SHARDS = sorted(Path("/tmp/mshep").glob("*.parquet"))
GOLD = Path("/tmp/gsm8k_gold.json")
OUT = (Path(__file__).resolve().parent.parent / "analysis_results"
       / "mathshepherd_arithmetic_lower_bound.json")

STEP = re.compile(r"Step\s+(\d+):\s*(.*?)\s*([+\-])\s*$")
ANSWER = re.compile(r"answer is:?\s*(.*)$", re.I)

# calculator annotations <<3+4=7>> are their own equations; strip the markers so
# the inner expression is parsed once, not twice
CALC = re.compile(r"<<([^>]*)>>")
NUM = r"\d[\d,]*(?:\.\d+)?"
EQ = re.compile(rf"({NUM}(?:\s*[+\-*/x×]\s*{NUM})+)\s*=\s*({NUM})")


def norm_num(s):
    s = re.sub(r"[^\d\.\-/]", "", str(s)).rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return s


def norm_text(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _eval_ltr(expr: str):
    toks = re.findall(r"\d+(?:\.\d+)?|[+\-*/]", expr)
    if not toks or len(toks) % 2 == 0:
        return None
    try:
        val = float(toks[0])
        for op, num in zip(toks[1::2], toks[2::2]):
            n = float(num)
            val = (val + n if op == "+" else val - n if op == "-"
                   else val * n if op == "*" else val / n)
        return val
    except Exception:
        return None


def _parse_eq(mm):
    lhs, rhs = mm.group(1), mm.group(2)
    expr = lhs.replace(",", "").replace("x", "*").replace("×", "*")
    if re.search(r"[^\d\.\s+\-*/]", expr):
        return None
    try:
        val_std = eval(expr, {"__builtins__": {}}, {})
        tgt = float(rhs.replace(",", ""))
    except Exception:
        return None
    return expr, tgt, val_std, _eval_ltr(expr)


def step_equations(text: str):
    out = []
    # calculator annotations <<3+4=7>> carry the equation verbatim; parse the
    # inside as authoritative, then DELETE the span so its fragments cannot pair
    # with surrounding numbers (the bug the control caught: "20 * 26 = <<20*26=520>>520"
    # unwrapped to "20 * 26 = 20").
    for m in CALC.finditer(text):
        mm = EQ.search(m.group(1))
        if mm:
            p = _parse_eq(mm)
            if p:
                out.append(p)
    text = CALC.sub(" ", text)
    for mm in EQ.finditer(text):
        p = _parse_eq(mm)
        if p:
            out.append(p)
    return out


def arithmetically_false(text: str):
    """True iff some equation is false under BOTH standard precedence and
    left-to-right chain evaluation (models often write "a + b * c = d" meaning
    (a+b)*c; accepting either reading keeps the bound conservative)."""
    eqs = step_equations(text)
    if not eqs:
        return None  # no parseable equation
    for _, tgt, val_std, val_ltr in eqs:
        denom = max(abs(tgt), 1e-9)
        std_ok = abs(val_std - tgt) / denom <= 1e-6
        ltr_ok = val_ltr is not None and abs(val_ltr - tgt) / denom <= 1e-6
        if not std_ok and not ltr_ok:
            return True
    return False


def main() -> None:
    gold = json.loads(GOLD.read_text())
    rows, n_seen = [], 0
    for shard in SHARDS:
        df = pd.read_parquet(shard, columns=["label", "task"])
        for lab in df[df["task"] == "GSM8K"]["label"]:
            n_seen += 1
            if "Step 1:" not in lab:
                continue
            problem = lab.split("Step 1:")[0].strip()
            key = norm_text(problem)
            if key not in gold:
                continue
            steps = []
            ok = True
            for line in lab[len(problem):].split("\n"):
                m = STEP.match(line.strip())
                if not m:
                    ok = False
                    break
                steps.append((m.group(2), 1 if m.group(3) == "+" else 0))
            if not ok or len(steps) < 3:
                continue
            m = ANSWER.search(steps[-1][0])
            if not m:
                continue
            rows.append((steps, int(norm_num(m.group(1)) == gold[key])))

    # ---- referee check against the published run
    wrong = [s for s, y in rows if y == 0]
    pos_in_failed = tot_in_failed = all_pos = 0
    for steps, y in rows:
        if y:
            continue
        body = steps[:-1]
        tot_in_failed += len(body)
        p = sum(l for _, l in body)
        pos_in_failed += p
        if body and p == len(body):
            all_pos += 1
    print(f"solutions parsed: {len(rows)}   failed: {len(wrong)}")
    print(f"non-final steps in failed: {tot_in_failed}  positive: {pos_in_failed} "
          f"({pos_in_failed/tot_in_failed:.4f})  all-positive solutions: {all_pos} "
          f"({all_pos/len(wrong):.4f})")

    # ---- arithmetic lower bound
    def tally(y_val):
        n_pos = n_cov = n_false = 0
        for steps, y in rows:
            if y != y_val:
                continue
            for text, l in steps[:-1]:
                if l != 1:
                    continue
                n_pos += 1
                r = arithmetically_false(text)
                if r is None:
                    continue
                n_cov += 1
                if r:
                    n_false += 1
        return n_pos, n_cov, n_false

    fp, fc, ff = tally(0)   # failing solutions
    sp, sc, sf = tally(1)   # succeeding solutions (control)
    cov_f = fc / fp if fp else 0.0
    print(f"\npositive steps in FAILED solutions:    {fp}, with equations {fc} "
          f"(coverage {cov_f:.3f}), arithmetically false {ff} "
          f"({ff/fc:.4f} of covered, {ff/fp:.4f} of all positive steps)")
    print(f"positive steps in SUCCEEDED solutions: {sp}, with equations {sc} "
          f"(coverage {sc/sp:.3f}), arithmetically false {sf} "
          f"({sf/sc:.4f} of covered, {sf/sp:.4f} of all positive steps)")
    if cov_f < 0.30:
        print("KILL GATE: equation coverage below 0.30 -- reporting only.")

    OUT.write_text(json.dumps({
        "referee_check": {
            "solutions": len(rows), "failed_solutions": len(wrong),
            "nonfinal_steps_in_failed": tot_in_failed,
            "positive_steps_in_failed": pos_in_failed,
            "positive_rate_in_failed": pos_in_failed / tot_in_failed,
            "all_positive_failed_solutions": all_pos,
            "all_positive_failed_rate": all_pos / len(wrong),
        },
        "coverage_kill_gate": 0.30,
        "failed": {"positive_steps": fp, "with_equations": fc,
                   "arithmetically_false": ff,
                   "false_rate_of_covered": ff / fc if fc else None,
                   "false_rate_of_all_positive": ff / fp if fp else None},
        "succeeded_control": {"positive_steps": sp, "with_equations": sc,
                              "arithmetically_false": sf,
                              "false_rate_of_covered": sf / sc if sc else None,
                              "false_rate_of_all_positive": sf / sp if sp else None},
    }, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
