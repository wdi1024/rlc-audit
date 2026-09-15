#!/usr/bin/env python3
"""Prevalence survey stage 3: the codings, and the estimate they support.

Stage 1 drew a fixed-seed random sample of 40 routing/cascade/deferral papers from
a screened pool of 125.  Stage 2 fetched all 40 and extracted candidate evidence.
This file records how each paper was coded, with the passage the coding rests on,
so that a reader can disagree with any individual call and recompute.

Categories, decided before coding:

  A  the paper reports an in-setup measurement relating its operative label to a
     construct-level label -- a human study on its own data, an agreement
     statistic against expert judgement, or explicit modelling of the gap between
     a proxy label and a gold one.
  B  the paper names the gap between its label and the construct it claims, but
     does not measure the agreement.
  C  neither.
  X  out of scope once the full text is read, with a reason.

Validity by citation -- "LLM judges have been shown to correlate with humans
[ref]" -- is category C, not A.  The claim under test is whether authors check
their own labels, and a citation to someone else's correlation on someone else's
data is exactly the substitution the paper is about.

  python3 prevalence_survey_code.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AR = ROOT / "analysis_results"

CODINGS: dict[str, tuple[str, str]] = {
    # --- A: measured, in setup -------------------------------------------------
    "2509.25535": ("A", "Builds routing on both gold-standard and preference-based labels and "
                        "models the shift between them explicitly; the discrepancy is the "
                        "paper's object rather than an assumption."),
    "2603.11957": ("A", "Reports QWK against expert graders on its own data and compares it to "
                        "inter-rater reliability among human graders."),
    "2604.25855": ("A", "'we run a small human evaluation on 200 randomly chosen training "
                        "samples and find that human annotations match the same g_coh label in "
                        "92% of cases.'"),
    # --- B: named, not measured ------------------------------------------------
    "2606.02581": ("B", "'The lexical quality proxy is a weak surrogate for true answer "
                        "quality; results may differ under human evaluation.' Named as a "
                        "limitation; no agreement measured."),
    "2607.08665": ("B", "Ablates sensitivity to verifier quality ('gains are verifier-gated, "
                        "shrinking as verifier quality degrades') but does not measure the "
                        "verifier's agreement with the construct."),
    # --- X: out of scope on full text -----------------------------------------
    "2601.20352": ("X", "Routes retrieval granularity inside one agent's memory; no selection "
                        "among models and no deferral decision."),
    "2608.16203": ("X", "Speech-retrieval benchmark; no routing or deferral decision in the "
                        "sense screened for."),
}
DEFAULT = ("C", "No in-setup check of the operative label against the claimed construct; "
                "where human agreement appears it is a citation to prior work or a "
                "reference-list entry.")


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> None:
    pool = json.loads((AR / "prevalence_pool.json").read_text())
    ev = {r["arxiv"]: r for r in json.loads((AR / "prevalence_evidence.json").read_text())}

    rows = []
    for pid in pool["sample"]:
        cat, why = CODINGS.get(pid, DEFAULT)
        rows.append({"arxiv": pid, "title": ev[pid]["title"], "year": ev[pid]["year"],
                     "category": cat, "justification": why,
                     "candidate_passages": ev[pid]["validation_evidence"][:3]})

    in_scope = [r for r in rows if r["category"] != "X"]
    n = len(in_scope)
    a = sum(1 for r in in_scope if r["category"] == "A")
    b = sum(1 for r in in_scope if r["category"] == "B")

    lo_a, hi_a = wilson(a, n)
    lo_ab, hi_ab = wilson(a + b, n)

    print(f"pool {pool['n_pool']} -> screened in {pool['n_included']} -> sampled "
          f"{pool['sample_n']} (seed {pool['seed']})")
    print(f"out of scope on full text: {len(rows)-n}; in scope: {n}\n")
    print(f"  A  measured in setup            {a:2d}/{n}  "
          f"{a/n:.3f}  95% CI [{lo_a:.3f}, {hi_a:.3f}]")
    print(f"  B  named but not measured       {b:2d}/{n}  {b/n:.3f}")
    print(f"  C  neither                      {n-a-b:2d}/{n}  {(n-a-b)/n:.3f}")
    print(f"\n  A or B (gap acknowledged at all) {a+b:2d}/{n}  "
          f"{(a+b)/n:.3f}  95% CI [{lo_ab:.3f}, {hi_ab:.3f}]")

    print("\ncoded A:")
    for r in in_scope:
        if r["category"] == "A":
            print(f"  {r['arxiv']} {r['title'][:64]}")
    print("coded B:")
    for r in in_scope:
        if r["category"] == "B":
            print(f"  {r['arxiv']} {r['title'][:64]}")

    out = {"n_pool": pool["n_pool"], "n_screened_in": pool["n_included"],
           "n_sampled": pool["sample_n"], "seed": pool["seed"],
           "n_in_scope": n, "n_A": a, "n_B": b, "n_C": n - a - b,
           "rate_A": a / n, "ci_A": [lo_a, hi_a],
           "rate_AB": (a + b) / n, "ci_AB": [lo_ab, hi_ab],
           "codings": rows}
    (AR / "prevalence_codings.json").write_text(json.dumps(out, indent=2))
    print("\n[wrote]", AR / "prevalence_codings.json")


if __name__ == "__main__":
    main()
