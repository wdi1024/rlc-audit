#!/usr/bin/env python3
"""Survey stage 4: does the release permit an off-span re-evaluation? (2026-08-21)

Stage 3 coded whether each paper checks its operative label against the construct
it claims.  The paper then used that survey to support a second, different claim:
that no surveyed paper releases enough for anyone to run our control.  A reviewer
caught the substitution, and it is a fair catch --- we asserted a property we had not
coded.  This stage codes it.

The control needs two things from a release, and they are separable:

  R1  per-example model outputs.  Without the generated text there is no span to
      re-read a proxy over, whatever else is published.
  R2  the proxy rule, in a form reproducible from the paper --- a keyword list, a
      string-match recipe, a released scorer checkpoint, an exact metric definition.

A paper permits the control only with both.  We look for R1 and R2 separately so the
failure mode is visible: our expectation is that R2 is common (methods sections
describe their metric) and R1 is rare, which would locate the gap in artifact
release rather than in method reporting.

Coding is keyword-triaged and then hand-confirmed against the matched sentence,
which is recorded for every paper so a reader can disagree with any single call.
A promise to release ("code will be available") is not a release: we code the
claim the paper makes about its own artifacts, and mark such cases separately
rather than silently counting them either way.

  python3 prevalence_survey_artifacts.py
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AR = ROOT / "analysis_results"
CACHE = Path("/tmp/prev_papers")

# R1: the paper states it releases per-example generations, not merely code or scores.
OUTPUT_PAT = re.compile(
    r"[^.]{0,220}(?:releas\w+|publish\w+|provide\w+|available at|we make [^.]{0,40}available)"
    r"[^.]{0,120}(?:generation|generated (?:text|output|response)|model output|"
    r"per-(?:example|instance|query) (?:output|response|generation)|"
    r"response(?:s)? (?:for|of) (?:each|every)|full (?:trace|transcript)|"
    r"raw (?:output|response|generation))[^.]{0,220}\.", re.I)

# R2: the proxy rule is reproducible from the paper.
RULE_PAT = re.compile(
    r"[^.]{0,200}(?:keyword|string match|exact match|substring|regular expression|regex|"
    r"we define [^.]{0,60}(?:score|metric)|refusal (?:list|prefix)|"
    r"released? (?:checkpoint|classifier|reward model)|scoring function)[^.]{0,200}\.", re.I)

# A promise is not an artifact.
PROMISE_PAT = re.compile(
    r"[^.]{0,120}(?:will be (?:made )?(?:publicly )?available|will be released|"
    r"upon acceptance|upon publication)[^.]{0,120}\.", re.I)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# Hand confirmations, made by reading the matched sentence in context. True where the
# keyword hit really is a statement that per-example generations are released.
# Every entry here overrides the keyword triage, and the reason is recorded.
R1_CONFIRMED: dict[str, bool] = {
    # "we utilize third-party API providers to obtain model outputs" describes how the
    # authors *got* outputs, not what they publish; the release is code only.
    "2604.09377": False,
}
# A second sweep for dataset-release phrasing (huggingface.co/datasets, Zenodo,
# "dataset is available") surfaced four more papers. All four were read and none
# releases per-example generations paired with the operative label:
#   2603.11957  links SciEntsBank, an input benchmark
#   2604.00660  cites pyGAM's Zenodo record, a software dependency
#   2605.05007  releases Uno-Curriculum, router training data, not scored generations
#   2605.12001  cites a Zenodo record for a third-party artifact
R1_RECALL_CHECKED = ["2603.11957", "2604.00660", "2605.05007", "2605.12001"]

# Papers whose only release statement is a promise.
PROMISE_ONLY: set[str] = set()


def main() -> None:
    pool = json.loads((AR / "prevalence_pool.json").read_text())
    coded = {r["arxiv"]: r for r in json.loads(
        (AR / "prevalence_codings.json").read_text())["codings"]}

    rows = []
    for pid in pool["sample"]:
        prev = coded.get(pid)
        if prev is None or prev["category"] == "X":
            continue
        path = CACHE / f"{pid}.txt"
        text = path.read_text(errors="replace") if path.exists() else ""
        if not text:
            rows.append({"arxiv": pid, "title": prev["title"], "fetched": False,
                         "r1": None, "r2": None, "permits": None,
                         "r1_evidence": [], "r2_evidence": []})
            continue
        o = [m.group(0).strip() for m in OUTPUT_PAT.finditer(text)][:3]
        r = [m.group(0).strip() for m in RULE_PAT.finditer(text)][:3]
        p = [m.group(0).strip() for m in PROMISE_PAT.finditer(text)][:2]
        r1 = R1_CONFIRMED.get(pid, bool(o)) and pid not in PROMISE_ONLY
        r2 = bool(r)
        rows.append({"arxiv": pid, "title": prev["title"], "fetched": True,
                     "r1": r1, "r2": r2, "permits": bool(r1 and r2),
                     "promise_only": pid in PROMISE_ONLY,
                     "r1_evidence": o, "r2_evidence": r, "promise_evidence": p})

    ok = [x for x in rows if x["fetched"]]
    n = len(ok)
    n_r1 = sum(1 for x in ok if x["r1"])
    n_r2 = sum(1 for x in ok if x["r2"])
    n_both = sum(1 for x in ok if x["permits"])

    print(f"in-scope papers with full text: {n} (of {len(rows)} coded in stage 3)\n")
    print(f"  R1  releases per-example generations   {n_r1:2d}/{n}  {n_r1/n:.3f}")
    print(f"  R2  proxy rule reproducible from paper {n_r2:2d}/{n}  {n_r2/n:.3f}")
    lo, hi = wilson(n_both, n)
    print(f"  both (control is runnable)             {n_both:2d}/{n}  {n_both/n:.3f}  "
          f"95% CI [{lo:.3f}, {hi:.3f}]")

    print("\npapers passing R1 (per-example generations released):")
    for x in ok:
        if x["r1"]:
            print(f"  {x['arxiv']}  {x['title'][:62]}")
            if x["r1_evidence"]:
                print(f"      \"{x['r1_evidence'][0][:150]}\"")
    if not any(x["r1"] for x in ok):
        print("  (none)")

    print(f"\nrecall check: {len(R1_RECALL_CHECKED)} further papers carrying dataset-release")
    print("  phrasing were read individually; none releases per-example generations.")
    print("  A paper that released outputs without saying so in its text would be missed,")
    print("  so this is a lower bound on release and an upper bound on our claim.")

    print("\nreading:")
    if n_both == 0:
        print(f"  no sampled paper releases both the outputs and the rule, so the control cannot")
        print(f"  be run on any of them. R2 holds for {n_r2}/{n}: the gap is in artifact release,")
        print("  not in method reporting, and that is the recommendation the paper should make.")
    else:
        print(f"  {n_both}/{n} papers release enough to run the control. The paper cannot say")
        print("  'none' and must name these.")

    out = {"n_in_scope": len(rows), "n_with_text": n,
           "n_r1": n_r1, "n_r2": n_r2, "n_permits": n_both,
           "rate_permits": n_both / n if n else None, "ci_permits": [lo, hi],
           "rows": rows}
    (AR / "prevalence_artifacts.json").write_text(json.dumps(out, indent=2))
    print("\n[wrote]", AR / "prevalence_artifacts.json")


if __name__ == "__main__":
    main()
