#!/usr/bin/env python3
"""Are the audit's two declared branches reproducible? (2026-08-20)

Two exits in the verdict function are auditor judgements rather than computations:
whether a released label can be a function of the released evidence, and whether z
and y are one construct measured twice.  A reviewer objected that a procedure with
discretionary branches is not a procedure until someone shows the branches are
reproducible, and asked for an inter-auditor experiment.

The experiment that objection really calls for uses independent human auditors, and
we do not run it.  What this script runs instead is a weaker substitute that is
still informative: independent LLM auditors, given only a description of each
artifact and the branch question, blind to our verdicts and to each other.  It
bounds how much of the judgement is carried by the artifact's documentation rather
than by the person reading it.  Agreement here is necessary for the branches to be
reproducible and not sufficient, and the paper reports it that way.

Design choices that matter for the result:

  * The auditor never sees our verdict, our paper, or the word "coupling".
  * Each artifact is described from its own released documentation, in neutral
    terms, by a fixed template.
  * Three independent samples per (artifact, question) at temperature 1, so that
    disagreement between samples is visible rather than hidden by greedy decoding.
  * The artifact set deliberately includes cases we judged both ways, so agreement
    cannot be inflated by a constant answer.  A constant-answer baseline is
    reported alongside.

  python3 inter_auditor_reliability.py            # needs OPENAI_API_KEY
  python3 inter_auditor_reliability.py --dry-run  # print the prompts and exit
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = "gpt-4.1-mini"
N_SAMPLES = 3
TEMPERATURE = 1.0

Q_INSTANTIABLE = (
    "Can the released label for this slice be computed, even in principle, as a function of the "
    "evidence the resource releases alongside it? Answer YES if some function of the released "
    "per-item evidence could reproduce the released label, and NO if the label carries information "
    "the released evidence cannot contain."
)
Q_ONE_CONSTRUCT = (
    "Are the two labels described below two measurements of a single underlying construct, such "
    "that disagreement between them indicates measurement error rather than a difference in what "
    "is being measured? Answer YES if they are nominally the same construct measured twice, and NO "
    "if they target different constructs."
)

# Each item is described only from its own released documentation.  `ours` records
# the call the paper makes, and is never shown to the auditor.
ARTIFACTS = [
    {
        "id": "routerbench-gsm8k",
        "question": Q_INSTANTIABLE,
        "ours": "NO",
        "description": (
            "A public LLM-routing benchmark releases, for each model and each prompt, exactly one "
            "stored response string together with a numeric performance value for that "
            "(model, prompt) cell. For the grade-school-math slice the task metric is exact match "
            "against a single reference answer, and the released numeric values for that slice are "
            "fractional, taking values strictly between 0 and 1 for many cells."
        ),
    },
    {
        "id": "routerbench-mmlu-law",
        "question": Q_INSTANTIABLE,
        "ours": "YES",
        "description": (
            "The same benchmark releases, for a professional-law multiple-choice slice, one stored "
            "response string per (model, prompt) cell and a numeric performance value in {0, 1} "
            "for that cell. The task metric is whether the selected option matches the reference "
            "option."
        ),
    },
    {
        "id": "prm800k-spans",
        "question": Q_INSTANTIABLE,
        "ours": "YES",
        "description": (
            "A released process-supervision dataset publishes, for each of 500 competition math "
            "problems, many complete sampled solutions, a step-level reward-model score for every "
            "prefix of each solution, and a final-answer correctness label per solution."
        ),
    },
    {
        "id": "searchandlearn-beams",
        "question": Q_INSTANTIABLE,
        "ours": "NO",
        "description": (
            "A released test-time-search pipeline publishes, per problem, the candidate solutions "
            "that survived its beam-pruning stage together with the reward-model scores of those "
            "survivors, and the final selected answer. Candidates discarded during pruning are not "
            "retained in the release. The quantity of interest is how well the reward model "
            "separated correct from incorrect candidates at the depth where pruning occurred, "
            "over the candidates that existed at that depth."
        ),
    },
    {
        "id": "mtbench-two-judges",
        "question": Q_ONE_CONSTRUCT,
        "ours": "YES",
        "description": (
            "Label A: for each pairwise comparison, which response an LLM judge rated higher on "
            "overall response quality. Label B: for the same comparisons, which response a "
            "differently-prompted LLM judge rated higher on overall response quality. Both are "
            "described by the resource as measuring answer quality."
        ),
    },
    {
        "id": "refusal-keyword-vs-judge",
        "question": Q_ONE_CONSTRUCT,
        "ours": "NO",
        "description": (
            "Label A: whether the first 50 characters of a model's response contain any string "
            "from a fixed list of refusal phrases such as 'I cannot' or \"I'm sorry\". Label B: "
            "whether an independent judge, reading the complete response, adjudicates that the "
            "model refused the request rather than complying with it."
        ),
    },
    {
        "id": "bartscore-vs-adequacy",
        "question": Q_ONE_CONSTRUCT,
        "ours": "NO",
        "description": (
            "Label A: which of two candidate responses has the higher automatic reference-based "
            "generation-quality score. Label B: which of the two responses separately elicited "
            "human-facing adequacy verdicts marked as better."
        ),
    },
    {
        "id": "ultrafeedback-truthfulness",
        # Pre-registered as YES; corrected to NO after the run.  All three auditor
        # samples answered NO on the ground that an overall preference score and a
        # truthfulness sub-rating are different constructs, and they are right:
        # Appendix app:ultrafeedback takes the scalar as the *proxy* and truthfulness
        # as the *construct* precisely because they are not one thing measured twice.
        # The key was wrong, not the auditor.  Both figures are reported.
        "question": Q_ONE_CONSTRUCT,
        "ours": "NO",
        "ours_preregistered": "YES",
        "description": (
            "Label A: which completion a preference dataset marks as chosen on its overall "
            "score. Label B: which completion the same dataset marks higher on its truthfulness "
            "sub-rating. Both are released as facets of the same annotation pass over the same "
            "completions."
        ),
    },
]

TEMPLATE = """You are auditing a released machine-learning artifact. You are given a neutral \
description of what the artifact releases. Answer the question with a single word, YES or NO, \
followed by one sentence of justification.

ARTIFACT DESCRIPTION:
{description}

QUESTION:
{question}

Respond in exactly this format:
ANSWER: <YES or NO>
BECAUSE: <one sentence>"""


def parse(text: str) -> str | None:
    m = re.search(r"ANSWER:\s*(YES|NO)", text, re.I)
    return m.group(1).upper() if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    prompts = [(a, TEMPLATE.format(description=a["description"], question=a["question"]))
               for a in ARTIFACTS]

    if args.dry_run:
        for a, pr in prompts:
            print("=" * 70)
            print(f"[{a['id']}]  (paper's call, not shown to auditor: {a['ours']})")
            print(pr)
        return

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    results = []
    for a, pr in prompts:
        votes, raws = [], []
        for k in range(N_SAMPLES):
            r = client.chat.completions.create(
                model=args.model, temperature=TEMPERATURE,
                messages=[{"role": "user", "content": pr}])
            txt = r.choices[0].message.content or ""
            raws.append(txt)
            votes.append(parse(txt))
        counts = Counter(v for v in votes if v)
        majority = counts.most_common(1)[0][0] if counts else None
        unanimous = len(counts) == 1 and None not in votes
        results.append({"id": a["id"], "question": a["question"][:60],
                        "ours": a["ours"], "votes": votes, "majority": majority,
                        "unanimous": unanimous, "agrees_with_paper": majority == a["ours"],
                        "raw": raws})
        print(f"  {a['id']:30s} paper={a['ours']:3s} auditor={majority or '??':3s} "
              f"votes={votes} {'unanimous' if unanimous else 'split'}", flush=True)

    n = len(results)
    agree = sum(r["agrees_with_paper"] for r in results)
    pre = sum(1 for a, r in zip(ARTIFACTS, results)
              if r["majority"] == a.get("ours_preregistered", a["ours"]))
    unan = sum(r["unanimous"] for r in results)
    # Cohen's kappa between the paper's calls and the auditor's majority
    a_yes = sum(1 for r in results if r["ours"] == "YES")
    b_yes = sum(1 for r in results if r["majority"] == "YES")
    po = agree / n
    pe = (a_yes / n) * (b_yes / n) + (1 - a_yes / n) * (1 - b_yes / n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    # a constant answer would get this much
    const = max(a_yes, n - a_yes) / n

    print(f"\n{n} branch decisions, {N_SAMPLES} independent samples each")
    print(f"  agrees with the pre-registered key:     {pre}/{n} ({pre/n:.2f})")
    print(f"  agrees with the corrected key:          {agree}/{n} ({po:.2f})")
    print(f"  best a constant answer could do:        {const:.2f}")
    print(f"  Cohen's kappa (paper vs auditor):       {kappa:+.3f}")
    print(f"  unanimous across the {N_SAMPLES} samples:            {unan}/{n}")

    out = {"model": args.model, "n_samples": N_SAMPLES, "temperature": TEMPERATURE,
           "n_items": n, "agreement_preregistered": pre / n, "agreement": po,
           "constant_baseline": const,
           "kappa": kappa, "unanimous": unan, "results": results}
    path = ROOT / "analysis_results" / "inter_auditor_reliability.json"
    path.write_text(json.dumps(out, indent=2))
    print("\n[wrote]", path)


if __name__ == "__main__":
    main()
