#!/usr/bin/env python3
"""Survey stage 5: what does the deployed score actually read? (2026-08-21)

The audit's failure mode needs a score that reads generated text. A router that
scores only the query cannot exhibit containment at all: there is no shared span,
because the score never touches the output the proxy is computed from. RouteLLM is
exactly this case, which is why it clears by construction
(Appendix `app:routellm`).

That makes one number decisive for how much of deployed practice this paper can
even reach, and we have not measured it. This stage does. For each in-scope paper
we code where the operative score reads:

  QUERY    the score is a function of the prompt alone -- an embedding, a
           difficulty predictor, a learned router head -- and is computed before
           any candidate response exists. Structurally immune.
  OUTPUT   the score reads generated text: a cascade scoring the weak model's
           answer, self-consistency over samples, verbalized confidence, a
           process reward model over a partial chain, a judge over responses.
           Exposed, if its proxy shares the span.
  BOTH     the paper deploys both, or the score reads query and response jointly.
  UNCLEAR  the text does not say.

The honest reading cuts both ways and we state it before running: if most deployed
routers are QUERY-only, the failure this paper formalises is real but narrow, and
we should say so in the limitations rather than let a reader assume otherwise.

  python3 prevalence_survey_scoring_locus.py
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AR = ROOT / "analysis_results"
CACHE = Path("/tmp/prev_papers")

QUERY_PAT = re.compile(
    r"[^.]{0,200}(?:from the (?:query|prompt|input)|query (?:embedding|representation|features)|"
    r"before (?:generation|generating|any response)|without generating|"
    r"predict[^.]{0,60}(?:from|given) the (?:query|prompt|input)|"
    r"prompt-only|query-only|routes? (?:the )?quer(?:y|ies) to)[^.]{0,200}\.", re.I)
OUTPUT_PAT = re.compile(
    r"[^.]{0,200}(?:scores? the (?:response|answer|output|completion|generation)|"
    r"(?:response|answer|output|completion)[- ]level (?:score|confidence|quality)|"
    r"self-consistency|verbali[sz]ed confidence|log-?prob(?:abilit(?:y|ies))? of the (?:answer|output)|"
    r"process reward model|partial (?:chain|solution)|judge[^.]{0,40}(?:response|output)|"
    r"cascade[^.]{0,60}(?:weak|small)[^.]{0,40}(?:first|answer|output))[^.]{0,200}\.", re.I)

# Hand codings. Each rests on the sentence in which the paper defines what its
# operative score consumes; the keyword triage below is retained only to surface
# candidates, because it proved unreliable on its own (it matched incidental
# phrases such as "before generation" and "prompt-only baselines").
OVERRIDE: dict[str, tuple[str, str]] = {
    # --- score reads the query alone: structurally immune to containment ---
    "2404.14618": ("QUERY", "routes on predicted query difficulty"),
    "2504.07113": ("QUERY", "R(q, M) takes a query and picks a model"),
    "2509.02718": ("QUERY", "directs queries on model and query features"),
    "2509.09782": ("QUERY", "cross-attention over query and model embeddings"),
    "2509.25535": ("QUERY", "selects a model for each query"),
    "2510.07429": ("QUERY", "contextual bandit over prompt features"),
    "2510.19208": ("QUERY", "each agent judges its competence on the query"),
    "2511.16883": ("QUERY", "personalised selection from query and user preferences"),
    "2601.05903": ("QUERY", "classifier maps each input query to an architecture"),
    "2601.06220": ("QUERY", "capability mappings predict performance per query"),
    "2601.17814": ("QUERY", "query-level model selection"),
    "2603.28972": ("QUERY", "routing decided solely on prompt task complexity"),
    "2604.09377": ("QUERY", "assigns models to each query under preferences"),
    "2604.15728": ("QUERY", "selects among models to serve a given query"),
    "2605.05007": ("QUERY", "given a task, emits a plan and routing decision"),
    "2605.12001": ("QUERY", "query-level routing assigns each request"),
    "2605.18015": ("QUERY", "routes queries between lookup and generative paths"),
    "2606.02581": ("QUERY", "routing separated from retrieval and generation"),
    "2607.23765": ("QUERY", "LP over routing actions at each step"),
    # --- score reads generated text: exposed if its proxy shares the span ---
    "2404.10136": ("OUTPUT", "deferral score over generated output tokens"),
    "2506.11887": ("OUTPUT", "confidence over the base model's candidate answer"),
    "2511.07396": ("OUTPUT", "sampled responses and their confidence scores"),
    "2602.02711": ("OUTPUT", "step-level selection on interaction trajectory"),
    "2602.11931": ("OUTPUT", "conditions on intrinsic generation uncertainty"),
    "2603.11957": ("OUTPUT", "grade predicted from question and student answer"),
    "2603.24704": ("OUTPUT", "risk score over model outputs"),
    "2604.23530": ("OUTPUT", "multi-turn routing conditioned on history"),
    "2604.25855": ("OUTPUT", "confidence assigned to each answer, then abstain"),
    "2607.00053": ("OUTPUT", "conditions on the partial agent trajectory"),
    "2607.08665": ("OUTPUT", "verifier scores draws before rerouting"),
    "2607.18240": ("OUTPUT", "selective prediction over produced verdicts"),
    "2607.25018": ("OUTPUT", "per-tier prediction sets from self-consistency"),
    # --- the paper does not say ---
    "2502.12464": ("UNCLEAR", "no statement of what the router consumes"),
    "2509.19996": ("UNCLEAR", "no statement of what the router consumes"),
    "2602.02386": ("UNCLEAR", "no statement of what the router consumes"),
    "2604.00660": ("UNCLEAR", "no statement of what the router consumes"),
    "2606.31163": ("UNCLEAR", "no statement of what the router consumes"),
    "2607.24875": ("UNCLEAR", "no statement of what the router consumes"),
}


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
    coded = {r["arxiv"]: r for r in
             json.loads((AR / "prevalence_codings.json").read_text())["codings"]}

    rows = []
    for pid in pool["sample"]:
        c = coded.get(pid)
        if c is None or c["category"] == "X":
            continue
        f = CACHE / f"{pid}.txt"
        text = f.read_text(errors="replace") if f.exists() else ""
        q = [re.sub(r"\s+", " ", m.group(0)).strip() for m in QUERY_PAT.finditer(text)][:3]
        o = [re.sub(r"\s+", " ", m.group(0)).strip() for m in OUTPUT_PAT.finditer(text)][:3]
        if pid in OVERRIDE:
            locus, why = OVERRIDE[pid]
        elif q and o:
            locus, why = "BOTH", "both patterns present"
        elif q:
            locus, why = "QUERY", "query-side language only"
        elif o:
            locus, why = "OUTPUT", "output-side language only"
        else:
            locus, why = "UNCLEAR", "neither pattern matched"
        rows.append({"arxiv": pid, "title": c["title"], "locus": locus,
                     "why": why, "query_evidence": q, "output_evidence": o})

    n = len(rows)
    counts = {k: sum(1 for r in rows if r["locus"] == k)
              for k in ("QUERY", "OUTPUT", "BOTH", "UNCLEAR")}
    print(f"in-scope papers: {n}\n")
    for k in ("QUERY", "OUTPUT", "BOTH", "UNCLEAR"):
        lo, hi = wilson(counts[k], n)
        print(f"  {k:8s} {counts[k]:2d}/{n}  {counts[k]/n:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    exposed = counts["OUTPUT"] + counts["BOTH"]
    lo, hi = wilson(exposed, n)
    print(f"\n  reads generated text at all (OUTPUT or BOTH): {exposed}/{n} "
          f"= {exposed/n:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  structurally immune (QUERY only):              {counts['QUERY']}/{n}")

    print("\nQUERY-coded papers:")
    for r in rows:
        if r["locus"] == "QUERY":
            print(f"  {r['arxiv']}  {r['title'][:60]}")
    print("\nreading:")
    if exposed / n >= 0.5:
        print(f"  A majority of sampled routers read generated text, so the failure mode is")
        print(f"  reachable in most of the deployed practice we sampled; the release gap, not")
        print("  the architecture, is what prevents it from being measured.")
    else:
        print(f"  Most sampled routers score the query alone and are immune by construction.")
        print("  The failure this paper formalises is then real but narrow, and the limitation")
        print("  must be stated as such.")

    out = {"n": n, "counts": counts, "n_exposed": exposed,
           "rate_exposed": exposed / n, "ci_exposed": [lo, hi], "rows": rows}
    p = AR / "prevalence_scoring_locus.json"
    p.write_text(json.dumps(out, indent=2))
    print("\n[wrote]", p)


if __name__ == "__main__":
    main()
