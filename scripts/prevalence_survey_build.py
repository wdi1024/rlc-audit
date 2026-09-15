#!/usr/bin/env python3
"""Prevalence survey: how often is the routing label validated? (2026-08-20)

The paper's appendix survey covers eight routing systems chosen because we already
cited them, which supports a statement about what those eight report and nothing
more. Reviewers asked, twice, for a number. A number needs a population and a
sample, so this script builds both.

  Population.  arXiv papers from 2023 onward returned by six fixed queries about
  LLM routing, cascades, deferral and selective prediction, deduplicated by arXiv
  ID.  The queries are listed in QUERIES and are not tuned after seeing results.

  Inclusion.  A paper is in scope if it (a) makes a per-query decision among
  models or between answering and deferring, and (b) evaluates that decision
  against a label.  Surveys, position papers, and work whose contribution is
  privacy, serving infrastructure, or hardware are out of scope.  Screening is
  done on the abstract by the rules in `screen()`, and every decision is recorded
  so the screen can be audited.

  Sample.  A fixed-seed random sample of the in-scope pool, so the estimate has a
  denominator and a confidence interval rather than being a list of examples we
  happened to have read.

Stage 1 (this script) builds and screens the pool and draws the sample.
Stage 2 (`prevalence_survey_code.py`) fetches each sampled paper and extracts the
evidence used to code it.

  python3 prevalence_survey_build.py
"""
from __future__ import annotations

import json
import re
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path

import certifi

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis_results"
SEED = 20260820
SAMPLE_N = 40
CTX = ssl.create_default_context(cafile=certifi.where())

QUERIES = [
    'all:"LLM routing" AND all:"large language model"',
    'all:"model cascade" AND all:"language model"',
    'abs:"router" AND abs:"large language models" AND abs:"cost"',
    'abs:"query routing" AND abs:"LLM"',
    'abs:"cascade" AND abs:"deferral" AND abs:"language model"',
    'abs:"selective prediction" AND abs:"large language model"',
]

# --- screening vocabulary, fixed before the pool was inspected ----------------
DECISION = ["rout", "cascad", "defer", "abstain", "select which", "model selection",
            "escalat", "fallback", "选择"]
OUT_OF_SCOPE = ["survey", "position paper", "we survey", "a review of",
                "differential privacy", "federated", "kv cache", "serving system",
                "gpu kernel", "quantization", "speculative decoding"]
EVAL = ["benchmark", "accuracy", "we evaluate", "experiments", "dataset"]


def arxiv(query: str, n: int = 60) -> list[dict]:
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": query, "start": 0, "max_results": n, "sortBy": "relevance"})
    with urllib.request.urlopen(url, timeout=60, context=CTX) as r:
        x = r.read().decode()
    out = []
    for e in re.findall(r"<entry>(.*?)</entry>", x, re.S):
        out.append({
            "id": re.search(r"<id>(.*?)</id>", e).group(1).split("/abs/")[-1],
            "title": re.sub(r"\s+", " ", re.search(r"<title>(.*?)</title>", e, re.S).group(1)).strip(),
            "abstract": re.sub(r"\s+", " ", re.search(r"<summary>(.*?)</summary>", e, re.S).group(1)).strip(),
            "year": int(re.search(r"<published>(\d{4})", e).group(1)),
        })
    return out


def screen(paper: dict) -> tuple[bool, str]:
    """Abstract-level inclusion. Returns (included, reason)."""
    text = (paper["title"] + " " + paper["abstract"]).lower()
    for bad in OUT_OF_SCOPE:
        if bad in text:
            return False, f"out of scope: {bad!r}"
    if not any(k in text for k in DECISION):
        return False, "no per-query decision among models or to defer"
    if not any(k in text for k in EVAL):
        return False, "no empirical evaluation described"
    return True, "in scope"


def main() -> None:
    pool: dict[str, dict] = {}
    for q in QUERIES:
        try:
            for p in arxiv(q):
                if p["year"] >= 2023:
                    pool[p["id"].split("v")[0]] = p
        except Exception as exc:                                  # network hiccup
            print(f"  query failed ({type(exc).__name__}): {q}")
        time.sleep(3)
    print(f"pool from {len(QUERIES)} queries, 2023+: {len(pool)} unique papers")

    scored = []
    for pid, p in sorted(pool.items()):
        ok, why = screen(p)
        scored.append({**p, "arxiv": pid, "included": ok, "screen_reason": why})
    included = [p for p in scored if p["included"]]
    print(f"in scope after screening: {len(included)}")
    from collections import Counter
    print("  exclusion reasons:",
          dict(Counter(p["screen_reason"] for p in scored if not p["included"]).most_common(6)))

    import random
    rng = random.Random(SEED)
    sample = rng.sample(included, min(SAMPLE_N, len(included)))
    sample.sort(key=lambda p: p["arxiv"])
    print(f"\ndrew {len(sample)} at seed {SEED}:")
    for p in sample:
        print(f"  {p['year']} {p['arxiv']:12s} {p['title'][:74]}")

    OUT.mkdir(exist_ok=True)
    (OUT / "prevalence_pool.json").write_text(json.dumps(
        {"queries": QUERIES, "seed": SEED, "n_pool": len(pool),
         "n_included": len(included), "sample_n": len(sample),
         "papers": scored, "sample": [p["arxiv"] for p in sample]}, indent=2))
    print("\n[wrote]", OUT / "prevalence_pool.json")


if __name__ == "__main__":
    main()
