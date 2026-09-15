#!/usr/bin/env python3
"""Prevalence survey stage 2: fetch the sampled papers and extract coding evidence.

Stage 1 drew a fixed-seed random sample of routing/cascade/deferral papers. This
stage downloads each one's full text and pulls out the passages a human needs in
order to code it, rather than coding it by keyword count. Two things are extracted:

  label_evidence       where the paper says what its decision is evaluated against
  validation_evidence  any passage suggesting the paper checked that this label
                       agrees with the construct it claims -- a human study, an
                       agreement statistic, a correlation with human ratings

The output is a JSON file with the quotes attached to each paper, so the coding
step is auditable and a reader can disagree with any individual call.

Nothing here decides anything. Keyword hits are candidate evidence; a paper with
no hits still has to be read before it is coded as "no validation", and the
absence of hits is recorded as such.

  python3 prevalence_survey_fetch.py
"""
from __future__ import annotations

import json
import re
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

import certifi

ROOT = Path(__file__).resolve().parent.parent
AR = ROOT / "analysis_results"
CACHE = Path("/tmp/prev_papers")
CTX = ssl.create_default_context(cafile=certifi.where())
UA = {"User-Agent": "Mozilla/5.0 (research survey; contact via paper)"}

LABEL_PAT = re.compile(
    r"(ground[- ]truth|gold label|we (?:use|adopt|evaluate|measure)[^.]{0,120}"
    r"(?:accuracy|exact match|correctness|win rate|judge|score)|"
    r"label(?:ed|s)? (?:are|is|by)[^.]{0,120})", re.I)
VALIDATION_PAT = re.compile(
    r"([^.]{0,200}(?:human (?:evaluation|annotat|study|judg|agreement)|"
    r"inter-annotator|inter-rater|Cohen'?s kappa|Krippendorff|"
    r"agreement with human|correlat\w+ with human|validate[^.]{0,60}label|"
    r"label (?:quality|validity|noise) (?:check|study|analysis))[^.]{0,200}\.)", re.I)


def fetch(arxiv_id: str) -> str | None:
    CACHE.mkdir(exist_ok=True)
    cached = CACHE / f"{arxiv_id}.txt"
    if cached.exists():
        return cached.read_text(errors="replace")
    for url in (f"https://arxiv.org/html/{arxiv_id}v1",
                f"https://arxiv.org/html/{arxiv_id}",
                f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=45, context=CTX) as r:
                html = r.read().decode("utf-8", "replace")
            if len(html) < 4000:
                continue
            text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"&[a-z]+;", " ", text)
            text = re.sub(r"\s+", " ", text)
            if len(text) < 3000:
                continue
            cached.write_text(text)
            return text
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            continue
        except Exception:
            continue
    return None


def main() -> None:
    pool = json.loads((AR / "prevalence_pool.json").read_text())
    by_id = {p["arxiv"]: p for p in pool["papers"]}
    out = []
    for i, pid in enumerate(pool["sample"], 1):
        meta = by_id[pid]
        text = fetch(pid)
        rec = {"arxiv": pid, "title": meta["title"], "year": meta["year"],
               "abstract": meta["abstract"], "fetched": text is not None}
        if text:
            rec["n_chars"] = len(text)
            rec["label_evidence"] = list(dict.fromkeys(
                m.group(0).strip()[:260] for m in LABEL_PAT.finditer(text)))[:6]
            rec["validation_evidence"] = list(dict.fromkeys(
                m.group(1).strip()[:320] for m in VALIDATION_PAT.finditer(text)))[:8]
        else:
            rec["label_evidence"] = []
            rec["validation_evidence"] = []
        out.append(rec)
        flag = "" if text else "  [FETCH FAILED]"
        print(f"{i:3d}/{len(pool['sample'])} {pid:12s} "
              f"label-hits {len(rec['label_evidence']):2d}  "
              f"validation-hits {len(rec['validation_evidence']):2d}{flag}", flush=True)
        if not (CACHE / f"{pid}.txt").exists():
            time.sleep(2)

    n_ok = sum(r["fetched"] for r in out)
    n_val = sum(1 for r in out if r["validation_evidence"])
    print(f"\nfetched {n_ok}/{len(out)}; "
          f"{n_val} have at least one candidate validation passage "
          f"(candidates, not codings)")
    (AR / "prevalence_evidence.json").write_text(json.dumps(out, indent=2))
    print("[wrote]", AR / "prevalence_evidence.json")


if __name__ == "__main__":
    main()
