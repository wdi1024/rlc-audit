#!/usr/bin/env python3
"""Does the 0/38 release finding survive looking at the repositories? (2026-08-22)

Our release condition R1 -- whether a paper publishes the per-example generations a
span-recomputable proxy would need -- was coded from paper text alone, and we said so
in the paper: "a paper that released generations without saying so in its text would
be missed." A reviewer points out that this makes a headline number rest on the
weakest available evidence, and that our own manuscript contains the counterexample:
we report that RouteLLM releases 109,101 rows of full responses.

So we open the repositories. For each sampled paper we extract GitHub and Hugging
Face links from its text, resolve the repository file listing through the API, and
look for artifacts that would carry per-example generations -- files or directories
named for outputs, generations, responses, completions, predictions or samples, in
a serialization that stores one record per example.

Three codings result, and the third is the one text-only coding hid:

  RELEASED        a repository we can reach contains per-example generation files
  NOT RELEASED    a repository we can reach, with no such artifact
  UNDETERMINED    no link in the text, or a link we cannot resolve

We do not fold UNDETERMINED into either side. Filename evidence is weaker than
opening every file, and we mark the coding as such rather than claiming to have
read the contents.

  python3 prevalence_survey_repo_audit.py
"""
from __future__ import annotations

import json
import re
import ssl
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import certifi

CTX = ssl.create_default_context(cafile=certifi.where())
UA = {"User-Agent": "release-audit/1.0", "Accept": "application/vnd.github+json"}
CACHE = Path("/tmp/prev_papers")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis_results" / "prevalence_repo_audit.json"

# Repositories that belong to a framework or a model release rather than to the
# paper's own artifacts; a link to one of these is not evidence the paper released
# anything.
# A link to a framework or a model release is not evidence that the paper released
# anything: the first pass coded 2603.28972 as RELEASED on the strength of test
# fixtures inside litellm and NeMo-Guardrails, which are tools it uses.
THIRD_PARTY = re.compile(
    r"github\.com/(langchain-ai|meta-llama|huggingface|openai|google-research|"
    r"pytorch|facebookresearch|vllm-project|EleutherAI|tatsu-lab|BerriAI|NVIDIA|"
    r"ollama|dswah|microsoft|google|deepmind|Significant-Gravitas|run-llama|"
    r"scikit-learn|numpy|pandas-dev|explosion|allenai|stanfordnlp)/", re.I)

GEN_HINT = re.compile(
    r"(generation|generations|output|outputs|response|responses|completion|"
    r"completions|prediction|predictions|sample|samples|trace|traces|dump)", re.I)
DATA_EXT = re.compile(r"\.(jsonl|json|csv|tsv|parquet|pkl|zip|gz)$", re.I)


CACHE_FILE = Path("/tmp/repo_audit_cache.json")
_cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}


def _save():
    CACHE_FILE.write_text(json.dumps(_cache))


def api(url, _retries=2):
    """Unauthenticated GitHub allows 60 calls an hour, and a run that trips the limit
    reports 403 for repositories that are perfectly reachable -- which silently
    inflates UNDETERMINED, the one category this audit exists to measure honestly.
    So a 403 waits for the window to reset instead of being recorded as a result."""
    if url in _cache:
        return _cache[url]
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                    timeout=30, context=CTX) as r:
            out = json.loads(r.read())
        _cache[url] = out
        _save()
        return out
    except urllib.error.HTTPError as e:
        if e.code == 403 and _retries > 0:
            reset = e.headers.get("X-RateLimit-Reset")
            wait = max(60, int(reset) - int(time.time()) + 10) if reset else 900
            print(f"    rate limited; waiting {wait//60} min for the window to reset",
                  flush=True)
            time.sleep(wait)
            return api(url, _retries - 1)
        return {"_err": f"HTTP {e.code}"}
    except Exception as e:
        return {"_err": type(e).__name__}


def repo_files(owner, repo):
    meta = api(f"https://api.github.com/repos/{owner}/{repo}")
    if "_err" in meta:
        return None, meta["_err"]
    branch = meta.get("default_branch", "main")
    tree = api(f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1")
    if "_err" in tree:
        return None, tree["_err"]
    return [t["path"] for t in tree.get("tree", []) if t.get("type") == "blob"], None


def links(text):
    out = set()
    for m in re.finditer(r"https?://(?:www\.)?github\.com/([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)", text):
        o, r = m.group(1), m.group(2).rstrip(".,);")
        if r.lower() in ("blob", "tree"):
            continue
        if THIRD_PARTY.search(m.group(0)):
            continue
        out.add((o, r))
    return sorted(out)


def main() -> None:
    papers = sorted(CACHE.glob("*.txt"))
    print(f"{len(papers)} cached papers\n")
    results, calls = {}, 0
    for f in papers:
        aid = f.stem
        text = f.read_text(encoding="utf-8", errors="replace")
        cands = links(text)
        if not cands:
            results[aid] = {"coding": "UNDETERMINED", "reason": "no own-repo link in text",
                            "repos": []}
            continue
        coded, detail = None, []
        for owner, repo in cands[:2]:                 # at most two repos per paper
            files, err = repo_files(owner, repo)
            calls += 2
            time.sleep(0.5)
            if files is None:
                detail.append({"repo": f"{owner}/{repo}", "error": err})
                continue
            hits = [p for p in files if GEN_HINT.search(p) and DATA_EXT.search(p)]
            dirs = sorted({p.split("/")[0] for p in files if GEN_HINT.search(p.split("/")[0])})
            detail.append({"repo": f"{owner}/{repo}", "n_files": len(files),
                           "gen_files": hits[:8], "gen_dirs": dirs[:5]})
            if hits:
                coded = "RELEASED"
        if coded is None:
            reachable = any("error" not in d for d in detail)
            coded = "NOT RELEASED" if reachable else "UNDETERMINED"
        results[aid] = {"coding": coded, "repos": detail}
        mark = {"RELEASED": "  <-- releases generations", "UNDETERMINED": "  (unresolved)"}.get(coded, "")
        print(f"{aid:16s} {coded:14s} {', '.join(d['repo'] for d in detail)[:48]}{mark}")

    tally = Counter(v["coding"] for v in results.values())
    n = len(results)
    print(f"\napi calls: {calls}")
    print(f"tally over {n} papers: {dict(tally)}")
    rel = [k for k, v in results.items() if v["coding"] == "RELEASED"]
    print(f"\nRELEASED ({len(rel)}): {', '.join(rel) if rel else 'none'}")
    for k in rel:
        for d in results[k]["repos"]:
            if d.get("gen_files"):
                print(f"  {k} -> {d['repo']}: {d['gen_files'][:3]}")
    print("\nreading:")
    print("  The text-only coding reported 0 of 38 releasing per-example generations. Any")
    print("  RELEASED row here is a paper that coding missed, and the UNDETERMINED rows are")
    print("  the uncertainty the single number was concealing. Filename evidence is weaker")
    print("  than reading files, so RELEASED here means 'an artifact of the right shape is")
    print("  present', not that we verified its contents.")
    OUT.write_text(json.dumps({"tally": dict(tally), "papers": results}, indent=2))
    print("\n[wrote]", OUT)


if __name__ == "__main__":
    main()
