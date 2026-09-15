#!/usr/bin/env python3
"""Does every cited work exist, and does the entry describe it correctly? (2026-08-21)

EMNLP 2026 screens camera-ready papers for hallucinated, fabricated or otherwise
unverifiable references and desk-rejects on a finding, and the notification says
explicitly not to rely on automated tools for this. So this script is not a
clearance: it is a triage that narrows what a human has to read.

For each entry we look the work up in whatever registry can identify it --- Crossref
by DOI, Crossref by title, arXiv by identifier or title --- and compare the returned
title and first author with the entry. Three outcomes:

  OK        a registry returned a record whose title matches closely
  CHECK     a record was found but the title or first author differs enough to read
  NOT FOUND nothing matched, which for a real paper usually means a venue registry
            we do not query (ACL Anthology pre-DOI, workshop papers, model cards)
            rather than a fabrication --- these are exactly the ones to read by hand

Model cards, blog posts and dataset releases legitimately have no registry entry;
they are reported separately so the NOT FOUND list stays meaningful.

  python3 verify_bibliography.py [--bib paper/acl_arr_refs.bib]
"""
from __future__ import annotations

import argparse
import json
import re
import ssl
import time
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

import certifi

CTX = ssl.create_default_context(cafile=certifi.where())
UA = {"User-Agent": "bib-verify/1.0 (mailto:wdi1024@kookmin.ac.kr)"}
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis_results" / "bibliography_verification.json"
NON_PAPER = re.compile(r"model card|blog|huggingface\.co|technical report|@misc", re.I)


def get(url, timeout=25):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                    timeout=timeout, context=CTX) as r:
            return json.loads(r.read())
    except Exception:
        return None


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def sim(a, b):
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def field(body, name):
    m = re.search(rf"\b{name}\s*=\s*[{{\"]", body, re.I)
    if not m:
        return ""
    i = m.end() - 1
    if body[i] == '"':
        j = body.index('"', i + 1)
        return body[i + 1:j]
    depth, j = 0, i
    for j in range(i, len(body)):
        depth += (body[j] == "{") - (body[j] == "}")
        if depth == 0:
            break
    return body[i + 1:j]


def clean(s):
    return re.sub(r"[{}]", "", re.sub(r"\s+", " ", s)).strip()


def arxiv(title, ident=""):
    """arXiv is where most of this bibliography actually lives, and unlike Semantic
    Scholar (which rate-limits unauthenticated callers to the point of uselessness)
    it answers reliably. Atom, not JSON, so parse the two fields we need."""
    if ident:
        url = "http://export.arxiv.org/api/query?id_list=" + ident
    else:
        q = urllib.parse.quote(f'ti:"{title[:180]}"')
        url = f"http://export.arxiv.org/api/query?search_query={q}&max_results=3"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                    timeout=30, context=CTX) as r:
            xml = r.read().decode("utf-8", "replace")
    except Exception:
        return None
    best = None
    for m in re.finditer(r"<entry>(.*?)</entry>", xml, re.S):
        e = m.group(1)
        tm = re.search(r"<title>(.*?)</title>", e, re.S)
        if not tm:
            continue
        t = clean(tm.group(1))
        am = re.search(r"<author>\s*<name>(.*?)</name>", e, re.S)
        cand = (t, clean(am.group(1)) if am else "")
        if best is None or sim(t, title) > sim(best[0], title):
            best = cand
    if best and (ident or sim(best[0], title) >= 0.82):
        return ("arxiv", best[0], best[1])
    return None


def s2(title):
    """Semantic Scholar indexes arXiv and the ACL Anthology, which is where most of
    this bibliography lives and which Crossref covers poorly. Querying Crossref alone
    reported 36 of 80 entries as unfound, nearly all of them real papers, and a
    report that noisy is worse than none."""
    q = urllib.parse.quote(title[:250])
    j = get("https://api.semanticscholar.org/graph/v1/paper/search?query=" + q
            + "&limit=3&fields=title,authors,year,externalIds", timeout=30)
    if not j or not j.get("data"):
        return None
    best = max(j["data"], key=lambda d: sim(d.get("title", ""), title))
    if sim(best.get("title", ""), title) < 0.82:
        return None
    au = best.get("authors") or []
    return ("semanticscholar", best.get("title", ""), au[0]["name"] if au else "")


def lookup(title, doi, arxiv_id):
    if doi:
        j = get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
        if j:
            m = j["message"]
            auths = m.get("author") or []
            first = (f"{auths[0].get('given','')} {auths[0].get('family','')}".strip()
                     if auths else "")
            return ("crossref-doi", (m.get("title") or [""])[0], first)
    if arxiv_id or title:
        r = arxiv(title, arxiv_id)
        time.sleep(0.4)
        if r:
            return r
    if title:
        q = urllib.parse.quote(title[:200])
        j = get(f"https://api.crossref.org/works?query.bibliographic={q}&rows=1")
        if j and j["message"]["items"]:
            m = j["message"]["items"][0]
            t = (m.get("title") or [""])[0]
            if sim(t, title) > 0.75:
                a = m.get("author") or []
                return ("crossref-title", t,
                        f"{a[0].get('given','')} {a[0].get('family','')}".strip() if a else "")
        j = get("https://api.crossref.org/works?query.title=" + q + "&rows=1")
        if j and j["message"]["items"]:
            m = j["message"]["items"][0]
            t = (m.get("title") or [""])[0]
            if sim(t, title) > 0.75:
                a = m.get("author") or []
                return ("crossref-title", t,
                        f"{a[0].get('given','')} {a[0].get('family','')}".strip() if a else "")
    return (None, "", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bib", default=str(ROOT / "paper" / "acl_arr_refs.bib"))
    a = ap.parse_args()
    src = Path(a.bib).read_text()
    entries = re.findall(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", src, re.S)
    print(f"{len(entries)} entries in {Path(a.bib).name}\n")

    ok, check, missing, nonpaper = [], [], [], []
    for kind, key, body in entries:
        title = clean(field(body, "title"))
        author = clean(field(body, "author"))
        doi = clean(field(body, "doi"))
        note = f"{kind} {field(body,'howpublished')} {field(body,'note')} {field(body,'url')}"
        m = re.search(r"(\d{4}\.\d{4,5})", body)
        arxiv_id = m.group(1) if m else ""
        first = author.split(" and ")[0] if author else ""
        first_last = first.split(",")[0].strip() if "," in first else first.split()[-1] if first else ""

        if NON_PAPER.search(note) and not doi:
            nonpaper.append((key, title))
            continue
        src_, got_title, got_author = lookup(title, doi, arxiv_id)
        time.sleep(0.12)
        if not src_:
            missing.append((key, title))
            print(f"NOT FOUND  {key:28s} {title[:64]}")
        elif sim(got_title, title) < 0.82:
            # a near-miss title is a different paper, not a metadata slip; report it
            # as unmatched so CHECK stays a list of real discrepancies
            missing.append((key, title))
            print(f"NOT FOUND  {key:28s} {title[:60]}")
            print(f"           {'':28s} (closest: {got_title[:52]})")
        elif first_last and got_author and norm(first_last) not in norm(got_author):
            check.append((key, f"author {first_last}", got_author))
            print(f"CHECK-AUTH {key:28s} bib first author {first_last} vs {got_author}")
        else:
            ok.append(key)

    print(f"\nOK {len(ok)} | CHECK {len(check)} | NOT FOUND {len(missing)} | "
          f"no-registry-expected {len(nonpaper)}")
    if nonpaper:
        print("\nno registry expected (model cards, blogs, dataset releases) --"
              " confirm these by URL, not by DOI:")
        for k, t in nonpaper:
            print(f"  {k:28s} {t[:60]}")
    print("\nreading:")
    print("  NOT FOUND is not evidence of fabrication: ACL Anthology entries without a")
    print("  DOI and workshop papers routinely fail a Crossref lookup. It is the list a")
    print("  human should open one by one. CHECK rows found a record that disagrees with")
    print("  the entry, which is where a wrong year, venue or author usually hides.")
    OUT.write_text(json.dumps({"ok": ok, "check": check, "not_found": missing,
                               "no_registry_expected": nonpaper}, indent=2))
    print("\n[wrote]", OUT)


if __name__ == "__main__":
    main()
