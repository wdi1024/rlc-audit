#!/usr/bin/env python3
"""Re-parse trace JSONs in-place using the latest parse_tagged_output().

Useful when traces were generated with an older lenient parser and you want to
recover finals from previously-flagged missing_final cases without re-running
GPU trace generation. raw_trace stays untouched; only `reasoning`, `final`,
`parse_status`, `is_refusal`, `is_refusal_raw`, and `is_refusal_final` are
recomputed.

Usage:
    python scripts/reparse_traces.py data/phase13_clean_xstest450_advbench100_traces_qwen3.5-2b.json [...]
    python scripts/reparse_traces.py --glob 'data/phase13_clean_*_traces_*.json'
"""
from __future__ import annotations

import argparse
import glob as glob_mod
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from extract_clean_regeneration_subset import parse_tagged_output, is_refusal  # noqa: E402


def reparse(path: Path) -> tuple[Counter, Counter]:
    payload = json.loads(path.read_text())
    records = payload.get("records", [])
    before = Counter(r.get("parse_status") for r in records)

    for r in records:
        raw = r.get("raw_trace", "")
        parsed = parse_tagged_output(raw)
        r["reasoning"] = parsed["reasoning"]
        r["final"] = parsed["final"]
        r["parse_status"] = parsed["parse_status"]
        r["is_refusal_raw"] = is_refusal(raw)
        r["is_refusal_final"] = is_refusal(parsed["final"])
        r["is_refusal"] = r["is_refusal_final"] or r["is_refusal_raw"]

    after = Counter(r.get("parse_status") for r in records)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return before, after


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="*", help="Trace JSON files to re-parse.")
    p.add_argument("--glob", help="Glob pattern (alternative to listing files).")
    args = p.parse_args()

    paths: list[Path] = [Path(x) for x in args.paths]
    if args.glob:
        paths.extend(Path(x) for x in glob_mod.glob(args.glob))
    if not paths:
        print("usage: reparse_traces.py FILE [FILE ...] [--glob 'pattern']",
              file=sys.stderr)
        sys.exit(2)

    for path in paths:
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        before, after = reparse(path)
        print(f"\n{path}")
        print(f"  before: {dict(before)}")
        print(f"  after:  {dict(after)}")
        n_total = sum(after.values())
        n_clean = n_total - after.get("missing_final", 0)
        print(f"  → final non-empty: {n_clean}/{n_total} ({100*n_clean/n_total:.1f}%)")


if __name__ == "__main__":
    main()
