#!/usr/bin/env python3
"""Fetch JailbreakBench behaviors and emit a phase-meta JSON.

Output schema matches the existing extract_clean_regeneration_subset.py meta
contract: a JSON file with a top-level "items" list, each entry providing
{"id", "prompt"} so the trace-extraction script can iterate.

Usage:
    python scripts/fetch_jailbreakbench.py
    python scripts/fetch_jailbreakbench.py --slug JailbreakBench/JBB-Behaviors

Output:
    data/phase15_jailbreakbench_meta.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "phase15_jailbreakbench_meta.json"

CANDIDATE_SLUGS = [
    ("JailbreakBench/JBB-Behaviors", "behaviors"),
    ("walledai/JailbreakBench", None),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slug", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_new_tokens", type=int, default=80)
    args = p.parse_args()

    from datasets import load_dataset

    slugs = [(args.slug, None)] if args.slug else CANDIDATE_SLUGS
    ds = None
    used_slug = None
    for slug, config in slugs:
        if not slug:
            continue
        try:
            print(f"[fetch] trying {slug} (config={config})...")
            kwargs = {"revision": args.revision} if args.revision else {}
            if config:
                ds = load_dataset(slug, config, **kwargs)
            else:
                ds = load_dataset(slug, **kwargs)
            used_slug = f"{slug}:{config}" if config else slug
            break
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}")

    if ds is None:
        print("ERROR: no JailbreakBench slug loaded. Pass --slug.", file=sys.stderr)
        sys.exit(1)

    items = []
    splits = ds.keys() if hasattr(ds, "keys") else ["default"]
    for split in splits:
        rows = ds[split] if hasattr(ds, "keys") else ds
        for r in rows:
            goal = r.get("Goal") or r.get("goal") or r.get("prompt") or r.get("question")
            if goal is None:
                continue
            cat = r.get("Category") or r.get("category") or split
            items.append({
                "id": f"jbb_{split}_{len(items):04d}",
                "prompt": str(goal),
                "category": str(cat),
                "split": str(split),
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "phase": "phase15_jailbreakbench",
        "source_slug": used_slug,
        "source_revision": args.revision,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "prompts": items,
    }, indent=2, ensure_ascii=False))
    print(f"[fetch] wrote {args.out} ({len(items)} items)")


if __name__ == "__main__":
    main()
