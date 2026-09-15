#!/usr/bin/env python3
"""Prepare additional safety benchmarks (Week 4-5 task).

Downloads and reformats AdvBench and SimpleSafetyTests into the same JSONL
format as data/xstest.jsonl so extract_traces.py can run on them with minimal
changes.

Output:
    data/advbench.jsonl         — AdvBench harmful_behaviors (520 prompts, all unsafe)
    data/simplesafetytests.jsonl — SimpleSafetyTests (100 prompts, all unsafe)

Format (matches xstest.jsonl):
    {"prompt": "...", "label": "unsafe"|"safe", "type": "<benchmark-specific>",
     "source": "advbench"|"simplesafetytests"}

Notes:
  - AdvBench is from Zou et al. 2023 (arXiv 2307.15043). Direct source on GitHub.
  - SimpleSafetyTests is from Vidgen et al. 2023 (arXiv 2311.08370).
  - Both have permissive licenses for research.
"""
import argparse
import csv
import io
import json
import os
import urllib.request


ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
SIMPLESAFETY_URL = "https://raw.githubusercontent.com/bertiev/SimpleSafetyTests/main/data/SimpleSafetyTests.csv"


def fetch(url, timeout=60):
    """Download URL contents as text. Returns (status_code, text)."""
    req = urllib.request.Request(url, headers={"User-Agent": "research-bench-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", errors="replace")


def write_jsonl(records, out_path):
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(records)} records → {out_path}")


def prepare_advbench(out_dir):
    out_path = os.path.join(out_dir, "advbench.jsonl")
    if os.path.exists(out_path):
        print(f"  AdvBench already exists at {out_path}, skipping.")
        return
    print(f"  fetching {ADVBENCH_URL}")
    status, text = fetch(ADVBENCH_URL)
    if status != 200:
        print(f"  ERROR: HTTP {status}")
        return
    reader = csv.DictReader(io.StringIO(text))
    records = []
    for row in reader:
        prompt = row.get("goal", "").strip() or row.get("behavior", "").strip()
        if not prompt:
            continue
        records.append({
            "prompt": prompt,
            "label": "unsafe",
            "type": "advbench_harmful",
            "source": "advbench",
        })
    write_jsonl(records, out_path)


def prepare_simplesafety(out_dir):
    """Fetch SimpleSafetyTests from HuggingFace (walledai/SimpleSafetyTests)."""
    out_path = os.path.join(out_dir, "simplesafetytests.jsonl")
    if os.path.exists(out_path):
        print(f"  SimpleSafetyTests already exists at {out_path}, skipping.")
        return
    print("  loading from HuggingFace: walledai/SimpleSafetyTests (both splits)")
    try:
        from datasets import load_dataset
        ds_instruct = load_dataset("walledai/SimpleSafetyTests", split="instruct")
        ds_info = load_dataset("walledai/SimpleSafetyTests", split="info")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        print("  try: pip install datasets, or download manually from "
              "huggingface.co/datasets/walledai/SimpleSafetyTests")
        return

    records = []
    for split_name, ds in [("instruct", ds_instruct), ("info", ds_info)]:
        for ex in ds:
            prompt = ex.get("prompt") or ex.get("instruct") or ""
            if not prompt:
                continue
            category = ex.get("harm_area") or ex.get("category") or "simplesafety"
            records.append({
                "prompt": prompt.strip(),
                "label": "unsafe",
                "type": f"simplesafety_{split_name}_{str(category).lower().replace(' ', '_')}",
                "source": "simplesafetytests",
            })
    write_jsonl(records, out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/")
    p.add_argument("--bench", default="advbench,simplesafetytests",
                   help="comma-separated list: advbench,simplesafetytests")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bench = set(args.bench.split(","))

    if "advbench" in bench:
        print("AdvBench:")
        prepare_advbench(args.out_dir)
    if "simplesafetytests" in bench:
        print("SimpleSafetyTests:")
        prepare_simplesafety(args.out_dir)

    print("\nMulti-benchmark prep DONE")
    print("Next: run extract_traces.py on each new file (separate phase tag).")
    print("  e.g.: python extract_traces.py --max_xstest 0 --max_hotpot 0 \\")
    print("           --data_dir data/ --phase phase4_advbench \\")
    print("           --models qwen3.5-2b,gemma-4-e2b")


if __name__ == "__main__":
    main()
