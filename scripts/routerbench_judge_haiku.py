#!/usr/bin/env python3
"""Independent Haiku adjudication of RouterBench mtbench responses.

For each of the 80 mtbench conversations and each model in the audited pair,
asks claude-haiku-4-5 for a binary adequacy verdict (would a careful grader
score this response >= 7/10). Output feeds the external RLC-Audit as the
semantic construct label, against RouterBench's stored GPT-4 scores as proxy.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/routerbench_audit/mtbench_pair.json"
OUT = ROOT / "results/routerbench_audit/mtbench_haiku_judgments.json"
MODEL = "claude-haiku-4-5-20251001"

PROMPT = """You are grading an AI assistant's answers in a two-turn conversation, in the style of MT-Bench single-answer grading.

User turns:
{turns}

Assistant's responses (in order):
{response}

Judge the overall quality of the assistant's responses to BOTH turns: correctness, helpfulness, relevance, depth. Would a careful grader score this at 7 or higher on a 1-10 scale?

Answer with exactly one word: ADEQUATE (>=7) or INADEQUATE (<7)."""


def judge(client: anthropic.Anthropic, turns: list[str], response: str) -> bool | None:
    text = PROMPT.format(
        turns="\n\n".join(f"[Turn {i+1}] {t}" for i, t in enumerate(turns)),
        response=response[:8000],
    )
    for attempt in range(5):
        try:
            msg = client.messages.create(
                model=MODEL, max_tokens=8,
                messages=[{"role": "user", "content": text}],
            )
            out = msg.content[0].text.strip().upper()
            if re.search(r"\bINADEQUATE\b", out):
                return False
            if re.search(r"\bADEQUATE\b", out):
                return True
            return None
        except anthropic.RateLimitError:
            time.sleep(10 * (attempt + 1))
        except anthropic.APIError:
            time.sleep(5 * (attempt + 1))
    return None


def main() -> None:
    data = json.loads(SRC.read_text())
    client = anthropic.Anthropic()
    records = []
    for i, r in enumerate(data["records"]):
        rec = {"sample_id": r["sample_id"], "eval_name": r["eval_name"]}
        rec["adequate_a"] = judge(client, r["prompt_turns"], r["response_a"])
        rec["adequate_b"] = judge(client, r["prompt_turns"], r["response_b"])
        records.append(rec)
        if (i + 1) % 10 == 0:
            print(f"{i+1}/{len(data['records'])}")
    OUT.write_text(json.dumps({
        "judge_model": MODEL,
        "model_a": data["model_a"], "model_b": data["model_b"],
        "records": records,
    }, indent=1))
    n_null = sum(1 for r in records if r["adequate_a"] is None or r["adequate_b"] is None)
    print(f"wrote {OUT} ({len(records)} rows, {n_null} with null verdicts)")


if __name__ == "__main__":
    main()
