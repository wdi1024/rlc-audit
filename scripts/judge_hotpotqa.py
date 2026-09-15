#!/usr/bin/env python3
"""LLM-as-judge for HotpotQA correctness — re-evaluation of §5.5 with sentence + judge.

The original §5.5 used TF-IDF cosine + substring-match correctness (a noisy proxy
parallel to keyword refusal). This script uses Claude-Haiku-4.5 to judge whether
each model's response contains the gold answer (semantically, not just verbatim).

Pipeline:
  1. Load Phase 2 HotpotQA prompts + Qwen/Gemma traces + gold answers from meta
  2. For each (prompt, gold, trace) triple, ask judge: "is this answer correct?"
  3. Save per-trace correctness labels
  4. Run from user terminal where ANTHROPIC_API_KEY is set

Output:
  results/disagree_routing/phase2_judge_correctness_qwen3.5-2b_anthropic_haiku.json
  results/disagree_routing/phase2_judge_correctness_gemma-4-e2b_anthropic_haiku.json

Cost (~250 prompts × 2 SLMs = 500 calls):
  Claude Haiku 4.5 ≈ $0.20

Usage:
  cd <workspace>
  python3 infra-data/scripts/disagree_routing/judge_hotpotqa.py
"""
import json
import os
import time

from anthropic import Anthropic

SYSTEM = """You are a binary correctness classifier for question-answering research. \
Given a question, the gold (reference) answer, and an AI's response, decide whether \
the AI's response contains a CORRECT answer to the question.

Counts as correct: the AI's response includes the gold answer (verbatim, paraphrased, \
or semantically equivalent — e.g., "Paris" vs. "the city of Paris" vs. "France's \
capital, Paris" all count).

Counts as incorrect: the AI's response is wrong, refuses, gives an unrelated answer, \
gives a partial answer that does not match the gold, or generates only a thinking \
process without committing to an answer.

Reply ONLY with valid JSON of the form:
{"correct": true|false, "rationale": "<one short sentence>"}"""

USER_TEMPLATE = """Question:
\"\"\"{question}\"\"\"

Gold answer:
\"\"\"{gold}\"\"\"

AI response:
\"\"\"{trace}\"\"\"

Reply with JSON only."""


def parse_judgment(text):
    try:
        d = json.loads(text)
        return bool(d.get("correct")), str(d.get("rationale", ""))
    except Exception:
        pass
    import re
    m = re.search(r'\{[^}]*"correct"\s*:\s*(true|false)[^}]*\}', text, re.IGNORECASE)
    if m:
        try:
            d = json.loads(m.group(0))
            return bool(d.get("correct")), str(d.get("rationale", ""))
        except Exception:
            pass
    t = text.lower()
    if '"correct": true' in t or '"correct":true' in t:
        return True, text[:120]
    if '"correct": false' in t or '"correct":false' in t:
        return False, text[:120]
    return None, text[:200]


def main():
    rdir = "results/disagree_routing"
    meta = {m["id"]: m for m in json.load(open(f"{rdir}/phase2_meta.json"))["prompts"]}

    for slm in ["qwen3.5-2b", "gemma-4-e2b"]:
        traces_path = f"{rdir}/phase2_traces_{slm}.json"
        if not os.path.exists(traces_path):
            print(f"  [{slm}] traces not found: {traces_path}")
            continue
        traces = json.load(open(traces_path))["records"]
        # filter to HotpotQA only
        hp_traces = [t for t in traces
                     if meta.get(t["id"], {}).get("task") == "hotpotqa"]
        print(f"\n[{slm}] HotpotQA traces: {len(hp_traces)}")
        save_path = (f"{rdir}/phase2_judge_correctness_{slm}_"
                     f"anthropic_claude-haiku-4-5-20251001.json")

        # Resume support
        if os.path.exists(save_path):
            existing = json.load(open(save_path))["records"]
            done_ids = {r["id"] for r in existing
                        if r.get("correct_judge") is not None}
            out = list(existing)
            print(f"  Resuming: {len(done_ids)} already judged.")
        else:
            done_ids = set()
            out = []

        client = Anthropic()
        t0 = time.time()
        n_total = len(hp_traces)
        for i, rec in enumerate(hp_traces):
            pid = rec["id"]
            if pid in done_ids:
                continue
            m_rec = meta.get(pid, {})
            question = m_rec.get("question", "")
            gold = m_rec.get("answer", "")
            trace = rec.get("trace", "")
            try:
                msg = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=120,
                    temperature=0,
                    system=SYSTEM,
                    messages=[{"role": "user",
                               "content": USER_TEMPLATE.format(
                                   question=question, gold=gold, trace=trace[:2000])}],
                )
                text = msg.content[0].text
                correct, rationale = parse_judgment(text)
            except Exception as e:
                correct, rationale = None, f"__ERROR__: {type(e).__name__}: {e}"
            out.append({
                "id": pid,
                "question": question,
                "gold": gold,
                "correct_judge": correct,
                "rationale": rationale[:160],
            })
            if (len(out) - len(done_ids)) % 25 == 0:
                elapsed = time.time() - t0
                done_now = len(out) - len(done_ids)
                rate = done_now / max(elapsed, 1)
                remaining = (n_total - len(out)) / max(rate, 0.001)
                print(f"  [{slm}] {len(out)}/{n_total} "
                      f"({rate:.2f}/s, ~{remaining/60:.1f}min left)", flush=True)
                json.dump({"records": out, "judge_model": "claude-haiku-4-5-20251001",
                           "phase": "phase2_hotpotqa_correctness", "slm": slm},
                          open(save_path, "w"), ensure_ascii=False)

        json.dump({"records": out, "judge_model": "claude-haiku-4-5-20251001",
                   "phase": "phase2_hotpotqa_correctness", "slm": slm},
                  open(save_path, "w"), ensure_ascii=False)
        elapsed = time.time() - t0
        print(f"[{slm}] done: {len(out)} judgments in {elapsed/60:.1f}min")
        print(f"  saved: {save_path}")
        valid = [r for r in out if r.get("correct_judge") is not None]
        correct_n = sum(1 for r in valid if r["correct_judge"])
        print(f"  Correct rate (judge): {correct_n}/{len(valid)} = "
              f"{correct_n/max(len(valid),1):.1%}")


if __name__ == "__main__":
    main()
