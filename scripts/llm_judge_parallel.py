#!/usr/bin/env python3
"""Parallel wrapper around the correctness judges (same prompt, parser, and
output format as llm_judge_correctness_{gsm8k,hotpot}.py, same resume files).

    python3 llm_judge_parallel.py --task gsm8k --phase phase14_gsm8k_2k \
        --models qwen3.5-2b --backend anthropic --model claude-haiku-4-5-20251001
"""
import argparse
import importlib.util
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_judge_module(task):
    path = HERE / f"llm_judge_correctness_{task}.py"
    spec = importlib.util.spec_from_file_location(f"judge_{task}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [str(path)]  # keep argparse in the module import-safe
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["gsm8k", "hotpot"], required=True)
    p.add_argument("--phase", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--backend", default="anthropic")
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
    p.add_argument("--results_dir", default="results/disagree_routing")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    mod = load_judge_module(args.task)
    rdir = args.results_dir
    meta = json.load(open(f"{rdir}/{args.phase}_meta.json"))["prompts"]
    # phase11 metas store the gold under "answer", phase14 metas under "gold_answer"
    gold_map = {m["id"]: {"question": m.get("question", ""),
                          "answer": m.get("answer") or m.get("gold_answer") or ""}
                for m in meta}
    n_empty = sum(1 for g in gold_map.values() if not g["answer"].strip())
    if n_empty:
        raise SystemExit(f"{n_empty}/{len(gold_map)} prompts have an empty gold answer -- refusing to judge")

    for slm in args.models.split(","):
        traces_path = f"{rdir}/{args.phase}_traces_{slm}.json"
        if not os.path.exists(traces_path):
            print(f"[SKIP] {traces_path} missing")
            continue
        traces = json.load(open(traces_path))["records"]
        out_path = (f"{rdir}/{args.phase}_correctness_{slm}_{args.backend}_"
                    f"{args.model.replace('/', '_')}.json")
        out, out_by_id = [], {}
        if os.path.exists(out_path):
            out = json.load(open(out_path)).get("records", [])
            out_by_id = {r["id"]: r for r in out}
        done = {r["id"] for r in out if r.get("correct_judge") is not None}
        todo = [r for r in traces if r["id"] not in done]
        print(f"[{slm}] {len(done)} done, {len(todo)} to go, workers={args.workers}")

        lock = threading.Lock()
        n = 0
        t0 = time.time()

        def save():
            tmp = out_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"model": slm, "backend": args.backend,
                           "judge_model": args.model, "records": out}, f, indent=2)
            os.replace(tmp, out_path)

        def work(r):
            import random
            g = gold_map.get(r["id"], {})
            for attempt in range(8):
                j = mod.judge_one(g.get("question", ""), r.get("trace", "") or "",
                                  g.get("answer", ""), args.backend, args.model)
                err = j.get("error") or ""
                if "RateLimit" not in err and "429" not in err and "Overloaded" not in err:
                    break
                time.sleep(min(2 ** attempt, 30) + random.random())
            return r["id"], g, j

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(work, r) for r in todo]
            for fut in as_completed(futs):
                pid, g, j = fut.result()
                rec = {"id": pid, "question": g.get("question", "")[:200],
                       "gold_answer": g.get("answer", ""),
                       "correct_judge": j["correct"], "rationale": j["rationale"],
                       "raw": j.get("raw"), "error": j.get("error")}
                with lock:
                    if pid in out_by_id:
                        out_by_id[pid].update(rec)
                    else:
                        out.append(rec)
                        out_by_id[pid] = rec
                    n += 1
                    if n % 50 == 0:
                        save()
                        rate = n / max(time.time() - t0, 1)
                        print(f"  {n}/{len(todo)}  {rate:.1f}/s  "
                              f"eta {(len(todo)-n)/max(rate,0.01):.0f}s", flush=True)
        save()
        ok = sum(1 for r in out if r.get("correct_judge") is not None)
        errs = sum(1 for r in out if r.get("error"))
        pos = sum(1 for r in out if r.get("correct_judge") == 1)
        print(f"[{slm}] wrote {out_path}: {ok} judged ({pos} correct), {errs} errors")


if __name__ == "__main__":
    main()
