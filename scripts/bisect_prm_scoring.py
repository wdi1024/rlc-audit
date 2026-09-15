#!/usr/bin/env python3
"""Which of our deviations from the model card flattened the PRM? (2026-08-21)

The cached rewards are confirmed broken, not merely suspicious. On ProcessBench
`gsm8k-0` -- which is the model card's own worked example, with its output printed
in the card -- we get

    ours  [0.2695, 0.2520, 0.2305, 0.2236]      flat, no dip
    card  [1.0,    0.1904, 0.9766, 1.0   ]      sharp dip at the annotated error

and across all 25,641 cached rewards not one exceeds 0.5352. The card's four-step
example alone contains three values above 0.97. So the score is squashed into a
narrow low band and discriminates nothing, which is why AUC sat at chance under
every aggregation. The checkpoint is fine (the `score.*` head is present in the
safetensors index and the load report reported no missing keys), so the fault is in
how we construct or call the model.

We deviated from the card in four places. This runs the card's own input through
each combination and diffs against the documented output:

    1. how pad_token_id is supplied -- the card cannot load at all without it on
       current transformers, so some patch is mandatory; the question is whether
       injecting a whole AutoConfig object is what breaks it, versus passing the
       one value as a kwarg and leaving the card's path otherwise untouched
    2. device_map "auto" (card) vs "cuda" (ours)
    3. use_cache left alone vs forced off on the config
    4. use_cache left alone vs forced off at the call site

Whatever reproduces the card is what the audit adopts. If nothing does, the fault is
not in these four and the ProcessBench card is dropped rather than patched again.

  python3 bisect_prm_scoring.py
"""
from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

MODEL = "Qwen/Qwen2.5-Math-PRM-7B"
EXPECTED = [1.0, 0.1904296875, 0.9765625, 1.0]

SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
QUERY = ("Sue lives in a fun neighborhood.  One weekend, the neighbors decided to play a "
         "prank on Sue.  On Friday morning, the neighbors placed 18 pink plastic flamingos "
         "out on Sue's front yard.  On Saturday morning, the neighbors took back one third "
         "of the flamingos, painted them white, and put these newly painted white flamingos "
         "back out on Sue's front yard.  Then, on Sunday morning, they added another 18 pink "
         "plastic flamingos to the collection. At noon on Sunday, how many more pink plastic "
         "flamingos were out than white plastic flamingos?")
RESPONSE = [
    "To find out how many more pink plastic flamingos were out than white plastic flamingos "
    "at noon on Sunday, we can break down the problem into steps. First, on Friday, the "
    "neighbors start with 18 pink plastic flamingos.",
    "On Saturday, they take back one third of the flamingos. Since there were 18 flamingos, "
    "(1/3 \\times 18 = 6) flamingos are taken back. So, they have (18 - 6 = 12) flamingos "
    "left in their possession. Then, they paint these 6 flamingos white and put them back "
    "out on Sue's front yard. Now, Sue has the original 12 pink flamingos plus the 6 new "
    "white ones. Thus, by the end of Saturday, Sue has (12 + 6 = 18) pink flamingos and 6 "
    "white flamingos.",
    "On Sunday, the neighbors add another 18 pink plastic flamingos to Sue's front yard. By "
    "the end of Sunday morning, Sue has (18 + 18 = 36) pink flamingos and still 6 white "
    "flamingos.",
    "To find the difference, subtract the number of white flamingos from the number of pink "
    "flamingos: (36 - 6 = 30). Therefore, at noon on Sunday, there were 30 more pink plastic "
    "flamingos out than white plastic flamingos. The answer is (\\boxed{30}).",
]


def make_step_rewards(logits, token_masks):
    p = F.softmax(logits, dim=-1) * token_masks.unsqueeze(-1)
    return [s[s != 0].view(-1, 2)[:, 1].float().cpu().tolist() for s in p]


def build(tok, device_map, pad_mode, cfg_no_cache):
    kw = dict(device_map=device_map, torch_dtype=torch.bfloat16, trust_remote_code=True)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad_mode == "kwarg":
        # leaves the card's construction path untouched apart from the one value
        kw["pad_token_id"] = pad
        if cfg_no_cache:
            kw["use_cache"] = False
    else:
        cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
        if getattr(cfg, "pad_token_id", None) is None:
            cfg.pad_token_id = pad
        if cfg_no_cache:
            cfg.use_cache = False
        kw["config"] = cfg
    return AutoModel.from_pretrained(MODEL, **kw).eval()


def main() -> None:
    print(f"target (model card, = ProcessBench gsm8k-0, human first error at index 1):")
    print(f"  {EXPECTED}")
    print(f"broken run produced: [0.2695, 0.252, 0.2305, 0.2236]\n")

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    sep = tok.encode("<extra_0>")[0]
    print(f"pad={tok.pad_token_id} eos={tok.eos_token_id} sep={sep}")
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": QUERY},
                {"role": "assistant", "content": "<extra_0>".join(RESPONSE) + "<extra_0>"}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    winner = None
    for pad_mode, device_map, cfg_no_cache in itertools.product(
            ("kwarg", "config"), ("auto", "cuda"), (False, True)):
        tag = f"pad={pad_mode:6s} device_map={device_map:4s} cfg.use_cache={not cfg_no_cache}"
        try:
            model = build(tok, device_map, pad_mode, cfg_no_cache)
        except Exception as e:
            print(f"{tag}  LOAD FAILED {type(e).__name__}: {str(e)[:70]}")
            continue
        ids = tok.encode(text, return_tensors="pt").to(model.device)
        masks = (ids == sep)
        with torch.no_grad():
            for call_no_cache in (False, True):
                kw = {"use_cache": False} if call_no_cache else {}
                try:
                    got = make_step_rewards(model(input_ids=ids, **kw)[0], masks)[0]
                except Exception as e:
                    print(f"{tag} call_no_cache={call_no_cache}  EXC {type(e).__name__}")
                    continue
                ok = (len(got) == len(EXPECTED)
                      and all(abs(a - b) < 0.02 for a, b in zip(got, EXPECTED)))
                print(f"{tag} call_no_cache={call_no_cache}  "
                      f"{[round(x, 4) for x in got]}  {'*** MATCH ***' if ok else 'differs'}")
                if ok and winner is None:
                    winner = (pad_mode, device_map, cfg_no_cache, call_no_cache)
        del model
        torch.cuda.empty_cache()

    print("\nreading:")
    if winner:
        print(f"  reproduces the card: pad={winner[0]}, device_map={winner[1]}, "
              f"cfg.use_cache={not winner[2]}, call use_cache=False:{winner[3]}")
        print("  The audit adopts this verbatim and rescores.")
    else:
        print("  Nothing reproduces the card. The fault is not in these four deviations,")
        print("  so the ProcessBench card is dropped rather than patched again.")


if __name__ == "__main__":
    main()
