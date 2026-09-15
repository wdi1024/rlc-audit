# Regeneration Phase Analysis

Phase: `phase_colab_qwen_mistral_xstest450_raw80`
Models: `qwen3.5-9b` vs `mistral-7b-instruct-v0.3`
Common prompts: 450

## Parse Coverage

| Model | n | final coverage | parse status |
|---|---:|---:|---|
| qwen3.5-9b | 450 | 0.000 | `{"raw_mode": 450}` |
| mistral-7b-instruct-v0.3 | 450 | 0.000 | `{"raw_mode": 450}` |

## AUC/AP Tables

| Score | Label | n pos | AUC | AP |
|---|---|---:|---:|---:|
| raw_prefix50 | kw_raw_prefix50_disagreement | 55/450 | 0.633 | 0.174 |
| raw_prefix50 | judge_raw_disagreement | 54/450 | 0.489 | 0.121 |
| raw_full | kw_raw_full_disagreement | 98/450 | 0.684 | 0.468 |
| raw_full | judge_raw_disagreement | 54/450 | 0.519 | 0.118 |
| final_full | kw_final_disagreement | 0/450 | - | - |
| final_full | judge_raw_disagreement | 54/450 | - | - |
