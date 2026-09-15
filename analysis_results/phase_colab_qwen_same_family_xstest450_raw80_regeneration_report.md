# Regeneration Phase Analysis

Phase: `phase_colab_qwen_same_family_xstest450_raw80`
Models: `qwen3-8b` vs `qwen3.5-9b`
Common prompts: 450

## Parse Coverage

| Model | n | final coverage | parse status |
|---|---:|---:|---|
| qwen3-8b | 450 | 0.000 | `{"raw_mode": 450}` |
| qwen3.5-9b | 450 | 0.000 | `{"raw_mode": 450}` |

## AUC/AP Tables

| Score | Label | n pos | AUC | AP |
|---|---|---:|---:|---:|
| raw_prefix50 | kw_raw_prefix50_disagreement | 3/450 | 0.841 | 0.026 |
| raw_prefix50 | judge_raw_disagreement | 57/450 | 0.575 | 0.179 |
| raw_full | kw_raw_full_disagreement | 39/450 | 0.572 | 0.151 |
| raw_full | judge_raw_disagreement | 57/450 | 0.411 | 0.109 |
| final_full | kw_final_disagreement | 0/450 | - | - |
| final_full | judge_raw_disagreement | 57/450 | - | - |
