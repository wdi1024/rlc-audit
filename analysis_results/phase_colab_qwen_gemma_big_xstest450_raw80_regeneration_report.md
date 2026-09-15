# Regeneration Phase Analysis

Phase: `phase_colab_qwen_gemma_big_xstest450_raw80`
Models: `qwen3.5-9b` vs `gemma-2-9b-it`
Common prompts: 450

## Parse Coverage

| Model | n | final coverage | parse status |
|---|---:|---:|---|
| qwen3.5-9b | 450 | 0.000 | `{"raw_mode": 450}` |
| gemma-2-9b-it | 450 | 0.000 | `{"raw_mode": 450}` |

## AUC/AP Tables

| Score | Label | n pos | AUC | AP |
|---|---|---:|---:|---:|
| raw_prefix50 | kw_raw_prefix50_disagreement | 143/450 | 0.508 | 0.326 |
| raw_prefix50 | judge_raw_disagreement | 71/450 | 0.509 | 0.160 |
| raw_full | kw_raw_full_disagreement | 218/450 | 0.725 | 0.715 |
| raw_full | judge_raw_disagreement | 71/450 | 0.504 | 0.162 |
| final_full | kw_final_disagreement | 0/450 | - | - |
| final_full | judge_raw_disagreement | 71/450 | - | - |
