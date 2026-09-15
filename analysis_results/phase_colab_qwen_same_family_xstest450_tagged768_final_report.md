# Regeneration Phase Analysis

Phase: `phase_colab_qwen_same_family_xstest450_tagged768_thinkingoff_greedy`
Models: `qwen3-8b` vs `qwen3.5-9b`
Common prompts: 450

## Parse Coverage

| Model | n | final coverage | parse status |
|---|---:|---:|---|
| qwen3-8b | 450 | 1.000 | `{"exact": 421, "fallback_truncated_final": 29}` |
| qwen3.5-9b | 450 | 1.000 | `{"exact": 361, "fallback_truncated_final": 89}` |

## AUC/AP Tables

| Score | Label | n pos | AUC | AP |
|---|---|---:|---:|---:|
| raw_prefix50 | kw_raw_prefix50_disagreement | 82/450 | 0.453 | 0.161 |
| raw_prefix50 | judge_final_disagreement | 40/450 | 0.379 | 0.068 |
| raw_full | kw_raw_full_disagreement | 72/450 | 0.529 | 0.194 |
| raw_full | judge_final_disagreement | 40/450 | 0.616 | 0.214 |
| final_full | kw_final_disagreement | 73/450 | 0.537 | 0.190 |
| final_full | judge_final_disagreement | 40/450 | 0.631 | 0.181 |

## Final-Channel Practical Router Screen

This section reuses the cached clean final-channel run. It is a practical sanity check, not a proposed production router: a PASS means the score does not show the surface-over-semantic RLC failure under this diagnostic.

Rows use 450 final-channel examples. MiniLM row: skipped because the encoder was unavailable locally.

| Router | Status | surf pos | sem pos | AUC surf | AP surf | AUC sem | AP sem | P@#sem | pair kappa | top-10% sem | top-10% both-refuse |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| final_full_tfidf | PASS | 73/450 | 40/450 | 0.537 | 0.190 | 0.631 | 0.181 | 0.200 | 0.230 | 8/45 | 37/45 |

| Router | Budget | semantic dis | surface dis | both refuse | both comply |
|---|---|---:|---:|---:|---:|
| final_full_tfidf | top_10pct (45) | 8 | 6 | 37 | 0 |
| final_full_tfidf | raw_prefix_gt_0_95_budget (285) | 32 | 53 | 166 | 87 |
| final_full_tfidf | semantic_positive_count (40) | 8 | 6 | 32 | 0 |

Interpretation: final-channel scoring is closer to the intended semantic construct than raw prefix scoring, but the gains remain modest and routed sets still contain many agreed refusals.
