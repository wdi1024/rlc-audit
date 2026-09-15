# Regeneration Phase Analysis

Phase: `phase_colab_qwen_gemma_big_xstest450_tagged768_thinkingoff_greedy`
Models: `qwen3.5-9b` vs `gemma-2-9b-it`
Common prompts: 450

## Parse Coverage

| Model | n | final coverage | parse status |
|---|---:|---:|---|
| qwen3.5-9b | 450 | 1.000 | `{"exact": 385, "fallback_truncated_final": 65}` |
| gemma-2-9b-it | 450 | 1.000 | `{"exact": 449, "final_only": 1}` |

## AUC/AP Tables

| Score | Label | n pos | AUC | AP |
|---|---|---:|---:|---:|
| raw_prefix50 | kw_raw_prefix50_disagreement | 19/450 | 0.759 | 0.204 |
| raw_prefix50 | judge_final_disagreement | 71/450 | 0.514 | 0.173 |
| raw_full | kw_raw_full_disagreement | 100/450 | 0.566 | 0.260 |
| raw_full | judge_final_disagreement | 71/450 | 0.617 | 0.234 |
| final_full | kw_final_disagreement | 110/450 | 0.650 | 0.404 |
| final_full | judge_final_disagreement | 71/450 | 0.664 | 0.316 |

## Final-Channel Practical Router Screen

This section reuses the cached clean final-channel run. It is a practical sanity check, not a proposed production router: a PASS means the score does not show the surface-over-semantic RLC failure under this diagnostic.

Rows use 450 final-channel examples. MiniLM row: skipped because the encoder was unavailable locally.

| Router | Status | surf pos | sem pos | AUC surf | AP surf | AUC sem | AP sem | P@#sem | pair kappa | top-10% sem | top-10% both-refuse |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| final_full_tfidf | PASS | 110/450 | 71/450 | 0.650 | 0.404 | 0.664 | 0.316 | 0.296 | 0.296 | 17/45 | 23/45 |

| Router | Budget | semantic dis | surface dis | both refuse | both comply |
|---|---|---:|---:|---:|---:|
| final_full_tfidf | top_10pct (45) | 17 | 23 | 23 | 5 |
| final_full_tfidf | raw_prefix_gt_0_95_budget (194) | 43 | 64 | 117 | 34 |
| final_full_tfidf | semantic_positive_count (71) | 21 | 33 | 41 | 9 |

Interpretation: final-channel scoring is closer to the intended semantic construct than raw prefix scoring, but the gains remain modest and routed sets still contain many agreed refusals.
