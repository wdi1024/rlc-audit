# Regeneration Phase Analysis

Phase: `phase_colab_qwen_mistral_xstest450_tagged768_thinkingoff_greedy`
Models: `qwen3.5-9b` vs `mistral-7b-instruct-v0.3`
Common prompts: 450

## Parse Coverage

| Model | n | final coverage | parse status |
|---|---:|---:|---|
| qwen3.5-9b | 450 | 1.000 | `{"exact": 361, "fallback_truncated_final": 89}` |
| mistral-7b-instruct-v0.3 | 450 | 1.000 | `{"exact": 226, "fallback_truncated_final": 224}` |

## AUC/AP Tables

| Score | Label | n pos | AUC | AP |
|---|---|---:|---:|---:|
| raw_prefix50 | kw_raw_prefix50_disagreement | 29/450 | 0.657 | 0.168 |
| raw_prefix50 | judge_final_disagreement | 66/450 | 0.440 | 0.129 |
| raw_full | kw_raw_full_disagreement | 134/450 | 0.632 | 0.393 |
| raw_full | judge_final_disagreement | 66/450 | 0.654 | 0.251 |
| final_full | kw_final_disagreement | 150/450 | 0.667 | 0.453 |
| final_full | judge_final_disagreement | 66/450 | 0.649 | 0.234 |

## Final-Channel Practical Router Screen

This section reuses the cached clean final-channel run. It is a practical sanity check, not a proposed production router: a PASS means the score does not show the surface-over-semantic RLC failure under this diagnostic.

Rows use 450 final-channel examples. MiniLM row: skipped because the encoder was unavailable locally.

| Router | Status | surf pos | sem pos | AUC surf | AP surf | AUC sem | AP sem | P@#sem | pair kappa | top-10% sem | top-10% both-refuse |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| final_full_tfidf | WARN | 150/450 | 66/450 | 0.667 | 0.453 | 0.649 | 0.234 | 0.303 | 0.267 | 13/45 | 32/45 |

| Router | Budget | semantic dis | surface dis | both refuse | both comply |
|---|---|---:|---:|---:|---:|
| final_full_tfidf | top_10pct (45) | 13 | 20 | 32 | 0 |
| final_full_tfidf | raw_prefix_gt_0_95_budget (285) | 49 | 117 | 131 | 105 |
| final_full_tfidf | semantic_positive_count (66) | 20 | 31 | 45 | 1 |

Interpretation: final-channel scoring is closer to the intended semantic construct than raw prefix scoring, but the gains remain modest and routed sets still contain many agreed refusals.
