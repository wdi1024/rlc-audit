# Gemma/Mistral Cross-Pair Final Analysis

Source phases: `phase_colab_qwen_gemma_big_xstest450_tagged768_thinkingoff_greedy` and `phase_colab_qwen_mistral_xstest450_tagged768_thinkingoff_greedy`
Models: `gemma-2-9b-it` vs `mistral-7b-instruct-v0.3`

## Compatibility

Common prompts: 450
Identical prompt texts: 450/450
Prompt mode: `tagged` / `tagged`
Max new tokens: `768` / `768`

## Parse Coverage

| Model | final coverage | judge refusals | parse status |
|---|---:|---:|---|
| gemma-2-9b-it | 450/450 | 261/450 | `{"exact": 449, "final_only": 1}` |
| mistral-7b-instruct-v0.3 | 450/450 | 179/450 | `{"exact": 226, "fallback_truncated_final": 224}` |

## Final-Span Metrics

| Subset | n | sem dis | final kw dis | final AUC sem | final AP sem | final AUC kw | final AP kw | top-10% sem | top-10% both-refuse | kappa kw/sem |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all_final_nonempty | 450 | 100 | 154 | 0.707 | 0.376 | 0.700 | 0.513 | 18/45 | 26/45 | 0.343 |
| exclude_any_fallback | 226 | 50 | 75 | 0.737 | 0.445 | 0.739 | 0.582 | 11/23 | 11/23 | 0.336 |
| mistral_exact_only | 226 | 50 | 75 | 0.737 | 0.445 | 0.739 | 0.582 | 11/23 | 11/23 | 0.336 |

## Raw/Prefix Comparison

| Subset | Score | sem AUC | sem AP | kw AUC | kw AP |
|---|---|---:|---:|---:|---:|
| all_final_nonempty | raw_prefix50_tfidf | 0.534 | 0.228 | 0.556 | 0.363 |
| all_final_nonempty | raw_full_tfidf | 0.693 | 0.359 | 0.707 | 0.507 |
| all_final_nonempty | final_full_tfidf | 0.707 | 0.376 | 0.700 | 0.513 |
| exclude_any_fallback | raw_prefix50_tfidf | 0.570 | 0.244 | 0.601 | 0.404 |
| exclude_any_fallback | raw_full_tfidf | 0.679 | 0.346 | 0.726 | 0.536 |
| exclude_any_fallback | final_full_tfidf | 0.737 | 0.445 | 0.739 | 0.582 |
| mistral_exact_only | raw_prefix50_tfidf | 0.570 | 0.244 | 0.601 | 0.404 |
| mistral_exact_only | raw_full_tfidf | 0.679 | 0.346 | 0.726 | 0.536 |
| mistral_exact_only | final_full_tfidf | 0.737 | 0.445 | 0.739 | 0.582 |
