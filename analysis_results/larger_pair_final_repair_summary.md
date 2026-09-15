# Larger-Pair Final-Span Repair Summary

All rows use matched XSTest 450 prompts, tagged `<reasoning>`/`<final>` generation, and final-channel OpenAI `gpt-4o-mini` refusal judgments.

| Pair | Role | n | sem dis | raw-prefix AUC | final AUC | final AP | trunc-excl final AUC | top-10% sem |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Gemma | heterogeneous repair | 450 | 71 | 0.514 | 0.664 | 0.316 | 0.684 | 17/45 |
| Qwen/Mistral | heterogeneous repair | 450 | 66 | 0.440 | 0.649 | 0.234 | 0.621 | 13/45 |
| Qwen/Phi | cross-family panel | 450 | 43 | 0.359 | 0.526 | 0.134 | 0.515 | 6/45 |
| Qwen3-8B/Phi | size-control panel | 450 | 50 | 0.470 | 0.579 | 0.181 | 0.538 | 9/45 |
| Gemma/Mistral | cross-pair repair check | 450 | 100 | 0.534 | 0.707 | 0.376 | 0.737 | 18/45 |
| Gemma/Qwen3-8B | size-control panel | 450 | 74 | 0.463 | 0.557 | 0.231 | 0.577 | 8/45 |
| Gemma/Phi | cross-family panel | 450 | 64 | 0.430 | 0.647 | 0.247 | 0.592 | 11/45 |
| Mistral/Qwen3-8B | size-control panel | 450 | 82 | 0.492 | 0.696 | 0.335 | 0.647 | 22/45 |
| Mistral/Phi | cross-family panel | 450 | 70 | 0.426 | 0.667 | 0.244 | 0.644 | 11/45 |
| Qwen/Llama | cross-family panel | 450 | 47 | 0.461 | 0.559 | 0.121 | 0.968 | 4/45 |
| Gemma/Llama | cross-family panel | 450 | 94 | 0.474 | 0.633 | 0.283 | 0.354 | 13/45 |
| Mistral/Llama | cross-family panel | 450 | 76 | 0.615 | 0.701 | 0.320 | 0.709 | 18/45 |
| Qwen3-8B/Llama | size-control panel | 450 | 66 | 0.513 | 0.664 | 0.226 | 0.760 | 11/45 |
| Llama/Phi | cross-family panel | 450 | 60 | 0.457 | 0.598 | 0.164 | 0.065 | 7/45 |
| Qwen/Qwen | same-family control | 450 | 40 | 0.379 | 0.631 | 0.181 | 0.688 | 8/45 |

## Parse Coverage

| Pair | Model | final coverage | parse status |
|---|---|---:|---|
| Qwen/Gemma | qwen3.5-9b | 450/450 | `{"exact": 385, "fallback_truncated_final": 65}` |
| Qwen/Gemma | gemma-2-9b-it | 450/450 | `{"exact": 449, "final_only": 1}` |
| Qwen/Mistral | qwen3.5-9b | 450/450 | `{"exact": 361, "fallback_truncated_final": 89}` |
| Qwen/Mistral | mistral-7b-instruct-v0.3 | 450/450 | `{"exact": 226, "fallback_truncated_final": 224}` |
| Qwen/Phi | qwen3.5-9b | 450/450 | `{"exact": 385, "fallback_truncated_final": 65}` |
| Qwen/Phi | phi-3.5-mini-instruct | 450/450 | `{"exact": 336, "fallback_truncated_final": 96, "final_only": 18}` |
| Qwen3-8B/Phi | qwen3-8b | 450/450 | `{"exact": 421, "fallback_truncated_final": 29}` |
| Qwen3-8B/Phi | phi-3.5-mini-instruct | 450/450 | `{"exact": 336, "fallback_truncated_final": 96, "final_only": 18}` |
| Gemma/Mistral | gemma-2-9b-it | 450/450 | `{"exact": 449, "final_only": 1}` |
| Gemma/Mistral | mistral-7b-instruct-v0.3 | 450/450 | `{"exact": 226, "fallback_truncated_final": 224}` |
| Gemma/Qwen3-8B | gemma-2-9b-it | 450/450 | `{"exact": 449, "final_only": 1}` |
| Gemma/Qwen3-8B | qwen3-8b | 450/450 | `{"exact": 421, "fallback_truncated_final": 29}` |
| Gemma/Phi | gemma-2-9b-it | 450/450 | `{"exact": 449, "final_only": 1}` |
| Gemma/Phi | phi-3.5-mini-instruct | 450/450 | `{"exact": 336, "fallback_truncated_final": 96, "final_only": 18}` |
| Mistral/Qwen3-8B | mistral-7b-instruct-v0.3 | 450/450 | `{"exact": 226, "fallback_truncated_final": 224}` |
| Mistral/Qwen3-8B | qwen3-8b | 450/450 | `{"exact": 421, "fallback_truncated_final": 29}` |
| Mistral/Phi | mistral-7b-instruct-v0.3 | 450/450 | `{"exact": 226, "fallback_truncated_final": 224}` |
| Mistral/Phi | phi-3.5-mini-instruct | 450/450 | `{"exact": 336, "fallback_truncated_final": 96, "final_only": 18}` |
| Qwen/Llama | qwen3.5-9b | 450/450 | `{"exact": 385, "fallback_truncated_final": 65}` |
| Qwen/Llama | llama-3.1-8b-instruct | 450/450 | `{"exact": 8, "fallback_raw_as_final": 80, "fallback_truncated_final": 362}` |
| Gemma/Llama | gemma-2-9b-it | 450/450 | `{"exact": 449, "final_only": 1}` |
| Gemma/Llama | llama-3.1-8b-instruct | 450/450 | `{"exact": 8, "fallback_raw_as_final": 80, "fallback_truncated_final": 362}` |
| Mistral/Llama | mistral-7b-instruct-v0.3 | 450/450 | `{"exact": 226, "fallback_truncated_final": 224}` |
| Mistral/Llama | llama-3.1-8b-instruct | 450/450 | `{"exact": 8, "fallback_raw_as_final": 80, "fallback_truncated_final": 362}` |
| Qwen3-8B/Llama | qwen3-8b | 450/450 | `{"exact": 421, "fallback_truncated_final": 29}` |
| Qwen3-8B/Llama | llama-3.1-8b-instruct | 450/450 | `{"exact": 8, "fallback_raw_as_final": 80, "fallback_truncated_final": 362}` |
| Llama/Phi | llama-3.1-8b-instruct | 450/450 | `{"exact": 8, "fallback_raw_as_final": 80, "fallback_truncated_final": 362}` |
| Llama/Phi | phi-3.5-mini-instruct | 450/450 | `{"exact": 336, "fallback_truncated_final": 96, "final_only": 18}` |
| Qwen/Qwen | qwen3-8b | 450/450 | `{"exact": 421, "fallback_truncated_final": 29}` |
| Qwen/Qwen | qwen3.5-9b | 450/450 | `{"exact": 361, "fallback_truncated_final": 89}` |
