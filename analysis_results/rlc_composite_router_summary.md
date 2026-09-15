# Audit-Guided Composite Repair Summary

Scores use cached final-channel outputs and final-channel semantic refusal judgments. The main fixed repair witness is `final_tfidf_marker_70_30`, an instance of `S_lambda = lambda z(final_tfidf) + (1-lambda) final_marker_disagreement` with lambda=0.7. The marker is a cheap final-span keyword disagreement computed only from observable final-answer text; it does not use semantic judge labels, human labels, judge rationales, or fitted weights. MiniLM and learned-router rows are diagnostic additions, not production-router claims.

| Pair | Score | sem dis | marker dis | AUC | AP | top-10% sem | top-10% both-refuse |
|---|---|---:|---:|---:|---:|---:|---:|
| Clean Qwen/Gemma | raw_prefix50_tfidf | 79/548 | 133/548 | 0.622 | 0.198 | 11/55 | 17/55 |
| Clean Qwen/Gemma | final_tfidf | 79/548 | 133/548 | 0.675 | 0.299 | 20/55 | 33/55 |
| Clean Qwen/Gemma | final_minilm | 79/548 | 133/548 | 0.651 | 0.297 | 20/55 | 31/55 |
| Clean Qwen/Gemma | final_tfidf_minilm_50_50 | 79/548 | 133/548 | 0.665 | 0.309 | 21/55 | 31/55 |
| Clean Qwen/Gemma | final_tfidf_marker_70_30 | 79/548 | 133/548 | 0.715 | 0.402 | 28/55 | 25/55 |
| Clean Qwen/Gemma | final_minilm_marker_70_30 | 79/548 | 133/548 | 0.689 | 0.343 | 23/55 | 29/55 |
| Clean Qwen/Gemma | rlc_composite_marker_fixed | 79/548 | 133/548 | 0.690 | 0.354 | 25/55 | 27/55 |
| Clean Qwen/Gemma | rlc_composite_marker_user_0_3 | 79/548 | 133/548 | 0.695 | 0.361 | 25/55 | 27/55 |
| Clean Qwen/Gemma | learned_logistic_oof | 79/548 | 133/548 | 0.741 | 0.421 | 29/55 | 21/55 |
| Clean Qwen/Gemma | final_marker_only | 79/548 | 133/548 | 0.728 | 0.291 | 21/55 | 27/55 |
| Qwen/Gemma | raw_prefix50_tfidf | 71/450 | 110/450 | 0.514 | 0.173 | 6/45 | 18/45 |
| Qwen/Gemma | final_tfidf | 71/450 | 110/450 | 0.664 | 0.316 | 17/45 | 23/45 |
| Qwen/Gemma | final_minilm | 71/450 | 110/450 | 0.661 | 0.291 | 14/45 | 27/45 |
| Qwen/Gemma | final_tfidf_minilm_50_50 | 71/450 | 110/450 | 0.676 | 0.300 | 15/45 | 24/45 |
| Qwen/Gemma | final_tfidf_marker_70_30 | 71/450 | 110/450 | 0.702 | 0.357 | 19/45 | 18/45 |
| Qwen/Gemma | final_minilm_marker_70_30 | 71/450 | 110/450 | 0.695 | 0.320 | 17/45 | 22/45 |
| Qwen/Gemma | rlc_composite_marker_fixed | 71/450 | 110/450 | 0.696 | 0.325 | 17/45 | 23/45 |
| Qwen/Gemma | rlc_composite_marker_user_0_3 | 71/450 | 110/450 | 0.701 | 0.331 | 17/45 | 23/45 |
| Qwen/Gemma | learned_logistic_oof | 71/450 | 110/450 | 0.757 | 0.497 | 24/45 | 13/45 |
| Qwen/Gemma | final_marker_only | 71/450 | 110/450 | 0.681 | 0.266 | 14/45 | 22/45 |
| Qwen/Mistral | raw_prefix50_tfidf | 66/450 | 150/450 | 0.440 | 0.129 | 2/45 | 12/45 |
| Qwen/Mistral | final_tfidf | 66/450 | 150/450 | 0.649 | 0.234 | 13/45 | 32/45 |
| Qwen/Mistral | final_minilm | 66/450 | 150/450 | 0.636 | 0.210 | 12/45 | 32/45 |
| Qwen/Mistral | final_tfidf_minilm_50_50 | 66/450 | 150/450 | 0.649 | 0.219 | 13/45 | 32/45 |
| Qwen/Mistral | final_tfidf_marker_70_30 | 66/450 | 150/450 | 0.687 | 0.317 | 18/45 | 27/45 |
| Qwen/Mistral | final_minilm_marker_70_30 | 66/450 | 150/450 | 0.680 | 0.237 | 14/45 | 30/45 |
| Qwen/Mistral | rlc_composite_marker_fixed | 66/450 | 150/450 | 0.673 | 0.238 | 13/45 | 32/45 |
| Qwen/Mistral | rlc_composite_marker_user_0_3 | 66/450 | 150/450 | 0.677 | 0.242 | 13/45 | 32/45 |
| Qwen/Mistral | learned_logistic_oof | 66/450 | 150/450 | 0.746 | 0.334 | 20/45 | 24/45 |
| Qwen/Mistral | final_marker_only | 66/450 | 150/450 | 0.704 | 0.251 | 13/45 | 17/45 |
| Qwen/Phi | raw_prefix50_tfidf | 43/450 | 114/450 | 0.359 | 0.072 | 2/45 | 28/45 |
| Qwen/Phi | final_tfidf | 43/450 | 114/450 | 0.526 | 0.134 | 6/45 | 38/45 |
| Qwen/Phi | final_minilm | 43/450 | 114/450 | 0.558 | 0.146 | 5/45 | 39/45 |
| Qwen/Phi | final_tfidf_minilm_50_50 | 43/450 | 114/450 | 0.535 | 0.143 | 7/45 | 37/45 |
| Qwen/Phi | final_tfidf_marker_70_30 | 43/450 | 114/450 | 0.541 | 0.138 | 5/45 | 39/45 |
| Qwen/Phi | final_minilm_marker_70_30 | 43/450 | 114/450 | 0.576 | 0.144 | 6/45 | 38/45 |
| Qwen/Phi | rlc_composite_marker_fixed | 43/450 | 114/450 | 0.545 | 0.143 | 8/45 | 36/45 |
| Qwen/Phi | rlc_composite_marker_user_0_3 | 43/450 | 114/450 | 0.547 | 0.143 | 8/45 | 36/45 |
| Qwen/Phi | learned_logistic_oof | 43/450 | 114/450 | 0.489 | 0.101 | 5/45 | 34/45 |
| Qwen/Phi | final_marker_only | 43/450 | 114/450 | 0.579 | 0.117 | 7/45 | 24/45 |
| Qwen3-8B/Phi | raw_prefix50_tfidf | 50/450 | 101/450 | 0.470 | 0.099 | 1/45 | 34/45 |
| Qwen3-8B/Phi | final_tfidf | 50/450 | 101/450 | 0.579 | 0.181 | 9/45 | 35/45 |
| Qwen3-8B/Phi | final_minilm | 50/450 | 101/450 | 0.592 | 0.207 | 10/45 | 35/45 |
| Qwen3-8B/Phi | final_tfidf_minilm_50_50 | 50/450 | 101/450 | 0.580 | 0.228 | 9/45 | 36/45 |
| Qwen3-8B/Phi | final_tfidf_marker_70_30 | 50/450 | 101/450 | 0.601 | 0.223 | 10/45 | 34/45 |
| Qwen3-8B/Phi | final_minilm_marker_70_30 | 50/450 | 101/450 | 0.611 | 0.212 | 9/45 | 36/45 |
| Qwen3-8B/Phi | rlc_composite_marker_fixed | 50/450 | 101/450 | 0.593 | 0.228 | 9/45 | 36/45 |
| Qwen3-8B/Phi | rlc_composite_marker_user_0_3 | 50/450 | 101/450 | 0.596 | 0.226 | 9/45 | 36/45 |
| Qwen3-8B/Phi | learned_logistic_oof | 50/450 | 101/450 | 0.569 | 0.180 | 12/45 | 28/45 |
| Qwen3-8B/Phi | final_marker_only | 50/450 | 101/450 | 0.621 | 0.158 | 10/45 | 26/45 |
| Gemma/Mistral | raw_prefix50_tfidf | 100/450 | 154/450 | 0.534 | 0.228 | 9/45 | 19/45 |
| Gemma/Mistral | final_tfidf | 100/450 | 154/450 | 0.707 | 0.376 | 18/45 | 26/45 |
| Gemma/Mistral | final_minilm | 100/450 | 154/450 | 0.636 | 0.319 | 13/45 | 32/45 |
| Gemma/Mistral | final_tfidf_minilm_50_50 | 100/450 | 154/450 | 0.682 | 0.349 | 13/45 | 32/45 |
| Gemma/Mistral | final_tfidf_marker_70_30 | 100/450 | 154/450 | 0.753 | 0.440 | 22/45 | 22/45 |
| Gemma/Mistral | final_minilm_marker_70_30 | 100/450 | 154/450 | 0.676 | 0.345 | 14/45 | 31/45 |
| Gemma/Mistral | rlc_composite_marker_fixed | 100/450 | 154/450 | 0.704 | 0.369 | 16/45 | 29/45 |
| Gemma/Mistral | rlc_composite_marker_user_0_3 | 100/450 | 154/450 | 0.708 | 0.371 | 16/45 | 29/45 |
| Gemma/Mistral | learned_logistic_oof | 100/450 | 154/450 | 0.761 | 0.461 | 22/45 | 21/45 |
| Gemma/Mistral | final_marker_only | 100/450 | 154/450 | 0.704 | 0.358 | 21/45 | 20/45 |
| Gemma/Qwen3-8B | raw_prefix50_tfidf | 74/450 | 85/450 | 0.463 | 0.159 | 4/45 | 20/45 |
| Gemma/Qwen3-8B | final_tfidf | 74/450 | 85/450 | 0.557 | 0.231 | 8/45 | 36/45 |
| Gemma/Qwen3-8B | final_minilm | 74/450 | 85/450 | 0.545 | 0.190 | 7/45 | 38/45 |
| Gemma/Qwen3-8B | final_tfidf_minilm_50_50 | 74/450 | 85/450 | 0.552 | 0.202 | 7/45 | 38/45 |
| Gemma/Qwen3-8B | final_tfidf_marker_70_30 | 74/450 | 85/450 | 0.620 | 0.336 | 21/45 | 23/45 |
| Gemma/Qwen3-8B | final_minilm_marker_70_30 | 74/450 | 85/450 | 0.589 | 0.215 | 8/45 | 37/45 |
| Gemma/Qwen3-8B | rlc_composite_marker_fixed | 74/450 | 85/450 | 0.579 | 0.222 | 9/45 | 36/45 |
| Gemma/Qwen3-8B | rlc_composite_marker_user_0_3 | 74/450 | 85/450 | 0.583 | 0.223 | 10/45 | 35/45 |
| Gemma/Qwen3-8B | learned_logistic_oof | 74/450 | 85/450 | 0.668 | 0.368 | 20/45 | 14/45 |
| Gemma/Qwen3-8B | final_marker_only | 74/450 | 85/450 | 0.702 | 0.320 | 21/45 | 14/45 |
| Gemma/Phi | raw_prefix50_tfidf | 64/450 | 122/450 | 0.430 | 0.120 | 2/45 | 36/45 |
| Gemma/Phi | final_tfidf | 64/450 | 122/450 | 0.647 | 0.247 | 11/45 | 33/45 |
| Gemma/Phi | final_minilm | 64/450 | 122/450 | 0.598 | 0.252 | 11/45 | 33/45 |
| Gemma/Phi | final_tfidf_minilm_50_50 | 64/450 | 122/450 | 0.631 | 0.273 | 10/45 | 35/45 |
| Gemma/Phi | final_tfidf_marker_70_30 | 64/450 | 122/450 | 0.695 | 0.297 | 15/45 | 27/45 |
| Gemma/Phi | final_minilm_marker_70_30 | 64/450 | 122/450 | 0.644 | 0.289 | 11/45 | 33/45 |
| Gemma/Phi | rlc_composite_marker_fixed | 64/450 | 122/450 | 0.655 | 0.295 | 10/45 | 35/45 |
| Gemma/Phi | rlc_composite_marker_user_0_3 | 64/450 | 122/450 | 0.660 | 0.294 | 11/45 | 34/45 |
| Gemma/Phi | learned_logistic_oof | 64/450 | 122/450 | 0.743 | 0.444 | 20/45 | 23/45 |
| Gemma/Phi | final_marker_only | 64/450 | 122/450 | 0.706 | 0.258 | 16/45 | 10/45 |
| Mistral/Qwen3-8B | raw_prefix50_tfidf | 82/450 | 137/450 | 0.492 | 0.166 | 2/45 | 16/45 |
| Mistral/Qwen3-8B | final_tfidf | 82/450 | 137/450 | 0.696 | 0.335 | 22/45 | 23/45 |
| Mistral/Qwen3-8B | final_minilm | 82/450 | 137/450 | 0.695 | 0.366 | 24/45 | 21/45 |
| Mistral/Qwen3-8B | final_tfidf_minilm_50_50 | 82/450 | 137/450 | 0.699 | 0.375 | 23/45 | 22/45 |
| Mistral/Qwen3-8B | final_tfidf_marker_70_30 | 82/450 | 137/450 | 0.728 | 0.392 | 23/45 | 22/45 |
| Mistral/Qwen3-8B | final_minilm_marker_70_30 | 82/450 | 137/450 | 0.730 | 0.400 | 23/45 | 22/45 |
| Mistral/Qwen3-8B | rlc_composite_marker_fixed | 82/450 | 137/450 | 0.717 | 0.398 | 24/45 | 21/45 |
| Mistral/Qwen3-8B | rlc_composite_marker_user_0_3 | 82/450 | 137/450 | 0.721 | 0.402 | 24/45 | 21/45 |
| Mistral/Qwen3-8B | learned_logistic_oof | 82/450 | 137/450 | 0.743 | 0.385 | 21/45 | 24/45 |
| Mistral/Qwen3-8B | final_marker_only | 82/450 | 137/450 | 0.724 | 0.329 | 13/45 | 14/45 |
| Mistral/Phi | raw_prefix50_tfidf | 70/450 | 120/450 | 0.426 | 0.135 | 4/45 | 32/45 |
| Mistral/Phi | final_tfidf | 70/450 | 120/450 | 0.667 | 0.244 | 11/45 | 31/45 |
| Mistral/Phi | final_minilm | 70/450 | 120/450 | 0.667 | 0.268 | 12/45 | 31/45 |
| Mistral/Phi | final_tfidf_minilm_50_50 | 70/450 | 120/450 | 0.675 | 0.272 | 11/45 | 32/45 |
| Mistral/Phi | final_tfidf_marker_70_30 | 70/450 | 120/450 | 0.690 | 0.271 | 11/45 | 32/45 |
| Mistral/Phi | final_minilm_marker_70_30 | 70/450 | 120/450 | 0.696 | 0.272 | 14/45 | 29/45 |
| Mistral/Phi | rlc_composite_marker_fixed | 70/450 | 120/450 | 0.689 | 0.272 | 12/45 | 32/45 |
| Mistral/Phi | rlc_composite_marker_user_0_3 | 70/450 | 120/450 | 0.693 | 0.274 | 12/45 | 32/45 |
| Mistral/Phi | learned_logistic_oof | 70/450 | 120/450 | 0.702 | 0.260 | 9/45 | 36/45 |
| Mistral/Phi | final_marker_only | 70/450 | 120/450 | 0.647 | 0.230 | 15/45 | 29/45 |
| Qwen/Llama | raw_prefix50_tfidf | 47/450 | 116/450 | 0.461 | 0.100 | 2/45 | 35/45 |
| Qwen/Llama | final_tfidf | 47/450 | 116/450 | 0.559 | 0.121 | 4/45 | 41/45 |
| Qwen/Llama | final_minilm | 47/450 | 116/450 | 0.606 | 0.128 | 3/45 | 42/45 |
| Qwen/Llama | final_tfidf_minilm_50_50 | 47/450 | 116/450 | 0.574 | 0.123 | 3/45 | 42/45 |
| Qwen/Llama | final_tfidf_marker_70_30 | 47/450 | 116/450 | 0.590 | 0.156 | 9/45 | 36/45 |
| Qwen/Llama | final_minilm_marker_70_30 | 47/450 | 116/450 | 0.635 | 0.140 | 3/45 | 42/45 |
| Qwen/Llama | rlc_composite_marker_fixed | 47/450 | 116/450 | 0.589 | 0.130 | 3/45 | 42/45 |
| Qwen/Llama | rlc_composite_marker_user_0_3 | 47/450 | 116/450 | 0.593 | 0.133 | 3/45 | 42/45 |
| Qwen/Llama | learned_logistic_oof | 47/450 | 116/450 | 0.644 | 0.197 | 12/45 | 31/45 |
| Qwen/Llama | final_marker_only | 47/450 | 116/450 | 0.653 | 0.164 | 7/45 | 28/45 |
| Gemma/Llama | raw_prefix50_tfidf | 94/450 | 126/450 | 0.474 | 0.187 | 3/45 | 42/45 |
| Gemma/Llama | final_tfidf | 94/450 | 126/450 | 0.633 | 0.283 | 13/45 | 32/45 |
| Gemma/Llama | final_minilm | 94/450 | 126/450 | 0.562 | 0.221 | 8/45 | 37/45 |
| Gemma/Llama | final_tfidf_minilm_50_50 | 94/450 | 126/450 | 0.601 | 0.239 | 9/45 | 36/45 |
| Gemma/Llama | final_tfidf_marker_70_30 | 94/450 | 126/450 | 0.688 | 0.361 | 20/45 | 23/45 |
| Gemma/Llama | final_minilm_marker_70_30 | 94/450 | 126/450 | 0.612 | 0.248 | 9/45 | 36/45 |
| Gemma/Llama | rlc_composite_marker_fixed | 94/450 | 126/450 | 0.631 | 0.264 | 11/45 | 34/45 |
| Gemma/Llama | rlc_composite_marker_user_0_3 | 94/450 | 126/450 | 0.636 | 0.269 | 11/45 | 34/45 |
| Gemma/Llama | learned_logistic_oof | 94/450 | 126/450 | 0.770 | 0.516 | 30/45 | 7/45 |
| Gemma/Llama | final_marker_only | 94/450 | 126/450 | 0.706 | 0.357 | 18/45 | 21/45 |
| Mistral/Llama | raw_prefix50_tfidf | 76/450 | 130/450 | 0.615 | 0.228 | 11/45 | 31/45 |
| Mistral/Llama | final_tfidf | 76/450 | 130/450 | 0.701 | 0.320 | 18/45 | 27/45 |
| Mistral/Llama | final_minilm | 76/450 | 130/450 | 0.717 | 0.378 | 17/45 | 28/45 |
| Mistral/Llama | final_tfidf_minilm_50_50 | 76/450 | 130/450 | 0.709 | 0.381 | 19/45 | 26/45 |
| Mistral/Llama | final_tfidf_marker_70_30 | 76/450 | 130/450 | 0.728 | 0.365 | 19/45 | 26/45 |
| Mistral/Llama | final_minilm_marker_70_30 | 76/450 | 130/450 | 0.741 | 0.399 | 19/45 | 26/45 |
| Mistral/Llama | rlc_composite_marker_fixed | 76/450 | 130/450 | 0.726 | 0.398 | 19/45 | 26/45 |
| Mistral/Llama | rlc_composite_marker_user_0_3 | 76/450 | 130/450 | 0.729 | 0.401 | 18/45 | 27/45 |
| Mistral/Llama | learned_logistic_oof | 76/450 | 130/450 | 0.725 | 0.350 | 19/45 | 26/45 |
| Mistral/Llama | final_marker_only | 76/450 | 130/450 | 0.698 | 0.288 | 19/45 | 18/45 |
| Qwen3-8B/Llama | raw_prefix50_tfidf | 66/450 | 105/450 | 0.513 | 0.138 | 2/45 | 43/45 |
| Qwen3-8B/Llama | final_tfidf | 66/450 | 105/450 | 0.664 | 0.226 | 11/45 | 34/45 |
| Qwen3-8B/Llama | final_minilm | 66/450 | 105/450 | 0.666 | 0.228 | 13/45 | 32/45 |
| Qwen3-8B/Llama | final_tfidf_minilm_50_50 | 66/450 | 105/450 | 0.671 | 0.228 | 12/45 | 33/45 |
| Qwen3-8B/Llama | final_tfidf_marker_70_30 | 66/450 | 105/450 | 0.707 | 0.296 | 19/45 | 26/45 |
| Qwen3-8B/Llama | final_minilm_marker_70_30 | 66/450 | 105/450 | 0.688 | 0.263 | 14/45 | 31/45 |
| Qwen3-8B/Llama | rlc_composite_marker_fixed | 66/450 | 105/450 | 0.688 | 0.263 | 14/45 | 31/45 |
| Qwen3-8B/Llama | rlc_composite_marker_user_0_3 | 66/450 | 105/450 | 0.691 | 0.267 | 14/45 | 31/45 |
| Qwen3-8B/Llama | learned_logistic_oof | 66/450 | 105/450 | 0.734 | 0.370 | 18/45 | 24/45 |
| Qwen3-8B/Llama | final_marker_only | 66/450 | 105/450 | 0.710 | 0.279 | 14/45 | 27/45 |
| Llama/Phi | raw_prefix50_tfidf | 60/450 | 114/450 | 0.457 | 0.115 | 2/45 | 43/45 |
| Llama/Phi | final_tfidf | 60/450 | 114/450 | 0.598 | 0.164 | 7/45 | 37/45 |
| Llama/Phi | final_minilm | 60/450 | 114/450 | 0.628 | 0.170 | 5/45 | 40/45 |
| Llama/Phi | final_tfidf_minilm_50_50 | 60/450 | 114/450 | 0.608 | 0.166 | 5/45 | 40/45 |
| Llama/Phi | final_tfidf_marker_70_30 | 60/450 | 114/450 | 0.622 | 0.192 | 12/45 | 33/45 |
| Llama/Phi | final_minilm_marker_70_30 | 60/450 | 114/450 | 0.644 | 0.179 | 5/45 | 40/45 |
| Llama/Phi | rlc_composite_marker_fixed | 60/450 | 114/450 | 0.621 | 0.174 | 6/45 | 39/45 |
| Llama/Phi | rlc_composite_marker_user_0_3 | 60/450 | 114/450 | 0.623 | 0.175 | 7/45 | 38/45 |
| Llama/Phi | learned_logistic_oof | 60/450 | 114/450 | 0.627 | 0.197 | 10/45 | 35/45 |
| Llama/Phi | final_marker_only | 60/450 | 114/450 | 0.613 | 0.180 | 12/45 | 17/45 |
| Qwen/Qwen | raw_prefix50_tfidf | 40/450 | 73/450 | 0.379 | 0.068 | 0/45 | 9/45 |
| Qwen/Qwen | final_tfidf | 40/450 | 73/450 | 0.631 | 0.181 | 8/45 | 37/45 |
| Qwen/Qwen | final_minilm | 40/450 | 73/450 | 0.613 | 0.215 | 9/45 | 36/45 |
| Qwen/Qwen | final_tfidf_minilm_50_50 | 40/450 | 73/450 | 0.624 | 0.217 | 9/45 | 36/45 |
| Qwen/Qwen | final_tfidf_marker_70_30 | 40/450 | 73/450 | 0.658 | 0.363 | 13/45 | 32/45 |
| Qwen/Qwen | final_minilm_marker_70_30 | 40/450 | 73/450 | 0.622 | 0.273 | 9/45 | 36/45 |
| Qwen/Qwen | rlc_composite_marker_fixed | 40/450 | 73/450 | 0.632 | 0.279 | 9/45 | 36/45 |
| Qwen/Qwen | rlc_composite_marker_user_0_3 | 40/450 | 73/450 | 0.633 | 0.290 | 9/45 | 36/45 |
| Qwen/Qwen | learned_logistic_oof | 40/450 | 73/450 | 0.591 | 0.379 | 16/45 | 17/45 |
| Qwen/Qwen | final_marker_only | 40/450 | 73/450 | 0.658 | 0.160 | 10/45 | 22/45 |

## Learned Router Coefficients

- Clean Qwen/Gemma: 5-fold OOF logistic, mean coefficients: final_tfidf=0.044, final_minilm=0.307, final_marker_disagreement=0.789, final_length_gap=0.242
- Qwen/Gemma: 5-fold OOF logistic, mean coefficients: final_tfidf=0.291, final_minilm=0.248, final_marker_disagreement=0.621, final_length_gap=0.519
- Qwen/Mistral: 5-fold OOF logistic, mean coefficients: final_tfidf=0.193, final_minilm=0.171, final_marker_disagreement=0.675, final_length_gap=-0.308
- Qwen/Phi: 5-fold OOF logistic, mean coefficients: final_tfidf=-0.218, final_minilm=0.322, final_marker_disagreement=0.316, final_length_gap=-0.108
- Qwen3-8B/Phi: 5-fold OOF logistic, mean coefficients: final_tfidf=0.197, final_minilm=-0.059, final_marker_disagreement=0.449, final_length_gap=0.109
- Gemma/Mistral: 5-fold OOF logistic, mean coefficients: final_tfidf=0.874, final_minilm=-0.155, final_marker_disagreement=0.655, final_length_gap=-0.088
- Gemma/Qwen3-8B: 5-fold OOF logistic, mean coefficients: final_tfidf=0.005, final_minilm=0.033, final_marker_disagreement=0.796, final_length_gap=-0.219
- Gemma/Phi: 5-fold OOF logistic, mean coefficients: final_tfidf=0.438, final_minilm=-0.021, final_marker_disagreement=0.753, final_length_gap=0.405
- Mistral/Qwen3-8B: 5-fold OOF logistic, mean coefficients: final_tfidf=0.267, final_minilm=0.133, final_marker_disagreement=0.753, final_length_gap=0.115
- Mistral/Phi: 5-fold OOF logistic, mean coefficients: final_tfidf=0.334, final_minilm=0.204, final_marker_disagreement=0.462, final_length_gap=-0.135
- Qwen/Llama: 5-fold OOF logistic, mean coefficients: final_tfidf=0.112, final_minilm=-0.000, final_marker_disagreement=0.556, final_length_gap=-0.069
- Gemma/Llama: 5-fold OOF logistic, mean coefficients: final_tfidf=0.953, final_minilm=-0.608, final_marker_disagreement=0.766, final_length_gap=-0.102
- Mistral/Llama: 5-fold OOF logistic, mean coefficients: final_tfidf=0.225, final_minilm=0.320, final_marker_disagreement=0.599, final_length_gap=-0.041
- Qwen3-8B/Llama: 5-fold OOF logistic, mean coefficients: final_tfidf=0.576, final_minilm=-0.191, final_marker_disagreement=0.731, final_length_gap=-0.117
- Llama/Phi: 5-fold OOF logistic, mean coefficients: final_tfidf=0.341, final_minilm=0.001, final_marker_disagreement=0.358, final_length_gap=-0.287
- Qwen/Qwen: 5-fold OOF logistic, mean coefficients: final_tfidf=0.101, final_minilm=0.093, final_marker_disagreement=0.529, final_length_gap=0.145

## Lambda Sensitivity

| Pair | lambda | AUC | AP | top-10% sem | top-10% both-refuse |
|---|---:|---:|---:|---:|---:|
| Clean Qwen/Gemma | 0.5 | 0.744 | 0.440 | 31/55 | 24/55 |
| Clean Qwen/Gemma | 0.6 | 0.730 | 0.422 | 31/55 | 22/55 |
| Clean Qwen/Gemma | 0.7 | 0.715 | 0.402 | 28/55 | 25/55 |
| Clean Qwen/Gemma | 0.8 | 0.701 | 0.380 | 25/55 | 28/55 |
| Clean Qwen/Gemma | 0.9 | 0.688 | 0.349 | 22/55 | 31/55 |
| Qwen/Gemma | 0.5 | 0.729 | 0.383 | 20/45 | 17/45 |
| Qwen/Gemma | 0.6 | 0.716 | 0.371 | 19/45 | 17/45 |
| Qwen/Gemma | 0.7 | 0.702 | 0.357 | 19/45 | 18/45 |
| Qwen/Gemma | 0.8 | 0.687 | 0.342 | 17/45 | 21/45 |
| Qwen/Gemma | 0.9 | 0.675 | 0.328 | 17/45 | 22/45 |
| Qwen/Mistral | 0.5 | 0.716 | 0.350 | 19/45 | 26/45 |
| Qwen/Mistral | 0.6 | 0.701 | 0.335 | 19/45 | 26/45 |
| Qwen/Mistral | 0.7 | 0.687 | 0.317 | 18/45 | 27/45 |
| Qwen/Mistral | 0.8 | 0.674 | 0.296 | 15/45 | 30/45 |
| Qwen/Mistral | 0.9 | 0.661 | 0.263 | 14/45 | 31/45 |
| Qwen/Phi | 0.5 | 0.557 | 0.144 | 6/45 | 37/45 |
| Qwen/Phi | 0.6 | 0.549 | 0.140 | 6/45 | 37/45 |
| Qwen/Phi | 0.7 | 0.541 | 0.138 | 5/45 | 39/45 |
| Qwen/Phi | 0.8 | 0.536 | 0.137 | 5/45 | 39/45 |
| Qwen/Phi | 0.9 | 0.530 | 0.133 | 5/45 | 39/45 |
| Qwen3-8B/Phi | 0.5 | 0.621 | 0.232 | 11/45 | 33/45 |
| Qwen3-8B/Phi | 0.6 | 0.609 | 0.226 | 11/45 | 34/45 |
| Qwen3-8B/Phi | 0.7 | 0.601 | 0.223 | 10/45 | 34/45 |
| Qwen3-8B/Phi | 0.8 | 0.593 | 0.217 | 9/45 | 35/45 |
| Qwen3-8B/Phi | 0.9 | 0.584 | 0.213 | 9/45 | 35/45 |
| Gemma/Mistral | 0.5 | 0.774 | 0.464 | 22/45 | 22/45 |
| Gemma/Mistral | 0.6 | 0.766 | 0.454 | 22/45 | 22/45 |
| Gemma/Mistral | 0.7 | 0.753 | 0.440 | 22/45 | 22/45 |
| Gemma/Mistral | 0.8 | 0.738 | 0.419 | 20/45 | 24/45 |
| Gemma/Mistral | 0.9 | 0.722 | 0.398 | 19/45 | 25/45 |
| Gemma/Qwen3-8B | 0.5 | 0.663 | 0.398 | 26/45 | 13/45 |
| Gemma/Qwen3-8B | 0.6 | 0.645 | 0.375 | 24/45 | 17/45 |
| Gemma/Qwen3-8B | 0.7 | 0.620 | 0.336 | 21/45 | 23/45 |
| Gemma/Qwen3-8B | 0.8 | 0.596 | 0.279 | 13/45 | 31/45 |
| Gemma/Qwen3-8B | 0.9 | 0.573 | 0.250 | 10/45 | 35/45 |
| Gemma/Phi | 0.5 | 0.732 | 0.330 | 14/45 | 29/45 |
| Gemma/Phi | 0.6 | 0.716 | 0.316 | 14/45 | 29/45 |
| Gemma/Phi | 0.7 | 0.695 | 0.297 | 15/45 | 27/45 |
| Gemma/Phi | 0.8 | 0.677 | 0.280 | 14/45 | 30/45 |
| Gemma/Phi | 0.9 | 0.660 | 0.263 | 11/45 | 33/45 |
| Mistral/Qwen3-8B | 0.5 | 0.751 | 0.414 | 23/45 | 22/45 |
| Mistral/Qwen3-8B | 0.6 | 0.739 | 0.402 | 23/45 | 22/45 |
| Mistral/Qwen3-8B | 0.7 | 0.728 | 0.392 | 23/45 | 22/45 |
| Mistral/Qwen3-8B | 0.8 | 0.718 | 0.381 | 22/45 | 23/45 |
| Mistral/Qwen3-8B | 0.9 | 0.708 | 0.369 | 21/45 | 24/45 |
| Mistral/Phi | 0.5 | 0.710 | 0.285 | 13/45 | 31/45 |
| Mistral/Phi | 0.6 | 0.700 | 0.278 | 13/45 | 31/45 |
| Mistral/Phi | 0.7 | 0.690 | 0.271 | 11/45 | 32/45 |
| Mistral/Phi | 0.8 | 0.681 | 0.265 | 12/45 | 31/45 |
| Mistral/Phi | 0.9 | 0.674 | 0.259 | 14/45 | 29/45 |
| Qwen/Llama | 0.5 | 0.617 | 0.180 | 14/45 | 30/45 |
| Qwen/Llama | 0.6 | 0.602 | 0.165 | 10/45 | 35/45 |
| Qwen/Llama | 0.7 | 0.590 | 0.156 | 9/45 | 36/45 |
| Qwen/Llama | 0.8 | 0.580 | 0.145 | 7/45 | 38/45 |
| Qwen/Llama | 0.9 | 0.570 | 0.131 | 6/45 | 39/45 |
| Gemma/Llama | 0.5 | 0.733 | 0.415 | 22/45 | 21/45 |
| Gemma/Llama | 0.6 | 0.711 | 0.386 | 22/45 | 21/45 |
| Gemma/Llama | 0.7 | 0.688 | 0.361 | 20/45 | 23/45 |
| Gemma/Llama | 0.8 | 0.667 | 0.338 | 16/45 | 27/45 |
| Gemma/Llama | 0.9 | 0.649 | 0.316 | 15/45 | 30/45 |
| Mistral/Llama | 0.5 | 0.742 | 0.375 | 19/45 | 26/45 |
| Mistral/Llama | 0.6 | 0.736 | 0.372 | 19/45 | 26/45 |
| Mistral/Llama | 0.7 | 0.728 | 0.365 | 19/45 | 26/45 |
| Mistral/Llama | 0.8 | 0.719 | 0.357 | 19/45 | 26/45 |
| Mistral/Llama | 0.9 | 0.710 | 0.347 | 20/45 | 25/45 |
| Qwen3-8B/Llama | 0.5 | 0.728 | 0.321 | 20/45 | 25/45 |
| Qwen3-8B/Llama | 0.6 | 0.719 | 0.313 | 20/45 | 25/45 |
| Qwen3-8B/Llama | 0.7 | 0.707 | 0.296 | 19/45 | 26/45 |
| Qwen3-8B/Llama | 0.8 | 0.692 | 0.273 | 16/45 | 29/45 |
| Qwen3-8B/Llama | 0.9 | 0.679 | 0.255 | 14/45 | 31/45 |
| Llama/Phi | 0.5 | 0.640 | 0.205 | 12/45 | 33/45 |
| Llama/Phi | 0.6 | 0.632 | 0.199 | 12/45 | 33/45 |
| Llama/Phi | 0.7 | 0.622 | 0.192 | 12/45 | 33/45 |
| Llama/Phi | 0.8 | 0.613 | 0.182 | 8/45 | 36/45 |
| Llama/Phi | 0.9 | 0.606 | 0.176 | 8/45 | 36/45 |
| Qwen/Qwen | 0.5 | 0.672 | 0.432 | 17/45 | 24/45 |
| Qwen/Qwen | 0.6 | 0.664 | 0.396 | 14/45 | 30/45 |
| Qwen/Qwen | 0.7 | 0.658 | 0.363 | 13/45 | 32/45 |
| Qwen/Qwen | 0.8 | 0.647 | 0.301 | 11/45 | 34/45 |
| Qwen/Qwen | 0.9 | 0.640 | 0.275 | 9/45 | 36/45 |
