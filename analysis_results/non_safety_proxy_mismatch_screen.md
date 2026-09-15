# Non-Safety Proxy Mismatch Screen

This is an experiment-prep screen over cached HotpotQA and GSM8K correctness traces.
Rows marked `MISMATCH_CANDIDATE` are candidates for a full non-safety RLC audit; they are not paper claims until inspected and validated.

| Task | Status | Proxy | Score | Surf+ | Sem+ | AUC surf | AUC sem | Gap | Kappa | Sem@50 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | MISMATCH_CANDIDATE | incomplete_surface | marker_jaccard_distance | 60 | 92 | 0.944 | 0.465 | 0.480 | -0.113 | 19/50 |
| hotpotqa | MISMATCH_CANDIDATE | long_trace_surface | length_absdiff | 126 | 60 | 0.808 | 0.356 | 0.452 | -0.068 | 4/50 |
| hotpotqa | MISMATCH_CANDIDATE | incomplete_surface | marker_jaccard_distance | 95 | 60 | 0.895 | 0.527 | 0.368 | 0.113 | 12/50 |
| gsm8k | MISMATCH_CANDIDATE | explicit_answer_marker | marker_jaccard_distance | 78 | 92 | 0.815 | 0.465 | 0.350 | -0.084 | 19/50 |
| gsm8k | MISMATCH_CANDIDATE | uncertainty_marker | marker_jaccard_distance | 35 | 92 | 0.790 | 0.465 | 0.325 | 0.062 | 19/50 |
| hotpotqa | MISMATCH_CANDIDATE | passage_or_evidence_marker | marker_jaccard_distance | 43 | 60 | 0.820 | 0.527 | 0.293 | -0.056 | 12/50 |
| gsm8k | MISMATCH_CANDIDATE | passage_or_evidence_marker | tfidf_full_distance | 2 | 92 | 0.871 | 0.605 | 0.266 | -0.016 | 26/50 |
| hotpotqa | MISMATCH_CANDIDATE | explicit_answer_marker | length_absdiff | 61 | 60 | 0.597 | 0.356 | 0.241 | -0.123 | 4/50 |
| gsm8k | MISMATCH_CANDIDATE | gold_in_prefix256 | tfidf_prefix128_distance | 5 | 92 | 0.676 | 0.437 | 0.238 | 0.025 | 13/50 |
| hotpotqa | MISMATCH_CANDIDATE | gold_in_prefix256 | tfidf_prefix128_distance | 16 | 60 | 0.682 | 0.477 | 0.205 | 0.093 | 10/50 |
| gsm8k | MISMATCH_CANDIDATE | long_trace_surface | length_absdiff | 90 | 92 | 0.804 | 0.608 | 0.195 | 0.032 | 27/50 |
| hotpotqa | MISMATCH_CANDIDATE | gold_in_prefix128 | length_absdiff | 12 | 60 | 0.548 | 0.356 | 0.192 | 0.094 | 4/50 |
| hotpotqa | MISMATCH_CANDIDATE | incomplete_surface | length_absdiff | 95 | 60 | 0.537 | 0.356 | 0.181 | 0.113 | 4/50 |
| hotpotqa | MISMATCH_CANDIDATE | gold_in_prefix256 | length_absdiff | 16 | 60 | 0.526 | 0.356 | 0.170 | 0.093 | 4/50 |
| hotpotqa | MISMATCH_CANDIDATE | explicit_answer_marker | marker_jaccard_distance | 61 | 60 | 0.694 | 0.527 | 0.167 | -0.123 | 12/50 |
| gsm8k | MISMATCH_CANDIDATE | explicit_answer_marker | tfidf_prefix128_distance | 78 | 92 | 0.600 | 0.437 | 0.162 | -0.084 | 13/50 |
| gsm8k | CAUTION | gold_in_suffix256 | tfidf_prefix128_distance | 65 | 92 | 0.573 | 0.437 | 0.135 | 0.130 | 13/50 |
| hotpotqa | CAUTION | gold_in_prefix128 | tfidf_prefix128_distance | 12 | 60 | 0.602 | 0.477 | 0.124 | 0.094 | 10/50 |
| gsm8k | CAUTION | gold_in_prefix256 | marker_jaccard_distance | 5 | 92 | 0.581 | 0.465 | 0.116 | 0.025 | 19/50 |
| hotpotqa | CAUTION | gold_in_suffix256 | tfidf_prefix256_distance | 94 | 60 | 0.533 | 0.464 | 0.068 | 0.284 | 8/50 |
| hotpotqa | CAUTION | gold_in_suffix256 | tfidf_prefix128_distance | 94 | 60 | 0.536 | 0.477 | 0.059 | 0.284 | 10/50 |
| hotpotqa | CAUTION | long_trace_surface | tfidf_prefix256_distance | 126 | 60 | 0.517 | 0.464 | 0.053 | -0.068 | 8/50 |
| hotpotqa | CAUTION | gold_in_suffix256 | tfidf_suffix256_distance | 94 | 60 | 0.703 | 0.654 | 0.049 | 0.284 | 19/50 |
| hotpotqa | CAUTION | long_trace_surface | tfidf_prefix128_distance | 126 | 60 | 0.514 | 0.477 | 0.037 | -0.068 | 10/50 |
| hotpotqa | CAUTION | gold_in_suffix256 | length_absdiff | 94 | 60 | 0.385 | 0.356 | 0.029 | 0.284 | 4/50 |
| hotpotqa | CAUTION | long_trace_surface | marker_jaccard_distance | 126 | 60 | 0.544 | 0.527 | 0.018 | -0.068 | 12/50 |
| hotpotqa | CAUTION | incomplete_surface | tfidf_prefix256_distance | 95 | 60 | 0.468 | 0.464 | 0.004 | 0.113 | 8/50 |
| hotpotqa | CAUTION | incomplete_surface | tfidf_prefix128_distance | 95 | 60 | 0.463 | 0.477 | -0.015 | 0.113 | 10/50 |
| hotpotqa | CAUTION | gold_in_suffix256 | marker_jaccard_distance | 94 | 60 | 0.492 | 0.527 | -0.034 | 0.284 | 12/50 |
| hotpotqa | CAUTION | long_trace_surface | tfidf_full_distance | 126 | 60 | 0.442 | 0.663 | -0.221 | -0.068 | 19/50 |
| hotpotqa | CAUTION | long_trace_surface | tfidf_suffix256_distance | 126 | 60 | 0.397 | 0.654 | -0.256 | -0.068 | 19/50 |
| gsm8k | ALIGNED_OR_LOW_GAP | uncertainty_marker | tfidf_prefix256_distance | 35 | 92 | 0.610 | 0.513 | 0.097 | 0.062 | 17/50 |
| gsm8k | ALIGNED_OR_LOW_GAP | long_trace_surface | tfidf_suffix256_distance | 90 | 92 | 0.627 | 0.532 | 0.095 | 0.032 | 25/50 |
| gsm8k | ALIGNED_OR_LOW_GAP | gold_in_full_trace | tfidf_prefix128_distance | 74 | 92 | 0.530 | 0.437 | 0.092 | 0.283 | 13/50 |
| gsm8k | ALIGNED_OR_LOW_GAP | gold_in_full_trace | marker_jaccard_distance | 74 | 92 | 0.553 | 0.465 | 0.088 | 0.283 | 19/50 |
| gsm8k | ALIGNED_OR_LOW_GAP | incomplete_surface | tfidf_prefix128_distance | 60 | 92 | 0.524 | 0.437 | 0.086 | -0.113 | 13/50 |
| hotpotqa | ALIGNED_OR_LOW_GAP | gold_in_prefix256 | tfidf_prefix256_distance | 16 | 60 | 0.549 | 0.464 | 0.085 | 0.093 | 8/50 |
| gsm8k | ALIGNED_OR_LOW_GAP | uncertainty_marker | length_absdiff | 35 | 92 | 0.691 | 0.608 | 0.082 | 0.062 | 27/50 |
| gsm8k | ALIGNED_OR_LOW_GAP | passage_or_evidence_marker | marker_jaccard_distance | 2 | 92 | 0.545 | 0.465 | 0.081 | -0.016 | 19/50 |
| hotpotqa | ALIGNED_OR_LOW_GAP | passage_or_evidence_marker | length_absdiff | 43 | 60 | 0.435 | 0.356 | 0.079 | -0.056 | 4/50 |

Next steps:
1. Inspect any `MISMATCH_CANDIDATE` rows for artifact mechanisms.
2. If the mechanism is plausible, freeze the proxy and score definition before adding new data.
3. Re-run with fresh labels or a held-out construct before promoting the row to the paper.
