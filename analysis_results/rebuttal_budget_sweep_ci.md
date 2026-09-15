# Budget sweep and bootstrap CIs (rebuttal prep)


## A. Primary prefix contract budget sweep (n=450, total sem dis=46)

| B | sem dis | both refuse | both comply | kw dis | sem precision | sem recall |
|---|---|---|---|---|---|---|
| 25 | 3 | 19 | 3 | 13 | 0.120 | 0.065 |
| 55 | 6 | 37 | 12 | 13 | 0.109 | 0.130 |
| 90 | 10 | 50 | 30 | 13 | 0.111 | 0.217 |
| 135 | 15 | 70 | 50 | 13 | 0.111 | 0.326 |
| 225 | 30 | 101 | 94 | 13 | 0.133 | 0.652 |

## B. Revision contracts budget sweep (n=548, total sem dis=79)

| Score | B | sem dis | both refuse | both comply | sem precision | sem recall |
|---|---|---|---|---|---|---|
| raw_prefix50_tfidf | 25 | 6 | 10 | 9 | 0.240 | 0.076 |
| raw_prefix50_tfidf | 55 | 11 | 17 | 27 | 0.200 | 0.139 |
| raw_prefix50_tfidf | 90 | 22 | 25 | 43 | 0.244 | 0.278 |
| raw_prefix50_tfidf | 135 | 30 | 46 | 59 | 0.222 | 0.380 |
| raw_prefix50_tfidf | 225 | 48 | 79 | 98 | 0.213 | 0.608 |
| final_tfidf | 25 | 11 | 12 | 2 | 0.440 | 0.139 |
| final_tfidf | 55 | 20 | 33 | 2 | 0.364 | 0.253 |
| final_tfidf | 90 | 28 | 60 | 2 | 0.311 | 0.354 |
| final_tfidf | 135 | 35 | 92 | 8 | 0.259 | 0.443 |
| final_tfidf | 225 | 52 | 154 | 19 | 0.231 | 0.658 |
| final_tfidf_marker_70_30 | 25 | 17 | 8 | 0 | 0.680 | 0.215 |
| final_tfidf_marker_70_30 | 55 | 28 | 25 | 2 | 0.509 | 0.354 |
| final_tfidf_marker_70_30 | 90 | 33 | 55 | 2 | 0.367 | 0.418 |
| final_tfidf_marker_70_30 | 135 | 41 | 86 | 8 | 0.304 | 0.519 |
| final_tfidf_marker_70_30 | 225 | 53 | 151 | 21 | 0.236 | 0.671 |

## C. Bootstrap 95% CIs

Primary prefix gap (existing, 10k boot): D=0.393 CI [0.324, 0.464], p_perm=0

| Revision score | AUC sem | 95% CI |
|---|---|---|
| raw_prefix50_tfidf | 0.622 | [0.556, 0.687] |
| final_tfidf | 0.675 | [0.605, 0.742] |
| final_tfidf_marker_70_30 | 0.715 | [0.642, 0.783] |

| Revision delta vs raw prefix | point | 95% CI | frac boot > 0 |
|---|---|---|---|
| final_tfidf | +0.053 | [-0.053, +0.155] | 0.843 |
| final_tfidf_marker_70_30 | +0.093 | [-0.011, +0.194] | 0.960 |

| Panel pair | gap (AUC kw - AUC sem) | 95% CI |
|---|---|---|
| Qwen/Gemma | -0.015 | [-0.092, +0.062] |
| Qwen/Mistral | +0.018 | [-0.060, +0.097] |
| Qwen/Phi | +0.128 | [+0.027, +0.232] |
| Qwen3-8B/Phi | +0.118 | [+0.029, +0.207] |
| Gemma/Mistral | -0.006 | [-0.068, +0.060] |
| Gemma/Qwen3-8B | +0.042 | [-0.030, +0.115] |
| Gemma/Phi | +0.015 | [-0.059, +0.086] |
| Mistral/Qwen3-8B | +0.068 | [+0.002, +0.136] |
| Mistral/Phi | +0.054 | [-0.024, +0.131] |
| Qwen/Qwen | -0.094 | [-0.189, +0.003] |
