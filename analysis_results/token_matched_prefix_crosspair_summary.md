# Token-Matched Prefix Cross-Pair Summary

This report makes no model-generation or judge calls. All rows join cached XSTest 450 prompt ids under the 512-token setting and evaluate the same raw prefix-50 TF-IDF contract.

| Pair | Role | n | surf n+ | sem n+ | AUC surf | AUC sem | gap | kappa | top-10% sem |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Gemma | baseline token-matched 512 setting | 450 | 15 | 44 | 0.991 | 0.567 | 0.424 | -0.017 | 4/45 |
| Qwen/Llama | Qwen + non-Gemma cross-pair | 450 | 167 | 58 | 0.589 | 0.475 | 0.114 | 0.137 | 2/45 |
| Gemma/Llama | cross-pair scope check | 450 | 154 | 52 | 0.655 | 0.486 | 0.169 | 0.049 | 7/45 |
