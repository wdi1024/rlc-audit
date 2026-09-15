# RLC-Audit — code and analysis outputs

Audit code and analysis outputs for *Auditing Proxy–Construct Agreement Across Text Spans*
(Daein Weon, Dongho Kang).

A score is often validated by its agreement with a cheap proxy label. When the score and the proxy are
computed from the same span of text, that agreement can come from surface evidence the two share
rather than from the construct the proxy stands for. The audit declares the score, its span, the proxy
and *its* span, and the construct as a validation contract, then re-evaluates the contract's own proxy
rule strictly outside the scored span, and reads what survives against a null estimated on the
contract itself.

## What is here

| Path | Contents |
|---|---|
| `scripts/` | Every script named in the paper. Each reported number is produced by one of these. |
| `analysis_results/` | The JSON and log outputs those scripts write, including the per-contract gaps, intervals, permutation nulls, precondition outcomes and intervention results. |

The paper names the script for each result, so a number can be traced by searching `scripts/` for that
filename. The released JSON uses the earlier exit name `ALIGNED` for what the paper calls
`NO COUPLING FLAG`.

## What is not here

- **The paper itself.** It is on arXiv.
- **Raw per-example generations.** The audited runs are ~2 GB of stored model outputs and do not fit a
  git repository. They are what the off-span control needs, so if you want to re-run the audit end to
  end rather than re-derive the reported numbers, open an issue and we will point you at an archive.
- **Human annotation sheets** used by a different paper in the same line of work.

## Reproducing a number

1. Find the script the paper names for the result.
2. Run it against `analysis_results/` (most scripts read the stored outputs rather than regenerating
   them, so no API keys are needed for the re-derivation path).
3. Scripts that call a provider API to regenerate traces are the exception and are marked by their
   imports; those need your own credentials.

Runs were made at temperature 0 through provider APIs and are therefore deterministic only up to the
provider.

## Citing

Please cite the arXiv version. Under review at TMLR.
