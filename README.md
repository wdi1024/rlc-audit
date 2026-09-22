# RLC-Audit

Code and analysis outputs for **Auditing Proxy-Based Validation Across Text Spans**
(Daein Weon, Dong Ho Kang). Paper: arXiv (link to be added on posting).

A score is often validated by its agreement with a cheap proxy label. When the score and the proxy are
computed from the same span of text, that agreement can come from surface evidence the two share
rather than from the construct the proxy stands for. This repository contains the audit that tests
for it: declare the score and its span, the proxy and *its* span, and the construct; then re-evaluate
the contract's own proxy rule strictly outside the scored span and read what survives against a null
estimated on the contract itself.

## Layout

```
scripts/           184 analysis scripts; the 51 the paper cites by name are listed in RESULTS_INDEX.md
analysis_results/  181 JSON and log outputs those scripts write — the numbers in the paper come from here
RESULTS_INDEX.md   which script produces which part of the paper
requirements.txt   dependencies (the re-derivation path needs only the first block)
```

## Reproducing a number

1. Find the result in the paper. Each reported number names its script.
2. Look the script up in [`RESULTS_INDEX.md`](RESULTS_INDEX.md) and run it:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # first block is enough for re-derivation
python3 scripts/<name>.py
```

Most scripts read the stored outputs in `analysis_results/` and recompute the reported statistic, so
no API keys are needed for that path. Scripts that regenerate model traces call a provider API and
need your own credentials; they are recognisable by their `openai` / `anthropic` imports.

Runs were made at temperature 0 through provider APIs and are therefore deterministic only up to the
provider.

## What is not here

- **The paper.** It is on arXiv.
- **Raw per-example generations (~2 GB).** These are what the off-span control consumes; they do not
  fit a git repository. Open an issue if you want to re-run the audit end to end rather than
  re-derive the reported numbers, and we will point you at an archive.
- **Human annotation sheets** belonging to a different paper in the same line of work.

## Naming

The released JSON predates two renamings in the paper: `ALIGNED` is what the paper calls `NO FLAG`,
and `SCORE_FAILURE` is what it calls `NO DEMONSTRATED CONSTRUCT RANKING`. The latter means the score was not shown
to rank the declared construct *at the tested span*, not that the score is uninformative everywhere.
The other exit names (`CONTAINMENT`, `DIVERGENCE`, `CAUTION`, `UNDECIDABLE`) appear unchanged.

## Citation

```bibtex
@article{weon2026rlc,
  title  = {Auditing Proxy-Based Validation Across Text Spans},
  author = {Weon, Daein and Kang, Dong Ho},
  year   = {2026},
  note   = {Preprint}
}
```

## License

MIT (see [LICENSE](LICENSE)). The analysis outputs derive from public benchmarks and from model
responses obtained through provider APIs; those sources keep their own terms.
