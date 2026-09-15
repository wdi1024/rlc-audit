#!/usr/bin/env python3
"""Emit LaTeX tables for the tagged768 multi-pair final-span panel.

Reads results/disagree_routing/larger_pair_final_repair_summary.json and writes
analysis_results/panel_tables.tex with (1) a main-text table over pairs whose
final spans parse without raw-as-final fallback, and (2) a full appendix table
with parse coverage for all pairs including Llama.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/disagree_routing/larger_pair_final_repair_summary.json"
OUT = ROOT / "analysis_results/panel_tables.tex"

SUBSET = "all_final_nonempty"


def has_raw_as_final(pair: dict) -> bool:
    return any(
        "fallback_raw_as_final" in m.get("parse_status", {})
        for m in pair["model_summaries"].values()
    )


def subset(pair: dict) -> dict:
    return next(s for s in pair["subsets"] if s["subset"] == SUBSET)


def fmt(x, nd=3):
    return f"{x:.{nd}f}" if x is not None else "--"


def main() -> None:
    pairs = json.loads(SRC.read_text())["pairs"]
    clean = [p for p in pairs if not has_raw_as_final(p)]
    excluded = [p for p in pairs if has_raw_as_final(p)]

    main_rows = []
    for p in clean:
        s = subset(p)
        kw = s["final_full_vs_keyword"]
        sem = s["final_full_vs_semantic"]
        top = s["final_top10pct"]
        gap = kw["auc"] - sem["auc"]
        main_rows.append(
            f"{p['pair']} & {s['n']} & {sem['n_pos']} & {kw['n_pos']} & "
            f"{fmt(s['keyword_semantic_kappa'])} & {fmt(kw['auc'])} & {fmt(sem['auc'])} & "
            f"{gap:+.3f} & {top['semantic_disagreement_n']}/{top['budget']} \\\\"
        )

    app_rows = []
    for p in pairs:
        s = subset(p)
        kw = s["final_full_vs_keyword"]
        sem = s["final_full_vs_semantic"]
        top = s["final_top10pct"]
        gap = kw["auc"] - sem["auc"]
        cov = []
        for name, m in p["model_summaries"].items():
            ps = m["parse_status"]
            exact = ps.get("exact", 0)
            raf = ps.get("fallback_raw_as_final", 0)
            cov.append(f"{exact}/{m['n']}" + (f" (raw-as-final {raf})" if raf else ""))
        tag = "excluded" if has_raw_as_final(p) else "panel"
        app_rows.append(
            f"{p['pair']} & {'; '.join(cov)} & "
            f"{fmt(s['keyword_semantic_kappa'])} & {fmt(kw['auc'])} & {fmt(sem['auc'])} & "
            f"{gap:+.3f} & {top['semantic_disagreement_n']}/{top['budget']} & {tag} \\\\"
        )

    tex = []
    tex.append("% ---- main-text panel table (pairs without raw-as-final fallback) ----")
    tex.append(r"""\begin{table}[!htbp]
\centering
\small
\resizebox{\linewidth}{!}{%
\begin{tabular}{lrrrrrrrr}
\toprule
Pair & n & sem n+ & kw n+ & $\kappa$ & AUC kw & AUC sem & $\Delta_{\mathrm{AUC}}$ & sem@B \\
\midrule""")
    tex.extend(main_rows)
    tex.append(r"""\bottomrule
\end{tabular}%
}
\caption{PLACEHOLDER main panel caption.}
\label{tab:pair-panel}
\end{table}""")
    tex.append("")
    tex.append("% ---- appendix full panel table with parse coverage ----")
    tex.append(r"""\begin{table}[!htbp]
\centering
\scriptsize
\resizebox{\linewidth}{!}{%
\begin{tabular}{llrrrrrl}
\toprule
Pair & exact parse coverage & $\kappa$ & AUC kw & AUC sem & $\Delta_{\mathrm{AUC}}$ & sem@B & Row \\
\midrule""")
    tex.extend(app_rows)
    tex.append(r"""\bottomrule
\end{tabular}%
}
\caption{PLACEHOLDER appendix panel caption.}
\label{tab:pair-panel-full}
\end{table}""")

    OUT.write_text("\n".join(tex) + "\n")
    print(f"clean pairs: {len(clean)}, excluded (raw-as-final): {len(excluded)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
