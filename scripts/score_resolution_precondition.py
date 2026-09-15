#!/usr/bin/env python3
"""The score-resolution precondition, run on every contract.

Section 3 states the precondition as: no single score value may carry more than
half the items. It had been applied by hand to the 50-character GSM8K span only.
This script computes the modal share of the prefix score on all twenty
contracts with the same score construction the audit uses, and records which
rows the stated rule withdraws. Run from the repository root.
"""
import importlib.util, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("c", ROOT / "scripts" / "complement_span_proxy_control.py")
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
OUT = ROOT / "analysis_results" / "score_resolution_precondition.json"

cap = {}
_orig = c.make_row
def _capture(label, family, span, s, z, z_perp, z_comp, y, empty, seed):
    key = label
    s = np.asarray(s, float); z = np.asarray(z, int)
    vals, cnt = np.unique(np.round(s, 6), return_counts=True)
    k = int(cnt.argmax())
    cap[key] = dict(n=int(len(s)), distinct_values=int(len(vals)), modal_share=float(cnt[k] / len(s)),
                    modal_value=float(vals[k]),
                    proxy_positives_on_modal=float(np.mean(np.round(s[z == 1], 6) == vals[k])) if z.sum() else None)
    return _orig(label, family, span, s, z, z_perp, z_comp, y, empty, seed)
c.make_row = _capture

def main():
    c.refusal_rows(); c.correctness_rows()
    # referee check: the two numbers the paper already prints
    assert abs(cap["XSTest 450 (primary run)"]["modal_share"] - 0.436) < 2e-3
    assert abs(cap["GSM8K, 50-char span"]["modal_share"] - 0.634) < 2e-3
    for k, v in cap.items():
        v["withdrawn_by_rule"] = v["modal_share"] > 0.5
    withdrawn = [k for k, v in cap.items() if v["withdrawn_by_rule"]]
    print(f"{'contract':28} {'n':>5} {'distinct':>8} {'modal':>6} {'z+ on modal':>11}")
    for k, v in cap.items():
        print(f"{k:28} {v['n']:5d} {v['distinct_values']:8d} {v['modal_share']:6.3f} "
              f"{'' if v['proxy_positives_on_modal'] is None else round(v['proxy_positives_on_modal'], 2):>11}"
              f"{'  <-- withdrawn' if v['withdrawn_by_rule'] else ''}")
    print("withdrawn by the stated rule:", withdrawn)
    OUT.write_text(json.dumps({"rule": "modal share of the score, rounded to 6 decimals, exceeds 0.5",
                               "contracts": cap, "withdrawn": withdrawn}, indent=1))
    print("wrote", OUT)

if __name__ == "__main__":
    main()
