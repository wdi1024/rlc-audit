#!/usr/bin/env python3
"""Run packaged-data reproductions without the original results/ layout.

Most archival scripts in this bundle expect files under
`results/disagree_routing/`. The public artifact stores inputs in `data/` and
outputs in `analysis_results/`. This wrapper creates a compatibility layout via
symlinks when possible, then runs selected reproduction scripts.

Examples:
    python scripts/run_packaged_analysis.py --setup-only
    python scripts/run_packaged_analysis.py --scripts core
    python scripts/run_packaged_analysis.py --scripts phase3,multibench
    python scripts/run_packaged_analysis.py --scripts core --include-encoders
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_DIR = ROOT / "results" / "disagree_routing"

SCRIPT_GROUPS = {
    "phase3": ["scripts/analyze_judge_robustness_phase3.py"],
    "scale": ["scripts/analyze_length_controlled_scale.py"],
    "mechanism": [
        "scripts/inspect_2b_disagree_cases.py",
        "scripts/intervention_strip_openings.py",
        "scripts/analyze_representation_level_evidence.py",
        "scripts/intervention_model_agnostic_normalizers.py",
        "scripts/intervention_strip_generalize.py",
        "scripts/analyze_same_family_control.py",
    ],
    "multibench": [
        "scripts/analyze_paradox_multibench.py",
        "scripts/test_gap_significance.py",
    ],
    "figures": ["scripts/make_paper_figures.py"],
    "submission": [
        "scripts/analyze_submission_robustness.py",
        "scripts/analyze_visible_response_sanity.py",
    ],
    "iclr": [
        "scripts/analyze_metric_family_coupling.py",
        "scripts/analyze_rlc_audit.py",
        "scripts/analyze_model_pair_boundary.py",
        "scripts/analyze_token_matched_prefix_crosspairs.py",
        "scripts/analyze_larger_pair_final_repair.py",
        "scripts/analyze_rlc_composite_router.py",
        "scripts/analyze_synthetic_prefix_injection.py",
        "scripts/analyze_debiased_router.py",
        "scripts/analyze_label_noise_robustness.py",
    ],
    "api_post": [
        "scripts/analyze_api_judge_extension.py",
        "scripts/analyze_visible_response_sanity.py",
    ],
}

ENCODER_SCRIPTS = [
    "scripts/analyze_sentence_embedding_paradox.py",
    "scripts/analyze_encoder_sweep.py",
    "scripts/analyze_encoder_orbench.py",
]

CORE_GROUPS = ["phase3", "scale", "mechanism", "multibench", "figures", "submission", "iclr"]


def link_or_copy(src: Path, dst: Path) -> None:
    """Create dst pointing at src, preserving existing compatible paths."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        rel_src = os.path.relpath(src, dst.parent)
        dst.symlink_to(rel_src, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def setup_legacy_layout() -> None:
    LEGACY_DIR.mkdir(parents=True, exist_ok=True)

    for src_dir_name in ["data", "analysis_results"]:
        src_dir = ROOT / src_dir_name
        if not src_dir.exists():
            continue
        for src in src_dir.iterdir():
            if src.is_file():
                link_or_copy(src, LEGACY_DIR / src.name)

    for dirname in ["figures", "human_annotation", "logs"]:
        src = ROOT / dirname
        if src.exists():
            link_or_copy(src, LEGACY_DIR / dirname)


def expand_scripts(spec: str, include_encoders: bool) -> list[str]:
    if spec == "core":
        scripts = [s for group in CORE_GROUPS for s in SCRIPT_GROUPS[group]]
    else:
        scripts = []
        for item in [x.strip() for x in spec.split(",") if x.strip()]:
            if item in SCRIPT_GROUPS:
                scripts.extend(SCRIPT_GROUPS[item])
            elif item.endswith(".py"):
                scripts.append(item)
            else:
                valid = ", ".join(sorted(SCRIPT_GROUPS))
                raise SystemExit(f"Unknown script/group '{item}'. Valid groups: core, {valid}")
    if include_encoders:
        scripts.extend(ENCODER_SCRIPTS)
    return scripts


def run_scripts(scripts: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    (ROOT / ".mplconfig").mkdir(exist_ok=True)
    for rel_script in scripts:
        script = ROOT / rel_script
        print(f"\n[run] {rel_script}", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=str(ROOT), env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scripts",
        default="core",
        help="Script group(s) or .py path(s). Use core, phase3, scale, mechanism, multibench, figures, submission, iclr, api_post.",
    )
    parser.add_argument(
        "--include-encoders",
        action="store_true",
        help="Also run sentence-transformer analyses. These may download encoder weights.",
    )
    parser.add_argument("--setup-only", action="store_true")
    args = parser.parse_args()

    setup_legacy_layout()
    print(f"[setup] compatibility layout ready at {LEGACY_DIR.relative_to(ROOT)}")

    if args.setup_only:
        return

    run_scripts(expand_scripts(args.scripts, args.include_encoders))


if __name__ == "__main__":
    main()
