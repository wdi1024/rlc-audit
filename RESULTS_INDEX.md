# Where each result comes from

The paper names a script for every reported number. This index lists those scripts by the part of
the paper they appear in, so a result can be traced without reading the LaTeX source.

### A documentation-consistency check on the two declared branches.
- `scripts/inter_auditor_reliability.py`

### A second coded question: can the control be run at all?
- `scripts/prevalence_survey_*.py`
- `scripts/prevalence_survey_artifacts.py`

### An estimate from the flagged subset.
- `scripts/mathshepherd_arithmetic_lower_bound.py`
- `scripts/mathshepherd_arithmetic_lower_bound_v2.py`

### Anchoring the Constants on Real Generations
- `scripts/semisynthetic_calibration.py`

### Auditing a Published Recipe (FrugalGPT-style Correctness Routing)
- `scripts/audit_frugalgpt_recipe.py`

### Budget Sensitivity and Statistical Uncertainty
- `scripts/multiplicity_correction.py`

### Calibration Controls
- `scripts/analyze_model_pair_boundary.py`
- `scripts/analyze_token_matched_prefix_crosspairs.py`
- `scripts/encoder_contracts_full_rule.py`

### Comparison with a cheaper baseline.
- `scripts/control_agreement_spearman.py`

### Composition of the surviving certificate.
- `scripts/opening_strip_span_independent.py`

### Correcting the judge's over-call.
- `scripts/judge_overcall_downsample.py`

### Error Rates on Structured Cases, and a Same-Precondition Baseline
- `scripts/diagnostic_error_rates.py`

### Hybrid LLM and MixInstruct
- `scripts/audit_hybridllm_mixinstruct.py`
- `scripts/audit_hybridllm_surface_axes.py`
- `scripts/hybridllm_cluster_bootstrap.py`
- `scripts/hybridllm_routed_composition.py`

### Keyword versus span in the repair.
- `scripts/external_refusal_cue.py`
- `scripts/judge_marker_stripped.py`
- `scripts/marker_strip_symmetry.py`
- `scripts/nli_cue_repair.py`

### Making the Orientation Check a Test
- `scripts/orientation_significance.py`

### Multi-Pair Final-Span Panel
- `scripts/analyze_larger_pair_final_repair.py`
- `scripts/build_panel_tables.py`
- `scripts/cross_judge_panel_full.py`

### Normalising the selection-harm rates.
- `scripts/audit_prm800k_scored.py`
- `scripts/audit_searchandlearn_beam.py`

### Open ends of the null.
- `scripts/disjoint_null_distribution.py`

### Sensitivity to the two thresholds.
- `scripts/band_sweep_disjoint_nokappa.py`
- `scripts/verdict_cutoff_sensitivity.py`

### The Containment Controls, Contract by Contract
- `scripts/complement_span_proxy_control.py`

### The Keyword Proxy Is Not a Straw Man
- `scripts/validated_proxy_audit.py`

### The construct mismatch, measured.
- `scripts/mathshepherd_estimand_vs_construct.py`

### The contract.
- `scripts/audit_routellm_deployed.py`

### The ordering claim across a grid, not a line.
- `scripts/synthetic_grid.py`

### The same baseline on the twenty contracts.
- `scripts/baseline_same_preconditions.py`

### The two rows our preconditions withdrew.
- `scripts/audit_score_resolution.py`
- `scripts/prefix_span_correctness_probe.py`
- `scripts/span_closure_fine_sweep.py`

### Two Further Controls
- `scripts/opening_strip_hotpot.py`
- `scripts/question_paraphrase_intervention.py`
- `scripts/verdict_rule_and_ablation.py`

### UltraFeedback's Selection Target
- `scripts/audit_ultrafeedback.py`

### What Was Fixed Before the Results Were Seen
- `scripts/recount_without_preconditions.py`

### What the design can detect.
- `scripts/power_and_calibration.py`

### What the grade-school-math labels describe.
- `scripts/routerbench_gsm8k_label_provenance.py`

### What the repair is and is not.
- `scripts/marker_tie_sensitivity.py`

### When a length asymmetry reaches the queue.
- `scripts/hybridllm_penetration_sweep.py`

### Which null.
- `scripts/significance_bar_comparison.py`

## Not indexed here

`scripts/` holds 182 files; the 51 above are the ones the paper cites by name. The rest are helpers
those scripts import, one-off regeneration jobs, and earlier analyses kept so that the chronology in
the appendix on post-hoc rules can be checked against real files.
