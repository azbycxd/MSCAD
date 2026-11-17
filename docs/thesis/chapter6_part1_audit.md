# Chapter 6 Part 1 audit

## 1. Scope and preconditions

Audit target: `docs/thesis/chapters/chapter6_experiments_part1.md` and `docs/thesis/chapter6_protocol_registry.csv`.

| Gate | Evidence | Result |
|---|---|---|
| Experimental phase closed | `experimental_phase_freeze.md` | PASS — `THESIS_EXPERIMENTAL_PHASE_CLOSED` |
| Chapter 4 correctness frozen | `chapter4_final_correctness_review.md` | PASS — `THESIS_CHAPTER4_CORRECTNESS_FROZEN` |
| Chapter 5 correctness frozen | `chapter5_final_correctness_review.md` | PASS — `THESIS_CHAPTER5_CORRECTNESS_FROZEN` |
| Method figures ready | `method_figure_audit.md` | PASS — `THESIS_METHOD_FIGURES_READY` |

Only Sections 6.1 and 6.2 were drafted. Sections 6.3–6.8 remain placeholders. No performance conclusion, ranking statement, new claim or method change was introduced.

## 2. Source trace

- Thesis scope and claim boundaries: `final_thesis_outline.md`, `final_claim_audit.md`, `final_experiment_findings.md`.
- Unified baseline split, score and q99 rules: `baseline_protocol.md`, `outputs/thesis_baselines/baseline_manifest.json`.
- SWaT source, labels and boundaries: `swat2015_source_audit.md`, `swat2015_protocol_adjudication.md`, `outputs/thesis_t25_swat2015_protocol/segment_manifest.json`, `protocol_decision.json`.
- HAI21 files, variables, process groups and events: `hai21_a0_dataset_attack_audit.md`, `outputs/thesis_hai21_a0_audit/audit_summary.json`.
- WADI variables, missingness, boundaries and Attack 13: `wadi_a0_dataset_attack_audit.md`, `wadi_a05_missingness_binding_freeze.md`, `outputs/thesis_wadi_a0_audit/dataset_manifest.json`, `outputs/thesis_wadi_a05_protocol/protocol_freeze.json`.
- Historical CoMuT provenance and scale: `comut_n1_historical_reproduction.md`, `outputs/thesis_comut_n1_historical/source_manifest.json` and dataset manifest.
- Controlled SWaT: `innovation2d1_swat_controlled_dual_view.md`, `outputs/thesis_innovation2d1_swat_controlled/manifest.json`, frozen donor manifest.
- Supplementary protocol layers: `evaluation_e2_r2_complete_literature_compatible.md`, E2 common-mask manifest, `evaluation_e3_2026_literature_comparison.md`, `stability_multiseed_results.md`.
- Registered experiment status: `outputs/thesis_registry/experiment_registry_frozen.csv`.
- Runtime and efficiency protocol: `environment_audit.md`, `outputs/thesis_baselines/efficiency_protocol.json`, `efficiency_e1_final_method.md`.

All dataset row counts, variable counts, boundary counts and event counts in the chapter are copied from these frozen records; none was recalculated.

## 3. Required self-audit

| Check | Result | Finding |
|---|---|---|
| A. WADI incorrectly promoted to Primary | PASS | WADI is consistently labeled `DOMAIN_SHIFT_STRESS_TEST` and supplementary. |
| B. Oracle described as deployable | PASS | F1-Oracle and PA-F1-Oracle are explicitly retrospective diagnostics using Test labels. |
| C. PA mixed into Strict | PASS | Primary Strict explicitly excludes PA; PA appears only in literature-compatible supplementary rows. |
| D. 2026 reported methods mixed with unified baselines | PASS | LSTM-AE/USAD/GDN are unified measured baselines; four 2026 methods are paper-reported supplementary comparisons only. |
| E. Historical Strict confused with common mask | PASS | Historical method-specific endpoints remain Primary; E2 common masks are supplementary and do not overwrite them. |
| F. SWaT 36 events confused with 35 point intervals | PASS | Chapter explicitly distinguishes 36 attack-list identities from 35 continuous point-label intervals. |
| G. HAI file boundary lost | PASS | Train split, window, lag, smoothing and PA all preserve official file boundaries. |
| H. Test tuning entered Primary | PASS | Primary permits Test only after model, score, calibration and q99 freeze; Test tuning/POT/SPOT/Test-best are excluded. |

## 4. Additional consistency checks

- The final baseline manifest confirms the common binary rule is strict `score > threshold`; no historical main-table exception was found.
- The unified event reporting rule is the frozen event hit fraction `>=0.10`. The earlier T2.5 SWaT adjudication's basic “at least one legal alarm” definition is provenance-level event semantics; final unified experiment manifests consistently use the later `0.10` reporting criterion. The chapter states the latter and does not alter point predictions.
- SWaT hard-gap, HAI file-boundary, WADI session/invalid-point and PA boundary rules are all preserved.
- HAI HIGH19 is identified as a frozen event grouping; DUAL is identified as deterministic in the three-seed protocol, with no artificial standard deviation.
- Efficiency text separates trainable parameters from non-trainable Tail calibration storage.
- The chapter contains exactly two tables and neither includes method performance results.
- `chapter6_protocol_registry.csv` contains protocol metadata only; it contains no performance value or ranking.

## 5. Negative-action assertions

NO TRAINING; NO INFERENCE; NO NEW EXPERIMENT; NO RESULT RECALCULATION; NO METHOD CHANGE; NO NEW CLAIM; NO NEW LITERATURE SEARCH; NO COMMIT; NO PUSH.

## 6. Final status

`THESIS_CHAPTER6_PART1_READY`

