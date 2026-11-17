# Innovation-2B Diagnostic: Effective Corruption Strength Audit

Status: **BLOCKED_MISSING_FITTED_MODEL_ARTIFACT**.

The requested audit cannot be completed without violating its explicit no-retraining constraint. Innovation-2B fitted each target's joint Ridge model in process, used it to produce aggregate metrics, then exited without persisting the fitted estimator, its complete coefficient vector, or per-sample clean/corrupted joint predictions.

## Evidence

The frozen Innovation-2B output directory contains only:

- `controlled_metrics.csv`
- `corruption_manifest.json`
- `innovation2b_summary.json`
- `score_semantics.csv`
- `target_summary.csv`

None contains the complete `w_cross` coefficients or sample-level predictions required to calculate

`abs(w_cross @ x_cross_corrupt - w_cross @ x_cross_clean)`.

The execution script creates `joint_model` by calling `fit_ridge(...)`, uses the estimator in memory, deletes the design matrices, and does not serialize the model. Innovation-2A2 likewise stores only aggregate source importance and the single largest coefficient per source, not all source-by-lag coefficients. Those summaries are mathematically insufficient to reconstruct `w_cross`.

Consequently, the following requested quantities remain **NOT COMPUTED**:

- per-target effective prediction-shift distributions;
- prediction-shift ratios relative to calibration joint residuals;
- prediction-shift versus collaboration-AUROC Spearman correlation;
- the preregistered effective-corruption diagnosis.

Feature-distance and discrete/continuous source-change summaries alone would not resolve the missing coefficient-weighted diagnostic and are therefore not presented as a completed audit.

## Required unlock

Completion requires explicit authorization for one exact deterministic refit of the already frozen six Ridge models, with no protocol, data, corruption, score, feature, lag, alpha, sample, or threshold change, followed by persistence of the fitted coefficients or required sample-level prediction shifts. No such refit was performed in this task.

**NO RETRAINING; NO NEW CORRUPTION; NO SCORE CHANGE; NO ATTACK DATA; NO PARAMETER SEARCH; NO COMMIT; NO PUSH.**
