# Evaluation E2-R1D — Deterministic Artifact Recovery

## Outcome

Status: `E2_RAW_SCORES_COMPLETED`.

All 12 frozen dataset × method combinations reproduced their historical Strict metrics with a maximum absolute metric difference no greater than `1e-6`. The frozen common masks remain unchanged:

- SWaT: 449,791 endpoints.
- HAI 21.03: 401,685 endpoints.

This phase performed inference and deterministic reconstruction of permitted preprocessing/calibration artifacts only. It did not train a model, run backward propagation, construct an optimizer, refit Ridge models, select targets/checkpoints, tune against Test, apply point adjustment, or overwrite historical Strict result files.

## Recovery boundary

Every reconstructed artifact is labelled `DETERMINISTIC_RECONSTRUCTION`. The reconstruction was allowed only after the historical formal runner established one unique combination of dataset version, ordered Train/Validation split, feature names/order, missing-value handling, dtype, and algorithm.

The following were reconstructed:

- Train-only StandardScaler parameters for LSTM-AE, GDN, and the final REC/DUAL pipeline.
- Train-only `MinMaxScaler([0,1], clip=False)` parameters for USAD.
- Validation-only per-sensor median/IQR for GDN.
- Validation-only per-target median/MAD disagreement references for DUAL, using already persisted Ridge coefficients and frozen targets/lags.

The following were not reconstructed or changed:

- model weights or checkpoint identity;
- Ridge coefficients;
- target selection, lag definitions, architecture, or hyperparameters;
- tail-fusion definition or persisted Calibration-A ECDF arrays;
- any historical Strict metric artifact.

## Strict reproduction results

| Dataset | Method | Strict endpoints | Maximum absolute metric difference | Result |
|---|---:|---:|---:|---|
| SWaT | LSTM-AE | 449,791 | 0 | `RAW_SCORE_REPRODUCED` |
| SWaT | USAD | 449,791 | 0 | `RAW_SCORE_REPRODUCED` |
| SWaT | GDN | 449,791 | 0 | `RAW_SCORE_REPRODUCED` |
| SWaT | Ours-REC | 449,793 | 1.11e-16 | `RAW_SCORE_REPRODUCED` |
| SWaT | Ours-DUAL | 449,793 | 1.01e-8 | `RAW_SCORE_REPRODUCED` |
| SWaT | Ours-TAIL | 449,793 | 0 | `RAW_SCORE_REPRODUCED` |
| HAI21 | LSTM-AE | 401,685 | 0 | `RAW_SCORE_REPRODUCED` |
| HAI21 | USAD | 401,685 | 0 | `RAW_SCORE_REPRODUCED` |
| HAI21 | GDN | 401,685 | 0 | `RAW_SCORE_REPRODUCED` |
| HAI21 | Ours-REC | 401,690 | 6.94e-18 | `RAW_SCORE_REPRODUCED` |
| HAI21 | Ours-DUAL | 401,690 | 7.98e-17 | `RAW_SCORE_REPRODUCED` |
| HAI21 | Ours-TAIL | 401,690 | 2.39e-7 | `RAW_SCORE_REPRODUCED` |

Strict reproduction was always evaluated on each method's historical endpoint set. Mapping to the already frozen E2 common mask was performed only after that method passed Strict reproduction. Ours uses endpoint 63 historically, so the first endpoint of each SWaT segment or HAI Test file is excluded only in the later common-mask view; baseline methods already use endpoint 64.

## SWaT mismatch diagnosis

Both E2-R1 mismatches have the same earliest inconsistent pipeline stage: `B_STANDARD_SCALER_STATISTICS`.

E2-R1 reused the persisted S1 A0 scaler, whose statistics were calculated in the float32 path. The formal LSTM-AE/GDN runner instead explicitly casts the frozen SWaT Normal Train split to float64 before fitting `data.transforms.StandardScaler`, then casts transformed arrays to float32. The two scalers differ by up to 0.3729675970 in channel means and 0.2029349537 in effective scales.

Following the single source-defined baseline path restores both methods exactly:

- LSTM-AE: prior maximum metric difference 0.0087962143; recovered difference 0.
- GDN: prior maximum metric difference 0.0180023848; recovered difference 0.

Checkpoint hashes, feature order, local endpoint 64, two hard-gap segments, evaluation mode, score aggregation, `score > threshold`, q99 threshold, and historical endpoint mask all matched. For GDN, source inspection additionally fixed `+1e-2` IQR stabilization, a segment-local four-point trailing mean, exactly zero scores for the first three forecast endpoints of each segment, and top-1/max-sensor aggregation. No alternative variant was tried or selected by metric proximity.

## Provenance and outputs

- `recovery/original_pipeline_trace.csv` maps every combination to its formal training/evaluation code, checkpoint or frozen coefficient artifact, calibration logic, and historical result.
- `recovery/artifact_recovery_manifest.csv` records reconstructed artifacts, raw archives, Strict status, and no-training/no-refit assertions.
- `score_availability.csv` now marks all 12 raw-score archives available as `RAW_SCORE_REPRODUCED`.
- `recovery/reconstructed_artifacts/` contains scaler, GDN Validation median/IQR, and DUAL Validation median/MAD artifacts with source split/hash, feature-order hash, dtype, formula, code location, creation time, and artifact hash.
- Each `raw_scores/<dataset>/<method>/` directory contains the raw Test archive, provenance, Strict reproduction report, and post-PASS common-mask validation.

The raw score artifacts are now eligible for the future E2-R2 comparison stage. This document does not perform or report literature-compatible re-ranking, PA, oracle thresholding, or any new scientific conclusion.

