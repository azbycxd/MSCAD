# Evaluation-E2-R1: raw-score persistence and common endpoint mask

## Status

`E2_RAW_SCORE_REPRODUCTION_REQUIRES_REVIEW`

E2-R1 froze both dataset-level common endpoint masks and audited all 12 dataset-method combinations. Three SWaT raw-score archives were persisted. Only the pre-existing SWaT Ours-REC archive reproduced its frozen Strict result within `1e-6`. Evaluation-only inference for SWaT LSTM-AE and GDN produced finite, correctly aligned scores, but their Strict metrics did not reproduce within tolerance. Those combinations were stopped without altering their scores. The remaining nine combinations lack a required frozen normalization or relation-reference artifact and were not inferred.

This status does not invalidate the historical aggregate results. It means the current repository does not contain enough frozen state to bind 11 newly persisted/reconstructed raw-score vectors to those historical Strict rows under the E2-R1 tolerance.

## E2 common endpoint policy

`E2_COMMON_MASK` is the exact intersection of legal endpoint identities within each dataset. It uses dataset identity plus segment/file identity, original local row, global row, and timestamp. No array was trimmed by length.

| Dataset | Candidate endpoints | Common endpoints | Exclusion rule |
|---|---:|---:|---|
| Official SWaT 2015 Physical | 449,793 | 449,791 | Drop local index 63 once in each of the two hard-gap-separated Test segments; common endpoints begin at local index 64. |
| HAI 21.03 | 401,690 | 401,685 | Drop local index 63 once in each of the five official Test files; common endpoints begin at local index 64. |

SWaT common segment counts are 298,773 and 151,018. HAI common file counts are 43,137, 118,737, 107,937, 39,537, and 92,337 for `test1.csv.gz` through `test5.csv.gz`. The SWaT mask SHA256 is `4c0d20a967c67caae58384b6b13b2d78312e2c351c6c48d13de722136bf9e0b1`; the HAI mask SHA256 is `5f066250841ec5e79d51d49b841cc45dfaf626a8b77720d408e365862563ddb7`.

The masks are stored under `outputs/thesis_evaluation/literature_compatible/common_masks/`. They preserve the SWaT hard session boundary and every HAI file boundary.

## Artifact audit

| Dataset | Method | Raw archive | Strict binding | Reason/status |
|---|---|---|---|---|
| SWaT | Ours-REC | Persisted by repackaging the unchanged S1 archive | PASS | Existing archive SHA256 was verified; no inference was run. |
| SWaT | LSTM-AE | Persisted | `RAW_SCORE_REPRODUCTION_MISMATCH` | Max Strict metric difference `0.0087962143`; stopped without adjustment. |
| SWaT | GDN | Persisted | `RAW_SCORE_REPRODUCTION_MISMATCH` | Max Strict metric difference `0.0180023848`; Validation median/IQR was deterministically reconstructed as permitted, then the combination was stopped. |
| SWaT | USAD | Not produced | `FROZEN_ARTIFACT_UNAVAILABLE` | Train-only MinMax parameters were not persisted. |
| SWaT | Ours-DUAL | Not produced | `FROZEN_ARTIFACT_UNAVAILABLE` | Ridge coefficients exist, but per-target disagreement median/MAD references were not persisted. |
| SWaT | Ours-TAIL | Not produced | `FROZEN_ARTIFACT_UNAVAILABLE` | ECDF arrays exist, but the frozen DUAL input branch is unavailable. |
| HAI21 | LSTM-AE | Not produced | `FROZEN_ARTIFACT_UNAVAILABLE` | Train-only StandardScaler parameters were not persisted. |
| HAI21 | USAD | Not produced | `FROZEN_ARTIFACT_UNAVAILABLE` | Train-only MinMax parameters were not persisted. |
| HAI21 | GDN | Not produced | `FROZEN_ARTIFACT_UNAVAILABLE` | Train-only StandardScaler parameters were not persisted, so Validation calibration cannot be reconstructed without forbidden refitting. |
| HAI21 | Ours-REC | Not produced | `FROZEN_ARTIFACT_UNAVAILABLE` | Train-only StandardScaler parameters were not persisted. |
| HAI21 | Ours-DUAL | Not produced | `FROZEN_ARTIFACT_UNAVAILABLE` | Coefficients exist, but the scaler and per-target disagreement median/MAD references were not persisted. |
| HAI21 | Ours-TAIL | Not produced | `FROZEN_ARTIFACT_UNAVAILABLE` | ECDF arrays exist, but neither frozen raw input branch can be generated from persisted state. |

The mismatched LSTM-AE and GDN archives remain as diagnostic evidence and are marked `available=false` in `score_availability.csv`. They must not be used by E2-R2 unless their provenance mismatch is resolved without tuning or refitting.

## Strict reproduction details

SWaT Ours-REC reproduced all frozen Strict metrics, with the largest absolute difference `1.11e-16`. Its common-mask evaluation is stored separately as `COMMON_MASK_REEVALUATION`; it does not overwrite the 449,793-point historical Strict row.

SWaT LSTM-AE produced AUROC `0.8185715152`, AUPR `0.7060044842`, F1 `0.2530784703`, and FPR `0.7354050156`, compared with frozen values `0.8182424435`, `0.7060434385`, `0.2544952156`, and `0.7266088013`.

SWaT GDN produced AUROC `0.8295957447`, AUPR `0.7303045593`, F1 `0.7132597341`, and FPR `0.0263582762`, compared with frozen values `0.8276650995`, `0.7281936622`, `0.7202160648`, and `0.0236581724`.

No attempt was made to select a different scaler, reinterpret a checkpoint, change score orientation, modify smoothing, or tune against Test labels.

## Persistence format

Each produced `test_raw_scores.npz` contains `score`, `label`, `segment_id`/`file_id`, `local_index`, `global_index`, `valid_endpoint`, `score_orientation`, and timestamp. Every method directory also contains `provenance.json`, `strict_reproduction.json`, and `common_mask_validation.json`. Unavailable directories contain the three JSON records and intentionally omit `test_raw_scores.npz`.

## Boundary and leakage checks

- SWaT windows and GDN four-point smoothing restart at the frozen 82-second hard session boundary.
- HAI common identities restart at each official Test file.
- Test labels were stored and evaluated only after deterministic scores were produced. They did not affect normalization, score transformation, model selection, threshold selection, smoothing, fusion, or calibration.
- No point adjustment, oracle threshold, POT, new fusion, or Test-driven parameter selection was run.

## Stop point

E2-R2 must not begin from the mismatched or unavailable combinations. The next admissible action is an artifact-provenance review: locate the exact scaler/reference artifacts used by the historical baseline and final-method runs, or formally accept that those Strict raw vectors are not reproducible from the current frozen repository.

**NO TRAINING. NO BACKWARD. NO OPTIMIZER. NO CHECKPOINT SELECTION. NO SCALER FIT. NO RIDGE FIT. NO TAIL RECALIBRATION. NO TEST TUNING. NO PA. NO ORACLE. NO STRICT RESULT OVERWRITE. NO COMMIT. NO PUSH.**
