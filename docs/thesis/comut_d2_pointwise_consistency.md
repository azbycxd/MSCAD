# CoMuT-D2 Pointwise Cross-Channel Consistency Diagnostic

Status: **POINTWISE_CROSS_CHANNEL_CONSISTENCY_PARTIAL**.

This diagnostic used only the 13 Historical Official CoMuT cases in `dataset/CoMuT_historical_v1`, generated with GutenTAG 1.0.0 from the frozen configuration SHA-256 `4D90D3871BFBB6C088F0AE4A3EAFE279D365FDBE4CACB4F85397908826570717`. Every scaler and Ridge model was fitted exclusively on that case's `train_no_anomaly.csv`; `test.csv` was transformed only and its `is_anomaly` column was read solely for final AUROC/AUPR evaluation.

## Frozen PCC diagnostic

For each target channel, Ridge regression with `alpha=1.0` predicts the standardized target from every other channel at the same timestamp. The target itself, historical lags, future values, and test data are excluded from its inputs. Per-target absolute residuals are calibrated by the training median and `max(1.4826 * MAD, 1e-6)`. The pointwise PCC score is the maximum non-negative calibrated residual over all targets. No target or channel pair is selected.

## Aggregate results

| Category | Cases | Median PCC AUROC | Median PCC AUPR |
|---|---:|---:|---:|
| Short | 3 | 1.000000 | 1.000000 |
| Medium | 3 | 1.000000 | 1.000000 |
| Large | 3 | 1.000000 | 1.000000 |
| Multiple-short | 4 | 1.000000 | 1.000000 |
| All | 13 | 1.000000 | 1.000000 |

The comparator values below are read directly from the existing Historical D1R result and were not recomputed by retraining AE:

| Score | Median AUROC |
|---|---:|
| AE endpoint | 1.000000 |
| RawCorr window | 0.676580 |
| V2 window | 0.680844 |
| PCC pointwise | 1.000000 |

PCC exceeds V2 on 13/13 cases. It exceeds AE by at least 0.10 on 0/13 cases, because the historical AE comparator is already essentially perfect. Therefore the preregistered `SUPPORTED` gate is not satisfied even though PCC itself separates all 13 cases perfectly; the mandatory status is `POINTWISE_CROSS_CHANNEL_CONSISTENCY_PARTIAL`.

## Representative score semantics

| Case | Normal median | Normal q99 | Anomaly median | Orientation |
|---|---:|---:|---:|---|
| short-on-2 | 0.079468 | 3.324238 | 46.274632 | PASS |
| medium-on-2 | 0.074341 | 3.338723 | 47.242907 | PASS |
| large-on-2 | 0.106341 | 3.287516 | 46.638167 | PASS |

The anomaly median is more than an order of magnitude above the normal q99 in every representative case, with the registered positive-score orientation unchanged. Short cases reach AUROC 1.0 without a rolling window, providing strong mechanism evidence that Historical CoMuT correlation anomalies are detectable by normal-only instantaneous cross-channel consistency and that the weak short-case RawCorr/V2 values arise from their rolling local-Pearson representation. This diagnostic does not establish superiority over reconstruction because the existing AE comparator is equally strong on this synthetic benchmark.

## Integrity checks

All 12 preregistered checks passed: 13 historical cases, train-only scaler, train-only Ridge, target exclusion, no lag, fixed `alpha=1.0`, no target selection, no test tuning, fixed score orientation, finite scores, anomaly label equal to 1, and no point adjustment. SHA-256 checks before and after the run confirm that the Historical CoMuT CSV files were unchanged.

**NO TEST TUNING; NO WINDOW SEARCH; NO LAG SEARCH; NO POINT ADJUSTMENT; NO FUSION; NO AE RETRAINING; NO COMMIT; NO PUSH.**
