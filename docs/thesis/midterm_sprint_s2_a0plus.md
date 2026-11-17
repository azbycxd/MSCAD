# Official SWaT 2015 A0+ preliminary scoring result

## Frozen protocol

A0+ reuses the exact S1 epoch-13 A0 checkpoint, 11,730-parameter architecture, Train-only StandardScaler, 64-point windows, and 449,793 legal Test endpoints. Per-channel median and `max(1.4826 * MAD, 1e-6)` are fitted only on Normal Train endpoint residuals. The primary score is the fixed mean of the largest 6 calibrated endpoint residuals; both score thresholds are fixed Normal Validation q99. There is no retraining, point adjustment, Test-driven tuning, threshold search, or k search.

## A0 versus A0+

| Score | Threshold | Precision | Recall | F1 | AUPR | AUROC | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 raw endpoint mean | 0.183962 | 0.150140 | 0.917870 | 0.258067 | 0.743353 | 0.834887 | 0.718133 |
| A0+ primary calibrated top-6 | 53.962557 | 0.359431 | 0.787865 | 0.493654 | 0.256131 | 0.768496 | 0.194078 |
| A0+ secondary calibrated mean-51 | 7.289773 | 0.227893 | 0.839769 | 0.358499 | 0.258308 | 0.774797 | 0.393259 |

Primary confusion matrix: TN=318478, FP=76694, FN=11587, TP=43034.

| A0+ primary minus A0 | Precision | Recall | F1 | AUPR | AUROC | FPR |
|---|---:|---:|---:|---:|---:|---:|
| Delta | +0.209291 | -0.130005 | +0.235586 | -0.487221 | -0.066390 | -0.524055 |

## Scientific conclusion

Status: `ROBUST_TOPK_SCORING_NOT_SUPPORTED`.

A0+ materially reduces fixed-q99 false positives and improves Precision/F1, so heterogeneous residual scale and 51-channel averaging do contribute to A0's operating-point weakness. However, both calibrated variants substantially degrade AUPR and also reduce AUROC. Therefore the robust top-k score is not supported as a strong reconstruction baseline candidate under the frozen S2 protocol. The result remains `VALID_DIAGNOSTIC / PRELIMINARY`; no alternative calibration, aggregation, k, epsilon, or threshold was tried.
