# A0 Oracle-Test-F1 diagnostic — NOT A FORMAL RESULT

> 该实验使用 Test labels 搜索最优阈值，仅用于分析 A0 score 的理论决策上限，不得作为正式 benchmark 结果。

Scientific status: `ORACLE_DIAGNOSTIC_ONLY / INVALID_FOR_FORMAL_BENCHMARK`.

The diagnostic reads the exact S1 A0 Test endpoint score, label, and timestamp arrays. It contains 449,793 endpoints from the same two independently windowed Test segments and does not cross the 82-second gap. No inference, retraining, score modification, model selection, or formal threshold update was performed.

| Result | Threshold | Precision | Recall | F1 | FPR | AUPR | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Formal A0 Validation q99 | 0.183962 | 0.150140 | 0.917870 | 0.258067 | 0.718133 | 0.743353 | 0.834887 |
| Test-label oracle diagnostic | 0.808602 | 0.975515 | 0.634591 | 0.768959 | 0.002202 | 0.743353 | 0.834887 |

Oracle confusion matrix: TN=394302, FP=870, FN=19959, TP=34662. Relative to formal q99, F1 changes by +0.510892, the threshold changes by +0.624640, and the threshold ratio is 4.395480.

Ranking invariant: `PASS`. AUPR and AUROC are identical to A0 within the required `1e-6` tolerance.

Diagnosis: `THRESHOLD_TRANSFER_DOMINATES_F1_FAILURE`. This descriptive oracle result shows that the unchanged A0 score has substantial point-wise separability and that Validation-to-Test threshold transfer dominates its formal fixed-threshold F1 failure. Because Test labels selected the threshold, the oracle F1 is invalid for formal comparison and must not be reused by any later detector, V3 component, fusion, or selection step.
