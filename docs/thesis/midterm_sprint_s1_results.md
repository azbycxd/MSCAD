# Official SWaT 2015 preliminary results

## Protocol

Official Physical data, 51 variables, Normal v1 ordered 396000/99000 raw-row Train/Validation split, Train-only StandardScaler, independent 64-point stride-1 windows, two hard-boundary Test segments, seed 1, Validation q99 thresholds, and no point adjustment or Test-driven tuning.

## A0 Reconstruction

Status: `PRELIMINARY_OFFICIAL_SWAT2015`; this is a single-seed diagnostic result, not a final multi-seed benchmark.

| Score | F1 | Precision | Recall | AUPR | AUROC | FPR | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| Primary: endpoint MAE, Validation q99=0.183962 | 0.258067 | 0.150140 | 0.917870 | 0.743353 | 0.834887 | 0.718133 | 11,730 |
| Secondary: full-window MAE, Validation q99=0.138601 | 0.256255 | 0.148403 | 0.937808 | 0.737771 | 0.834542 | 0.743841 | 11,730 |

The checkpoint was epoch 13 with Validation SmoothL1 loss 0.00603307. Evaluation contains 449793 legal endpoints from the two frozen Test segments. Primary score medians were 0.0511484 (Train), 0.0598919 (Validation), and 0.467249 (Test). The high fixed-q99 Test FPR is retained as a scientific result; no score, threshold, window, split, model, or optimization setting was changed after seeing Test performance.

## V3-A Normal Collaboration Gate

| Eligible targets | Gain >=5% | Gain >=10% | Gain >=20% | Median gain | Max gain | Top-10 targets by gain |
|---:|---:|---:|---:|---:|---:|---|
| 37 | 6 | 4 | 3 | -0.182297 | 0.406098 | FIT601 0.406098; FIT301 0.315733; FIT201 0.272803; DPIT301 0.173826; AIT202 0.066193; PIT501 0.051354; FIT501 0.049480; FIT401 0.034916; LIT101 0.031251; LIT301 0.029852 |

Status: `PROMISING_INCREMENTAL_CROSS_CHANNEL_SIGNAL`. The gate used Normal Train/Validation only, lags `[1,5,10,20,30]`, and Ridge alpha 1.0. The canonical solve used float64 because an initial float32 solve emitted an ill-conditioned-matrix warning; this numerical correction changed no data, lag, alpha, target, or gate threshold, and the warning output remains preserved outside Git.

## Scientific conclusion

官方 SWaT 上已建立可信 Rec 初步基线；正常数据中至少 4 个目标存在超过 10% 的跨通道增量预测信息，达到预先冻结的继续门禁，V3 值得后续继续研究。该信号并非全局一致（eligible targets 的 median gain 为负），因此本结论只支持后续研究，不构成 V3 anomaly detector 或最终方法已经实现的证据。
