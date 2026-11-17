# Innovation-2C0-B Historical CoMuT Dynamic Dependency Structure Test

Status: **DYNAMIC_DEPENDENCY_STRUCTURE_COMUT_NOT_SUPPORTED**.

This test applies the frozen Innovation-2C0-A dynamic dependency signature to all 13 Historical Official CoMuT cases. It uses each case's complete `train_no_anomaly.csv` for train-only scaling and structure learning and reads `test.csv` labels only for final evaluation. All signal channels form one local process group.

## Binding and frozen protocol

- Dataset root: `dataset/CoMuT_historical_v1`; all 13 train/test pairs exist.
- Median across-case Normal-train pairwise absolute Pearson: `0.997505`, confirming Historical rather than compatibility data.
- Delta, directed dependency formula, window `64`, and lags `{0,1,5,10,20,30}` exactly reuse 2C0-A.
- Train signature stride is 5; every legal Test endpoint is evaluated at stride 1.
- Nondegenerate features use Train `std > 1e-6` only. PCA uses chronological Train first 80%, with the smallest k reaching 95% explained variance. The final 20% supplies the normal median/MAD reference and q99 threshold.
- Primary labels are endpoint labels. Window-overlap labels are diagnostic only. No point adjustment or Test threshold search is used.

## Case results

| Case | Category | AUROC | AUPR | Recall | FPR | Anomaly/normal structure-error ratio |
|---|---|---:|---:|---:|---:|---:|
| 2-short / 2 different | Multiple-short | 0.697488 | 0.012762 | 0.000000 | 0.016080 | 1.362096 |
| 2-short / 2 same | Multiple-short | 0.501432 | 0.007356 | 0.000000 | 0.037961 | 1.010317 |
| 3-short / 3 different | Multiple-short | 0.681629 | 0.093504 | 0.333333 | 0.022788 | 1.573133 |
| 3-short / 3 same | Multiple-short | 0.870324 | 0.182723 | 0.666667 | 0.053342 | 3.080767 |
| large-on-2 | Large | 0.617004 | 0.020965 | 0.000000 | 0.041350 | 1.342714 |
| large-on-3 | Large | 0.543783 | 0.123192 | 0.250000 | 0.008414 | 0.841193 |
| large-on-4 | Large | 0.733257 | 0.599715 | 0.650000 | 0.029961 | 5.907775 |
| medium-on-2 | Medium | 0.517403 | 0.007598 | 0.000000 | 0.013230 | 1.137199 |
| medium-on-3 | Medium | 0.419939 | 0.007209 | 0.000000 | 0.014757 | 0.741255 |
| medium-on-4 | Medium | 0.848484 | 0.801615 | 0.800000 | 0.071240 | 3.755676 |
| short-on-2 | Short | 0.841073 | 0.021083 | 0.000000 | 0.000304 | 1.987675 |
| short-on-3 | Short | 0.732473 | 0.007145 | 0.000000 | 0.008007 | 1.410772 |
| short-on-4 | Short | 0.397813 | 0.003135 | 0.000000 | 0.019055 | 1.006817 |

Overall medians are endpoint AUROC `0.681629`, endpoint AUPR `0.020965`, q99 recall `0`, and q99 FPR `0.019055`.

## Category summary

| Category | Cases | Median AUROC | Median AUPR |
|---|---:|---:|---:|
| Short | 3 | 0.732473 | 0.007145 |
| Medium | 3 | 0.517403 | 0.007598 |
| Large | 3 | 0.617004 | 0.123192 |
| Multiple-short | 4 | 0.689559 | 0.053133 |

## Short alignment diagnostic

All three single-Short anomalies contain 40 anomalous points, shorter than the fixed 64-point window.

| Case | Endpoint AUROC/AUPR | Overlap AUROC/AUPR | Anomaly peak | Post-anomaly 64-tail peak |
|---|---:|---:|---:|---:|
| short-on-2 | 0.841073 / 0.021083 | 0.639466 / 0.017386 | 1.546948 | 0.642681 |
| short-on-3 | 0.732473 / 0.007145 | 0.585160 / 0.011246 | 0.748435 | 0.471697 |
| short-on-4 | 0.397813 / 0.003135 | 0.767079 / 0.598217 | 0.106969 | 3.127260 |

The overlap diagnostic does not consistently improve alignment: it decreases AUROC for short-on-2 and short-on-3 but exposes a strong post-anomaly tail for short-on-4. It remains diagnostic-only and does not replace endpoint evaluation.

## Existing Historical comparators

| Method | Median endpoint AUROC | Median endpoint AUPR |
|---|---:|---:|
| AE | 1.000000 | 1.000000 |
| RawCorr | 0.676580 | 0.012751 |
| V2 | 0.680844 | 0.012751 |
| PCC | 1.000000 | 1.000000 |
| Dynamic Dependency Structure | 0.681629 | 0.020965 |

The new structure score is close to RawCorr/V2 in median AUROC and slightly higher in median AUPR, but it does not stably detect relationship anomalies across channel counts and durations. No requirement was imposed to beat AE; the representation fails its own preregistered overall threshold.

## Gates

| Gate | Result |
|---|---|
| A: overall median AUROC >= 0.80 | FAIL |
| B: overall median AUPR >= 0.80 | FAIL |
| C: Short median AUROC >= 0.70 | PASS |
| D: Medium median AUROC >= 0.80 | FAIL |
| E: Large median AUROC >= 0.90 | FAIL |
| F: Multiple-short median AUROC >= 0.75 | FAIL |

Because overall median AUROC is below `0.70`, the mandatory status is `DYNAMIC_DEPENDENCY_STRUCTURE_COMUT_NOT_SUPPORTED`. No window, lag, PCA, feature, threshold, or alignment adjustment was made after observing this result.

All 15 integrity tests pass.

**HISTORICAL OFFICIAL COMUT ONLY; TRAIN-ONLY STRUCTURE MODEL; NO WINDOW SEARCH; NO LAG SEARCH; NO PCA SEARCH; NO TEST TUNING; NO PA; NO FUSION; NO COMMIT; NO PUSH.**
