# Innovation-2D0 Dual-View Predictive Disagreement on Historical CoMuT

Status: **DUAL_VIEW_RELATION_MECHANISM_SUPPORTED**.

This Normal-trained mechanism test compares two target predictions: a self view from the target's own fixed lags `{1,5,10,20,30}`, and a cross view from every other channel at fixed lags `{0,1,5,10,20,30}`. Both are Ridge models with `alpha=1.0`. The anomaly evidence is exclusively `abs(p_self-p_cross)`; neither observed-target reconstruction residual enters the score.

Each Historical Official CoMuT case uses a train-only StandardScaler. Legal Normal Train points are split chronologically 80/20 for model fitting and disagreement calibration. Per-target scores use calibration median/MAD, and the case score is their maximum. Test labels are read only after scores exist. The q99 threshold comes only from Train calibration.

## Overall and category results

Overall median AUROC is `0.995989`, median AUPR is `0.686927`, median q99 recall is `0.937500`, and median FPR is `0.010313`. All `13/13` cases have AUROC at least `0.80`.

| Category | Cases | Median AUROC | Median AUPR | Median recall | Median FPR |
|---|---:|---:|---:|---:|---:|
| Short | 3 | 0.995730 | 0.403800 | 0.9250 | 0.009668 |
| Medium | 3 | 0.995043 | 0.506890 | 0.9000 | 0.010111 |
| Large | 3 | 0.996165 | 0.751877 | 0.9625 | 0.011213 |
| Multiple-short | 4 | 0.997180 | 0.756346 | 0.9375 | 0.010537 |

## Case results and relation response

| Case | AUROC | AUPR | Anomaly/normal disagreement ratio | Most frequent anomaly argmax target |
|---|---:|---:|---:|---|
| 2-short / 2 different | 0.998487 | 0.814484 | 69.7411 | value-0 |
| 2-short / 2 same | 0.998371 | 0.813979 | 71.7012 | value-0 |
| 3-short / 3 different | 0.992256 | 0.476940 | 51.1409 | value-0 |
| 3-short / 3 same | 0.995989 | 0.698713 | 52.3826 | value-0 |
| large-on-2 | 0.996430 | 0.922275 | 63.9352 | value-1 |
| large-on-3 | 0.996165 | 0.751877 | 53.8898 | value-0 |
| large-on-4 | 0.994500 | 0.631728 | 45.4173 | value-0 |
| medium-on-2 | 0.998736 | 0.829612 | 69.4397 | value-0 |
| medium-on-3 | 0.995043 | 0.506890 | 51.5907 | value-0 |
| medium-on-4 | 0.989398 | 0.253839 | 42.3971 | value-0 |
| short-on-2 | 0.998139 | 0.686927 | 69.1001 | value-0 |
| short-on-3 | 0.995730 | 0.403800 | 52.9352 | value-0 |
| short-on-4 | 0.994358 | 0.289238 | 45.1346 | value-0 |

The anomaly median disagreement is 42–72 times the Test-normal median in every case. This response is pointwise at lag 30 and does not use a 64-point window, overlap labels, or point adjustment.

## Existing comparator summary

| Method | Median AUROC | Median AUPR |
|---|---:|---:|
| AE | 1.000000 | 1.000000 |
| RawCorr | 0.676580 | 0.012751 |
| V2 | 0.680844 | 0.012751 |
| PCC | 1.000000 | 1.000000 |
| Dynamic Dependency Structure | 0.681629 | 0.020965 |
| Dual-View Disagreement | 0.995989 | 0.686927 |

The dual-view score nearly matches PCC in ranking while remaining explicitly independent of the observed target value in its anomaly evidence. Normal self/cross fit and calibration MAEs plus calibration disagreement quantiles for every target are retained in `target_normal_sanity.csv`; they had no target-selection effect.

## Preregistered gates

| Gate | Requirement | Result |
|---|---|---|
| A | Median AUROC >= 0.90 | PASS |
| B | Median AUPR >= 0.50 | PASS |
| C | At least 10/13 AUROC >= 0.80 | PASS: 13/13 |
| D | Short median AUROC >= 0.80 | PASS |
| E | Medium median AUROC >= 0.80 | PASS |
| F | Large median AUROC >= 0.90 | PASS |
| G | Multiple-short median AUROC >= 0.80 | PASS |

All gates pass, so `DUAL_VIEW_RELATION_MECHANISM_SUPPORTED` is the mandatory status. This authorizes a separately preregistered SWaT controlled validation; none was run here.

All 15 integrity tests pass. No source, lag, alpha, feature, window, threshold, or target was selected using Test results.

**HISTORICAL OFFICIAL COMUT; SELF VIEW vs CROSS VIEW ONLY; NO OBSERVED-TARGET RESIDUAL SCORE; NO TEST TUNING; NO FEATURE SEARCH; NO LAG SEARCH; NO FUSION; NO COMMIT; NO PUSH.**
