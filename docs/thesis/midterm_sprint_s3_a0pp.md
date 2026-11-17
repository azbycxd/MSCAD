# Official SWaT 2015 A0++ static POT result

## Why S3

S2 showed that changing the reconstruction score damages ranking. S3 therefore keeps the original A0 raw endpoint 51-channel MAE unchanged and asks only whether a Normal-only static POT tail fit transfers better than empirical Validation q99.

## POT protocol

Only 98,937 Normal Validation endpoint scores are used. The initialization threshold is the frozen q95 (`u=0.10108516`); 4,947 strict exceedances are fitted with a GPD using `loc=0`, shape `0.36358702`, and scale `0.03101275`. With fixed alpha 0.01 and conditional quantile `0.80000606`, the static threshold is `0.16892305`. The observed Validation exceedance rate at this final threshold is `0.01207839`.

## A0 versus A0++

| Method | Threshold | Precision | Recall | F1 | FPR | AUPR | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 empirical q99 | 0.18396227 | 0.150140 | 0.917870 | 0.258067 | 0.718133 | 0.743353 | 0.834887 |
| A0++ static POT | 0.16892305 | 0.148442 | 0.920269 | 0.255647 | 0.729703 | 0.743353 | 0.834887 |

A0++ confusion matrix: TN=106814, FP=288358, FN=4355, TP=50266. Deltas versus A0 are threshold -0.0150392, Precision -0.0016984, Recall +0.0023983, F1 -0.0024201, and FPR +0.0115696.

## Ranking invariant check

`PASS`: AUPR and AUROC are exactly equal to the stored A0 values at reported precision, with differences below the frozen `1e-6` tolerance. The same S1 Test score, labels, and 449,793 endpoint timestamps were read directly; no score recomputation or Test-based parameter estimation occurred.

## Scientific conclusion

Status: `STATIC_POT_NOT_SUFFICIENT`.

Static POT gives a decision point very close to, and slightly worse than, empirical q99. The limitation is therefore not the empirical q99 estimator alone; it is more consistent with an overall Normal Validation-to-Test score distribution shift. This is `VALID_DIAGNOSTIC / PRELIMINARY`, not a final result. No alternative POT parameter or threshold was tried.
