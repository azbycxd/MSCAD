# V3 Repair — Selective cross-only conditional collaboration

> `NORMAL-ONLY MECHANISM VALIDATION`. This is not SWaT real-Attack performance; no Attack cache or labels were read.

## Frozen design

All models use Official SWaT 2015 Normal v1, the A0 Train-only scaler, float64, lags `[1,5,10,20,30]`, and Ridge alpha 1.0. R1 fits every target on Train endpoints at stride 4 while lag values still come from the original 1 Hz sequence. Each initial 250-feature model excludes all five target-history inputs; predictor importance freezes five other channels, followed by a 25-feature Ridge refit. Validation remains stride 1. Stability requires overall R² >=.30, first-half R² >=.20, second-half R² >=.20, and nonzero Train target variance.

## R1 cross-only discovery

Status: `CROSS_ONLY_DEPENDENCY_SUPPORTED`. Stable target count is 23 and median eligible overall R² is 0.955361.

| Target | R² overall | R² first | R² second | Frozen Top5 predictors |
|---|---:|---:|---:|---|
| FIT101 | .976695 | .975589 | .977818 | LIT101, MV101, AIT203, LIT301, LIT401 |
| LIT101 | .595775 | .555167 | .634893 | AIT203, PIT503, PIT501, FIT201, AIT504 |
| MV101 | .943358 | .940219 | .946567 | FIT101, PIT501, PIT503, AIT203, LIT301 |
| P101 | .992862 | .992472 | .993259 | FIT201, MV201, P203, LIT401, AIT201 |
| FIT201 | .999217 | .999198 | .999237 | P101, AIT201, AIT203, PIT501, PIT503 |
| MV201 | .986250 | .985605 | .986908 | P101, P203, FIT201, AIT201, AIT203 |
| P203 | .997270 | .997077 | .997466 | FIT201, AIT201, P101, P205, MV201 |
| P205 | .977190 | .976638 | .977751 | AIT203, PIT501, PIT503, AIT202, P203 |
| DPIT301 | .995878 | .995831 | .995924 | FIT301, PIT503, PIT501, LIT401, MV302 |
| FIT301 | .992669 | .992121 | .993223 | DPIT301, P302, LIT301, LIT401, MV302 |
| MV301 | .729755 | .732964 | .726243 | FIT301, DPIT301, P602, LIT401, LIT101 |
| MV302 | .946543 | .945780 | .947316 | DPIT301, FIT301, P302, LIT401, FIT601 |
| MV303 | .300222 | .302361 | .297872 | P302, DPIT301, FIT301, LIT301, LIT401 |
| MV304 | .573674 | .570495 | .577006 | DPIT301, FIT301, LIT301, MV302, P302 |
| P302 | .965708 | .964909 | .966515 | DPIT301, FIT301, LIT401, MV302, LIT301 |
| FIT401 | .946421 | .964911 | .858195 | FIT501, PIT501, PIT503, UV401, P402 |
| FIT501 | .960253 | .975994 | .882955 | FIT401, PIT503, PIT501, UV401, P402 |
| FIT503 | .825424 | .885433 | .597659 | PIT501, PIT503, P501, UV401, FIT504 |
| FIT504 | .475188 | .556632 | .298928 | PIT503, PIT501, FIT503, P501, AIT504 |
| PIT501 | .953543 | .984586 | .785036 | PIT503, FIT501, FIT401, P402, UV401 |
| PIT503 | .955361 | .985443 | .784123 | PIT501, FIT501, FIT401, AIT203, P402 |
| FIT601 | .982928 | .983365 | .982451 | DPIT301, P602, FIT301, P302, LIT401 |
| P602 | .778848 | .781739 | .775692 | DPIT301, FIT301, FIT601, LIT401, LIT301 |

For every listed predictor, lag 1 has the largest absolute coefficient within its five frozen lag coefficients. This fact is reported, not used to change the lag set.

## R2 controlled validation

R2 uses exactly the 23 R1-frozen targets. Validation valid times are split chronologically into disjoint calibration/evaluation halves. Each of 46 cells has 2,000 clean and 2,000 corrupted samples. Shift is fixed at 60 seconds. Context Swap uses the preregistered sorted-time cyclic half-offset rule with >=300-second separation. Corruption changes only each target's frozen Top5 predictor histories; the target trajectory and all other channels remain unchanged. A0 is the unchanged epoch-13 checkpoint.

| Injection | Median V3 AUROC | Median V3 AUPR | Median A0 AUROC |
|---|---:|---:|---:|
| Temporal Shift | .512473 | .531878 | .510246 |
| Deterministic Context Swap | .677006 | .750209 | .743877 |
| All 46 cells | .523134 | .557664 | — |

V3 exceeds A0 AUROC by more than .05 in 8/46 cells. Median corruption recall is .0785 and median clean-Normal FPR is .0090.

| Gate | Requirement | Result | Status |
|---|---|---:|---|
| A | Median V3 AUROC >= .80 | .523134 | FAIL |
| B | V3 > A0+.05 in >=75% cells | 8/46 | FAIL |
| C | Median corruption recall >= .50 | .0785 | FAIL |
| D | Median clean-Normal FPR <= .03 | .0090 | PASS |

## Scientific conclusion

Final status: `SELECTIVE_CROSS_ONLY_NOT_SUPPORTED`.

Cross-only Normal prediction is stable for many targets, but that predictive fit does not become a reliable anomaly mechanism under either frozen relation-only corruption. The selective cross-only repair therefore fails its last preregistered controlled gate. No target, predictor count, lag, alpha, shift, donor, threshold, or sample was changed in response to the result. This finding does not authorize V3 real-Attack evaluation, fusion, or a new V4 direction.
