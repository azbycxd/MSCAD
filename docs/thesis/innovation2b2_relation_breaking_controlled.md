# Innovation-2B2 Input-space Relation-Breaking Controlled Validation

Status: **STAGE_JOINT_LAGGED_SCORE_NOT_SUPPORTED**.

This Normal-only stress test changes only the controlled donor construction from Innovation-2B. For every frozen evaluation endpoint it forms a chronological grid of at most 256 legal Normal donors separated by at least 600 seconds, applies the preregistered discrete-context guard where available, and selects the earliest donor attaining the maximum standardized cross-feature input distance. Ridge coefficients, predictions, residuals, anomaly scores, A0 scores, and labels are excluded from donor selection.

The six targets, evaluation endpoints, target value and history, other stages, fitted Ridge artifacts, feature order, lags `{1,5,10,20,30}`, `alpha=1.0`, calibration reference, q99 threshold, score formula, and frozen A0 checkpoint are unchanged.

## Results and original 2B comparison

| Target | Old shift ratio | New shift ratio | New feature distance | Zero shift | Old Collab AUROC | New Collab AUROC/AUPR | A0 AUROC/AUPR | Self AUROC | Recall | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AIT202 | 0.373491 | 0.625442 | 10.099728 | 0 | 0.528153 | 0.562010 / 0.570762 | 0.990630 / 0.974145 | 0.500000 | 0.0170 | 0.0100 |
| FIT201 | 4.630653 | 130.483613 | 8.844096 | 0 | 0.760974 | 0.996292 / 0.997484 | 0.990007 / 0.971184 | 0.500000 | 0.9950 | 0.0090 |
| DPIT301 | 0.225244 | 19.979927 | 26.003643 | 0 | 0.698020 | 0.980919 / 0.981836 | 0.999087 / 0.997081 | 0.500000 | 0.8230 | 0.0130 |
| FIT301 | 2.206788 | 136.537403 | 26.171099 | 0 | 0.711839 | 0.994473 / 0.993351 | 0.999107 / 0.996932 | 0.500000 | 0.9905 | 0.0115 |
| FIT401 | 0.073541 | 0.058071 | 3.637225 | 0 | 0.499256 | 0.498557 / 0.499446 | 0.843216 / 0.776992 | 0.500000 | 0.0055 | 0.0065 |
| FIT601 | 0.000000 | 4477.176120 | 19.451982 | 0 | 0.511751 | 0.999744 / 0.999750 | 0.997895 / 0.998323 | 0.500000 | 0.9995 | 0.0115 |

Corruption strength and collaboration sensitivity rise together strongly for FIT201, DPIT301, FIT301, and FIT601. AIT202 receives a much larger input-space displacement but remains below a unit median shift ratio and improves only slightly. FIT401's feature distance increases while its coefficient-weighted prediction shift remains weak and its collaboration ranking does not improve. Thus input distance creates effective model-space relation breaks for four targets, but not uniformly across all six frozen targets.

The same input-distance stress also makes A0 highly discriminative: its AUROC is above 0.99 for five targets except FIT401 (`0.843216`). Consequently, collaboration does not exceed A0 by 0.10 for any target, even though its own median AUROC is high. This is a controlled-stress-test comparison, not a real-attack result.

## FIT601 discrete context

| Source | Lag | State-change fraction |
|---|---:|---:|
| P602 | 1 | 0.9940 |
| P602 | 5 | 0.9980 |
| P602 | 10 | 0.9965 |
| P602 | 20 | 0.0325 |
| P602 | 30 | 0.0185 |

The new input-distance donor eliminates FIT601's former effective-corruption failure: median feature distance is `19.451982`, median shift ratio is `4477.176120`, and zero-shift fraction is zero.

## Preregistered gates

| Gate | Requirement | Result |
|---|---|---|
| V | At least 4/6 median shift ratios >= 1 | PASS: 4/6 |
| A | Median Collab AUROC >= 0.80 | PASS: 0.987696 |
| B | At least 4/6 Collab AUROC >= A0 AUROC + 0.10 | FAIL: 0/6 |
| C | Median corrupted recall >= 0.50 | PASS: 0.906750 |
| D | Median clean FPR <= 0.03 | PASS: 0.010750 |
| E | All Self AUROC in `[0.49,0.51]` | PASS: all 0.500000 |

Median Collab AUPR is `0.987593`. Gate V confirms that the benchmark now forms an effective model-space relation corruption for the required number of targets. Because Detection Gate B fails, the preregistered final status is `STAGE_JOINT_LAGGED_SCORE_NOT_SUPPORTED`. Under this protocol the current collaboration score is not established as the Innovation-2 final method, and this score route stops here.

All 15 integrity checks pass. Normal data, frozen A0 checkpoint, and A0 scaler hashes are identical before and after execution. No Attack data or labels were read.

**NORMAL-ONLY RELATION-BREAKING STRESS TEST; INPUT-DISTANCE DONOR ONLY; NO SCORE-DRIVEN DONOR; NO MODEL-DRIVEN DONOR; NO ATTACK DATA; NO FEATURE SEARCH; NO LAG SEARCH; NO THRESHOLD SEARCH; NO FUSION; NO COMMIT; NO PUSH.**
