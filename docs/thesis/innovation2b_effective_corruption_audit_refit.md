# Innovation-2B Diagnostic Unlock: Exact Refit and Effective Corruption Audit

Refit checkpoint: **DETERMINISTIC_REFIT_REPRODUCED**. Final diagnosis: **CORRUPTION_EFFECTIVELY_WEAK**.

This audit used the explicitly authorized one-time exact deterministic refit of the six frozen Innovation-2B Ridge models. It retained the original Normal-only data, train-only scaler, targets, stage sources, feature ordering, lags `{1,5,10,20,30}`, `alpha=1.0`, float64 computation, splits, 2,000 evaluation samples, deterministic donors, corruption, and score formula.

## Reproduction gate

For every target, collaboration AUROC, collaboration AUPR, corrupted recall, and clean FPR reproduce the original Innovation-2B values. The maximum absolute difference over all 24 comparisons is `5.551115123125783e-17`, below the required `1e-8` tolerance.

## Effective corruption strength

| Target | Stage | 2A2 gain | Median cross-feature distance | Median prediction shift | Median shift ratio | P90 shift ratio | Zero-shift fraction | Collab AUROC | A0 AUROC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIT202 | P2 | 0.072088 | 1.710785 | 0.000416605 | 0.373491 | 1.163118 | 0.0000 | 0.528153 | 0.741890 |
| FIT201 | P2 | 0.053537 | 1.855815 | 0.002921624 | 4.630653 | 130.572330 | 0.0000 | 0.760974 | 0.744461 |
| DPIT301 | P3 | 0.190883 | 2.365099 | 0.000372627 | 0.225244 | 24.366812 | 0.0000 | 0.698020 | 0.706980 |
| FIT301 | P3 | 0.330211 | 2.365555 | 0.001757251 | 2.206788 | 159.647049 | 0.0000 | 0.711839 | 0.700617 |
| FIT401 | P4 | 0.095624 | 1.939271 | 0.000152294 | 0.073541 | 0.188416 | 0.0000 | 0.499256 | 0.648423 |
| FIT601 | P6 | 0.613691 | 0.000000 | 0.000000000 | 0.000000 | 0.000000 | 0.9705 | 0.511751 | 0.504404 |

The ratio denominator is each target's calibration-normal median joint residual. None of the six denominators required the `1e-6` protection.

Three targets meet the preregistered weak-corruption condition: DPIT301 (`0.225244`), FIT401 (`0.073541`), and FIT601 (`0`). Only FIT201 and FIT301 have median prediction shifts larger than their normal median joint residual, so the alternative four-target score-insensitivity condition is not met.

## Continuous-source changes

The target-level median / p90 absolute standardized continuous-feature changes are:

| Target | Median | P90 |
|---|---:|---:|
| AIT202 | 0.212752 | 2.287387 |
| FIT201 | 0.218924 | 0.682226 |
| DPIT301 | 0.680976 | 2.688266 |
| FIT301 | 0.735690 | 2.623760 |
| FIT401 | 0.102852 | 1.041695 |
| FIT601 | N/A | N/A |

FIT601 has no continuous cross source; its sole source is the discrete actuator P602.

## FIT601 discrete-source audit

| Source | Lag | State-change fraction |
|---|---:|---:|
| P602 | 1 | 0.0145 |
| P602 | 5 | 0.0150 |
| P602 | 10 | 0.0155 |
| P602 | 20 | 0.0145 |
| P602 | 30 | 0.0150 |

P602@lag5 changes state in only `1.50%` of corrupted samples. Across the complete five-lag cross-feature vector, `97.05%` of FIT601 samples are exactly unchanged. Its median cross-feature distance and median prediction shift are therefore both exactly zero. The equality of the original clean and corrupted median joint errors (`0.000227156` each) is directly consistent with the input data: a deterministic donor window usually carries the same P602 state at all five evaluated lags.

## Descriptive associations

- Spearman A, 2A2 validation gain versus median prediction shift: `-0.600000`.
- Spearman B, median prediction shift versus collaboration AUROC: `0.885714`.
- Spearman C, median shift ratio versus collaboration AUROC: `0.885714`.

These correlations are descriptive only and had no selection effect.

## Diagnosis

Exactly `3/6` targets have a median prediction shift below one quarter of their calibration-normal median joint residual. The preregistered diagnosis is therefore `CORRUPTION_EFFECTIVELY_WEAK`. This identifies a limitation of the frozen donor corruption's effective model-space perturbation, especially for FIT601; it does not alter the original Innovation-2B score or its reported result.

The refit artifacts, complete coefficients and masks, sample indices and predictions, discrete state-change table, and continuous feature-change table are stored under `outputs/thesis_innovation2b_diagnostic_refit/`. The original 2A2 and 2B files were not modified.

**EXACT DETERMINISTIC REFIT ONLY; NO MODEL SELECTION; NO FEATURE SEARCH; NO LAG SEARCH; NO ALPHA SEARCH; NO NEW CORRUPTION; NO SCORE CHANGE; NO ATTACK DATA; NO TEST TUNING; NO COMMIT; NO PUSH.**
