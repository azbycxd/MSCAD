# Innovation-2D1 SWaT Controlled Dual-View Collaboration Validation

Status: **DUAL_VIEW_SWAT_CONTROLLED_SUPPORTED**.

This Normal-only test applies the frozen Innovation-2D0 dual-view disagreement score to the exact six targets, 2,000 clean endpoints, donor endpoints, and source replacements from Innovation-2B2. Donors and corruption are read from the frozen 2B2 manifest and are not regenerated.

Each self Ridge uses target lags `{1,5,10,20,30}`. Each cross Ridge excludes the target and uses all other nonconstant same-stage sources at `{0,1,5,10,20,30}`. Both use `alpha=1.0`, float64, the Train-only scaler, and only Normal Train `[0,396000)` for fitting. Validation first half supplies the normal median/MAD reference and q99 threshold; Validation second-half frozen endpoints supply clean and corrupted evaluation.

The score is exclusively the calibrated `abs(pred_self-pred_cross)`. Observed-target residuals do not enter the score.

## Target results

| Target | Stage | Self cal MAE | Cross cal MAE | Clean median d | Corrupt median d | Ratio | Dual AUROC/AUPR | Old joint AUROC | A0 AUROC | Recall | FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AIT202 | P2 | 0.003930 | 0.164943 | 0.250924 | 0.370209 | 1.475 | 0.710757 / 0.709433 | 0.562010 | 0.990630 | 0.2140 | 0.0105 |
| FIT201 | P2 | 0.001525 | 0.016840 | 0.008694 | 2.283535 | 262.655 | 0.999887 / 0.999891 | 0.996292 | 0.990007 | 0.9975 | 0.0110 |
| DPIT301 | P3 | 0.004734 | 0.027514 | 0.009502 | 2.874631 | 302.525 | 0.998018 / 0.997909 | 0.980919 | 0.999087 | 0.8875 | 0.0105 |
| FIT301 | P3 | 0.003187 | 0.018535 | 0.006087 | 2.595814 | 426.451 | 0.995807 / 0.995952 | 0.994473 | 0.999107 | 0.8580 | 0.0085 |
| FIT401 | P4 | 0.005982 | 0.116027 | 0.078473 | 0.073649 | 0.939 | 0.512903 / 0.519511 | 0.498557 | 0.843216 | 0.0000 | 0.0000 |
| FIT601 | P6 | 0.005284 | 0.010862 | 0.000457 | 11.328278 | 24784.768 | 0.999999 / 0.999999 | 0.999744 | 0.997895 | 1.0000 | 0.0070 |

Overall median Dual AUROC is `0.996913`, median AUPR is `0.996931`, median corrupted recall is `0.872750`, and median clean FPR is `0.009500`. Four of six targets have AUROC at least `0.80`.

## Old joint-residual comparison

| Target | Old AUROC | Dual AUROC | Delta |
|---|---:|---:|---:|
| AIT202 | 0.562010 | 0.710757 | +0.148747 |
| FIT201 | 0.996292 | 0.999887 | +0.003595 |
| DPIT301 | 0.980919 | 0.998018 | +0.017099 |
| FIT301 | 0.994473 | 0.995807 | +0.001334 |
| FIT401 | 0.498557 | 0.512903 | +0.014346 |
| FIT601 | 0.999744 | 0.999999 | +0.000255 |

Dual-view disagreement substantially improves AIT202 and retains near-ceiling performance for FIT201, both P3 targets, and FIT601. FIT401 remains unsupported: its corrupted median disagreement is slightly below its clean median. No target was removed and no protocol was altered in response.

## Gates

| Gate | Requirement | Result |
|---|---|---|
| A | Median Dual AUROC >= 0.85 | PASS: 0.996913 |
| B | At least 4/6 targets AUROC >= 0.80 | PASS: 4/6 |
| C | Median corrupted recall >= 0.50 | PASS: 0.872750 |
| D | Median clean FPR <= 0.03 | PASS: 0.009500 |
| E | Every cross model excludes its target | PASS: 6/6 |

All preregistered gates pass, so the mandatory status is `DUAL_VIEW_SWAT_CONTROLLED_SUPPORTED`. This authorizes a separately requested Innovation-2D2 Official SWaT real-attack complementarity test; it was not run here.

All 15 integrity tests pass. No Attack workbook, new corruption, donor change, target/source selection, lag/alpha search, threshold search, point adjustment, or fusion was used.

**FROZEN SWAT RELATION-BREAKING BENCHMARK; SELF-VIEW VS CROSS-VIEW ONLY; NO OBSERVED-TARGET RESIDUAL SCORE; NO ATTACK DATA; NO NEW CORRUPTION; NO FEATURE SEARCH; NO LAG SEARCH; NO THRESHOLD SEARCH; NO FUSION; NO COMMIT; NO PUSH.**
