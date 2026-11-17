# Innovation-2B2 Diagnostic: A0 Reconstruction Score Decomposition

Status: **A0_TARGET_RECONSTRUCTION_CAPTURES_CONTEXT_VIOLATION**.

This diagnostic reuses the exact six Innovation-2B2 targets, 2,000 clean and corrupted windows, saved donor endpoints, frozen A0 checkpoint, and frozen A0 scaler. It decomposes the existing 51-channel endpoint absolute reconstruction-error vector without retraining, changing donors, tuning scores, or reading Attack data.

## Reproduction

The recomputed A0 GLOBAL AUROC matches the stored Innovation-2B2 value for every target. Maximum absolute difference: `0.0`, satisfying the required `1e-8` tolerance.

## Component ranking metrics

Each entry is `AUROC / AUPR`.

| Target | Collab AUROC | GLOBAL | TARGET ONLY | CORRUPTED SOURCES | UNAFFECTED | STAGE ALL |
|---|---:|---:|---:|---:|---:|---:|
| AIT202 | 0.562010 | 0.990630 / 0.974145 | 0.500967 / 0.569524 | 0.998942 / 0.999314 | 0.889622 / 0.810621 | 0.999156 / 0.999397 |
| FIT201 | 0.996292 | 0.990007 / 0.971184 | 0.994389 / 0.980062 | 0.999060 / 0.999281 | 0.894521 / 0.817976 | 0.998970 / 0.998829 |
| DPIT301 | 0.980919 | 0.999087 / 0.997081 | 0.983558 / 0.989745 | 0.998948 / 0.995429 | 0.982120 / 0.976752 | 0.999036 / 0.996930 |
| FIT301 | 0.994473 | 0.999107 / 0.996932 | 0.991177 / 0.993288 | 0.999040 / 0.997167 | 0.981463 / 0.975495 | 0.999046 / 0.997034 |
| FIT401 | 0.498557 | 0.843216 / 0.776992 | 0.587638 / 0.594549 | 0.970058 / 0.972269 | 0.720108 / 0.669912 | 0.970505 / 0.972703 |
| FIT601 | 0.999744 | 0.997895 / 0.998323 | 0.993389 / 0.989394 | 0.998346 / 0.998919 | 0.997857 / 0.998348 | 0.997593 / 0.997231 |

## Median error magnitudes

Each entry is `clean → corrupt (corrupt/clean ratio)`.

| Target | GLOBAL | TARGET ONLY | CORRUPTED SOURCES | UNAFFECTED | STAGE ALL |
|---|---|---|---|---|---|
| AIT202 | 0.069779 → 0.193580 (2.77×) | 0.083695 → 0.081626 (0.98×) | 0.113610 → 0.946650 (8.33×) | 0.063206 → 0.089668 (1.42×) | 0.113869 → 0.823195 (7.23×) |
| FIT201 | 0.069779 → 0.186043 (2.67×) | 0.047984 → 0.908134 (18.93×) | 0.125194 → 0.781229 (6.24×) | 0.063206 → 0.090999 (1.44×) | 0.113869 → 0.803772 (7.06×) |
| DPIT301 | 0.069779 → 0.304785 (4.37×) | 0.098327 → 2.891307 (29.41×) | 0.060320 → 0.744399 (12.34×) | 0.070404 → 0.154508 (2.19×) | 0.064979 → 0.984647 (15.15×) |
| FIT301 | 0.069779 → 0.303205 (4.35×) | 0.065545 → 2.343751 (35.76×) | 0.065359 → 0.783632 (11.99×) | 0.070404 → 0.155646 (2.21×) | 0.064979 → 0.957787 (14.74×) |
| FIT401 | 0.069779 → 0.095017 (1.36×) | 0.028090 → 0.038212 (1.36×) | 0.113331 → 0.259214 (2.29×) | 0.065864 → 0.079110 (1.20×) | 0.099497 → 0.223221 (2.24×) |
| FIT601 | 0.069779 → 0.469188 (6.72×) | 0.013495 → 2.480678 (183.82×) | 0.046308 → 3.019418 (65.20×) | 0.071306 → 0.374216 (5.25×) | 0.030658 → 2.778142 (90.62×) |

No clean component median was near zero, so no ratio was suppressed.

## Interpretation

Corrupted-source AUROC is high for all six targets, confirming that direct errors on replaced source channels contribute strongly to A0 GLOBAL. However, this does not fully explain the four high-collaboration cases:

- FIT201 TARGET ONLY: `0.994389` versus GLOBAL `0.990007`.
- DPIT301 TARGET ONLY: `0.983558` versus GLOBAL `0.999087`.
- FIT301 TARGET ONLY: `0.991177` versus GLOBAL `0.999107`.
- FIT601 TARGET ONLY: `0.993389` versus GLOBAL `0.997895`.

Thus TARGET ONLY is itself close to GLOBAL for four of six targets, including every specifically requested high-Collab target. The prescribed descriptive category is `A0_TARGET_RECONSTRUCTION_CAPTURES_CONTEXT_VIOLATION`. AIT202 and FIT401 remain exceptions: their target-only ranking is weak and their GLOBAL score is primarily explained by the corrupted-source/stage components.

The high UNAFFECTED scores for P3 and P6 also show that the reconstruction network's response propagates beyond the directly replaced endpoint channels. This is descriptive model behavior, not a new detection score or success gate.

**NO RETRAINING; NO NEW CORRUPTION; NO DONOR CHANGE; NO FEATURE SEARCH; NO SCORE TUNING; NO THRESHOLD; NO ATTACK DATA; NO FUSION; NO COMMIT; NO PUSH.**
