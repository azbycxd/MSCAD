# V3-I1 Innovation-level Dynamic Collaboration Discovery

## Scope and preregistration

V3-I1 is a Normal-only mechanism diagnostic. It uses only Official SWaT 2015 `SWaT_Dataset_Normal_v1.xlsx`, with the frozen ordered split of the first 396,000 rows as Train and the final 99,000 rows as Validation. No Attack data or labels, synthetic anomaly, reconstruction retraining, fusion, threshold, or test-driven selection is involved.

All choices were fixed before observing the result: Ridge `alpha=1.0`, lags `[1, 5, 10, 20, 30]`, Train fit stride 4, and Top5 predictor channels. A stable target must have overall relative innovation gain at least 0.10, gain in both chronological Validation halves at least 0.05, and overall innovation R² at least 0.10.

## Audited channel scope

Classification is read directly from the T2 Normal v1 `variable_profiles` and is not outcome-driven:

- continuous targets: 25
- non-constant discrete transition predictors: 15
- constant channels excluded: 11

The Normal v1 audit differs from an Attack-file-based count because several actuators are constant throughout Normal v1. For this Train/Validation discovery they are therefore excluded exactly as the audited Normal schema specifies.

## Method

For every continuous channel, a Train-only self Ridge predicts its current standardized value from only its own values at the five frozen lags. The innovation is the signed residual from that frozen self model. Validation innovations use the unchanged Train-fitted self models.

Each cross model predicts a target innovation from lagged innovations of other continuous channels and lagged transitions of non-constant discrete channels. The target's own innovation history is excluded. Persistent actuator raw states and continuous raw levels never enter the cross model. An initial all-candidate Train Ridge ranks channels by the sum of absolute lag coefficients; the fixed Top5 channels are then refit using Train only.

The baseline is the Train median target innovation. Relative gain and innovation R² are evaluated on Validation, including its chronological first and second halves.

## Result

Stable target count: **0**.

Status: **INNOVATION_COLLABORATION_NOT_SUPPORTED**.

Consequently, median stable gain and median stable innovation R² are undefined. The highest overall relative gain was only 0.013950 (DPIT301), below the preregistered 0.10 threshold; its innovation R² was 0.099015, also below 0.10. No target satisfied the two chronological-half gain criteria together with the overall criteria.

The raw-level/innovation contrast is especially clear for targets with very high V3-R R1 R²:

| Target | Raw R1 R² | Innovation gain | Innovation R² | Gain first | Gain second | Top5 predictors |
|---|---:|---:|---:|---:|---:|---|
| FIT201 | 0.999217 | 0.001029 | 0.280694 | 0.017524 | -0.016934 | AIT201; P101; LIT401; LIT301; P203 |
| DPIT301 | 0.995878 | 0.013950 | 0.099015 | 0.014267 | 0.013633 | FIT301; P302; LIT101; FIT601; LIT301 |
| FIT301 | 0.992669 | -0.120590 | 0.407127 | -0.111450 | -0.129872 | DPIT301; P302; FIT101; LIT301; FIT601 |
| FIT601 | 0.982928 | -0.803199 | 0.355264 | -0.760516 | -0.849642 | DPIT301; FIT301; LIT401; LIT301; P602 |
| FIT101 | 0.976695 | -0.037168 | 0.029767 | -0.039231 | -0.035152 | LIT101; MV101; LIT301; AIT502; AIT201 |
| FIT501 | 0.960253 | -0.012571 | 0.016060 | -0.013384 | -0.011772 | PIT501; UV401; PIT503; FIT401; AIT504 |
| PIT503 | 0.955361 | -0.007975 | 0.040024 | -0.008483 | -0.007457 | PIT501; UV401; FIT501; FIT401; FIT503 |
| PIT501 | 0.953543 | -0.009287 | 0.033495 | -0.008568 | -0.010032 | PIT503; UV401; FIT501; FIT401; AIT504 |

High raw-level predictability therefore does not translate into preregistered, chronologically stable MAE gain after removing each target's self-predictable component. Some innovation R² values remain positive while MAE gain is negative; this does not satisfy the registered collaboration definition and must not be promoted by changing the metric after the result.

## Artifacts and decision

Ignored diagnostic artifacts are under `outputs/thesis_v3i_innovation/`:

- `v3i1_target_metrics.csv`
- `v3i1_predictor_manifest.json`
- `v3i1_summary.json`

The negative gate result stops this line at V3-I1. V3-I2 is not authorized by the preregistered gate. Existing S4, V3-B, and V3-R files and results were retained without overwrite.
