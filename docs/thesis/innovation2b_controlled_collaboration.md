# Innovation-2B SWaT Stage-local Joint Lagged Collaboration Controlled Validation

Status: **STAGE_JOINT_LAGGED_CONTROLLED_NOT_SUPPORTED**.

This Normal-only controlled diagnostic tests whether the six stable dependencies frozen by Innovation-2A2 can identify a deterministic mismatch between a target's unchanged trajectory and the lagged context of the other nonconstant variables in the same audited physical stage. It is not an Attack-data benchmark and is not evidence that the score detects real SWaT attacks.

## Frozen protocol

- Data: Official SWaT Normal v1 only. Train is `[0,396000)`; Validation is `[396000,495000)`.
- Validation is split chronologically into calibration `[396000,445500)` and controlled evaluation `[445500,495000)`.
- Scaling: the frozen train-only `StandardScaler`; all relation computations use float64.
- Targets: exactly the six stable Innovation-2A2 targets: P2 `AIT202`, `FIT201`; P3 `DPIT301`, `FIT301`; P4 `FIT401`; P6 `FIT601`.
- Models: per-target self-history Ridge and stage-joint lagged Ridge, both `alpha=1.0`, with fixed lags `{1,5,10,20,30}`. The joint model uses every other nonconstant same-stage variable; no source or lag is selected in this experiment.
- Collaboration residual: `c(t)=e_joint(t)-e_self(t)`. Per-target normal reference is the calibration median and `max(1.4826*MAD, 1e-6)`; the score is `max(0,(c-median)/scale)`.
- Threshold: per-target calibration-normal score q99 only.
- Evaluation samples: 2,000 clean and 2,000 corrupted legal 64-point windows per target, selected by a fixed uniform `np.linspace` index rule.
- Corruption: retain the target's current value and complete history, retain all other stages, and replace every other nonconstant same-stage source window with the deterministic Normal donor `(k+floor(N/2)) mod N`. The observed donor separation is 24,718–24,719 seconds, exceeding the frozen 600-second guard.
- Comparators: the self-only score and the frozen A0 reconstruction checkpoint (`11,730` parameters). A0 was evaluated without retraining.

Neither the Attack workbook nor Attack labels were read. Controlled class labels are defined solely by this preregistered context-swap construction.

## Results

| Target | Stage | 2A2 gain | Collab AUROC | Collab AUPR | A0 AUROC | A0 AUPR | Self AUROC | Corrupt recall | Clean FPR | Joint degradation | Self change |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AIT202 | P2 | 0.072088 | 0.528153 | 0.538296 | 0.741890 | 0.787216 | 0.500000 | 0.0230 | 0.0100 | 0.000034 | 0.000000 |
| FIT201 | P2 | 0.053537 | 0.760974 | 0.813198 | 0.744461 | 0.782499 | 0.500000 | 0.4840 | 0.0090 | 0.001247 | 0.000000 |
| DPIT301 | P3 | 0.190883 | 0.698020 | 0.752276 | 0.706980 | 0.746322 | 0.500000 | 0.3580 | 0.0130 | 0.000231 | 0.000000 |
| FIT301 | P3 | 0.330211 | 0.711839 | 0.759603 | 0.700617 | 0.748741 | 0.500000 | 0.3530 | 0.0115 | 0.000722 | 0.000000 |
| FIT401 | P4 | 0.095624 | 0.499256 | 0.500888 | 0.648423 | 0.601525 | 0.500000 | 0.0050 | 0.0065 | 0.000002 | 0.000000 |
| FIT601 | P6 | 0.613691 | 0.511751 | 0.517439 | 0.504404 | 0.514945 | 0.500000 | 0.0345 | 0.0115 | 0.000000 | 0.000000 |

Overall medians are: collaboration AUROC `0.613086`, collaboration AUPR `0.645286`, corrupted recall `0.193750`, and clean FPR `0.010750`. Collaboration exceeds A0 AUROC by at least 0.10 for `0/6` targets. Both P3 targets are below 0.80 AUROC. FIT601, despite the largest 2A2 Normal validation gain, has controlled AUROC `0.511751` and ranks fifth of six by detectability. The descriptive Spearman association between 2A2 gain and controlled AUROC is `-0.314286`; it was not used for any decision.

## Score semantics

| Target | Clean median e_self | Clean median e_joint | Clean median c | Corrupt median e_self | Corrupt median e_joint | Corrupt median c |
|---|---:|---:|---:|---:|---:|---:|
| AIT202 | 0.001066 | 0.001276 | -0.000062 | 0.001066 | 0.001280 | -0.000009 |
| FIT201 | 0.000695 | 0.000621 | -0.000003 | 0.000695 | 0.002717 | 0.001347 |
| DPIT301 | 0.002112 | 0.001633 | -0.000599 | 0.002112 | 0.005244 | 0.000166 |
| FIT301 | 0.000891 | 0.000721 | -0.000080 | 0.000891 | 0.001748 | 0.000776 |
| FIT401 | 0.003004 | 0.001919 | -0.000388 | 0.003004 | 0.001897 | -0.000394 |
| FIT601 | 0.001033 | 0.000227 | -0.000805 | 0.001033 | 0.000227 | -0.000805 |

Every target has exactly zero median self-error change and self AUROC 0.5, confirming that the controlled operation leaves the target and its self-history unchanged. FIT201 and the two P3 targets show some joint-error response, but this response is not sufficiently strong or consistent across the frozen target set. FIT601's donor swap leaves its joint error unchanged at the reported precision; this result is retained as observed and was not used to alter the donor rule.

## Preregistered gates

| Gate | Requirement | Result |
|---|---|---|
| A | Median collaboration AUROC >= 0.80 | FAIL: 0.613086 |
| B | At least 4/6 targets beat A0 AUROC by >= 0.10 | FAIL: 0/6 |
| C | Median corrupted recall >= 0.50 | FAIL: 0.193750 |
| D | Median clean FPR <= 0.03 | PASS: 0.010750 |
| E | Every self AUROC is in `[0.49,0.51]` | PASS: all 0.500000 |

The median collaboration AUROC is also below the preregistered 0.70 partial-support boundary. The required status is therefore `STAGE_JOINT_LAGGED_CONTROLLED_NOT_SUPPORTED`. This falsifies the tested score-and-corruption construction as a broadly reliable controlled detector under the frozen protocol; it does not retroactively invalidate the Normal-only predictive gains observed in Innovation-2A2, and it does not adjudicate performance on real attacks.

## Integrity

All 15 protocol tests pass. The Normal cache, frozen A0 checkpoint, and A0 scaler SHA-256 values are identical before and after execution. No feature search, lag search, threshold search, point adjustment, score fusion, model retraining, or test-driven tuning was performed.

**NORMAL-ONLY CONTROLLED VALIDATION; NO ATTACK DATA; NO ATTACK LABELS; NO TEST TUNING; NO FEATURE SEARCH; NO LAG SEARCH; NO POINT ADJUSTMENT; NO FUSION; NO RETRAINING; NO COMMIT; NO PUSH.**
