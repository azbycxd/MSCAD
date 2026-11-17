# SWaT-PCC1 Stage-local Pointwise Cross-Channel Consistency

Status: **STAGE_LOCAL_PCC_NOT_SUPPORTED_ON_REAL_SWAT**.

This first real-data transfer of the frozen CoMuT-D2 mechanism used only Official SWaT 2015 Physical data and the previously frozen T2.5/S1/S6 protocol. It did not modify or retrain A0 and did not use Attack data for fitting, feature selection, threshold choice, or any other model decision.

## Frozen protocol

- Normal Train: first 396,000 raw points of `SWaT_Dataset_Normal_v1.xlsx`.
- Normal Validation: final 99,000 raw points; used only for the PCC q99 threshold.
- Test: all 449,919 raw points of `SWaT_Dataset_Attack_v0.xlsx`.
- P1–P6 membership, variable kind, and constant status: T2 Normal-v1 schema audit only.
- Predictors: every nonconstant channel in the target's physical stage, including discrete channels.
- Targets: every nonconstant continuous channel; discrete channels are not targets.
- Model: one float64 `Ridge(alpha=1.0)` per continuous target, using all other same-stage channels at the same timestamp. No self input, lag, future value, target selection, or stage selection.
- Residual calibration: Train median and `max(1.4826 * MAD, 1e-6)` per target.
- Stage score: maximum calibrated target residual; standalone PCC: maximum over all six stages.
- Threshold: Normal Validation q99, `2400.920587`, fixed before Attack was opened.
- The 82-second gap remains a hard session boundary. PCC is strictly pointwise and performs no cross-row or cross-gap operation.

The T2 schema yields 25 continuous targets across all six stages. Constants are excluded, while discrete nonconstant channels remain eligible predictors exactly as preregistered.

## Global results

| Method | AUROC | AUPR | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|
| PCC | 0.855231 | 0.572923 | 0.084755 | 0.008275 | 0.015078 | 0.012348 |
| Existing A0 | 0.834887 | 0.743353 | 0.150140 | 0.917870 | 0.258067 | 0.718133 |

PCC ranks test points reasonably well and has a much lower fixed-threshold FPR than A0, but its Validation q99 transfers as an extremely conservative threshold: only 452 attack points are detected, and none recovers an A0-missed attack point. The existing A0 values and threshold are read unchanged from S1; A0 was neither rerun nor rescored.

## Frozen contextual subset

| ID | A0 hit ratio | PCC hit ratio | PCC max validation percentile | PCC top stage | PCC top target |
|---:|---:|---:|---:|---|---|
| 13 | 0.000000 | 0.000000 | 87.356566 | P5 | PIT501 |
| 14 | 0.023202 | 0.000000 | 98.806061 | P2 | FIT201 |
| 29 | 1.000000 | 0.000000 | 96.421212 | P2 | AIT201 |

The top stage/target is the contributor at each event's maximum standalone PCC score. No Attack-driven stage or target was selected. PCC hits 0/3 preregistered `HIGH_CONTEXTUAL_CANDIDATE` events at the fixed event-hit ratio of 0.10. In particular, it does not recover IDs 13 or 14, which were the decisive contextual cases.

## Complementarity and gates

Complementarity is evaluated only on the 449,793 legal A0 endpoint timestamps from the two frozen test segments. There are zero `A0 miss / PCC hit` anomaly points and zero recovered official events. PCC adds 49 normal alarms among points classified normal by A0. Because the recovered-TP denominator is zero, `added_FP_per_recovered_TP` is undefined rather than zero.

| Gate | Requirement | Result |
|---|---|---|
| A | PCC hits at least 2/3 contextual events | FAIL: 0/3 |
| B | ID13 or ID14 has PCC hit ratio >=0.10 and is at least 0.10 above A0 | FAIL |
| C | PCC test FPR <=0.10 | PASS: 0.012348 |
| D | Recover at least one A0 completely/nearly missed official event | FAIL: 0 |

The experiment therefore provides no evidence of real-event complementarity under the preregistered stage-local, no-lag, q99 protocol. The difference between AUROC 0.855 and the near-zero decision recall is a threshold-transfer observation, not permission to search another threshold in this experiment.

## Integrity and tests

All 16 preregistered checks passed. The scaler and Ridge fits use Train only; target self-input is excluded; only continuous targets are predicted; discrete predictors are allowed; constants are excluded; `alpha=1.0`; no lag, point adjustment, fusion, or test-driven selection is used; the canonical 36 events and contextual IDs `{13,14,29}` remain unchanged; all scores are finite. SHA-256 hashes of all four frozen S1 cache arrays match before and after the run.

**NO ATTACK-DRIVEN TRAINING; NO FEATURE SELECTION; NO LAG SEARCH; NO THRESHOLD SEARCH; NO POINT ADJUSTMENT; NO FUSION; NO COMMIT; NO PUSH.**
