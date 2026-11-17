# Innovation-2D2: Official SWaT Real-Attack Dual-View Test

## Scope and frozen protocol

This diagnostic evaluates the already frozen Local Lagged Dual-View Collaboration Consistency mechanism on the official SWaT 2015 Physical Attack data. It does not redesign or tune the method. The six targets are exactly AIT202, FIT201, DPIT301, FIT301, FIT401, and FIT601. Self View uses target lags `[1,5,10,20,30]`; Cross View excludes the target and uses every other nonconstant same-stage channel at lags `[0,1,5,10,20,30]`. Both models are `Ridge(alpha=1.0)`.

The scaler and both views are fitted only on Normal v1 rows `[0,396000)`. Robust disagreement references and the global q99 threshold use only the first half of Normal Validation, with legal lagged endpoints `[396030,445500)`. Attack is opened only after model fitting, calibration, max-over-six aggregation, and threshold freezing. The Dual score is strictly

`max_j max(0, (abs(pred_self_j - pred_cross_j) - median_j) / scale_j)`.

Observed-target residuals are not part of this score. The 82-second gap is a hard session boundary. The first 30 points of both Attack segments are invalid for lag features; Dual/A0 comparisons use the same 449,793 frozen A0 endpoint timestamps, which are a subset of the legal Dual endpoints. No point adjustment is used.

## Global results

| Method | AUROC | AUPR | Precision | Recall | F1 | FPR | Frozen threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dual-View | 0.337473 | 0.090193 | 0.088143 | 0.008220 | 0.015038 | 0.011754 | 236954.843306 |
| A0 aligned | 0.834887 | 0.743353 | 0.150140 | 0.917870 | 0.258067 | 0.718133 | 0.1839622706 |

The very large frozen Dual threshold is a direct consequence of the preregistered Normal Validation-front-half q99 calibration; it was not altered after observing Attack performance.

## Point complementarity

Among real Attack endpoints, A0 missed and Dual detected 7 points. Dual added 368 normal false positives on points where A0 was correct, or 52.5714 added false positives per recovered true positive. The full Attack partition is: both hit 442, A0-only hit 49,693, Dual-only hit 7, and both miss 4,479.

## Contextual events

| Event | Components/stage | A0 hit rate | Dual hit rate | Most frequent Dual argmax |
|---|---|---:|---:|---|
| 13 | MV304 / P3 | 0.000000 | 0.000000 | FIT601 / P6 |
| 14 | MV303 / P3 | 0.023202 | 0.000000 | FIT601 / P6 |
| 29 | P201; P203; P205 / P2 | 1.000000 | 0.000000 | AIT202 / P2 |

For IDs 13 and 14, the most frequent attribution is not a frozen P3 target; for ID 29 it is the expected P2 target AIT202. These observations are explanatory only and were not used for feature, target, lag, or threshold selection.

## Canonical event results and gates

Of 36 canonical physical events, 2 have Dual hit rate at least 0.10, 7 are A0 near-misses (`A0 hit rate < 0.10`), and none satisfies the preregistered Dual-recovery definition. Therefore the recovered-event ID list is empty.

| Gate | Result | Evidence |
|---|---|---|
| A | FAIL | 0/3 contextual events have Dual hit rate >= 0.10 |
| B | FAIL | Neither ID13 nor ID14 reaches 0.10 or improves over A0 |
| C | PASS | Dual normal FPR = 0.011754 <= 0.10 |
| D | FAIL | No A0 near-miss event is recovered |

Final status: `DUAL_VIEW_REAL_SWAT_COMPLEMENTARITY_NOT_SUPPORTED`.

This outcome does not revise the controlled 2D0/2D1 mechanism evidence. It establishes that the frozen form does not provide the preregistered real-SWaT contextual complementarity under its fixed Normal-calibrated threshold.

## Integrity and artifacts

All 20 requested integrity conditions passed, including train-only scaling, fixed targets/lags/alpha, target exclusion from Cross View, validation-only calibration and thresholding, gap isolation, canonical-36 identity, frozen A0 binding, aligned masks, and finite outputs. Source artifact hashes are recorded in `outputs/thesis_innovation2d2_real_swat/manifest.json`.

No A0 retraining, Attack-driven selection, target change, source/feature search, lag search, alpha search, threshold search, POT, test-best F1, point adjustment, fusion, commit, or push was performed.
