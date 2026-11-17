# Innovation-2A2 SWaT Stage-level Joint Lagged Dependency Discovery

Status: **STAGE_LOCAL_JOINT_LAGGED_DEPENDENCY_SUPPORTED**. The preregistered gate permits a separately authorized Innovation-2B experiment; 2B was not run here.

This Normal-only diagnostic tests whether all other nonconstant variables in an audited physical stage jointly add predictive information beyond a continuous target's own history. It uses no Attack data, Attack labels, anomaly scores, thresholds, source selection, or lag selection.

## Frozen protocol

- Official Normal v1 split: Train `[0,396000)`, Validation `[396000,495000)`.
- StandardScaler: fitted only on all 396,000 Train rows; float64 throughout.
- Discovery Fit/Select: chronological `[0,316800)` and `[316800,396000)`, each with a 30-point boundary guard and `fit_stride=4`.
- Official Validation: stride 1 and excluded from design decisions.
- Targets: all 25 continuous, nonconstant channels from the T2 schema.
- Joint features: target self-history plus every other same-stage nonconstant channel at all fixed lags `{1,5,10,20,30}`. Constants and cross-stage channels are excluded; lag zero is forbidden.
- Ridge: `alpha=1.0`. No source, lag, alpha, or Top-k feature search.
- Stability: discovery/full-validation gain >=0.05 and both validation-half gains >=0.02.
- Source importance: post-hoc sum of absolute full-Train coefficients over five lags. It does not change fitting, target status, or gates.

## Stage outcome

Six of 25 targets are stable, covering four stages.

| Stage | Continuous targets | Stable targets |
|---|---:|---|
| P1 | 2 | None |
| P2 | 4 | AIT202, FIT201 |
| P3 | 3 | DPIT301, FIT301 |
| P4 | 4 | FIT401 |
| P5 | 11 | None |
| P6 | 1 | FIT601 |

## Stable joint targets

| Target | Stage | Cross sources | Discovery gain | Validation gain | First | Second | Top contributing sources / strongest lag |
|---|---|---:|---:|---:|---:|---:|---|
| AIT202 | P2 | 6 | 0.064607 | 0.072088 | 0.071357 | 0.072782 | AIT203@1; P205@30; P203@30 |
| FIT201 | P2 | 6 | 0.063554 | 0.053537 | 0.075042 | 0.030100 | AIT201@1; MV201@10; AIT203@5 |
| DPIT301 | P3 | 8 | 0.180625 | 0.190883 | 0.188940 | 0.192828 | FIT301@1; P302@1; MV302@1 |
| FIT301 | P3 | 8 | 0.335935 | 0.330211 | 0.332654 | 0.327731 | DPIT301@5; P302@5; P301@5 |
| FIT401 | P4 | 5 | 0.050332 | 0.095624 | 0.088062 | 0.103505 | P402@10; UV401@1; AIT402@1 |
| FIT601 | P6 | 1 | 0.613463 | 0.613691 | 0.617911 | 0.609174 | P602@5 |

The median stable validation gain is `0.143254`.

## Pairwise 2A comparison

All three pairwise 2A stable targets remain stable in the joint model: `DPIT301`, `FIT301`, and `FIT601`. The joint model adds `AIT202`, `FIT201`, and `FIT401`. Thus three targets failed to obtain any stable pairwise source in 2A but pass the fixed joint stability rule in 2A2. This is direct Normal-only evidence that some incremental collaboration is multivariate at stage level rather than adequately represented by a single source-target pair.

## Preregistered gates

| Gate | Requirement | Result |
|---|---|---|
| A | At least 6 stable targets | PASS: 6 |
| B | Stable targets cover at least 3 stages | PASS: 4 |
| C | Median stable validation gain >=0.05 | PASS: 0.143254 |
| D | Every stable target has both half gains >=0.02 | PASS |

All four gates pass, so `STAGE_LOCAL_JOINT_LAGGED_DEPENDENCY_SUPPORTED` is the required status. This authorizes, but does not itself execute, a separately preregistered controlled relation-anomaly validation.

## Integrity

All 15 protocol checks pass. The frozen Normal cache SHA-256 is identical before and after execution. No Attack source or labels were opened, and the post-hoc source ranking had no selection effect.

**NORMAL-ONLY JOINT DEPENDENCY DISCOVERY; NO ATTACK DATA; NO ATTACK LABELS; NO SOURCE SELECTION; NO LAG SEARCH; NO ANOMALY METRICS; NO FUSION; NO COMMIT; NO PUSH.**
