# Innovation-2A SWaT Stage-local Lagged Dependency Discovery

Status: **STAGE_LOCAL_LAGGED_DEPENDENCY_NARROW**. Automatic progression to Innovation-2B is stopped for manual review.

This experiment asked only whether Official SWaT 2015 Normal data contains stable, directed, same-stage lagged dependencies that add predictive information beyond each target's own fixed history. It did not open Attack data or labels and did not compute anomaly metrics.

## Frozen protocol

- Source: `SWaT_Dataset_Normal_v1.xlsx` through the frozen S1 Normal cache.
- Official Train/Validation: first 396,000 / final 99,000 raw points.
- Train-internal discovery: `[0,316800)` for fitting and `[316800,396000)` for selection; both exclude their first 30 dependent samples so no lagged sample crosses the boundary.
- StandardScaler: fitted once on the complete 396,000-point Train split; Validation is transform-only.
- Targets: all 25 continuous, nonconstant variables from the T2 Official schema.
- Sources: all other nonconstant variables in the same audited P1–P6 stage; continuous and discrete sources are eligible, constants and cross-stage sources are excluded.
- Self history and candidate cross lags: exactly `{1,5,10,20,30}` at the original 1 Hz timeline; lag zero is forbidden.
- Ridge: `alpha=1.0`, float64. Discovery Fit/Select use `fit_stride=4`; final full-Train refitting, coefficient halves, and Official Validation use stride 1.
- Lag and source discovery use only the Train-internal Select split. Official Validation cannot choose or change either.
- Stability thresholds are unchanged: discovery/full-validation gains >=0.05, both validation-half gains >=0.02, and Train-half source coefficients have the same sign.
- At most three stable sources per target are selected by descending full-validation gain.

## Discovery outcome

There are 196 eligible directed source-target pairs. Three pass every stability condition, and all three remain after the fixed Top-3 rule. They cover three unique targets in two stages.

| Stage | Continuous targets | Candidates | Stable | Selected |
|---|---:|---:|---:|---:|
| P1 | 2 | 6 | 0 | 0 |
| P2 | 4 | 24 | 0 | 0 |
| P3 | 3 | 24 | 2 | 2 |
| P4 | 4 | 20 | 0 | 0 |
| P5 | 11 | 121 | 0 | 0 |
| P6 | 1 | 1 | 1 | 1 |

## Frozen selected relations

| Source -> target | Stage | Lag | Discovery gain | Validation gain | First-half gain | Second-half gain |
|---|---|---:|---:|---:|---:|---:|
| P302 -> DPIT301 | P3 | 1 | 0.058351 | 0.052238 | 0.049875 | 0.054601 |
| P302 -> FIT301 | P3 | 1 | 0.052406 | 0.063403 | 0.061923 | 0.064905 |
| P602 -> FIT601 | P6 | 5 | 0.418902 | 0.423834 | 0.425029 | 0.422556 |

All three source coefficients retain their positive sign between the first and second Train halves. The median selected validation gain is `0.063403`. Selected lag counts are lag 1: two; lag 5: one; lag 10/20/30: zero.

## Preregistered gate

| Gate | Requirement | Result |
|---|---|---|
| A | At least 6 selected stable relations | FAIL: 3 |
| B | At least 4 unique stable targets | FAIL: 3 |
| C | At least two stages each have at least two selected relations | FAIL: only P3 has 2 |
| D | Median selected validation gain >=0.05 | PASS: 0.063403 |

The evidence supports a small, interpretable dependency set, especially `P602(t-5) -> FIT601(t)`, but not the breadth required for the full `SUPPORTED` status. The preregistered result is therefore `STAGE_LOCAL_LAGGED_DEPENDENCY_NARROW`, and no controlled-corruption experiment is launched automatically.

## Raw cross-prediction comparison

Among the current continuous targets, 12 were previously marked `cross_only_stable_target=true` by the Normal-only V3-R raw cross-prediction diagnostic. Nine of those have no stable incremental dependency here: `FIT101`, `FIT201`, `FIT401`, `FIT501`, `FIT503`, `FIT504`, `LIT101`, `PIT501`, and `PIT503`. This comparison does not affect selection. It shows that high raw cross-channel predictability can reflect common operating state or redundancy without adding information after target self-history is present.

## Integrity

All 15 protocol checks pass. Only the frozen Normal cache was loaded; its SHA-256 is unchanged before and after execution. Attack data, Attack labels, thresholds, anomaly scores, point adjustment, and fusion were not used.

**NORMAL-ONLY DEPENDENCY DISCOVERY; NO ATTACK DATA; NO ATTACK LABELS; NO TEST-DRIVEN SELECTION; NO ANOMALY METRICS; NO FUSION; NO COMMIT; NO PUSH.**
