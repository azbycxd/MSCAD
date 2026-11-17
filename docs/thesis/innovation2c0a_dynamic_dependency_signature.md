# Innovation-2C0-A SWaT Normal-only Dynamic Dependency Signature Feasibility

Status: **DYNAMIC_DEPENDENCY_SIGNATURE_FEASIBLE**.

This audit evaluates only whether the preregistered stage-local lagged dynamic-dependency signature is nondegenerate, low-rank, and stable from Official SWaT Normal Train to Official Normal Validation. It does not perform anomaly detection.

## Frozen representation

- Normal v1 Train `[0,396000)` and Validation `[396000,495000)`; the StandardScaler is fitted only on Train.
- Official audited P1–P6 stages; all nonconstant continuous and discrete channels retained, constants excluded.
- Dynamic values: `delta_z_i(t)=z_i(t)-z_i(t-1)` with no filtering or smoothing.
- Window `64`, stride `5`, fixed lags `{0,1,5,10,20,30}`.
- Directed feature: `D_(i→j,l)(t) = mean_tau[delta_z_i(tau-l) * delta_z_j(tau)]`, for every ordered pair `i != j` over the legal 64-point window. No local-variance normalization or Pearson correlation is used.
- Nondegenerate features are selected solely by Train standard deviation greater than `1e-6`.
- Each stage's PCA is fitted chronologically on the first 80% of retained Train signatures. Its dimension is the smallest `k` reaching 95% cumulative explained variance. The final 20% is Train Holdout; Validation is never used for feature filtering or PCA fitting.
- Structure error is mean absolute signature reconstruction error over retained features. No anomaly threshold or anomaly metric is computed.

There are `79,182` Train signatures and `19,782` Validation signatures per stage.

## Stage feasibility

| Stage | Channels | Raw dim | Nondegenerate dim/ratio | k95 | Compression | Train holdout median/q95 | Validation median/q95 | Median ratio | q95 ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 | 4 | 72 | 62 / 0.861111 | 2 | 0.032258 | 2.649e-06 / 1.183e-04 | 2.611e-06 / 9.037e-05 | 0.985734 | 0.763922 |
| P2 | 7 | 252 | 227 / 0.900794 | 4 | 0.017621 | 4.180e-06 / 2.300e-05 | 4.177e-06 / 5.037e-06 | 0.999098 | 0.218978 |
| P3 | 9 | 432 | 286 / 0.662037 | 11 | 0.038462 | 4.338e-05 / 2.064e-03 | 4.338e-05 / 1.506e-03 | 0.999996 | 0.729446 |
| P4 | 6 | 180 | 90 / 0.500000 | 1 | 0.011111 | 8.775e-07 / 1.386e-06 | 8.676e-07 / 1.411e-06 | 0.988776 | 1.017909 |
| P5 | 12 | 792 | 719 / 0.907828 | 8 | 0.011127 | 5.203e-05 / 8.898e-05 | 5.509e-05 / 9.352e-05 | 1.058632 | 1.050983 |
| P6 | 2 | 12 | 12 / 1.000000 | 2 | 0.166667 | 1.901e-05 / 1.901e-05 | 1.901e-05 / 1.901e-05 | 1.000000 | 1.000000 |

All six stages exceed the nondegeneracy requirement, all six meet the low-rank compression requirement, and all six meet both Normal stability ratios.

## Preregistered P3 sanity features

| Lag | P302→FIT301 std | P302→DPIT301 std |
|---:|---:|---:|
| 0 | 0.000205174 | 0.000135779 |
| 1 | 0.000111538 | 0.000644684 |
| 5 | 0.003308422 | 0.002752474 |
| 10 | 0.001911670 | 0.001040872 |
| 20 | 0.000252846 | 0.000565127 |
| 30 | 0.000062676 | 0.000037138 |

Both directed P3 relations are expressible and most variable at the preregistered lag 5 among the six inspected lags. This is explanatory only and does not alter the lag set.

## Preregistered P6 sanity feature

| Lag | P602→FIT601 std |
|---:|---:|
| 0 | 0.001812064 |
| 1 | 0.002470149 |
| 5 | 0.053132429 |
| 10 | 0.032392454 |
| 20 | 0.000475608 |
| 30 | 0.000105083 |

P602→FIT601 is strongly nondegenerate at lag 5 and lag 10, with lag 5 largest. This is also explanatory only.

## Feasibility gates

| Gate | Requirement | Result |
|---|---|---|
| A | At least four stages have nondegenerate ratio >= 0.20 | PASS: 6/6 |
| B | At least three stages have compression ratio <= 0.50 | PASS: 6/6 |
| C | At least three stages have median ratio <= 2 and q95 ratio <= 3 | PASS: 6/6 |

All gates pass. The representation is authorized for a separately requested Innovation-2C0-B Historical CoMuT pure-relation anomaly test, but that test was not run here.

All 15 protocol tests pass. The feature-statistics artifact retains the fixed order, Train standard deviation, nondegenerate decision, and variability rank for every directed dependency; the Top-10 per stage is interpretation-only.

**NORMAL-ONLY STRUCTURE FEASIBILITY; NO ATTACK DATA; NO ATTACK LABELS; NO WINDOW SEARCH; NO LAG SEARCH; NO ANOMALY METRICS; NO FUSION; NO COMMIT; NO PUSH.**
