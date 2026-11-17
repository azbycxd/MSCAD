# Ablation A1: Innovation-2 Ridge Results

## Scope and evidence class

This batch executed the preregistered Innovation-2 Ridge ablations R5--R10 from `ablation_p0_design.md`. All new results are **RETROSPECTIVE_ABLATION** evidence. The frozen main method remains unchanged. No reconstruction model was trained or rerun.

The measured end-to-end runtime was **147.219 seconds (2.454 minutes)**. This includes SWaT and HAI Normal-only Ridge fitting, Historical CoMuT's 13 cases, the frozen SWaT controlled cases, score persistence, and D3 subset evaluation. CSV artifact authoring and tests are separate from this experiment runtime.

## Frozen definitions

- D0 Final is read-only: Self lags `[1,5,10,20,30]`, Cross lags `[0,1,5,10,20,30]`, Ridge alpha `1`, Final Unified Reliability Gate.
- D1 changes only the lag set to Self `[1]` and Cross `[0,1]`. The same Normal-only gate is recomputed after the short-lag refit.
- D2 retains the full D0 lag set and score, but uses all frozen base candidates: 25 on SWaT and 56 on HAI.
- D3 reuses the D2 candidate-level matrices without refitting. A/B/C/D correspond to candidate-only, Discovery skill, Validation skill, and stability filters.
- Every operating threshold is the corresponding Normal Validation score's fixed q99. Test labels are evaluation-only. There is no point adjustment, oracle threshold, Test-best selection, lag search, or hyperparameter search.

## Real-data ablation

| Dataset | Variant | Targets | AUROC | AUPR | F1-q99 | FPR | Event hit |
|---|---|---:|---:|---:|---:|---:|---:|
| SWaT | D0 Final | 15 | 0.675631 | 0.167510 | 0.015663 | 0.012455 | 2/36 |
| SWaT | D1 Short-lag | 16 | 0.865991 | 0.565012 | 0.015249 | 0.012301 | 2/36 |
| SWaT | D2 No-Gate | 25 | 0.665518 | 0.165144 | 0.015663 | 0.012455 | 2/36 |
| HAI | D0 Final | 54 | 0.916334 | 0.570193 | 0.610556 | 0.012443 | 44/50 |
| HAI | D1 Short-lag | 54 | 0.882610 | 0.426064 | 0.556056 | 0.009882 | 44/50 |
| HAI | D2 No-Gate | 56 | 0.915105 | 0.570880 | 0.610556 | 0.012443 | 44/50 |

D1 changed the gate membership rather than reusing D0's list. SWaT retained 16 short-lag targets, including `FIT301`, compared with 15 D0 targets. HAI retained 54 targets under both D0 and D1, but this equality is an observed count rather than a forced target-list reuse.

The factual deltas versus D0 are:

- SWaT D1: AUROC `+0.190360`, AUPR `+0.397502`, F1 `-0.000414`, FPR `-0.000154`.
- HAI D1: AUROC `-0.033724`, AUPR `-0.144129`, F1 `-0.054500`, FPR `-0.002561`.
- SWaT D2: AUROC `-0.010113`, AUPR `-0.002367`; q99 F1/FPR are unchanged to numerical precision.
- HAI D2: AUROC `-0.001229`, AUPR `+0.000687`; q99 F1/FPR are unchanged to numerical precision.

These are descriptive retrospective contrasts. They do not by themselves establish that long lags or the gate are universally necessary or harmful.

## Gate decomposition

| Dataset | Stage | Targets | AUROC | AUPR | F1-q99 | FPR |
|---|---|---:|---:|---:|---:|---:|
| SWaT | A | 25 | 0.665518 | 0.165144 | 0.015663 | 0.012455 |
| SWaT | B | 20 | 0.675220 | 0.167406 | 0.015663 | 0.012455 |
| SWaT | C | 18 | 0.675524 | 0.167476 | 0.015663 | 0.012455 |
| SWaT | D | 15 | 0.675631 | 0.167510 | 0.015663 | 0.012455 |
| HAI | A | 56 | 0.915105 | 0.570880 | 0.610556 | 0.012443 |
| HAI | B | 54 | 0.916334 | 0.570193 | 0.610556 | 0.012443 |
| HAI | C | 54 | 0.916334 | 0.570193 | 0.610556 | 0.012443 |
| HAI | D | 54 | 0.916334 | 0.570193 | 0.610556 | 0.012443 |

The target sets are monotonically non-increasing. D3-D exactly matches the frozen D0 target lists. AUROC, AUPR, Precision, Recall, F1, and FPR all reproduce D0 within `1e-6` on both datasets.

## Mechanism ablation

| Dataset | Variant | Median AUROC | Median AUPR | Median recall | Median FPR |
|---|---|---:|---:|---:|---:|
| Historical CoMuT | D0 Final | 0.995989 | 0.686927 | 0.937500 | 0.010313 |
| Historical CoMuT | D1 Short-lag | 0.995305 | 0.638043 | 0.925000 | 0.010775 |
| SWaT controlled | D0 Final | 0.996913 | 0.996931 | 0.872750 | 0.009500 |
| SWaT controlled | D1 Short-lag | 0.995790 | 0.995124 | 0.839750 | 0.009750 |
| SWaT controlled | D2 No-Gate | 0.630637 | 0.550076 | 0.008000 | 0.008000 |

Historical CoMuT D1 completed all 13 cases without an industrial gate; all 13 have AUROC at least 0.8. Category median AUROC is 0.991430 Short, 0.996067 Medium, 0.995305 Large, and 0.996026 Multiple-short.

The controlled donor manifest hash is unchanged. D1 evaluates the same six frozen target-local scenarios. D2 uses the preregistered all-candidate global maximum: only cross sources belonging to the corrupted stage are replaced, while the controlled target remains clean when it is an input to another candidate. Its much lower q99 recall is recorded as an aggregation/calibration observation, not used to alter the method.

## Persistence and boundaries

Each SWaT/HAI D1 and D2 directory contains Ridge coefficients and intercepts, source indices, target lists, gate diagnostics, Normal calibration statistics, Validation candidate matrices, Test raw-disagreement and calibrated candidate matrices, global scores, labels/timeline, thresholds, event summaries, and a manifest. D3 persists every stage's Validation/Test global scores and target manifest. CoMuT persists per-case candidate matrices, labels/timeline, coefficients, thresholds, and metrics. Controlled D1/D2 persist all six clean/corrupt score vectors and the unchanged donor hash.

SWaT preserves the two hard Test segments and evaluates only endpoint index 63 or later within each segment. HAI never constructs windows across source files. CoMuT keeps the historical 13-case protocol and does not introduce the industrial reliability gate.

## Integrity result

All 13 preregistered checks passed, including D1 lag binding, alpha binding, candidate counts, Normal-only selection, monotonic D3 sets, D3-D reproduction, controlled donor immutability, CoMuT gate exclusion, q99 provenance, raw-score persistence, and the absence of point adjustment/Test-best evaluation.

**Final status: `ABLATION_A1_INNOVATION2_COMPLETED`.**

No REC training, GPU REC ablation, hyperparameter search, lag search, Test tuning, point adjustment, oracle evaluation, method change, new target rule, commit, or push was performed.
