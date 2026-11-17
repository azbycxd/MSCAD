# Evaluation E2-R2 — Complete Literature-Compatible Evaluation

## Status and scope

Final status: `LITERATURE_COMPATIBLE_EVAL_COMPLETED`.

All 12 SWaT/HAI21 × six-method combinations passed the E2-R1D prerequisite and were evaluated from frozen raw-score archives. This phase performed no inference, training, scaler reconstruction, Ridge reconstruction, tail recalibration, checkpoint access, score transformation, model change, or parameter search.

Historical Strict results remain the thesis Primary results. Oracle and classic Point Adjustment results are retrospective literature-compatible diagnostics and must not be presented as deployable operating points.

## Endpoint and boundary protocol

- SWaT common mask: 449,791 endpoints. Mapping used exact segment/local/global/timestamp identity matching.
- HAI21 common mask: 401,685 endpoints. Mapping used exact official-file/local/global identity matching.
- SWaT PA intervals never cross the hard session gap.
- HAI PA intervals never cross official Test-file boundaries.
- Common-mask point labels form 35 SWaT intervals and 50 HAI intervals. SWaT's 35 point-label intervals are not the same object as the 36 attack-list event identities.
- Every q99 result uses the original method-specific frozen Normal Validation q99 threshold.

## SWaT comparison

| Method | AUROC common | AUPR common | F1 q99 | F1 Oracle | PA-F1 q99 | PA-F1 Oracle |
|---|---:|---:|---:|---:|---:|---:|
| LSTM-AE | 0.818242 | 0.706043 | 0.254495 | 0.762499 | 0.266571 | 0.884087 |
| USAD | 0.803816 | 0.698928 | 0.255184 | 0.741845 | 0.269202 | 0.805217 |
| GDN | 0.827665 | 0.728194 | 0.720216 | 0.772086 | 0.870736 | 0.900728 |
| Ours-REC | 0.834887 | 0.743353 | 0.258068 | 0.768959 | 0.272041 | 0.887275 |
| Ours-DUAL | 0.675632 | 0.167511 | 0.015663 | 0.406974 | 0.187814 | 0.809482 |
| Ours-TAIL | 0.708374 | 0.206747 | 0.260790 | 0.339804 | 0.274334 | 0.398977 |

Ours-REC retains the highest common-mask AUROC and AUPR. Under Oracle and PA-Oracle, GDN is numerically first and Ours-REC second; Ours-REC remains above LSTM-AE and USAD on both metrics. The large Oracle deltas for LSTM-AE, USAD, and Ours-REC show that their frozen q99 operating points transfer poorly despite substantially stronger raw-score ranking. GDN is comparatively less threshold-sensitive and strongly PA-sensitive.

The historical Ours-REC regression passed:

- F1-Oracle: 0.768959436 versus reference 0.768959.
- PA-F1@q99: 0.272040727 versus reference 0.272041.
- PA-F1-Oracle: 0.887275006 versus reference 0.887275.

The common-mask version removes exactly the first endpoint of each SWaT segment from methods historically starting at local index 63. The resulting change is recorded without modifying historical results.

## HAI21 comparison

| Method | AUROC common | AUPR common | F1 q99 | F1 Oracle | PA-F1 q99 | PA-F1 Oracle |
|---|---:|---:|---:|---:|---:|---:|
| LSTM-AE | 0.907084 | 0.577510 | 0.512393 | 0.637348 | 0.644794 | 0.883002 |
| USAD | 0.856873 | 0.494578 | 0.399760 | 0.579345 | 0.572777 | 0.873148 |
| GDN | 0.877643 | 0.497775 | 0.562053 | 0.590018 | 0.762172 | 0.840044 |
| Ours-REC | 0.852899 | 0.536294 | 0.520594 | 0.639274 | 0.730542 | 0.908513 |
| Ours-DUAL | 0.916334 | 0.570194 | 0.610556 | 0.654843 | 0.785479 | 0.842294 |
| Ours-TAIL | 0.901850 | 0.617897 | 0.617350 | 0.686865 | 0.768181 | 0.919477 |

The strict-protocol advantage remains visible under the literature-compatible views, but it is metric-dependent rather than universal. Ours-TAIL has the highest AUPR, F1-Oracle, and PA-F1-Oracle. Ours-DUAL has the highest AUROC and PA-F1@q99 and the second-highest F1-Oracle. Ours-REC is second on PA-F1-Oracle. GDN remains competitive on PA-F1@q99, while LSTM-AE ranks third on PA-F1-Oracle. These are numerical orderings, not a claim of universal superiority.

## Interpretation boundaries

- `F1-Oracle` uses the exact Test-best threshold, evaluates every unique score boundary, and resolves exact ties using the largest numeric threshold.
- `PA-F1@q99` applies classic PA only after frozen Validation q99 predictions.
- `PA-F1-Oracle` combines exact Test-best threshold selection and PA.
- PA never deletes a false positive and never modifies normal intervals.
- AUROC/AUPR are computed from raw scores on the common endpoint set; no AUROC/AUPR is computed from PA-adjusted binary predictions.
- No code path emits SOTA, best-method, or universally-superior conclusions.

## Outputs

The `outputs/thesis_evaluation/literature_compatible/final/` directory contains immutable Historical Strict Table A, common-mask q99 Table B, literature-compatible Table C, per-dataset comparison tables, threshold-transfer diagnostics, Oracle details, PA boundary audits, a local/external literature interface schema, and a manifest with source hashes and all negative-action assertions.

External paper values remain blank for Park-Shim 2026, LatentFlow 2026, MAD-TCN 2026, and CCM-TAD 2026. No network access or literature-number transcription was performed in E2-R2.

