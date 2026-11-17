# Evaluation-E2: Literature-compatible evaluation

## Status

`LITERATURE_COMPATIBLE_EVAL_PARTIAL`

Only one of the 12 required dataset-method combinations has a persisted raw Test score, aligned binary labels, legal endpoint timeline, frozen q99 threshold, and provenance: **SWaT Ours-REC**. The other 11 combinations retain valid Strict aggregate metrics, but their raw score/label/mask vectors were not persisted. E2 did not load checkpoints, rerun inference, or train any model to fill those gaps.

## Scope and protocol

Datasets are Official SWaT 2015 Physical and HAI 21.03. Methods are LSTM-AE, USAD, GDN, Ours-REC, Ours-DUAL, and Ours-TAIL. WADI is excluded from the Primary tables. LatentFlow, MAD-TCN, CCM-TAD, and Park-Shim are not run.

Four result meanings remain separate:

1. `F1_q99` is the frozen Strict result. Prediction is `score > Normal Validation q99`; it has no Test tuning and no point adjustment.
2. `F1_oracle` is an exact retrospective Test-label threshold sweep without PA. Every unique score boundary is evaluated. The largest numeric threshold wins exact F1 ties.
3. `PA_F1_q99` starts from the frozen q99 predictions and applies classic point adjustment within each aligned label-one interval. It never changes normal positions and never crosses the SWaT hard gap or an HAI file boundary.
4. `PA_F1_oracle` is the most permissive diagnostic: exact Test-best threshold plus classic PA.

AUROC and AUPR are copied from the unchanged Strict raw-score results. E2 never computes AUROC/AUPR from PA-adjusted binary predictions.

## Score availability audit

| Dataset | Method | Raw score and aligned labels | E2 literature metrics |
|---|---|---|---|
| SWaT | Ours-REC | Available: frozen S1 `test_scores.npz` | Completed |
| SWaT | LSTM-AE, USAD, GDN, Ours-DUAL, Ours-TAIL | `SCORE_NOT_PERSISTED` | Not recomputed |
| HAI21 | all six methods | `SCORE_NOT_PERSISTED` | Not recomputed |

The available SWaT archive contains 449,793 legal Ours-REC endpoints from two separately windowed Test segments. Completed baseline metrics record 449,791 common endpoints because those protocols start at local index 64 rather than the reconstruction endpoint's local index 63. E2 preserves every method's frozen Strict metric and does not silently redefine the historical main table. A future fully common-mask rerun therefore requires persisted raw scores for all methods and an explicit protocol decision.

## Table A: Strict evaluation

No Test tuning. No point adjustment.

### SWaT

| Method | AUROC | AUPR | F1-q99 | FPR-q99 |
|---|---:|---:|---:|---:|
| LSTM-AE | 0.818242 | 0.706043 | 0.254495 | 0.726609 |
| USAD | 0.803816 | 0.698928 | 0.255184 | 0.716859 |
| GDN | 0.827665 | 0.728194 | 0.720216 | 0.023658 |
| Ours-REC | 0.834887 | 0.743353 | 0.258067 | 0.718133 |
| Ours-DUAL | 0.675631 | 0.167510 | 0.015663 | 0.012455 |
| Ours-TAIL | 0.708375 | 0.206747 | 0.260789 | 0.692531 |

### HAI21

| Method | AUROC | AUPR | F1-q99 | FPR-q99 |
|---|---:|---:|---:|---:|
| LSTM-AE | 0.907084 | 0.577510 | 0.512393 | 0.024441 |
| USAD | 0.856873 | 0.494578 | 0.399760 | 0.031484 |
| GDN | 0.877643 | 0.497775 | 0.562053 | 0.011529 |
| Ours-REC | 0.852899 | 0.536294 | 0.520594 | 0.016133 |
| Ours-DUAL | 0.916334 | 0.570193 | 0.610556 | 0.012443 |
| Ours-TAIL | 0.901850 | 0.617897 | 0.617350 | 0.012627 |

## Table B: Literature-compatible diagnostics

| Dataset | Method | F1-q99 | F1-Oracle | PA-F1@q99 | PA-F1-Oracle | Status |
|---|---|---:|---:|---:|---:|---|
| SWaT | Ours-REC | 0.258067 | 0.768959 | 0.272041 | 0.887275 | Completed |
| SWaT | other five methods | frozen values available | — | — | — | `SCORE_NOT_PERSISTED` |
| HAI21 | all six methods | frozen values available | — | — | — | `SCORE_NOT_PERSISTED` |

`F1-Oracle` is a Test-best threshold diagnostic. `PA` is classic point adjustment. `PA-F1-Oracle` combines Test-best threshold selection and point adjustment. None is a deployable threshold or a replacement for the Strict result.

## SWaT Ours-REC exact results

The unchanged score archive SHA256 is `B71B9B6A82985D93F61DCC2D9672AFCB7D64DEC9040B50CBCE7CB75E4335AFDA`.

| Metric | q99 | Oracle | PA at q99 | PA Oracle |
|---|---:|---:|---:|---:|
| Threshold | 0.183962 | 0.808285 | 0.183962 | 0.671267 |
| Precision | 0.150140 | 0.975515 | 0.158062 | 0.921443 |
| Recall | 0.917870 | 0.634591 | 0.975394 | 0.855550 |
| F1 | 0.258067 | 0.768959 | 0.272041 | 0.887275 |
| FPR | 0.718133 | 0.002202 | 0.718133 | 0.010082 |

- `delta_oracle = +0.510892`.
- `delta_pa_q99 = +0.013973`.
- `delta_pa_oracle = +0.629208`.
- Exact sweep boundaries evaluated: 439,111.
- Point-label anomaly intervals: 35, split independently at the hard session boundary.
- q99 hits 32 intervals; PA-Oracle hits 18 intervals. These counts describe threshold hits on Physical point-label intervals, not the separate 36-event attack-list evaluation.

Strict recomputation matches the frozen AUROC, AUPR, Precision, Recall, F1, and FPR within `1e-6`; the largest observed absolute difference is `1.11e-16`. The new exact `score > threshold` Oracle F1 is 0.768959436, within `4.36e-7` of the historical approximately 0.768959 diagnostic. Its numeric threshold differs from the earlier report because that report used the `score >= threshold` convention; the selected prediction set and F1 remain consistent.

## Interpretation

For SWaT Ours-REC, the evidence supports **good ranking with poor Validation-to-Test threshold transfer**. Oracle thresholding raises F1 by 0.511 and reduces FPR from 0.718 to 0.002. Classic PA at the frozen q99 adds only 0.014 F1 because PA cannot remove the large number of normal-point false positives. PA adds further benefit only after Test-best threshold selection, producing the non-deployable upper diagnostic F1 of 0.887.

No mechanism conclusion is drawn for the remaining methods. Aggregate Strict metrics alone cannot determine their Oracle or PA behavior. In particular, the requested HAI comparison of Ours-DUAL/Ours-TAIL against GDN/LSTM-AE remains unavailable until raw aligned scores are intentionally persisted by an authorized evaluation-only inference run.

## Literature interface

`literature_comparison_schema.csv` contains placeholders for Park-Shim 2026, LatentFlow 2026, MAD-TCN 2026, and CCM-TAD 2026. E2 copies no external paper number. Each future entry must record its evaluation protocol, PA use, Test-best threshold use, source, and direct comparability before it can be placed beside local results.

## Scientific use

- Table A remains the only Primary result table.
- Table B is retrospective and literature-compatible diagnostic evidence.
- Missing literature-compatible cells must remain missing; they may not be inferred from AUROC, AUPR, event hit rates, or checkpoints.
- No existing Strict artifact was overwritten.

**NO TRAINING. NO MODEL CHANGE. NO SCORE CHANGE. NO SCORE TRANSFORMATION. NO NEW FUSION. NO TEST-DRIVEN MODEL OR CHECKPOINT SELECTION. NO PA IN THE STRICT TABLE. NO COMMIT. NO PUSH.**
