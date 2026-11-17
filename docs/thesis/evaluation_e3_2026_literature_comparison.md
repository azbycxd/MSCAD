# Evaluation-E3: 2026 Recent Literature Reported-Result and Protocol Compatibility Audit

## Decision

**Status: `E3_2026_LITERATURE_AUDIT_PARTIAL_FINAL`.**

The audit used only official publisher/proceedings pages, official arXiv, official institutional records, and official repositories. CCM-TAD was already fully recoverable. In E3-R1, the official ACM eReader restored LatentFlow Tables 1 and 7. Park & Shim yielded only the publisher's narrative SWaT F1-gain range and WADI maximum gain, while MAD-TCN still exposed no exact SWaT performance-table cells. Remaining inaccessible values are permanently retained as `NOT_EXTRACTED`; no number was inferred from a blog, secondary paper, search snippet, or repository behavior.

No training, inference, reimplementation, score conversion, metric conversion, Test tuning, commit, or push was performed.

E3-R1 closes the restricted-source search with **47 external records: 33 extracted and 14 `NOT_EXTRACTED`**. The 20 previously verified CCM-TAD records are unchanged, and all 24 local E2 rows remain in the unified comparison schema. No `SOURCE_CONFLICT` was found.

## Source access ledger

| Work | Official paper status | Official repository | Result extraction status |
|---|---|---|---|
| Park & Shim 2026 | Publisher narrative and abstract accessible; Table 3 cells blocked by robot verification | `plugin-mtsad`, commit `84e0d145de4c53b4d4733e3b33164e6068a3d5e9` | SWaT F1 gain `+0.028–+0.050` and WADI maximum gain `+0.417` retained; absolute Table 3 scores remain `NOT_EXTRACTED` |
| LatentFlow KDD 2026 | Official ACM eReader/PDF accessible under CC BY 4.0 | `LatentFlow`, commit `3695f54317fcc1a3fdbfa0a7037d1cdf4369db27` | Tables 1 and 7 fully recovered for SWaT/WADI Aff-P, Aff-R, Aff-F1, VUS-ROC, and VUS-PR |
| MAD-TCN 2026 | Publisher full text inaccessible; official abstract available | `MAD-TCN`, commit `24d42ee0e0fc5a5460d437026cf4b89f05381293` | SWaT table `NOT_EXTRACTED`; approximate 0.026 M parameter claim retained as abstract-only evidence |
| CCM-TAD ICML 2026 | Official arXiv PDF and appendices accessible | `CCM-TAD`, commit `3a8bc8263b648a707cd0658bf4f06517fb9e52c5` | Main and supplementary SWaT/WADI results extracted with table/page trace |

## Table D1. 2026 Recent Methods — Reported by original papers

`NOT_EXTRACTED` means the requested value was not directly visible in an allowed first-party source. It does not mean zero or not reported by the paper.

| Method | Venue | Dataset | Reported metric | Reported value | Evaluation protocol | Source |
|---|---|---|---:|---:|---|---|
| Park-Shim Plug-in | ESWA 2026 | SWaT | Precision / Recall / F1 / AUROC / AUPR, all bases ± plug-in | NOT_EXTRACTED | Table 3 cells inaccessible; paper protocol incomplete | Publisher DOI and official repo |
| Park-Shim Plug-in | ESWA 2026 | SWaT | F1 gain range | +0.028–+0.050 | Publisher narrative summary across all base models; no absolute scores inferred | Official ScienceDirect HTML |
| Park-Shim Plug-in | ESWA 2026 | WADI | Maximum F1 gain | 0.417 | Abstract-only; exact base model/table/protocol unavailable | Official ScienceDirect abstract |
| Park-Shim Plug-in | ESWA 2026 | WADI | Precision / Recall / F1 / AUROC / AUPR, all bases ± plug-in | NOT_EXTRACTED | Full paper inaccessible; paper protocol unknown | Publisher DOI and official repo |
| LatentFlow | KDD 2026 | SWaT | Aff-P / Aff-R / Aff-F1 | 65.01±0.74 / 93.59±2.28 / 76.70±0.50 | Affiliation metrics; SPOT threshold; three repetitions | Table 1, p. 4503 |
| LatentFlow | KDD 2026 | SWaT | VUS-ROC / VUS-PR | 78.47 / 59.45 | Threshold-agnostic range-aware VUS | Table 7, p. 4509 |
| LatentFlow | KDD 2026 | WADI | Aff-P / Aff-R / Aff-F1 | 63.72±0.30 / 95.71±0.73 / 76.51±0.13 | Affiliation metrics; SPOT threshold; three repetitions | Table 1, p. 4503 |
| LatentFlow | KDD 2026 | WADI | VUS-ROC / VUS-PR | 47.69 / 6.15 | Threshold-agnostic range-aware VUS | Table 7, p. 4509 |
| MAD-TCN | Neurocomputing 2026 | SWaT | AUC / F1 / Precision / Recall | NOT_EXTRACTED | `PROTOCOL_NOT_FULLY_SPECIFIED` | Publisher page and official repo |
| MAD-TCN | Neurocomputing 2026 | Cross-dataset | Parameters | ≈0.026 M | Publisher abstract claim only | Official publisher abstract |
| CCM-TAD | ICML 2026 | SWaT | Point-based Best-F1 (PF1 in repo) | 0.882 | Test-best point threshold; no classic PA | Table 4, p. 8 |
| CCM-TAD | ICML 2026 | SWaT | Sequential Best-F1 (SF1 in repo) | 0.883 | Joint Test-label maximization over alpha and h; boundary correction | Table 2, p. 7; Appendix F |
| CCM-TAD | ICML 2026 | SWaT | PR AUC | 0.911 | Sequential score | Table 3, p. 8 |
| CCM-TAD | ICML 2026 | SWaT | VUS-PR / VUS-ROC | 0.779 / 0.758 | Range-aware VUS metrics | Table 10, p. 19 |
| CCM-TAD | ICML 2026 | SWaT | Range-F1 / Affiliation-F1 | 0.681 / 0.781 | Range/Affiliation metrics | Table 11, p. 19 |
| CCM-TAD | ICML 2026 | WADI | Point-based Best-F1 (PF1 in repo) | 0.739 | Test-best point threshold; no classic PA | Table 4, p. 8 |
| CCM-TAD | ICML 2026 | WADI | Sequential Best-F1 (SF1 in repo) | 0.761 | Joint Test-label maximization over alpha and h; boundary correction | Table 2, p. 7; Appendix F |
| CCM-TAD | ICML 2026 | WADI | PR AUC | 0.720 | Sequential score | Table 3, p. 8 |
| CCM-TAD | ICML 2026 | WADI | VUS-PR / VUS-ROC | 0.776 / 0.822 | Range-aware VUS metrics | Table 10, p. 19 |

CCM-TAD Table 12 additionally reports five-seed mean ± standard deviation: SWaT F1 `0.877 ± 0.004`, PR AUC `0.907 ± 0.004`, VUS-ROC `0.749 ± 0.011`, VUS-PR `0.770 ± 0.009`; WADI F1 `0.744 ± 0.015`, PR AUC `0.698 ± 0.024`, VUS-ROC `0.813 ± 0.017`, VUS-PR `0.756 ± 0.016`. These remain under the same incompatible Test-driven and dataset-preprocessing protocol.

## Table D2. Protocol Compatibility

| Method | Metric | PA | Test-best | Test-driven HP/selection | Range/Affiliation | Direct comparability with Ours |
|---|---|---|---|---|---|---|
| Park-Shim paper | Point Precision/Recall/F1/AUROC/AUPR | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | `PARTIAL`: metric names overlap, but full paper protocol and dataset version are unverified |
| Park-Shim repository | Point F1 and PA-F1 over fused scores | Yes for the range path | Yes | Yes: 11 fusion weights are all evaluated on Test; base error uses Test median/IQR | PA path present | `NOT_DIRECT` for PA-F1; repository behavior cannot substitute for paper values |
| LatentFlow | Affiliation-F1 | No classic PA | Not shown as a Test-best sweep in current repo | Fixed dataset q; its derivation is undocumented; SPOT updates on Test stream | Yes | `NOT_DIRECT` |
| LatentFlow | VUS-ROC / VUS-PR | No | N/A | Checkpoint uses Normal Validation loss; paper result values inaccessible | Yes | `NOT_DIRECT` |
| MAD-TCN | AUC | UNKNOWN | N/A for AUC | UNKNOWN | UNKNOWN | `PARTIAL`: metric name overlaps AUROC, but dataset/protocol are unverified |
| MAD-TCN | Precision / Recall / F1 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | `PARTIAL`: names overlap, but threshold, PA, Test use, and preprocessing are unverified |
| CCM-TAD | PF1 | No | Yes | Yes: Test PF1 checkpoint/threshold behavior in repository | No | `PARTIAL` with Ours F1-Oracle only; preprocessing and endpoint streams differ |
| CCM-TAD | SF1 | No classic PA | Yes | Yes: alpha and h jointly maximize Test-label F1 | Sequential evidence + boundary correction | `NOT_DIRECT` |
| CCM-TAD | PR AUC | No | N/A for metric | Yes: model/score selection remains Test-driven | Sequential score | `PARTIAL` with Ours AUPR; not a strict comparison |
| CCM-TAD | VUS-ROC / VUS-PR | No | N/A | Yes: source model and hyperparameters are Test-selected | Yes | `NOT_DIRECT` |
| CCM-TAD | Range-F1 / Affiliation-F1 | No classic PA | Paper table does not fully state threshold | Yes: source model/hyperparameters are Test-selected | Yes | `NOT_DIRECT` |

No external row qualifies as `DIRECT`. The nearest numerical comparison is `PARTIAL`, not evidence that either method is superior.

## Dataset-version verification

### CCM-TAD

The paper explicitly reports SWaT as 51 channels, Train 495,000, Test 449,919, and WADI as 123 channels, Train 1,209,601, Test 172,801. It then downsamples SWaT/WADI by five, zeros SWaT P201 and LIT401, applies a moving average, clips transformed Test values to `[-4,+4]`, and applies separate WADI missing-value/startup handling. This is not the thesis Official SWaT 2015 Physical pipeline, so even point AUROC/AUPR/F1 comparisons are only `PARTIAL`.

### Park-Shim, LatentFlow, MAD-TCN

The allowed accessible sources do not establish a bitwise match to the thesis Official SWaT 2015 Physical data, its 51-variable input, exact labels, or common endpoint mask. Their SWaT metric-name matches therefore cannot be promoted to `DIRECT`. WADI has no Ours E2 comparator and remains contextual only.

HAI 21.03 stays the thesis independent real-world validation dataset. None of these four papers supplies a verified HAI result, so HAI values are excluded from direct literature claims.

## Paper protocol versus repository behavior

### Park & Shim 2026

The official abstract supports dependency-matrix reconstruction, a normal-only dependency reference, a dependency anomaly score, and fusion with a base score. The public repository instantiates GDN, reconstructs dense attention/dependency matrices, jointly trains forecast and plug-in losses, computes `a*S_base + (1-a)*S_dep` for all 11 `a` values, normalizes base errors with Test median/IQR, and evaluates both Test-best point F1 and Test-best PA-F1. Because the paper tables are unavailable, these source behaviors are recorded in parallel and are not asserted to be the exact publication protocol.

Method comparison with Ours:

| Dimension | Park & Shim | Ours |
|---|---|---|
| Explicit dependency anomaly | Reconstructs a base model's dependency matrix | Compares Self-View and Cross-View predictive evidence |
| Normal-only relation reference | Plug-in AE learns normal dependency matrices | Self/Cross predictors and calibration use normal-only data |
| Score construction | Dependency reconstruction deviation fused with base score | Dual-view disagreement/evidence; tail fusion is separately frozen |
| Base-model dependence | Requires a base model exposing dependency matrices; released code is GDN-specific | Uses lightweight fixed predictive views |
| Interpretability | Deviation matrix localizes changing variable links | View disagreement and target/source residuals localize evidence |
| Parameter cost | Adds an N²-input dependency AE | Deterministic/lightweight dual predictive evidence |

This is a factual structural comparison, not a ranking claim.

### LatentFlow 2026

LatentFlow learns a nonlinear, directed, self-blind top-k channel graph and evolves it through a latent gated dynamic inspired by continuous Ornstein-Uhlenbeck behavior. Its official score is endpoint reconstruction MSE; graph dynamics affect reconstruction but are not a separately fused score. Ours instead uses fixed lagged Self/Cross views and directly interprets their predictive disagreement. The two methods target related dependency anomalies, but are not mathematically identical. LatentFlow is learned and nonlinear; Ours is deterministic/lightweight dual predictive evidence.

The paper reports three-run mean ± standard deviation for Affiliation metrics and uses SPOT with a threshold estimated from training reconstruction-error distribution. The current repository selects checkpoints by Normal Validation loss. `q` affects SPOT-based Affiliation metrics (`0.016` on SWaT, `0.002` on WADI); VUS metrics are threshold-agnostic and range-aware. The repository does not prove how those dataset-specific q values were chosen, and SPOT processes the Test score stream, so Affiliation-F1 is not mapped to ordinary point F1.

Recovered official values are: SWaT Aff-P `65.01±0.74`, Aff-R `93.59±2.28`, Aff-F1 `76.70±0.50`, VUS-ROC `78.47`, VUS-PR `59.45`; WADI Aff-P `63.72±0.30`, Aff-R `95.71±0.73`, Aff-F1 `76.51±0.13`, VUS-ROC `47.69`, VUS-PR `6.15`. These values remain `NOT_DIRECT` relative to the thesis point-wise metrics.

### MAD-TCN 2026

The official model file verifies multi-scale local/global depthwise temporal convolution, cross-variable feed-forward mixing, and an adaptive gate. The repository has no dataset loader, training/evaluation entry point, SWaT configuration, threshold code, or anomaly-score implementation, and its constructor references an undeclared external `args.alpha_value`. Therefore the paper's SWaT AUC/F1/Precision/Recall and their protocol remain `NOT_EXTRACTED` / `PROTOCOL_NOT_FULLY_SPECIFIED`.

The approximately `0.026 M` parameter count is a paper-reported abstract claim. It is not a local measurement and is excluded from the local efficiency table.

### CCM-TAD 2026

CCM-TAD uses correlation-profile clustering, cluster-specific embeddings, a temporally masked causal mixer, endpoint reconstruction MSE, and sequential evidence with boundary correction. The paper's PF1-like point result and SF1-like sequential result are distinct. SF1 is not ordinary point-wise F1, and neither SF1, Affiliation-F1, Range-F1, nor VUS is converted to an Ours metric.

The paper defines Best-F1 by jointly maximizing `alpha` and `h` against ground-truth labels. The repository additionally evaluates Test every epoch, selects the checkpoint by Test PF1/SF1, sweeps point thresholds, and optimizes sequential alpha/threshold on Test labels. It also uses a centered moving average that consumes future points, despite the model's masked temporal mixer. Paper and repository behavior are recorded separately.

## Mapping to E2 results

E2 supplies Ours strict SWaT AUROC/AUPR/F1-q99/FPR-q99 and supplementary F1-Oracle, PA-F1@q99, and PA-F1-Oracle. Only like-for-like metric semantics may be juxtaposed:

- Ordinary AUC/PR AUC can be a `PARTIAL` numerical reference when the external dataset pipeline or score postprocessing differs.
- Test-best point F1 can be a `PARTIAL` reference to Ours F1-Oracle, never to Ours F1-q99.
- Affiliation-F1, Range-F1, VUS, and CCM-TAD SF1 are `NOT_DIRECT` and are not converted.
- PA-F1 is shown only against explicitly point-adjusted literature results. It is not treated as CCM-TAD SF1.
- HAI results remain independent validation and are not inserted into a direct 2026 literature comparison.

Accordingly, this audit does not support the statement that Ours exceeds all 2026 methods. Where `PARTIAL` is shown, the numbers are only a supplementary comparison under different protocols.

## Source trace

- Park & Shim publisher DOI: <https://doi.org/10.1016/j.eswa.2026.131906>
- Park & Shim official publisher page and abstract: <https://www.sciencedirect.com/science/article/pii/S0957417426008195>
- Park & Shim official repository: <https://github.com/bae301764/plugin-mtsad>
- LatentFlow ACM DOI: <https://doi.org/10.1145/3770855.3818034>
- LatentFlow official repository: <https://github.com/SunboTax/LatentFlow>
- MAD-TCN publisher page: <https://www.sciencedirect.com/science/article/pii/S0925231226003516>
- MAD-TCN official repository: <https://github.com/qianmo2001/MAD-TCN>
- CCM-TAD official arXiv paper: <https://arxiv.org/abs/2506.00188>
- CCM-TAD official repository: <https://github.com/Secure-and-Intelligent-Systems-Lab/CCM-TAD>

The machine-readable CSV files contain one source-trace record per requested metric, including paper title, table number, row, column, page, context, evidence status, and comparability level. `restricted_source_recovery_log.csv` records every final-access attempt and the reason for each remaining failure. Because Park & Shim absolute Table 3 scores and MAD-TCN's exact SWaT table remain unavailable from an allowed source, E3 is permanently closed as `PARTIAL_FINAL`; future work must not substitute secondary-source numbers.
