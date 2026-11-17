# Paper table layout draft

Date: 2026-09-23  
Status: pre-ablation layout; strict existing values are frozen, new ablation cells are `TBD`

Formatting rule: Historical Strict, literature-compatible diagnostics, and paper-reported external results must remain separate tables. `TBD` means the frozen ablation has not been executed; it is not zero and must never be imputed.

## Table 1. Strict baselines

Primary protocol: original legal endpoints, Normal Validation q99, no PA, no Test-best threshold.

| Dataset | Method | AUROC | AUPR | Precision | Recall | F1-q99 | FPR-q99 |
|---|---|---:|---:|---:|---:|---:|---:|
| SWaT | LSTM-AE | 0.818242 | 0.706043 | 0.147874 | 0.912250 | 0.254495 | 0.726609 |
| SWaT | USAD | 0.803816 | 0.698928 | 0.148539 | 0.904762 | 0.255184 | 0.716859 |
| SWaT | GDN | 0.827665 | 0.728194 | 0.793843 | 0.659087 | 0.720216 | 0.023658 |
| SWaT | Ours-REC | 0.834887 | 0.743353 | 0.150140 | 0.917870 | 0.258067 | 0.718133 |
| SWaT | Ours-DUAL | 0.675631 | 0.167510 | 0.087166 | 0.008605 | 0.015663 | 0.012455 |
| SWaT | Ours-TAIL | 0.708375 | 0.206747 | 0.152452 | 0.901228 | 0.260789 | 0.692531 |
| HAI21 | LSTM-AE | 0.907084 | 0.577510 | 0.399575 | 0.713982 | 0.512393 | 0.024441 |
| HAI21 | USAD | 0.856873 | 0.494578 | 0.300978 | 0.595060 | 0.399760 | 0.031484 |
| HAI21 | GDN | 0.877643 | 0.497775 | 0.537723 | 0.588689 | 0.562053 | 0.011529 |
| HAI21 | Ours-REC | 0.852899 | 0.536294 | 0.459109 | 0.601095 | 0.520594 | 0.016133 |
| HAI21 | Ours-DUAL | 0.916334 | 0.570193 | 0.554350 | 0.679446 | 0.610556 | 0.012443 |
| HAI21 | Ours-TAIL | 0.901850 | 0.617897 | 0.555963 | 0.693976 | 0.617350 | 0.012627 |

Caption note: F1 is the frozen q99 operating point; AUROC/AUPR are the main ranking metrics. SWaT exposes a boundary for DUAL/TAIL rather than positive main-method evidence.

## Table 2. Efficiency

| Dataset | Method | Params | State bytes | Mean ms | Median ms | P95 ms | Peak VRAM MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| SWaT | LSTM-AE | 66,547 | 269,357 | 0.983 | 0.537 | 2.975 | 38.295 |
| SWaT | USAD | 35,663,152 | 142,659,821 | 1.418 | 1.274 | 2.273 | 210.748 |
| SWaT | GDN | 8,001 | 40,050 | 0.747 | 0.681 | 1.211 | 29.211 |
| SWaT | Ours-REC | 11,730 | 63,981 | 0.722 | 0.641 | 1.055 | 12.873 |
| SWaT | Ours-DUAL | 771 | 68,436 | 0.232 | 0.211 | 0.356 | — |
| SWaT | Ours-FULL | 12,501 | 925,219 | 1.156 | 1.048 | 1.706 | 12.873 |
| SWaT | Ours-TAIL | 0 | 792,802 | — | — | — | — |
| HAI21 | LSTM-AE | 75,535 | 305,315 | 0.948 | 0.498 | 2.960 | 44.885 |
| HAI21 | USAD | 72,227,568 | 288,917,461 | 12.384 | 12.708 | 14.191 | 403.276 |
| HAI21 | GDN | 9,793 | 47,193 | 0.738 | 0.696 | 0.967 | 36.221 |
| HAI21 | Ours-REC | 15,482 | 77,981 | 0.786 | 0.679 | 1.271 | 19.709 |
| HAI21 | Ours-DUAL | 7,038 | 594,324 | 1.404 | 1.343 | 1.772 | — |
| HAI21 | Ours-FULL | 22,520 | 2,147,163 | 2.729 | 2.644 | 3.431 | 19.709 |
| HAI21 | Ours-TAIL | 0 | 1,474,858 | — | — | — | — |

Caption note: Params count trainable parameters only. TAIL is non-trainable calibration state. CPU/GPU component timing semantics must remain in the footnote. Do not claim Ours-FULL has fewer parameters than GDN.

## Table 3. Dual ablation

Primary panel: HAI21. Secondary boundary panel: SWaT. All rows use the strict q99 protocol.

| Dataset | Variant | Self lags | Cross lags | Reliability Gate | AUROC | AUPR | F1-q99 | HIGH19 median hit | FPR-q99 |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| HAI21 | D0 Final Dual | 1,5,10,20,30 | 0,1,5,10,20,30 | Unified Final | 0.916334 | 0.570193 | 0.610556 | 0.841085 | 0.012443 |
| HAI21 | D1 Without long lags | 1 | 0,1 | Unified logic reapplied | TBD | TBD | TBD | TBD | TBD |
| HAI21 | D2 Candidate only | 1,5,10,20,30 | 0,1,5,10,20,30 | None | TBD | TBD | TBD | TBD | TBD |
| HAI21 | D3-B + Discovery skill | 1,5,10,20,30 | 0,1,5,10,20,30 | Discovery | TBD | TBD | TBD | TBD | TBD |
| HAI21 | D3-C + Validation full/halves | 1,5,10,20,30 | 0,1,5,10,20,30 | Discovery + Validation | TBD | TBD | TBD | TBD | TBD |
| SWaT | D0 Final Dual | 1,5,10,20,30 | 0,1,5,10,20,30 | Unified Final | 0.675631 | 0.167510 | 0.015663 | — | 0.012455 |
| SWaT | D1 Without long lags | 1 | 0,1 | Unified logic reapplied | TBD | TBD | TBD | — | TBD |
| SWaT | D2 Candidate only | 1,5,10,20,30 | 0,1,5,10,20,30 | None | TBD | TBD | TBD | — | TBD |
| SWaT | D3-B + Discovery skill | 1,5,10,20,30 | 0,1,5,10,20,30 | Discovery | TBD | TBD | TBD | — | TBD |
| SWaT | D3-C + Validation full/halves | 1,5,10,20,30 | 0,1,5,10,20,30 | Discovery + Validation | TBD | TBD | TBD | — | TBD |

Layout note: D3-D equals D0 and should be shown by a bracket/arrow in the final typeset table rather than duplicated as a second numeric row.

Mechanism subtable:

| Variant | Historical CoMuT AUROC | Historical CoMuT AUPR | SWaT-controlled AUROC | SWaT-controlled AUPR |
|---|---:|---:|---:|---:|
| D0 Final Dual | 0.995989 median | 0.686927 median | 0.996913 median | 0.996931 median |
| D1 Without long lags | TBD | TBD | TBD | TBD |
| D2 Without Reliability Gate | — | — | TBD | TBD |

Caption note: the controlled and CoMuT numbers are mechanism evidence, not the industrial performance leaderboard.

## Table 4. Fusion ablation

| Dataset | Variant | Definition | AUPR | F1-q99 | FPR-q99 | Normal-tail REC share | Normal-tail DUAL share |
|---|---|---|---:|---:|---:|---:|---:|
| SWaT | F0 REC | REC endpoint score | 0.743353 | 0.258067 | 0.718133 | — | — |
| SWaT | F1 DUAL | calibrated disagreement max | 0.167510 | 0.015663 | 0.012455 | — | — |
| SWaT | F2 Simple/raw fusion | no frozen historical definition | — | — | — | — | — |
| SWaT | F3 Robust-z max | `max(Z_REC,Z_DUAL)` | 0.168979 | 0.015664 | 0.012453 | 0.000000 | 1.000000 |
| SWaT | F4 Tail-evidence max | `max(E_REC,E_DUAL)` | 0.206747 | 0.260789 | 0.692531 | 0.651911 | 0.348089 |
| HAI21 | F0 REC | REC endpoint score | 0.536294 | 0.520594 | 0.016133 | — | — |
| HAI21 | F1 DUAL | calibrated disagreement max | 0.570193 | 0.610556 | 0.012443 | — | — |
| HAI21 | F2 Simple/raw fusion | no frozen historical definition | — | — | — | — | — |
| HAI21 | F3 Robust-z max | `max(Z_REC,Z_DUAL)` | 0.576716 | 0.619503 | 0.012471 | 0.000000 | 1.000000 |
| HAI21 | F4 Tail-evidence max | `max(E_REC,E_DUAL)` | 0.617897 | 0.617350 | 0.012627 | 0.591549 | 0.408451 |

Caption note: normal-tail shares come from the frozen Validation Audit-B top 1% and explain branch comparability; they are not Test-selected fusion weights. WADI can appear in a supplementary stress panel only.

## Table 5. Literature-compatible supplementary

This table is retrospective and must not be merged with Table 1 or used for ablation selection.

| Dataset | Method | AUROC-common | AUPR-common | F1-q99-common | F1-Oracle | PA-F1@q99 | PA-F1-Oracle |
|---|---|---:|---:|---:|---:|---:|---:|
| SWaT | LSTM-AE | 0.818242 | 0.706043 | 0.254495 | 0.762499 | 0.266571 | 0.884087 |
| SWaT | USAD | 0.803816 | 0.698928 | 0.255184 | 0.741845 | 0.269202 | 0.805217 |
| SWaT | GDN | 0.827665 | 0.728194 | 0.720216 | 0.772086 | 0.870736 | 0.900728 |
| SWaT | Ours-REC | 0.834887 | 0.743353 | 0.258068 | 0.768959 | 0.272041 | 0.887275 |
| SWaT | Ours-DUAL | 0.675632 | 0.167511 | 0.015663 | 0.406974 | 0.187814 | 0.809482 |
| SWaT | Ours-TAIL | 0.708374 | 0.206747 | 0.260790 | 0.339804 | 0.274334 | 0.398977 |
| HAI21 | LSTM-AE | 0.907084 | 0.577510 | 0.512393 | 0.637348 | 0.644794 | 0.883002 |
| HAI21 | USAD | 0.856873 | 0.494578 | 0.399760 | 0.579345 | 0.572777 | 0.873148 |
| HAI21 | GDN | 0.877643 | 0.497775 | 0.562053 | 0.590018 | 0.762172 | 0.840044 |
| HAI21 | Ours-REC | 0.852899 | 0.536294 | 0.520594 | 0.639274 | 0.730542 | 0.908513 |
| HAI21 | Ours-DUAL | 0.916334 | 0.570194 | 0.610556 | 0.654843 | 0.785479 | 0.842294 |
| HAI21 | Ours-TAIL | 0.901850 | 0.617897 | 0.617350 | 0.686865 | 0.768181 | 0.919477 |

Caption note: F1-Oracle uses a Test-best threshold; PA columns use classic Point Adjustment. All are supplementary diagnostics.

## Table 6. 2026 recent literature protocol comparison

| Method | Dataset | Frozen first-party evidence | Metric/protocol semantics | Comparability |
|---|---|---|---|---|
| Park & Shim | SWaT | F1 gain +0.028–+0.050; absolute Table 3 not extracted | gain over paper baselines; absolute operating point unavailable | NOT_DIRECT/PARTIAL |
| Park & Shim | WADI | maximum F1 gain +0.417; absolute values not extracted | gain only | NOT_DIRECT |
| LatentFlow | SWaT | Aff-P 65.01±0.74; Aff-R 93.59±2.28; Aff-F1 76.70±0.50; VUS-ROC 78.47; VUS-PR 59.45 | Affiliation and VUS, not ordinary point-wise metrics | NOT_DIRECT |
| LatentFlow | WADI | Aff-P 63.72±0.30; Aff-R 95.71±0.73; Aff-F1 76.51±0.13; VUS-ROC 47.69; VUS-PR 6.15 | Affiliation and VUS | NOT_DIRECT |
| MAD-TCN | SWaT | AUC/F1/Precision/Recall not extracted | official code incomplete; exact performance unavailable | PARTIAL |
| MAD-TCN | cross-dataset claim | approximately 0.026M parameters; approximately 34% speed improvement | publisher narrative, not local unified evaluation | NOT_DIRECT |
| CCM-TAD | SWaT/WADI | 20 frozen official-PDF values across PF1, SF1, PR AUC, VUS, range/affiliation, and five-seed results | distinct metric semantics and Test-driven behavior retained | PARTIAL/NOT_DIRECT |

Caption note: no external 2026 row is directly comparable to the thesis strict table. No metric conversion or missing-value inference is permitted.
