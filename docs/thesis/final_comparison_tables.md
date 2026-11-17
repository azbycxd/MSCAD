# Final comparison tables

冻结日期：2026-09-23  
状态：`THESIS_PRE_ABLATION_STATE_FROZEN`

本文件固定四层评价语义。四层之间不得用同一列名混写，也不得用补充诊断覆盖严格主结果。

1. **Historical Strict**：论文主表；原始合法 endpoint、Normal Validation q99、无 Test tuning、无 PA。
2. **Common-mask Strict**：E2-R2 的同一 endpoint identity 交集；阈值仍为原冻结 Validation q99。
3. **Literature-compatible**：回顾性补充诊断；包含 Test-best Oracle 和经典 Point Adjustment。
4. **Paper-reported**：外部论文第一方报告值；保留其原指标语义，不做换算。

机器可读表：

- `outputs/thesis_tables/table_1_strict_baselines.csv`
- `outputs/thesis_tables/table_2_literature_compatible.csv`
- `outputs/thesis_tables/table_3_efficiency.csv`
- `outputs/thesis_tables/table_4_recent_2026_literature.csv`

## Table 1. Strict Unified Baseline

来源统一为 `E2 Historical Strict`。本表是 SWaT 与 HAI21 的主性能表；不得被 common-mask、Oracle 或 PA 数值覆盖。

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

## Table 2. E2-R2 common-mask literature-compatible diagnostics

| Dataset | Method | AUROC-common | AUPR-common | F1-q99-common | F1-Oracle | PA-F1@q99 | PA-F1-Oracle | FPR-q99 | FPR-Oracle |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SWaT | LSTM-AE | 0.818242 | 0.706043 | 0.254495 | 0.762499 | 0.266571 | 0.884087 | 0.726609 | 0.002151 |
| SWaT | USAD | 0.803816 | 0.698928 | 0.255184 | 0.741845 | 0.269202 | 0.805217 | 0.716859 | 0.000617 |
| SWaT | GDN | 0.827665 | 0.728194 | 0.720216 | 0.772086 | 0.870736 | 0.900728 | 0.023658 | 0.001258 |
| SWaT | Ours-REC | 0.834887 | 0.743353 | 0.258068 | 0.768959 | 0.272041 | 0.887275 | 0.718134 | 0.002202 |
| SWaT | Ours-DUAL | 0.675632 | 0.167511 | 0.015663 | 0.406974 | 0.187814 | 0.809482 | 0.012455 | 0.280173 |
| SWaT | Ours-TAIL | 0.708374 | 0.206747 | 0.260790 | 0.339804 | 0.274334 | 0.398977 | 0.692532 | 0.369269 |
| HAI21 | LSTM-AE | 0.907084 | 0.577510 | 0.512393 | 0.637348 | 0.644794 | 0.883002 | 0.024441 | 0.007672 |
| HAI21 | USAD | 0.856873 | 0.494578 | 0.399760 | 0.579345 | 0.572777 | 0.873148 | 0.031484 | 0.004960 |
| HAI21 | GDN | 0.877643 | 0.497775 | 0.562053 | 0.590018 | 0.762172 | 0.840044 | 0.011529 | 0.006602 |
| HAI21 | Ours-REC | 0.852899 | 0.536294 | 0.520594 | 0.639274 | 0.730542 | 0.908513 | 0.016133 | 0.003190 |
| HAI21 | Ours-DUAL | 0.916334 | 0.570194 | 0.610556 | 0.654843 | 0.785479 | 0.842294 | 0.012443 | 0.005897 |
| HAI21 | Ours-TAIL | 0.901850 | 0.617897 | 0.617350 | 0.686865 | 0.768181 | 0.919477 | 0.012627 | 0.005194 |

`F1-Oracle` 是 Test-best threshold 的诊断值。`PA-F1@q99` 与 `PA-F1-Oracle` 使用经典 Point Adjustment。三者均为回顾性、文献兼容的补充评价，不代表可部署阈值性能。

## Table 3. Efficiency

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

`Params` 只统计可训练参数。TAIL 是非可训练校准状态，故参数为 0；其存储量保留在 State bytes 中。没有独立测得的 TAIL latency/VRAM 均以 `—` 表示，没有估算。GDN 参数量小于 Ours-FULL，因此论文不得声称 Ours 在所有比较下“参数更少”或“普遍更轻”。

## Table 4. Recent 2026 literature

完整的 49 行逐指标表见 `table_4_recent_2026_literature.csv`：47 条冻结外部记录和 2 条仅在指标语义对应时加入的本地 Ours-TAIL 参照。外部 `DIRECT` 记录数为 **0**。

| Method | Dataset | Frozen reported evidence | Comparability |
|---|---|---|---|
| Park & Shim | SWaT | F1 gain `+0.028–+0.050`; absolute Table 3 values `NOT_EXTRACTED` | `NOT_DIRECT` for gain; unresolved absolute rows remain `PARTIAL` |
| Park & Shim | WADI | maximum F1 gain `+0.417`; absolute values `NOT_EXTRACTED` | `NOT_DIRECT` |
| LatentFlow | SWaT | Aff-P `65.01±0.74`, Aff-R `93.59±2.28`, Aff-F1 `76.70±0.50`, VUS-ROC `78.47`, VUS-PR `59.45` | `NOT_DIRECT` |
| LatentFlow | WADI | Aff-P `63.72±0.30`, Aff-R `95.71±0.73`, Aff-F1 `76.51±0.13`, VUS-ROC `47.69`, VUS-PR `6.15` | `NOT_DIRECT` |
| MAD-TCN | SWaT | AUC/F1/Precision/Recall `NOT_EXTRACTED` | `PARTIAL`; protocol incomplete |
| MAD-TCN | cross-dataset claim | parameters `≈0.026 M`; training/inference speed improvement `≈34%` | `NOT_DIRECT`; publisher narrative only |
| CCM-TAD | SWaT/WADI | 20 frozen values covering PF1, SF1, PR AUC, VUS, range/affiliation and five-seed results | `PARTIAL` or `NOT_DIRECT`; no metric conversion |

CCM-TAD 的 PF1、SF1、PR AUC、VUS、Range-F1 与 Affiliation-F1 保持不同语义；Test-driven checkpoint/threshold/alpha 行为原样登记。LatentFlow 的 Affiliation 与 VUS 值不转换为普通 point-wise F1/AUROC/AUPR。Park 与 MAD 的 `NOT_EXTRACTED` 单元格未被填充或反推。

## Consistency gate

- Table 1 与 E2 Historical Strict 原结果一致。
- Table 2 与 E2-R2 common-mask 和 Oracle/PA 产物一致。
- Historical Strict、Common-mask Strict、Literature-compatible、Paper-reported 四层没有合并成同一“主表”。
- 本轮未训练、未推理、未修改方法、未 Test tuning、未执行 PA 作为主评价。

