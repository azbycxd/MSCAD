# Ablation-A3：Innovation-1 最终消融发现

## 证据边界

本结论只合并 ABL3A SWaT 与 ABL3B HAI21 的 single-seed `RETROSPECTIVE_ABLATION`。主方法、正式结果、数据协议和阈值协议已冻结；消融结果不反向改写 Final 方法，也不构成新的 Test-driven architecture selection。

## Table Abl-1

| Dataset | Variant | Params | AUROC | AUPR | F1 | FPR | Mean latency (ms) | P95 latency (ms) | Peak VRAM (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SWaT | R0 Final | 11,730 | 0.834887 | 0.743353 | 0.258067 | 0.718133 | 0.722302 | 1.054905 | 12.8730 |
| SWaT | R1 | 11,436 | 0.862082 | 0.775608 | 0.268399 | 0.665935 | 0.886306 | 1.510880 | 17.4668 |
| SWaT | R2 | 25,715 | 0.831320 | 0.735137 | 0.258918 | 0.704556 | 0.711675 | 1.223740 | 17.1914 |
| HAI21 | R0 Final | 15,482 | 0.852899 | 0.536294 | 0.520594 | 0.016133 | 0.786008 | 1.271355 | 19.7090 |
| HAI21 | R1 | 15,132 | 0.872754 | 0.564453 | 0.535371 | 0.017362 | 0.604107 | 0.922420 | 27.2822 |
| HAI21 | R2 | 32,911 | 0.844836 | 0.508042 | 0.496158 | 0.016833 | 0.595262 | 1.314080 | 27.6064 |

## 冻结事实字段

| Fact | Value |
|---|---:|
| R1 vs R0 ΔAUPR SWaT | +0.032255 |
| R1 vs R0 ΔAUPR HAI21 | +0.028159 |
| R2 vs R0 ΔAUPR SWaT | -0.008216 |
| R2 vs R0 ΔAUPR HAI21 | -0.028252 |
| R1 Params/R0 SWaT | 0.974936 |
| R1 Params/R0 HAI21 | 0.977393 |
| R2 Params/R0 SWaT | 2.192242 |
| R2 Params/R0 HAI21 | 2.125759 |
| R1 mean latency/R0 SWaT | 1.227058 |
| R1 mean latency/R0 HAI21 | 0.768575 |
| R2 mean latency/R0 SWaT | 0.985287 |
| R2 mean latency/R0 HAI21 | 0.757323 |

跨数据集 AUPR 方向为 `CONSISTENT_DIRECTION`。参数比率是结构性事实；单次 latency 比率在两个数据集上方向不完全一致，因此不把 latency 当作 R1/R2 优越性的唯一依据。

## A. 减少 R0 temporal component 后如何变化？

R1 把三处 depthwise temporal kernel 从 `k=3` 改为 `k=1`，其他条件不变。它在 SWaT 减少 294 个参数，同时 AUROC/AUPR/F1 分别增加 0.027195/0.032255/0.010331；在 HAI21 减少 350 个参数，同时 AUROC/AUPR/F1 分别增加 0.019855/0.028159/0.014777。HAI event hit 由 45/50 增至 47/50，SWaT event hit 由 29/36 降为 28/36。

因此必须记录：`R1_DOMINATES_R0_ON_OBSERVED_RANKING_METRICS`。但 event 指标并非在两个数据集上全面单调改善，且这是已观察 Test 后的单 seed 消融，不能把 R1 重命名为 Final 或重跑主实验。

## B. Dense variant 是否稳定增加性能？

否。R2 把 SWaT/HAI 参数分别增至 R0 的 2.192×/2.126×，但 AUROC 与 AUPR 在两个数据集上都下降。F1 仅在 SWaT 上增加 0.000851，在 HAI 上下降 0.024437；事件命中也没有超过 R0。结论为 `DENSIFICATION_NO_CONSISTENT_GAIN`。

这支持 depthwise-separable factorization 作为轻量化手段：相较 dense 同宽卷积，它用显著更少参数保持或提高了本组观测性能。该结论不等于 R0 的全部结构均为最优。

## C. R0 能否声称为 best architecture？

**NO。** 冻结的 R1 在两个数据集上都以更少参数获得更高 AUROC 和 AUPR。现有证据明确不支持“R0 是最佳架构”。正确表述是：R0 是正式实验中冻结并使用的 reference architecture；内部消融发现存在更简单、在观测 ranking 指标上更好的 R1，但由于研究状态和主实验冻结，未将其提升为 Final 方法。

## D. Innovation-1 应表述为什么？

现有证据更支持 **lightweight performance-efficiency design**，不支持把它表述为已由消融验证的独立结构创新。

原因有两点：

1. depthwise-separable factorization 相对 dense R2 显著减少参数，而 R2 没有稳定性能收益，说明轻量化分解有明确性能—效率价值；
2. 去除 `k=3` temporal component 的 R1 反而在两个数据集的 AUROC/AUPR 上都提升，说明现有消融不能把该 temporal component 作为必要或有益的结构创新证据。

论文中可将 Innovation-1 定位为冻结 REC 分支的轻量化设计与工程取舍，并诚实报告 R1 的 retrospective finding。不得据此声称每个组件都“必要”、R0“最优”或该结论已完成多 seed 稳健性验证。

## 最终状态

- `ABLATION_A3_INNOVATION1_COMPLETED`
- `CONSISTENT_DIRECTION`
- `R1_DOMINATES_R0_ON_OBSERVED_RANKING_METRICS`
- `DENSIFICATION_NO_CONSISTENT_GAIN`
- NO R0 RETRAIN
- NO NEW REC VARIANT
- NO HP/ARCHITECTURE SEARCH
- NO TEST TUNING
- NO PA / NO ORACLE
- NO METHOD CHANGE
- NO COMMIT / NO PUSH
