# Stability-STAB-P1：冻结主方法三随机种子稳定性评价

## 1. 研究问题与冻结边界

本实验只回答：冻结主方法的结果对 REC 随机初始化呈现多大变异。

- 数据集：SWaT 2015 Official Physical、HAI 21.03。
- 方法：R0 Final REC，以及由每个 REC seed 与同一冻结 DUAL 组成的 Final TAIL。
- seeds：两个数据集均为正式 seed `1` 加预注册 seed `42`、`43`。
- REC 架构、优化器、学习率、batch、early stopping、数据划分、scaler 和 q99 协议不变。
- DUAL 在冻结数据、特征和 solver 协议下为确定性分支；本轮直接绑定既有系数、reference、targets 和 raw test score，没有重新拟合 Ridge、重新筛 target 或改变 lag/median/MAD。
- 每个 seed 的 Tail 使用该 REC seed 自身的 Normal REC Calibration-A；DUAL Calibration-A 固定。Threshold-B q99 按 seed 独立冻结，Test 只用于最终评价。
- 本轮未定义“稳定/不稳定”的人为阈值，只报告 sample std、range 与 formal-seed-minus-mean。

前置状态 `THESIS_CONTRIBUTION_STRUCTURE_FROZEN`、`ABLATION_FINAL_PACKAGE_READY`、`FINAL_CLAIMS_DEFENSIBLE` 和 `THESIS_NARRATIVE_STRUCTURE_READY` 均已通过。

## 2. 协议与完整性门禁

正式 seed 1 没有重新训练或覆盖。REC 与 TAIL 的 AUROC、AUPR、Precision、Recall、F1、FPR 均与冻结历史 artifact 完全一致，所有 absolute delta 为 `0`。正式 checkpoint、scaler、metrics、Tail summary，以及冻结 DUAL 的 coefficient、reference、target、calibration 和 raw-score artifact 在运行前后 SHA256 不变。

两个数据集的 R0 协议均为：window=64、stride=1、SmoothL1、Adam、lr=1e-3、weight decay=1e-4、batch=256、max epoch=15、patience=4，由 Normal Validation 最低 loss 选择 checkpoint。没有运行 R1/R2。

## 3. 每 seed 严格结果

### 3.1 SWaT

| Method | Seed | q99 threshold | AUROC | AUPR | Precision | Recall | F1 | FPR | Event hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| REC | 1 formal | 0.183962 | 0.834887 | 0.743353 | 0.150140 | 0.917870 | 0.258067 | 0.718133 | 29/36 |
| REC | 42 | 0.259005 | 0.834754 | 0.746213 | 0.156164 | 0.898226 | 0.266069 | 0.670867 | 28/36 |
| REC | 43 | 0.198652 | 0.819470 | 0.716855 | 0.148710 | 0.917358 | 0.255932 | 0.725854 | 29/36 |
| TAIL | 1 formal | 2.391137 | 0.708375 | 0.206747 | 0.152452 | 0.901228 | 0.260789 | 0.692531 | 28/36 |
| TAIL | 42 | 2.232415 | 0.810378 | 0.448953 | 0.160108 | 0.894711 | 0.271612 | 0.648733 | 28/36 |
| TAIL | 43 | 2.218141 | 0.648998 | 0.168898 | 0.149535 | 0.915655 | 0.257086 | 0.719811 | 29/36 |

SWaT REC 的 ranking 与 q99 operating point 在三个 seed 间呈现较窄范围；TAIL 的 AUROC/AUPR 范围明显更宽，尤其 seed 42 的 AUPR 高于另两个 seed。该观察记为 `SEED_VARIABILITY_OBSERVED`，不用于挑选 seed、替换历史主结果或改变 Tail 定义。

### 3.2 HAI21

| Method | Seed | q99 threshold | AUROC | AUPR | Precision | Recall | F1 | FPR | Events | HIGH19 median / hit | DIRECT31 median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| REC | 1 formal | 0.307940 | 0.852899 | 0.536294 | 0.459109 | 0.601095 | 0.520594 | 0.016133 | 45/50 | 0.650190 / 18 | 0.558824 |
| REC | 42 | 0.280297 | 0.855551 | 0.553537 | 0.453572 | 0.639320 | 0.530661 | 0.017546 | 48/50 | 0.645669 / 19 | 0.742424 |
| REC | 43 | 0.303451 | 0.846247 | 0.513330 | 0.187760 | 0.639879 | 0.290329 | 0.063059 | 46/50 | 0.650980 / 18 | 0.802083 |
| TAIL | 1 formal | 2.382045 | 0.901850 | 0.617897 | 0.555963 | 0.693976 | 0.617350 | 0.012627 | 45/50 | 0.838863 / 19 | 0.873418 |
| TAIL | 42 | 2.256539 | 0.895612 | 0.614072 | 0.523436 | 0.715212 | 0.604478 | 0.014834 | 48/50 | 0.844961 / 19 | 0.893939 |
| TAIL | 43 | 2.287726 | 0.897456 | 0.582540 | 0.292040 | 0.732760 | 0.417633 | 0.040467 | 48/50 | 0.844961 / 19 | 0.908333 |

HAI REC 与 TAIL 的 AUROC/AUPR ranking 变异较小，但 seed 43 的 q99 Precision、F1 与 FPR 明显偏离另外两个 seed。这表明 ranking 稳定程度与 Validation q99 operating-point 迁移稳定程度不能混为一谈。事件层面，TAIL 的 HIGH19 hit count 在三个 seed 中均为 19，HIGH19 median hit 范围为 0.838863–0.844961。

## 4. Table Stability：三 seed mean ± sample std

本表是 supplementary stability table。历史 Primary 表继续使用正式 seed 1，不被均值覆盖。

| Dataset | Method | AUROC mean±std | AUPR mean±std | F1 mean±std | FPR mean±std |
|---|---|---:|---:|---:|---:|
| SWaT | REC | 0.829704 ± 0.008863 | 0.735474 ± 0.016188 | 0.260023 ± 0.005344 | 0.704951 ± 0.029769 |
| SWaT | TAIL | 0.722584 ± 0.081623 | 0.274866 ± 0.151947 | 0.263162 ± 0.007548 | 0.687025 ± 0.035857 |
| HAI21 | REC | 0.851566 ± 0.004793 | 0.534387 ± 0.020171 | 0.447195 ± 0.135943 | 0.032246 ± 0.026694 |
| HAI21 | TAIL | 0.898306 ± 0.003205 | 0.604836 ± 0.019403 | 0.546487 ± 0.111776 | 0.022642 ± 0.015476 |

DUAL 不列为 `mean±0`：它不是三次随机实验，而是在冻结协议下确定的同一分支。

## 5. Range 与 formal-seed-minus-mean

| Dataset | Method | AUROC range | AUPR range | F1 range | FPR range | Formal minus mean: AUROC / AUPR / F1 / FPR |
|---|---|---|---|---|---|---|
| SWaT | REC | 0.819470–0.834887 | 0.716855–0.746213 | 0.255932–0.266069 | 0.670867–0.725854 | +0.005183 / +0.007879 / -0.001956 / +0.013182 |
| SWaT | TAIL | 0.648998–0.810378 | 0.168898–0.448953 | 0.257086–0.271612 | 0.648733–0.719811 | -0.014209 / -0.068119 / -0.002373 / +0.005506 |
| HAI21 | REC | 0.846247–0.855551 | 0.513330–0.553537 | 0.290329–0.530661 | 0.016133–0.063059 | +0.001334 / +0.001907 / +0.073399 / -0.016113 |
| HAI21 | TAIL | 0.895612–0.901850 | 0.582540–0.617897 | 0.417633–0.617350 | 0.012627–0.040467 | +0.003544 / +0.013060 / +0.070863 / -0.010016 |

## 6. 事件汇总

| Dataset | Method | Event hit mean±std | HIGH19 median hit mean±std | HIGH19 hit mean±std | DIRECT31 median hit mean±std |
|---|---|---:|---:|---:|---:|
| SWaT | REC | 28.667 ± 0.577 | — | — | — |
| SWaT | TAIL | 28.333 ± 0.577 | — | — | — |
| HAI21 | REC | 46.333 ± 1.528 | 0.648947 ± 0.002866 | 18.333 ± 0.577 | 0.701110 ± 0.126783 |
| HAI21 | TAIL | 47.000 ± 1.732 | 0.842928 ± 0.003521 | 19.000 ± 0.000 | 0.891897 ± 0.017547 |

## 7. 训练运行时间

正式 seed 未重训，其历史总训练时间未在本轮重新估算，记为 N/A。新增 seed 的实际训练时间：SWaT seed 42/43 为 223.414/218.791 秒；HAI seed 42/43 为 261.253/231.057 秒。效率参数、时延和显存结构不随 seed 改变，继续引用 Efficiency E1。

## 8. 论文使用方式与科学结论

1. Primary historical table 保持正式 seed 1 数值不变。
2. 本文增加本文件的 Table Stability 作为补充稳定性证据，不把三 seed 均值反向覆盖既有主表。
3. REC ranking 在两个数据集的 AUROC/AUPR 上呈现相对有限的 seed spread；HAI 的固定 q99 F1/FPR 对 seed 更敏感。
4. TAIL 在 HAI 的 AUROC/AUPR spread 较小，但 q99 operating point 仍有变异；SWaT TAIL 的 ranking spread 较大。
5. 因存在明显更好或更差的 seed 结果，记录 `SEED_VARIABILITY_OBSERVED`；不据此进行 best-seed selection、架构替换或方法修改。
6. 这些结果不会把 C5 的 HAI 正向互补证据升级为跨数据集普遍结论，也不会改变 C7 的 SWaT 边界。

## 9. 完整性结果

12 项门禁全部通过：恰好三个不同 seeds、正式 artifact 未覆盖、正式指标在 1e-6 内复现、仅 R0、无 R1/R2、Validation-only checkpoint、Validation-only q99、DUAL hash 不变、Tail 公式不变、REC calibration 与 seed 正确绑定、无 PA/Oracle/Test-best、全部 raw scores 持久化。

**Final status：`THESIS_MULTI_SEED_STABILITY_COMPLETED`。**

NO METHOD CHANGE；NO R1/R2；NO RIDGE REFIT；NO TARGET RESELECTION；NO HP SEARCH；NO BEST-SEED SELECTION；NO TEST TUNING；NO PA；NO ORACLE；NO BASELINE RERUN；NO COMMIT；NO PUSH。
