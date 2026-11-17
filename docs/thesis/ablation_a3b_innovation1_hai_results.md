# Ablation-ABL3B：Innovation-1 HAI21 REC 结构消融

## 状态与范围

- 状态：`ABLATION_A3_INNOVATION1_COMPLETED`
- 证据类别：`RETROSPECTIVE_ABLATION`
- 数据：HAI 21.03；3 个 Normal Train 文件、5 个 Official Test 文件、79 variables
- 固定协议：每个 Normal 文件前 80% 为 Train、后 20% 为 Normal Validation；window 64、stride 1；禁止跨文件构窗
- R0 只读复用；R1 与 R2 完全沿用 ABL3A 中已经冻结的结构定义，仅输入/输出通道数由 51 变为 79

本轮没有重训 R0，没有新增 variant，没有超参数或架构搜索，没有 Test 调优、Point Adjustment 或 Oracle。

## 结构与参数预检

| Variant | 唯一结构差异 | HAI Params | 参数审计 |
|---|---|---:|---|
| R0 Final | frozen depthwise-separable `k=3` reference | 15,482 | PASS |
| R1 | 仅三处 depthwise temporal kernel 从 `k=3` 改为 `k=1` | 15,132 | PASS |
| R2 | 仅将三个 depthwise-separable block 替换为同宽 dense `k=3` Conv-BN-GELU-Dropout | 32,911 | PASS |

三个版本均保留相同的 hidden/latent width、BN、GELU、dropout、output projection、full-window SmoothL1 loss、Adam、学习率、weight decay、batch、最大 epoch、patience、seed 与 Validation-only checkpoint rule。参数审计确认归一化结构签名与 SWaT R1/R2 完全一致，唯一数据集差异是 `C=79`。

## Scaler 证据说明

HAI 正式 R0 的可追溯实现与 SWaT S1 路径不同。HAI R0 使用 `sklearn.preprocessing.StandardScaler.partial_fit`，只在三个 Normal Train prefix 上以 float64 统计量拟合；每个窗口进入模型前转为 float32。E2-R1D 的 HAI Ours-REC reconstructed scaler 也明确记录该历史路径。R1/R2 精确复用了这个 HAI R0 路径，没有套用 LSTM-AE/GDN 的训练流程，也没有把 SWaT 自定义 float32 scaler 错套到 HAI。

## 训练结果

| Variant | Stop | Best epoch | Best Validation loss | Training time (s) | q99 threshold |
|---|---|---:|---:|---:|---:|
| R0 Final | existing frozen artifact | 3 | 0.0314837890 | N/A | 0.3079403102 |
| R1 | early stopping at epoch 7 | 3 | 0.0280056320 | 238.252 | 0.2469502498 |
| R2 | early stopping at epoch 7 | 3 | 0.0372072326 | 221.366 | 0.3146875134 |

R1 与 R2 严格串行运行。二者 checkpoint 均只由 Normal Validation reconstruction loss 选择；q99 阈值只由各自 Normal Validation endpoint MAE 冻结，之后才读取 Test。

## HAI point-wise 结果

| Variant | Params | AUROC | AUPR | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 Final | 15,482 | 0.852899 | 0.536294 | 0.459109 | 0.601095 | 0.520594 | 0.016133 |
| R1 | 15,132 | 0.872754 | 0.564453 | 0.458035 | 0.644127 | 0.535371 | 0.017362 |
| R2 | 32,911 | 0.844836 | 0.508042 | 0.437074 | 0.573712 | 0.496158 | 0.016833 |

相对 R0：

| Variant | ΔParams | ΔAUROC | ΔAUPR | ΔF1 | ΔFPR |
|---|---:|---:|---:|---:|---:|
| R1 | -350 | +0.019855 | +0.028159 | +0.014777 | +0.001230 |
| R2 | +17,429 | -0.008064 | -0.028252 | -0.024437 | +0.000700 |

## 50 个 official events

| Variant | Hit ≥0.10 | HIGH19 median hit | HIGH19 hit ≥0.10 | DIRECT31 median hit |
|---|---:|---:|---:|---:|
| R0 Final | 45/50 | 0.650190 | 18/19 | 0.558824 |
| R1 | 47/50 | 0.687259 | 19/19 | 0.764706 |
| R2 | 45/50 | 0.632184 | 18/19 | 0.558824 |

事件身份、边界、HIGH19/DIRECT31 分类和 hit≥0.10 判据均沿用正式 HAI 语义，未进行 Point Adjustment。

## 效率

| Variant | Params | State bytes | Mean latency (ms) | Median (ms) | P95 (ms) | Peak VRAM (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| R0 Final | 15,482 | 77,981 | 0.786008 | 0.678550 | 1.271355 | 19.7090 |
| R1 | 15,132 | 74,869 | 0.604107 | 0.533450 | 0.922420 | 27.2822 |
| R2 | 32,911 | 139,765 | 0.595262 | 0.457300 | 1.314080 | 27.6064 |

延迟均按 batch=1、100 warm-up、1,000 个合法 endpoint 在同一 RTX 4060 上测量。单次 benchmark 的 latency 与 peak VRAM 受运行时分配和内核状态影响，因此参数量、state size 是结构性证据；latency/VRAM 作为同机观测值报告，不外推为硬件无关结论。

## 科学结论

HAI 与 SWaT 的 AUPR 方向一致：R1 提升，R2 下降，故跨数据集方向状态为 `CONSISTENT_DIRECTION`。R1 在两个数据集上都满足 params≤R0 且 AUROC/AUPR≥R0，必须记录为 `R1_DOMINATES_R0_ON_OBSERVED_RANKING_METRICS`。这是 single-seed retrospective ablation，不能据此修改或重命名已冻结的主方法。

R2 在 SWaT 与 HAI 均把参数增加到 R0 的约 2.1 倍，却没有稳定提升 ranking；状态为 `DENSIFICATION_NO_CONSISTENT_GAIN`。结果支持 depthwise-separable factorization 的轻量化价值，但不支持把 R0 的 `k=3` temporal component 宣称为必要性能来源。

## 可追溯性与测试

R1/R2 分别保存 best model、完整 latest resume state、checkpoint SHA256、training history、scaler statistics、feature order、architecture manifest、Validation raw score、Test raw score、labels、file/raw-index/timestamp、q99、metrics、50-event rows 与 efficiency。R0 checkpoint SHA256 在运行前后保持不变。

专项测试覆盖任务列出的 12 项门禁，结果为 `12 passed`。

