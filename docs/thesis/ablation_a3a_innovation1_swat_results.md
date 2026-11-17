# Ablation-ABL3A：Innovation-1 SWaT REC 结构消融

## 状态与范围

- 状态：`ABLATION_A3A_INNOVATION1_SWAT_COMPLETED`
- 证据类别：`RETROSPECTIVE_ABLATION`
- 数据：SWaT 2015 Official Physical，51 variables
- 固定协议：Normal Train `[0,396000)`，Normal Validation `[396000,495000)`，window 64，stride 1；Test 的 82-second gap 为 hard session boundary。
- R0 只读复用。R1、R2 按 `ablation_p0_design.md` 和 `rec_architecture_trace.md` 的冻结定义串行训练。

本轮没有 HAI 训练、R0 重训练、架构搜索、超参数搜索、Test 调优、Point Adjustment 或 Oracle。

## 冻结架构差异

| Variant | Encoder / decoder blocks | Temporal component | Bottleneck | Output | Params |
|---|---|---|---|---|---:|
| R0 Final | 3 个 depthwise-separable block；每块为 depthwise `k=3`、BN、GELU、pointwise `k=1`、BN、GELU、dropout | 三层 `k=3`，effective receptive field 7 | 每个时间点 32 channels | `Conv1d(64,51,k=1,bias=True)` | 11,730 |
| R1 | 与 R0 相同，仅三处 depthwise kernel 改成 `k=1` | 三层 `k=1`，effective receptive field 1 | 同 R0 | 同 R0 | 11,436 |
| R2 | 三个同宽 dense `k=3` Conv-BN-GELU-Dropout block | 三层 dense `k=3`，effective receptive field 7 | 同 R0 | 同 R0 | 25,715 |

三个版本均使用 `[B,T,C] → [B,T,C]`，无 pooling、stride change、dilation、residual 或 skip。Conv1d 参数沿时间共享。

## Scaler dtype 证据澄清

任务文本提到 E2 的 float64 scaler。仓库中的 E2-R1D 报告实际区分了两个路径：LSTM-AE/GDN baseline runner 使用 float64 统计，而正式 S1 Ours-REC checkpoint 使用 float32 scaler 统计。ABL3A 的约束同时要求“除结构外与 R0 完全一致”并沿用正式 REC 实现，因此 R1/R2 使用了与 S1 A0 相同的 float32 source/statistics/model-input 路径。Scaler 只 fit Normal Train；Validation 和 Test 均为 transform-only。该差异没有被静默改写成 float64。

## 检测结果

| Variant | AUROC | AUPR | Precision | Recall | F1 | FPR | Event hit ≥0.10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 Final | 0.834887 | 0.743353 | 0.150140 | 0.917870 | 0.258067 | 0.718133 | 29/36 |
| R1 | 0.862082 | 0.775608 | 0.157662 | 0.901778 | 0.268399 | 0.665935 | 28/36 |
| R2 | 0.831320 | 0.735137 | 0.151021 | 0.906739 | 0.258918 | 0.704556 | 28/36 |

所有阈值均为各自 Normal Validation endpoint MAE 的固定 q99，比较符为 `>`。Test labels 只参与最终评价。

## 参数与效率

| Variant | Params | Ratio vs R0 | State/checkpoint bytes | Mean latency (ms) | Median (ms) | P95 (ms) | Peak VRAM (MiB) | Training time (s) | Best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 Final | 11,730 | 1.000000 | 63,981 | 0.722302 | 0.641150 | 1.054905 | 12.8730 | N/A | 13 |
| R1 | 11,436 | 0.974936 | 60,149 | 0.886306 | 0.715750 | 1.510880 | 17.4668 | 229.604 | 13 |
| R2 | 25,715 | 2.192242 | 111,029 | 0.711675 | 0.574250 | 1.223740 | 17.1914 | 175.288 | 8 |

R0 bytes 是 Efficiency-E1 已有正式 checkpoint footprint；R1/R2 列为本轮单独保存的 `state_dict.pt` bytes。延迟均按同一 RTX 4060、batch=1、100 warm-up、1,000 个合法 Normal endpoints 测量。R0 原训练耗时未持久化，因此保持 N/A。

R1 在参数量减少 294 的同时，mean latency 比 R0 高 0.164004 ms；R2 参数增加 13,985，但本次 mean latency 比 R0 低 0.010627 ms。参数量与单次延迟没有形成单调关系，本轮不把单次计时差异解释为结构优越性。

## 相对 R0 的事实差值

| Variant | ΔAUROC | ΔAUPR | ΔF1 | ΔParams | ΔMean latency (ms) |
|---|---:|---:|---:|---:|---:|
| R1 | +0.027195 | +0.032255 | +0.010331 | -294 | +0.164004 |
| R2 | -0.003567 | -0.008216 | +0.000851 | +13,985 | -0.010627 |

R1 达到 max epoch 15，最佳 Validation checkpoint 为 epoch 13。R2 在 epoch 12 达到 patience 4 并自然 early-stop，最佳 checkpoint 为 epoch 8。以上只描述冻结单 seed retrospective ablation 的观测结果，不使用“组件必不可少”“最优结构”或“全面优于”等结论。

## 可追溯性

每个变体均保存：best model、latest optimizer/resume state、训练历史、scaler、feature order、architecture manifest、Validation raw score、Test raw score、labels/timeline、q99 threshold、36-event evaluation、效率结果与 SHA256 manifest。R0 checkpoint 在本轮前后 SHA256 一致。

12 项专项测试覆盖冻结定义、R0 只读、split、scaler、feature order、window/gap、训练配置、Validation-only checkpoint/q99、raw score persistence 以及无 PA/Oracle/Test tuning。
