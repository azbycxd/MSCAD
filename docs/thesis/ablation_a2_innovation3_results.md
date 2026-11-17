# Ablation-ABL2：Innovation-3 异构证据校准消融

## 审计定位

- 状态：`ABLATION_A2_INNOVATION3_COMPLETED`
- 证据类别：`RETROSPECTIVE_ABLATION`
- 前置门禁：`ABLATION_PROTOCOL_READY`、`ABLATION_A1_INNOVATION2_COMPLETED`
- 主方法、数据划分、模型和评分分支均已冻结；本轮没有训练、Ridge 重拟合、参数搜索、测试集调优、PA、POT 或 oracle。
- Primary 数据集为 SWaT 2015 与 HAI 21.03。WADI.A2 只作为已有正式结果的 domain-shift stress test，不进入主结论。

本轮仅比较预注册的四个版本：

- F0 REC：冻结的 reconstruction branch。
- F1 DUAL：冻结的 dual-view branch。
- F3 Robust-Z：历史方案，分别进行单边 robust-z 后取 `max`。
- F4 Tail-Calibrated：最终方案，在 Normal Validation 的 Calibration-A 上估计经验尾概率，在 Audit-B 上冻结 q99 阈值，之后取两分支 evidence 的 `max`。

不存在 F2，也没有新增 simple、weighted 或 learned fusion。

## 冻结公式

F3 使用历史公式：

```text
z_b(x) = max(0, (S_b(x) - median_b) / scale_b)
S_F3(x) = max(z_REC(x), z_DUAL(x))
```

严格检测指标复用历史完整 Normal Validation 的 `median/scale` 与 q99 阈值；Normal-tail dominance 审计复用冻结的 Calibration-A/Audit-B 划分。

F4 使用最终公式：

```text
p_tail,b(x) = (1 + #{s in Calibration-A_b : s >= x}) / (N_b + 1)
E_b(x) = -log10(p_tail,b(x))
S_F4(x) = max(E_REC(x), E_DUAL(x))
```

ECDF 只来自 Calibration-A；阈值只来自 Audit-B q99。Test 标签不参与校准、阈值或选择。

## 原始分数尺度异质性

| Dataset | Branch | Calibration-A median | q99 | top-1% median |
|---|---:|---:|---:|---:|
| SWaT | REC | 0.050391 | 0.175359 | 0.216906 |
| SWaT | DUAL | 2.818878 | 237519.193548 | 390021.926608 |
| HAI21 | REC | 0.145365 | 0.303158 | 0.328209 |
| HAI21 | DUAL | 4.937569 | 38.646888 | 53.451039 |

SWaT 的 DUAL/REC q99 比为 1,354,474.02，HAI 为 127.48。原始分值并不处于可直接比较的数值尺度。

## Normal-tail 可比性

| Dataset | F3 top-1% REC/DUAL | F4 q99 balance ratio | F4 top-1% REC/DUAL |
|---|---:|---:|---:|
| SWaT | 0% / 100% | 1.044888 | 65.19% / 34.81% |
| HAI21 | 0% / 100% | 1.000280 | 59.15% / 40.85% |
| WADI.A2 | 0% / 100% | 1.035297 | 51.78% / 48.22% |

F3 在三套数据上都由 DUAL 100% 支配，说明按中心和 MAD/IQR 尺度进行线性 robust-z 并没有解决重尾分支之间的极端值可比性。F4 把三套数据的 Normal q99 evidence 比例压到接近 1，并恢复了两个分支在 Normal top-1% 区域的共同贡献。

这支持的是“尾校准改善异构证据可比性”，不是“融合必然改善检测性能”。

## Table Abl-4：Innovation-3 融合消融

| Variant | SWaT AUPR | SWaT F1 | SWaT FPR | HAI AUPR | HAI F1 | HAI FPR | Normal-tail balance |
|---|---:|---:|---:|---:|---:|---:|---|
| F0 REC | 0.743353 | 0.258067 | 0.718133 | 0.536294 | 0.520594 | 0.016133 | Single REC branch |
| F1 DUAL | 0.167510 | 0.015663 | 0.012455 | 0.570193 | 0.610556 | 0.012443 | Single DUAL branch |
| F3 Robust-Z | 0.168979 | 0.015664 | 0.012453 | 0.576716 | 0.619503 | 0.012471 | 三套数据 top-1% 均为 REC/DUAL = 0/100% |
| F4 Tail-Calibrated | 0.206747 | 0.260789 | 0.692531 | 0.617897 | 0.617350 | 0.012627 | SWaT 65/35；HAI 59/41；WADI 52/48 |

## 检测性能与事件表现

HAI21 上，F4 的 AUPR 为 0.617897，高于 F0 的 0.536294、F1 的 0.570193 与 F3 的 0.576716；F1/F3/F4 的 q99 operating-point F1 分别为 0.610556、0.619503、0.617350。F4 命中 45/50 个事件，HIGH19 median hit 为 0.838863，DIRECT31 median hit 为 0.873418。可比性改善与 HAI AUPR 提升同时出现，但本消融不把关联解释为跨数据集普遍因果。

SWaT 上，F4 的 AUPR 为 0.206747，显著低于 F0 REC 的 0.743353；F4 的 FPR 仍为 0.692531，且 28/36 事件命中略低于 F0 的 29/36。尾校准在 Normal 证据尺度上成功，但没有保留 REC 的测试排序优势，Normal-q99 阈值迁移仍不稳定。因此 SWaT 结论只能是 `TAIL_FUSION_SWAT_UNSTABLE`，不能表述为检测改进。

WADI.A2 上，四个版本 AUPR 都约为 0.052–0.055；F4 AUPR 0.054510、F1 0.051955、F3 0.054201。该结果仅说明 domain shift 下证据平衡不等于检测有效，不用于 Primary 性能声明。

## 冻结结论

1. `TAIL_CALIBRATION_BALANCES_NORMAL_EVIDENCE`：F4 在三套数据上恢复了 Normal-tail 证据可比性，解决了 F3 被 DUAL 单分支支配的问题。
2. HAI21 提供了正向检测关联：F4 AUPR 高于两个单分支及 F3，但 q99 F1 与 F3 接近。
3. SWaT 提供了清晰边界：校准成功不保证排序保持或阈值迁移成功；F4 不能替代该数据上的 REC 主分支。
4. Innovation-3 的论文定位是异构分数的 evidence calibration / score comparability，而不是 universal detection improvement。

## 可追溯性与测试

- SWaT/HAI 冻结 REC、DUAL、TAIL 原始测试分数及其 SHA256 已写入 manifest。
- Calibration-A ECDF artifact、历史阈值文件、3A/3B 正式指标与事件文件及其 SHA256 已写入 manifest。
- SWaT/HAI 的 F3 与 F4 fusion score、标签、timeline 已持久化为 NPZ，并记录 SHA256。
- 12 项专项门禁全部通过；所有正式数值与冻结 3A/3B 产物一致。

本轮没有修改主方法，没有训练模型，没有重新拟合 Ridge，没有创建新融合规则，没有使用 Test 调优，也没有提交或推送。
