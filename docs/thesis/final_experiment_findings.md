# Final experiment findings and method freeze

冻结日期：2026-09-23  
状态：`THESIS_PRE_ABLATION_STATE_FROZEN`

## Authoritative freeze

- **Ours：`FROZEN`**。
- Unified reliability gate：`FROZEN`。
- Self View / Cross View lag definitions：`FROZEN`。
- Ridge `alpha`：`FROZEN`。
- Dual score：`FROZEN`。
- Tail calibration：`FROZEN`。
- Primary datasets：SWaT 2015 Official Physical、HAI 21.03。
- WADI：`DOMAIN_SHIFT_STRESS_TEST`，不得提升为主性能证据。
- 下一阶段只允许消融冻结组件；不得借消融重新定义主方法、重选 lag/alpha/score，或根据 Test 结果改变主协议。

本文件与 `outputs/thesis_registry/registry_manifest.json` 是 E3-FINAL 之后的冻结入口。早期阶段文档继续作为历史证据，不得以其当时的“未冻结”状态覆盖本次冻结。

## Final findings

### A. SWaT reconstruction ranking is strong, but threshold transfer is poor

Ours-REC 在 SWaT Historical Strict 上的 AUROC/AUPR 为 `0.834887/0.743353`，在当前严格表中具有强 score ranking。可是 Validation q99 转移到 Test 后，F1 仅 `0.258067`、FPR 为 `0.718133`。因此可支持的结论是：raw-score 排序有效，但 Normal Validation 到 Test 的 operating-point transfer 很差；不能把低 q99 F1 解释为 ranking 完全失效，也不能隐藏高 FPR。

### B. Oracle and PA-Oracle show that ranking has not collapsed

在 common endpoint 上，Ours-REC 的 `F1-Oracle=0.768959`，`PA-F1-Oracle=0.887275`，明显高于严格 q99 F1。它们说明 score 中仍有可分信息，同时暴露阈值迁移和区间评价的影响。Oracle 使用 Test-best threshold，PA-Oracle 还使用经典 Point Adjustment；二者只允许作为回顾性诊断，不是部署性能或主结果。

### C. HAI supplies stable positive collaboration evidence

HAI21 上 Ours-DUAL 的 Historical Strict `AUROC=0.916334`、`AUPR=0.570193`、`F1-q99=0.610556`、`FPR-q99=0.012443`。相对 REC，DUAL 在该独立真实数据集上提供稳定的正向跨通道协作证据。该结论限定于冻结的 Self/Cross 定义与 HAI21 协议，不推广为“所有数据集均提升”。

### D. HAI tail calibration improves complementary evidence

HAI21 的 Ours-TAIL 获得 `AUPR=0.617897`、`F1-q99=0.617350`、`F1-Oracle=0.686865`、`PA-F1-Oracle=0.919477`。这支持 tail calibration 对 HAI score tail 与补充证据的价值。SWaT 上 TAIL 的 AUROC/AUPR 明显低于 REC，因此不允许写成跨数据集一致优势。

### E. Efficiency claims must be scoped

Ours-FULL 的可训练参数为 SWaT `12,501`、HAI21 `22,520`；对应本地 batch=1 平均延迟约 `1.156 ms` 与 `2.729 ms`。GDN 只有 `8,001/9,793` 个参数，因此不能声称 Ours 对所有强基线都“参数更少”或“普遍更轻”。可以报告 Ours 相对 USAD 的巨大规模差异，以及 REC/DUAL/FULL 各组件的实测时间；TAIL 的单独 latency/VRAM 未测得，不估算。

### F. WADI is a domain-shift stress test only

WADI.A2 的结果用于暴露 Normal-domain shift、缺失值处理和跨数据集迁移风险。它不进入 SWaT/HAI 主性能结论，不用于重选 reliability gate、lag、Ridge alpha、Dual score 或 Tail calibration。

### G. No 2026 external method is directly comparable

E3 冻结的 47 条外部记录中，`DIRECT=0`。Park & Shim 只恢复 gain 摘要；LatentFlow 使用 Affiliation/VUS；MAD-TCN 的 SWaT 性能未提取且官方代码不完整；CCM-TAD 有不同预处理、Test-driven selection 及多种非普通 point-wise 指标。所有并列只能标为 `PARTIAL` 或 `NOT_DIRECT`，不得据此宣称“超过所有 2026 方法”。

## What the thesis may claim

- 冻结的双视角方法在 HAI21 提供了正向、独立的协作证据。
- SWaT reconstruction score 有较强排序，但严格 q99 operating point 存在严重阈值迁移问题。
- Tail calibration 在 HAI21 的 AUPR 和补充 Oracle/PA-Oracle 诊断中表现突出。
- 方法具有可解释的 REC、Self/Cross DUAL 和非可训练 TAIL 组件，并有完整效率分解。

## Claims that remain prohibited

- 不得声称 SOTA。
- 不得声称在所有数据集都优于所有方法。
- 不得把 Test-best Oracle 当作可部署结果。
- 不得把 PA 后结果当作无 PA 的 point-wise F1。
- 不得把 WADI stress-test 结果写成主泛化结论。
- 不得把 Affiliation-F1、Range-F1、VUS、CCM-TAD SF1 换算成普通 F1/AUROC/AUPR。

## Next gate: ablation only

下一阶段应以已冻结主方法为对照，只做预注册消融：移除或隔离 REC、DUAL、Tail calibration、reliability gate 或固定视角。任何会改变主方法定义、Test protocol、阈值来源或已冻结超参数的实验，都不是消融，必须另立研究分支和新注册表项。

