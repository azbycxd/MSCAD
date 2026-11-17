# 最终消融结论与论文措辞冻结

## 轻量化重构分支的结构与效率分析

轻量化时序重构检测分支（Lightweight Temporal Reconstruction Branch）的消融结果表明，其贡献应定位为面向性能—效率权衡的轻量化设计，而非某一固定卷积核宽度的结构最优性。冻结主方法 R0 在 SWaT 与 HAI 21.03 上均保持万级参数规模；相较之下，将深度可分离卷积替换为同宽稠密卷积的 R2，其参数量分别达到 R0 的约 2.19 倍与 2.13 倍，但 AUROC 与 AUPR 均未获得稳定提升，说明深度可分离分解具有明确的参数效率价值。与此同时，仅将三处时间卷积核由 \(k=3\) 改为 \(k=1\) 的 R1 在两个数据集上均以更少参数获得更高 AUROC 与 AUPR，因而现有证据不支持将 R0 或 \(k=3\) 描述为最优结构。R1 属于观察测试结果后的单种子回顾性发现，不替换已经冻结并用于正式实验的 R0 主方法。

## 滞后双视角协同检测机制消融分析

局部滞后双视角协同一致性检测（Local Lagged Dual-View Collaboration Consistency Detection）通过 Self View、Cross View、局部滞后关系、Normal-only Reliability Gate 与预测分歧共同刻画通道协同异常。Historical CoMuT 的 D0 对合成关系异常表现出较强敏感性，Controlled SWaT 的 D0 也能有效识别冻结的关系破坏；当移除较长滞后、仅保留短滞后 D1 时，HAI 真实攻击上的 AUPR 与 F1 明显低于 D0，说明较长滞后对该数据中的真实协同异常具有贡献。另一方面，SWaT 真实攻击上的 D1 point-wise ranking 提升，而事件命中仍为 2/36，表明滞后贡献具有数据集依赖性，排序改善也不必然转化为事件覆盖改善。Controlled SWaT 中 D2 No-Gate 明显退化，支持 Reliability Gate 的主要作用是以正常数据约束候选可靠性，避免不可靠 target 经 max aggregation 污染协同异常证据；该 Gate 不是基于测试性能的优化器。

## 异构异常证据尾概率校准消融分析

基于正常尾概率校准的异构双证据融合（Normal-Tail Calibrated Heterogeneous Dual-Evidence Fusion）旨在改善 REC 与 DUAL 两类异构分数的证据可比性。历史 Robust-Z 方案在 SWaT、HAI 与 WADI 的 Normal top-1% 区域均由 DUAL 分支 100% 支配；尾概率校准后，REC/DUAL 贡献比例分别调整为 65.19%/34.81%、59.15%/40.85% 与 51.78%/48.22%，表明两类重尾分数在正常尾部的可比较性得到明显改善。HAI 上 Tail AUPR 为 0.617897，较最佳单分支提高 0.047703；但 SWaT 上 Tail AUPR 仅为 0.206747，未保持 REC 的 0.743353 排序优势。因此，该贡献应表述为正常尾概率校准对异构证据尺度与互补表达的改善，而不能表述为融合在所有数据集上普遍提升检测性能。

## 整体讨论与适用边界

三组消融共同界定了方法的有效范围：第一项贡献提供参数高效的重构参考，但不宣称冻结架构或固定卷积核宽度最优；第二项贡献说明双视角预测分歧能够响应关系破坏，长短滞后与 Reliability Gate 的作用随数据语义、事件结构和聚合方式而变化；第三项贡献解决的是异构异常证据的正常尾部可比性，并不保证校准后的融合排序在每个数据集上均优于最强单分支。所有结论均基于冻结协议下的回顾性消融，不涉及测试驱动调参，也不反向修改正式方法或既有主实验结论。

## 贡献层级冻结

论文正式采用“两个核心方法创新 + 一个轻量化设计贡献”的结构。ABL1/ABL2/ABL3 报告中的 Innovation-1/2/3 仅作为历史实验编号保留，不再作为论文贡献顺序。

| 层级 | 中文名称 | 英文名称 |
|---|---|---|
| 核心方法创新 1 | 局部滞后双视角协同一致性检测方法 | Local Lagged Dual-View Collaboration Consistency Detection |
| 核心方法创新 2 | 基于正常尾概率校准的异构双证据融合方法 | Normal-Tail Calibrated Heterogeneous Dual-Evidence Fusion |
| 轻量化设计贡献 | 轻量化时序重构与协同检测框架 | Lightweight Temporal Reconstruction and Collaboration Detection Framework |
