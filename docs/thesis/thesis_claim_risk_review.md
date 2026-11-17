# 论文主张与答辩风险审计

## Q1. R1 比 Final REC 还好，为什么不替换 Final？

**事实基础**

- R1 在 SWaT 与 HAI21 上均满足 Params 小于 R0、AUROC 与 AUPR 高于 R0。
- R1 是在主方法和正式结果冻结后完成的 single-seed retrospective ablation。
- 事件指标并非全面单调改善：SWaT event hit 由 29/36 降为 28/36。

**建议答法**

R0 是预先冻结并用于全部正式比较的 reference architecture。R1 的结果是在观察正式测试结果后获得的回顾性发现，若据此替换主方法并重做主表，会把测试信息引入架构选择并破坏实验冻结边界。因此论文如实报告 R1，使用它修正“固定 kernel 最优”的表述，但不将其提升为新的 Final。

**不能说**

- R0 在总体上仍然优于 R1。
- R1 只是偶然，因此可以忽略。
- 为保持论文叙事，可以不报告 R1。

## Q2. 既然 R1 更好，轻量化还能算贡献吗？

**事实基础**

- Final REC 在 SWaT/HAI21 上为 11,730/15,482 个参数，保持万级规模。
- dense R2 参数为 R0 的约 2.19 倍/2.13 倍，且 AUROC/AUPR 均未提升。
- R1 说明固定 kernel width 没有最优性证据，但不否定 depthwise-separable factorization 的参数效率。

**建议答法**

轻量化贡献来自整体 performance–efficiency design：depthwise-separable REC 控制参数规模，Dual 使用线性 Ridge，Tail 不引入可训练参数，并提供组件级参数、时延、显存与状态审计。消融支持相对 dense 结构的参数效率，但不支持把某个 kernel width 声称为独立最优结构。

**不能说**

- 消融证明 R0 是最优轻量网络。
- k=3 是轻量化贡献成立的必要条件。
- Ours 对所有基线都参数更少或速度更快；GDN 参数更少，部分时延也更低。

## Q3. SWaT 上 DUAL 不好，为什么协同检测仍成立？

**事实基础**

- Historical CoMuT D0 median AUROC 0.995989。
- SWaT controlled D0 median AUROC 0.996913。
- HAI21 DUAL AUROC/AUPR/F1 为 0.916334/0.570193/0.610556。
- SWaT official real attacks DUAL AUROC/AUPR 仅为 0.675631/0.167510。

**建议答法**

论文主张的是方法能够检测关系破坏，并在合成机制、真实轨迹控制实验和独立真实攻击 HAI 上获得分层支持。SWaT 正式攻击中大量异常更偏向局部幅值变化、标签边界和运行状态迁移，冻结 DUAL 对该数据的覆盖有限。因此协同机制成立，但其收益依赖异常是否真正破坏可学习的跨通道关系，不能推广为所有数据集均提升。

**不能说**

- SWaT DUAL 结果不重要或可以删除。
- Controlled 结果等价于正式真实攻击结果。
- HAI 一套数据足以证明跨数据集普适性。

## Q4. Tail 在 SWaT 变差，为什么还能算创新？

**事实基础**

- Robust-Z 在 SWaT、HAI、WADI 的 Normal top-1% 均为 DUAL 100% 支配。
- Tail calibration 后 REC/DUAL 约为 65/35、59/41、52/48，q99 balance 约为 1。
- HAI Tail AUPR 0.617897，比最佳单分支高 0.047703。
- SWaT Tail AUPR 0.206747，低于 REC 0.743353。

**建议答法**

该创新解决的是异构分数在正常尾部不可比较的问题。正常尾概率映射给不同分支提供统一的概率证据语义，三套数据的 tail-balance 审计直接支持这一点。HAI 显示校准后存在互补收益，SWaT 则说明证据可比性改善不保证测试排序同步改善。论文把后者作为方法边界，而不是回避它。

**不能说**

- Tail 在所有数据集都提高检测性能。
- 证据平衡本身等价于更高 AUROC/AUPR。
- 可根据 SWaT Test 结果重新选择融合规则。

## Q5. 为什么最新 2026 方法没有 DIRECT 数值排名？

**事实基础**

- E3 冻结 47 条外部记录，33 条提取，14 条保持 NOT_EXTRACTED。
- DIRECT 记录数为 0。
- LatentFlow 使用 Affiliation/VUS；CCM-TAD 包含 Test-driven selection 与不同预处理；Park & Shim 仅恢复 gain 摘要；MAD-TCN 主性能表未恢复。

**建议答法**

论文坚持指标语义、数据版本、阈值来源和后处理协议一致后才允许 DIRECT 比较。现有 2026 方法在 Affiliation、VUS、SF1、Test-best 行为、预处理或数据版本上与本论文严格 point-wise q99 协议不同，因此只做 PARTIAL 或 NOT_DIRECT 的结构与协议比较。缺失值保持 NOT_EXTRACTED，避免用二手来源或指标换算制造不可靠排名。

**不能说**

- 没有 DIRECT 排名说明本文优于这些方法。
- Affiliation-F1、VUS、SF1 可以直接换算成普通 F1/AUROC/AUPR。
- 根据摘要中的“最高”或 gain 可以反推出绝对分数。

## 总体风险控制

- 正式结构固定为两个核心方法创新与一个轻量化设计贡献。
- 所有正向结论均同时给出数据集、协议和指标范围。
- SWaT DUAL/TAIL、WADI domain shift、R1 retrospective finding 与 2026 非直接可比性必须保留。
- 禁止使用 SOTA、全面优于、最优结构、普适优越或绝对证明。

最终判定：THESIS_CONTRIBUTION_STRUCTURE_FROZEN。

