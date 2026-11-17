# 论文最终贡献结构

## 论文题目

轻量化多变量时序协同异常检测方法研究

题目保持不变。本轮只重构贡献层级，不修改方法、实验、数据协议或结论。

## 正式贡献结构

论文采用“两个核心方法创新 + 一个轻量化设计贡献”的结构，不再将具体 REC 卷积核配置作为独立算法创新。

### 核心方法创新 1：局部滞后双视角协同一致性检测方法

英文名称：Local Lagged Dual-View Collaboration Consistency Detection。

该方法分别构建 Self View 与 Cross View，利用多尺度局部滞后信息建立正常状态下的目标通道预测关系，以预测分歧形成协同异常证据，并通过 Unified Reliability Gate 在正常数据上约束候选 target 的可靠性。它是论文最核心的方法创新。

证据链如下：

| 证据层级 | 数据与实验 | 冻结事实 | 解释边界 |
|---|---|---|---|
| 合成机制 | Historical CoMuT，D0 | median AUROC 0.995989 | 证明对受控关系异常敏感，不等同于工业 Test 性能 |
| 真实轨迹控制实验 | SWaT controlled，D0 | median AUROC 0.996913，AUPR 0.996931 | 证明冻结关系破坏场景中的机制响应 |
| 独立真实攻击 | HAI21，DUAL | AUROC 0.916334，AUPR 0.570193，F1 0.610556 | 支持真实协同异常检测，不推广到所有数据集 |
| 滞后消融 | HAI21，D1 short-lag | AUPR 由 0.570193 降至 0.426064 | 较长滞后在 HAI 中有贡献；SWaT 上方向不同 |
| Gate 消融 | SWaT controlled，D2 No-Gate | AUROC 由 0.996913 降至 0.630637 | Gate 抑制不可靠 target 污染 max aggregation |
| 适用边界 | SWaT official real attacks | DUAL AUROC/AUPR 为 0.675631/0.167510 | 不宣称真实攻击上跨数据集一致提升 |

### 核心方法创新 2：基于正常尾概率校准的异构双证据融合方法

英文名称：Normal-Tail Calibrated Heterogeneous Dual-Evidence Fusion。

该方法针对 REC 与 DUAL 分数的尺度和尾部形态差异，使用正常校准集上的经验尾概率把两类分数映射到可比较的异常证据尺度，再进行 tail-calibrated fusion。核心贡献是异构证据的可比较性与互补表达，不是融合在所有数据集上的普遍性能优势。

证据链如下：

| 证据层级 | 数据与实验 | 冻结事实 | 解释边界 |
|---|---|---|---|
| 问题识别 | SWaT、HAI21、WADI.A2，Robust-Z | Normal top-1% 均为 DUAL 100% 支配 | 线性 robust-z 未解决重尾可比性 |
| 校准效果 | 三数据集，Tail-Calibrated | REC/DUAL 为 65.19/34.81、59.15/40.85、51.78/48.22；q99 balance 约为 1 | 支持正常尾部证据平衡 |
| 正向检测关联 | HAI21 | Tail AUPR 0.617897，较最佳单分支增加 0.047703 | 不能外推为普遍因果或普遍优势 |
| 适用边界 | SWaT | Tail AUPR 0.206747，低于 REC 的 0.743353 | 校准成功不保证保留最强单分支排序 |
| 域偏移边界 | WADI.A2 | 各版本 AUPR 约 0.052–0.055 | 仅作 domain-shift stress test |

### 轻量化设计贡献：轻量化时序重构与协同检测框架

英文名称：Lightweight Temporal Reconstruction and Collaboration Detection Framework。

该贡献包括 depthwise-separable REC、线性 Ridge Dual 分支，以及参数量、时延、显存和状态大小的完整效率审计。Final REC 在 SWaT 与 HAI21 上分别为 11,730 与 15,482 个参数，保持万级规模；dense R2 的参数分别达到 R0 的 2.192 倍与 2.126 倍，却没有稳定的 AUROC/AUPR 收益。R1 在两个数据集上均以更少参数取得更高 AUROC/AUPR，说明固定 kernel width 没有获得跨数据集最优性证据。因此，该项只能表述为 performance–efficiency trade-off 与系统轻量化设计贡献。

## 历史编号与正式贡献的映射

| 历史实验编号 | 正式论文归属 | 处理规则 |
|---|---|---|
| 旧 Innovation-1 / ABL3 | 轻量化设计贡献 | 历史文件名和实验 ID 保留；正文不再称其为第一算法创新 |
| 旧 Innovation-2 / ABL1 | 核心方法创新 1 | 作为论文第一核心方法创新 |
| 旧 Innovation-3 / ABL2 | 核心方法创新 2 | 作为论文第二核心方法创新 |

全局措辞审计未发现“本文提出的具体 k=3 结构性能最优”“消融证明当前 REC 结构最优”或“全面优于简化版本”等正向夸大表述。docs/thesis 中仍出现的 Innovation-1/2/3 主要是历史实验标题、文件名和注册编号，应作为可追溯别名保留，不进入摘要、绪论创新点或结论的正式贡献排序。

## 章节结构的最小修改方案

不推翻现有实验章节，只调整方法章节标题与贡献归属：

1. 第 3 章：轻量化重构基础检测模块。介绍 REC、训练协议和轻量化实现，不宣称具体 kernel 最优。
2. 第 4 章：局部滞后双视角协同异常检测方法。集中描述 Self/Cross、多尺度 lag、预测分歧与 Unified Reliability Gate。
3. 第 5 章：异构双证据校准与融合方法。描述 normal-tail probability mapping、证据尺度统一及融合。
4. 第 6 章：实验与分析。依次呈现主性能、效率、轻量化结构分析、双视角机制消融、尾概率校准消融及文献兼容性比较。

## Git 边界复核

既有 inventory 共 211 项：COMMIT 26、REVIEW 122、ARCHIVE_EXTERNALLY 61、GITIGNORE 2。源码、专项测试和论文文档已列为 COMMIT 候选；小型结果表和最终论文图因位于 ignored outputs 路径而保持 REVIEW，须人工确认后再纳入。raw score、checkpoint、candidate matrix、NPZ 与大型状态继续保持 ARCHIVE_EXTERNALLY 或 GITIGNORE。本轮未执行 git add、commit 或 push。

## 状态

THESIS_CONTRIBUTION_STRUCTURE_FROZEN

