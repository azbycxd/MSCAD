# MSCAD 中期报告前实验与公式审计

本目录是项目的逻辑入口。它不改变任何算法、数据、配置、输出或 checkpoint，只把散落在代码、配置、阶段报告和结果目录中的证据连接起来。

## 一页结论

### 我的课题是什么？

本课题研究多变量工业时间序列异常检测：以重构误差为基础，考察通道间短期、长期、滞后及状态相关结构是否能提供独立于重构的异常证据，并在严格无泄漏的数据协议下评估其有效性。

### 目前有几个方法版本？

共有三个逻辑版本：

1. **V1 — Original MSCAD**：重构主干与 `relation(X)` 对 `relation(X_hat)` 的结构差异，包含强联合训练、弱结构微调和 legacy score fusion。
2. **V2 — Independent Relation**：当前关系与正常参考关系比较，包含同步、delta、lagged 和 state-conditioned 诊断。
3. **V3 — Conditional Collaboration**：计划中的候选方向；目标是 other-channel conditional prediction 与 independent collaboration residual。尚未形成冻结的最终实现。

详细映射见 [method_versions.md](method_versions.md)。

### 哪个是旧方案？

V1 是旧方案。它解释了项目如何从原始 MSCAD 演进而来，但其中 legacy SWaT 实验受数据重叠影响，不能作为最终 benchmark 证据；`relation(X)-relation(X_hat)` 还存在与重构质量耦合的问题。

### 哪些结果已经失效？

所有使用旧 `Swat/normal.csv` 与 `Swat/merged.csv` 且落入已确认重叠协议的真实异常检测结果，统一为 `INVALID_FOR_FINAL`。它们只能作为历史过程，不可进入最终主结果表。

### 哪些结果仍可信？

- HAI 严格协议下的冻结 Tiny DSConv AE 重构基线是当前唯一 `VALID_FINAL_CANDIDATE`。
- HAI 的结构、稳定性和受控机制实验，以及 A12 2026 的 N0/N1/N2/M1/M2/M2R/M2S，保留为 `VALID_DIAGNOSTIC`。
- A12 M2P 只说明已做过原型诊断，不等于 V3 已完成。
- SWaT 2015 official Physical 尚未审计，状态为 `TODO`。

### 当前最终方法是否已经冻结？

没有。冻结的是 HAI 严格协议下的重构基线及其有限论文定位，不是一个已经完成并验证的统一 MSCAD 最终方法。V2 仍是诊断层，V3 仍是 proposed candidate。

### 下一步实验是什么？

下一步首先是审计 SWaT 2015 official Physical 的文件、标签、时间连续性和无重叠划分，建立正式主 benchmark 协议。审计通过前不运行正式训练。若继续 V3，应先定义独立 collaboration residual、正常参考和无泄漏验证规则，再决定是否实现和实验。

### SWaT 2015、A12、HAI 分别负责什么？

| 数据集 | 当前角色 | 当前状态 |
|---|---|---|
| SWaT 2015 official Physical | 计划重新建立的正式主 benchmark | `TODO / awaiting audit` |
| SWaT A12 2026 official | 协议工程、source alignment、结构候选与失败边界诊断 | `VALID_DIAGNOSTIC` |
| HAI 22.04 | 当前严格可信的主证据来源；重构基线冻结，结构结果用于限制结论 | 1 个 `VALID_FINAL_CANDIDATE`，其余主要为 `VALID_DIAGNOSTIC` |

## 审计注册表

- [method_versions.md](method_versions.md)：V1/V2/V3 的代码、配置、实验、输出、数据和科学状态。
- [experiment_registry.md](experiment_registry.md)：61 个实验家族及其协议、checkpoint、指标和论文可用性。
- [formula_registry.md](formula_registry.md)：F1–F7 七个公式版本及代码映射、首次实验、问题和状态。
- [result_registry.md](result_registry.md)：按 Legacy SWaT、A12 2026、HAI、SWaT 2015 official 四个数据版本分表。
- [decision_log.md](decision_log.md)：D1–D9 研究决策及证据链。
- [unresolved_artifacts.md](unresolved_artifacts.md)：重复命名、非正式输出、数据镜像和待闭合来源。

## 状态总览

| 状态 | 数量 | 含义 |
|---|---:|---|
| `VALID_FINAL_CANDIDATE` | 1 | 可作为最终论文候选证据，仍需遵守登记的结论边界 |
| `VALID_DIAGNOSTIC` | 35 | 协议或机制诊断可信，但不应伪装成最终主 benchmark |
| `EXPLORATORY` | 5 | 探索性实现或分析，只用于形成假设 |
| `INVALID_FOR_FINAL` | 19 | 历史保留，但不得用于最终论文主要证据 |
| `TODO` | 1 | 尚未完成必要审计或实验 |
| **合计** | **61** | 按实验家族计数，不按每个 seed、checkpoint 或 smoke run 重复计数 |

## 公式总览

共登记 7 个核心公式版本：F1 reconstruction loss；F2 original joint loss；F3 original structure discrepancy；F4 original score fusion；F5 independent relation score；F6 lag/state-conditioned representation；F7 conditional cross-channel score。

代码与历史文档存在重要冲突或表述错位：旧 SWaT 文档的真实指标结论被后续 data audit 否定；旧配置宣称的划分隔离与实际文件重叠不一致；早期结构收益表述与严格 HAI/A12 诊断不一致；M2P 原型存在不代表 V3 已实现或冻结。详见 [formula_registry.md](formula_registry.md) 中的红色冲突标记。

## 使用规则

- 查结论前先确认 dataset version 和 protocol status。
- 不跨数据版本合并主结果表。
- 不从目录名反推实验参数；无法闭合时登记为未决。
- 不引用 smoke、failed、retry 或 dry-run 指标作为正式结果。
- legacy SWaT 资产保留但标注 `historical only, not usable as final benchmark evidence`。

## 本次审计边界

- 修改算法：No
- 修改模型、评分公式或数据：No
- 运行训练：No
- 删除或移动文件、outputs、checkpoints：No
- reset、clean、force push：No
- 新增内容：仅本目录下的逻辑索引文档
