# 论文实验阶段最终冻结声明

## 1. 冻结结论

截至 2026-09-25，论文实验阶段正式关闭：

- `method design closed`
- `baseline search closed`
- `literature search closed`
- `ablation closed`
- `stability closed`

全局机器可读标志：`experimental_phase_closed = true`。

最终状态：`THESIS_EXPERIMENTAL_PHASE_CLOSED`。

## 2. 冻结依据

| 工作流 | 冻结状态 | 主要入口 |
|---|---|---|
| 贡献结构 | `THESIS_CONTRIBUTION_STRUCTURE_FROZEN` | `docs/thesis/final_contribution_structure.md` |
| 论文叙事结构 | `THESIS_NARRATIVE_STRUCTURE_READY` | `docs/thesis/final_thesis_outline.md` |
| 消融包 | `ABLATION_FINAL_PACKAGE_READY` | `outputs/thesis_ablation/ablation_final_summary.json` |
| 三随机种子稳定性 | `THESIS_MULTI_SEED_STABILITY_COMPLETED` | `outputs/thesis_stability/manifest.json` |
| 最终实验注册 | `COMPLETED` | `outputs/thesis_registry/experiment_registry_frozen.csv` 中 `MULTI_SEED_STABILITY` |

三随机种子结果只作为 Stability/Robustness 补充证据。历史 Primary 表继续使用原正式 seed，不用 mean 结果重写。

## 3. 后续允许工作

- 论文写作与章节整合。
- 已冻结结果的图表排版、编号和交叉引用。
- 已有公式、符号和算法伪代码的整理与表述校对。
- 答辩材料、答辩问答和关键数字的准备。
- 对已有 artifact 的只读一致性检查。

以上工作只能整理既有证据，不得产生新的实验选择。

## 4. 后续禁止工作

- 基于已见 Test 结果修改模型、score、gate、lag、fusion、threshold、数据划分或预处理。
- 新增 seed、重跑 baseline、搜索超参数或挑选 best seed。
- 用 Oracle、PA 或 Test-best 结果覆盖严格 Primary。
- 用三 seed 均值回写历史正式结果。
- 因 SWaT TAIL 或 q99 指标波动而反向调整方法。
- 扩大现有 claim 的适用范围或证据强度。

任何未来方法改动均应进入独立、预注册的新研究周期，不属于本论文冻结实验阶段。

## 5. 终局边界

本次冻结确认：没有基于已见 Test 结果修改方法；没有新增训练、推理或实验；没有改变历史 Primary；没有把单数据集正向结果包装为普遍结论。

