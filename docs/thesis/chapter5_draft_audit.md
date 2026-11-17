# 第 5 章草稿审计

## 1. 前置状态

| 门禁 | 结果 | 证据 |
|---|---|---|
| 实验阶段已关闭 | PASS | `docs/thesis/experimental_phase_freeze.md`: `THESIS_EXPERIMENTAL_PHASE_CLOSED` |
| 贡献结构已冻结 | PASS | `docs/thesis/final_contribution_structure.md`: `THESIS_CONTRIBUTION_STRUCTURE_FROZEN` |
| 第 4 章正确性已冻结 | PASS | `docs/thesis/chapter4_final_correctness_review.md`: `THESIS_CHAPTER4_CORRECTNESS_FROZEN` |
| 论文题目一致 | PASS | 《轻量化多变量时序协同异常检测方法研究》 |

本轮只写方法章节和可追溯注册表，没有训练、推理、实验、重新拟合、参数搜索、融合搜索或结果改写。

## 2. 正式实现绑定

| 项目 | 冻结定义 | 源码/产物依据 | 结果 |
|---|---|---|---|
| Calibration split | 每个连续 Normal Validation segment/file 内按时间前 50% 为 Calibration-A，后 50% 为 Threshold-B；两分支共用 endpoint 与边界 | `run_innovation3b_tail_fusion_final.py:72-80`; final 3B manifest `datasets.*.validation_split` | PASS |
| Endpoint alignment | 每段 REC/DUAL 长度必须一致；DUAL 裁剪到 REC 的 window-64 common endpoint | `run_innovation3b_tail_fusion_final.py:76,114-124` | PASS |
| 分支内部校准边界 | Chapter 4 DUAL 的 target 级 Normal Validation 中位数/MAD 保持冻结；A/B 只拆分最终 branch score 用于 Tail 映射和阈值 | `run_innovation3b_tail_fusion_final.py:100-110,114-124` | PASS |
| 经验上尾 | `(1 + count{s in C_b: s >= x})/(N_b+1)` | `run_innovation3b_tail_fusion_final.py:56-59` | PASS |
| Tie convention | `searchsorted(..., side="left")`，因此所有 `s == x` 均计入 `s >= x` | `run_innovation3b_tail_fusion_final.py:57-58` | PASS |
| 有限样本修正 | 分子 `+1`、分母 `N+1`；最小概率 `1/(N+1)`，证据有限 | `run_innovation3b_tail_fusion_final.py:59` | PASS |
| 证据变换 | `-log10(p_tail)` | `run_innovation3b_tail_fusion_final.py:59` | PASS |
| 融合 | `max(E_REC,E_DUAL)`，无权重 | `run_innovation3b_tail_fusion_final.py:86-88`; final 3B manifest `score` | PASS |
| 阈值 | Threshold-B 融合分数 `np.quantile(...,0.99)`；未显式指定 method，冻结 NumPy 2.2.5 默认 linear | `run_innovation3b_tail_fusion_final.py:81-82`; `requirements.txt:6` | PASS |
| 二值比较 | `score > threshold`，不是 `>=` | `run_innovation3b_tail_fusion_final.py:62-69` | PASS |
| Test 隔离 | Calibration-A 与 Threshold-B 均在读取 Test 前冻结；Test 不用于映射或阈值 | final 3B manifest tests: `test_not_in_calibration`, `no_test_adaptation` | PASS |

### 历史 3A 与最终 3B 的阈值来源差异

`outputs/thesis_innovation3a_tail_calibration/manifest.json` 的可行性审计把阈值记为 `Calibration-A q99 only`，并在 Audit-B 上检查正常 FPR；最终 `run_innovation3b_tail_fusion_final.py` 则先由 Calibration-A 建立上尾映射，再在后半段 Threshold-B 的融合证据上冻结 q99。`outputs/thesis_innovation3b_tail_fusion_final/manifest.json` 与 `docs/thesis/ablation_a2_innovation3_results.md` 均确认最终方案为 `Calibration-A ECDF + Threshold-B q99`。第 5 章遵循最终 3B 定义；3A 仅作为历史校准可行性审计保留，不用于覆盖最终阈值规则。该差异已显式解释，不构成未解决的方法冲突。

## 3. Draft Audit A–I

| 检查 | 结果 | 审计说明 |
|---|---|---|
| A. 是否误把 E 称为异常概率 | PASS | 正文称其为经验正常上尾极端性证据，明确不是 `P(Y=anomaly|x)` |
| B. 是否误称 formal p-value | PASS | 明确不是严格假设检验 p-value，也不声称 conformal coverage |
| C. 是否使用 Test 数据 calibration | PASS | Calibration-A 建图、Threshold-B 定阈值，均来自 Normal Validation；Test 仅冻结推理 |
| D. 是否修改最终 fusion 公式 | PASS | 保持 `max(E_REC,E_DUAL)`，无权重、平均或搜索 |
| E. 是否声称 universal improvement | PASS | 明确 comparability 不等于 universal gain；数值结果留第 6 章 |
| F. 是否混淆 Calibration 和 Threshold | PASS | 分别定义 Calibration-A 上尾映射与 Threshold-B q99 operating point |
| G. 是否混淆 Chapter 4 target max 与 Chapter 5 branch max | PASS | 分别标记 target-level aggregation 与 branch-level evidence fusion |
| H. 是否把 Robust-Z 写成 Final | PASS | Robust-Z 只作为动机和消融参照；Normal-tail 为最终方法 |
| I. 是否隐藏 SWaT/WADI 边界 | PASS | 方法边界预告 SWaT 阈值/排序风险与 WADI domain-shift stress-test 定位；不提前放数值 |

## 4. 复杂度与存储审计

| 检查 | 结果 | 说明 |
|---|---|---|
| 数学查找复杂度 | PASS | 排序后单点二分查找为 `O(log N)`，批量为 `O(M log N)` |
| 当前工程实现 | PASS | 明确 `tail_evidence` 每次调用内部重新 `np.sort`，未虚构持久化排序优化 |
| Trainable 参数 | PASS | Tail 新增 0 个可训练参数 |
| Non-trainable 状态 | PASS | 明确保存/持有 Calibration-A 参考数组与冻结阈值，未把其从完整状态大小中隐去 |
| 有限样本限制 | PASS | 说明分辨率、台阶、饱和与 `log10(N+1)` 上界 |

## 5. 输出完整性

| 输出 | 结果 | 内容 |
|---|---|---|
| `docs/thesis/chapters/chapter5_tail_calibration.md` | PASS | 冻结 5.1–5.7 结构、表 5-1、算法 5-1、公式（5-1）至（5-9） |
| `docs/thesis/figures_src/tail_calibration_method_flow.mmd` | PASS | Normal calibration 与 frozen Test inference 分区，未画 Test adaptation |
| `docs/thesis/chapter5_formula_registry.csv` | PASS | 9 条公式/性质，均绑定最终 runner 或冻结 manifest |
| 引文处理 | PASS | 仅使用 `[CITATION:ECDF]` 与 `[CITATION:robust-statistics]` 占位，未联网或虚构编号 |

## 6. 冻结结论

第 5 章草稿与最终 Innovation-3B runner、最终 manifest、冻结消融定义和贡献边界一致。唯一需要区分的历史差异是 3A 可行性审计的阈值来源；正文已明确采用最终 3B 的 Threshold-B q99，不存在未解决的方法定义冲突。

Final status：`THESIS_CHAPTER5_DRAFT_READY`。

NO TRAINING；NO INFERENCE；NO EXPERIMENT；NO NEW FUSION；NO METHOD CHANGE；NO NEW CLAIM；NO NEW LITERATURE SEARCH；NO COMMIT；NO PUSH。
