# 第 5 章公式、算法与源码正确性复核

## 1. 复核范围与前置门禁

本轮逐项对照第 5 章正文、公式注册表、算法 5-1、Tail 流程图、最终 Innovation-3B runner、3B manifest、ABL2 Innovation-3 manifest 与 Efficiency-E1 Tail 状态记录。

前置状态均已确认：

- `THESIS_CHAPTER5_DRAFT_READY`
- `THESIS_CHAPTER4_CORRECTNESS_FROZEN`
- `ABLATION_A2_INNOVATION3_COMPLETED`
- `OURS_EFFICIENCY_AUDIT_COMPLETED`

本轮没有训练、推理、实验、结果重算或方法变更。未修改 runner、manifest、score、threshold 或实验产物。

## 2. 最小文字修订

复核未发现公式或冻结方法错误，但发现三处可产生实现歧义的表达，已在 `chapter5_tail_calibration.md` 中做最小修订：

1. 将“Test 不参与 endpoint 对齐”限定为 Test 不参与 **Validation calibration endpoint alignment**。冻结 Test 推理仍按预设 window、segment/file 边界和 common endpoint 规则对齐两个分数，但 Test 标签不选择该掩码。
2. 将算法 5-1 第 8 步改为“每次 lookup 调用内部排序并使用 `searchsorted(side="left")`”，避免暗示最终 3B runner 已持久化排序结果。
3. 区分最终 3B runner 与 Efficiency-E1：3B 的 `freeze_tail` 保存未排序 `rec_a/dual_a`，`tail_evidence` 每次调用重新排序；E1 为效率与部署状态审计另行持久化两条 sorted ECDF arrays 和 threshold。

上述修改只提高源码对应精度，没有改变任何数学公式、融合规则或实验结论。公式注册表和流程图无需修改。

## 3. 九个核心公式核验

`chapter5_formula_source_audit.csv` 共记录 9 项，结果为 `MATCH=9`、`MISMATCH=0`。

| 公式 | 结论 | 核验重点 |
|---|---|---|
| (5-1) | MATCH | REC/DUAL 在 window-64 common endpoint 上对齐；只是统一符号 |
| (5-2) | MATCH | 历史 F3 为 median、`1.4826*MAD`、`1e-6` floor 和非负截断；只作消融参照 |
| (5-3) | MATCH | 每个连续 Validation 段前 `floor(n/2)` 连接成 Calibration-A；重复值保留 |
| (5-4) | MATCH | `(1+count{s>=x})/(N+1)`；`side="left"` 包含全部并列值 |
| (5-5) | MATCH | `-log10(p_tail)`；不是异常后验概率、formal p-value 或 conformal guarantee |
| (5-6) | MATCH | 最小尾概率 `1/(N+1)`，最大证据 `log10(N+1)`；等 N 仅限当前冻结对齐协议 |
| (5-7) | MATCH | 唯一最终融合为 `max(E_REC,E_DUAL)`，无权重或搜索 |
| (5-8) | MATCH | Threshold-B 融合证据 `np.quantile(...,0.99)`；NumPy 2.2.5 默认 linear |
| (5-9) | MATCH | 二值判定严格使用 `score > threshold` |

## 4. Calibration-A / Threshold-B 与 endpoint 对齐

最终 3B 方法对每个连续 Normal Validation segment/file 独立执行：

1. REC 与 DUAL 先裁剪到相同合法 endpoint；长度不一致立即报错。
2. 前 `floor(n/2)` 个 endpoint 进入 Calibration-A。
3. 后 `n-floor(n/2)` 个 endpoint 进入 Threshold-B。
4. Calibration-A 只建立两个分支的经验上尾映射。
5. Threshold-B 只产生映射后的融合正常分数并冻结 q99。

SWaT Normal Validation 为单段；SWaT Attack 推理继续遵守两个 hard-session segments。HAI Validation 和 Test 保持文件边界。WADI Validation 保持缺失值切分后的连续运行段。正文没有把两个分支写成不同 timeline 上的独立校准。

DUAL 的 target-level median/MAD 仍由第 4 章协议在完整 Normal Validation 上建立。第 5 章 A/B split 只作用于已冻结的最终 REC/DUAL branch scores，不会重新拟合 DUAL target reference。

## 5. 历史 3A 与最终 3B 隔离

历史 Innovation-3A 是校准可行性审计：Calibration-A 建 ECDF，并曾以 Calibration-A q99 在 Audit-B 上检查正常稳定性。最终 Innovation-3B 与 ABL2 F4 已冻结为：

- Calibration-A：ECDF only；
- Threshold-B（历史材料中有时称 Audit-B）：final Tail q99 only；
- Test：frozen mapping and threshold inference only。

第 5 章正文、公式（5-8）、算法 5-1 和流程图均采用最终 3B 定义，没有把历史 3A 阈值规则写回正式方法。

## 6. Algorithm 5-1 逐步追踪

`chapter5_algorithm_source_audit.csv` 共记录 18 步，结果为 `MATCH=18`、`MISMATCH=0`。

关键步骤核验如下：

- Step 3：`boundary=len(rec)//2`，按时间前后切分。
- Step 7：只冻结 `rec_a`、`dual_a`、threshold 和 segment sizes。
- Step 8：正式 3B helper 每次调用内部 `np.sort`，随后 `searchsorted(side="left")`。
- Steps 9–10：Threshold-B 分别使用本分支 Calibration-A reference。
- Step 12：在 `max(e_rec_b,e_dual_b)` 上调用 q99。
- Steps 14–16：Test 只应用冻结 reference、max fusion 和严格 `>` 阈值。

runner 采用分段向量化计算而不是 Python 逐点循环，但数学动作与算法逐 endpoint 表述一致。binary predictions 在 metrics/event 逻辑中即时使用，没有单独持久化为数组；这不改变式（5-9）的定义。

## 7. Test 隔离

三个数据集均先完成基础分支冻结、Calibration-A 映射和 Threshold-B q99，再打开 Test/Attack 数据。Test score 只在冻结推理阶段按预设 common endpoint 规则对齐与打分；Test label 不决定 endpoint mask，也不参与 Calibration-A、ECDF、Threshold-B q99 或 fusion definition。SWaT 中的标签数组一致性检查只是对预先由时间戳确定的索引进行完整性验证，不反向选择 endpoint。

因此不存在 Test calibration、Test threshold search、Test-selected fusion 或 Test adaptation。

## 8. Chapter 4 / Chapter 5 max 边界

- Chapter 4：`max_j S_j(t)`，在 Unified Reliability Gate 保留的 target 间执行 target-level aggregation。
- Chapter 5：`max(E_REC(t),E_DUAL(t))`，在两条已校准分支间执行 branch-level evidence fusion。

正文和流程图均保持这一层级区分。流程图只显示 REC/DUAL branch evidence 的 max，不把 target 集合重新画入 Tail calibration，也没有 Test 回流箭头。

## 9. Complexity 与 storage

最终 3B runner 的事实为：

- `freeze_tail` 保存未排序的 `rec_a`、`dual_a`；
- `tail_evidence` 每次调用执行 `np.sort`；
- 一次批量映射实际包含 `O(N log N + M log N)`；
- 多文件调用会重复排序。

Efficiency-E1 则显式保存 `rec_ecdf_sorted`、`dual_ecdf_sorted`、threshold 和 segment-size metadata，并在单点时延路径直接执行 `searchsorted`。其状态记录与正文“0 trainable Tail params，但存在 non-trainable calibration storage”一致：SWaT 为 98,937 个 calibration elements，HAI21 为 184,135 个，均等于两条 Calibration-A 数组元素数加一个 threshold。

正文已经区分 3B runner 的实际复杂度和 E1 的持久化状态，没有把未实现的 3B 一次排序复用写成既成事实，也没有从完整状态大小中隐藏 Tail arrays。

## 10. Flow 与术语审计

流程图明确分成 Normal Validation calibration phase 和 Frozen Test inference phase。Calibration-A 只流向两个 tail mappings；Threshold-B 经映射与 max 后流向 q99；冻结 mappings 与 threshold 单向进入 Test inference。不存在 Test score 回流或 Test label 节点。

正文统一使用“经验正常上尾概率”“尾部证据”“Tail-Calibrated Fusion Score”“Calibration-A”和“Threshold-B”。“异常概率”“posterior probability”“formal p-value”和“conformal”只出现在明确否定语境中。

## 11. 最终问题回答

| 问题 | 回答 |
|---|---|
| A. 9 个核心公式是否全部 MATCH？ | 是，`9/9 MATCH` |
| B. Algorithm 5-1 是否逐步 MATCH？ | 是，`18/18 MATCH`；第 8 步已做实现级措辞澄清 |
| C. A/B split 是否无歧义？ | 是；逐连续段前半 Calibration-A、后半 Threshold-B |
| D. 是否存在 Test calibration？ | 否 |
| E. 3A/3B 历史差异是否已经隔离？ | 是；3A 只作历史可行性审计，正式方法遵循 3B |
| F. Fusion 与 threshold 是否完全对应最终 3B？ | 是；`max` + Threshold-B q99 + strict `>` |
| G. complexity/storage 描述是否真实？ | 是；3B 每调用排序，E1 持久化 sorted ECDF arrays，Tail trainable params 为 0 |
| H. 是否可以进入 Chapter 6？ | 是 |

## 12. 冻结结论

公式、算法、流程图、复杂度、存储和主张边界均与最终 3B 正式实现一致。未发现源码与冻结方法冲突。

Final status：`THESIS_CHAPTER5_CORRECTNESS_FROZEN`。

NO TRAINING；NO INFERENCE；NO EXPERIMENT；NO NEW FUSION；NO METHOD CHANGE；NO NEW CLAIM；NO NEW LITERATURE SEARCH；NO GIT ADD；NO COMMIT；NO PUSH。
