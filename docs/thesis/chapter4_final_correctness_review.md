# 第4章公式、算法与源码正确性复核

## 1. 复核范围与结论

本轮仅核验第4章方法表达与正式 SWaT、HAI unified runner、冻结 manifest 和参数审计产物的一致性。未运行训练、推理或实验，也未修改 runner、manifest、实验结果或方法定义。

复核前状态 `THESIS_CHAPTER4_DRAFT_READY` 已确认。公式逐项证据见 `chapter4_formula_source_audit.csv`，算法逐步证据见 `chapter4_algorithm_source_audit.csv`。

## 2. 已执行的最小文字修订

| 修订项 | 原因 | 方法是否改变 |
|---|---|---|
| 明确 Discovery fit/select 为 Normal Train 内逐连续段的时间顺序 80%/20% 划分 | 区分 Discovery fit、Discovery select、完整 Normal Train 与 Normal Validation | 否 |
| 明确 Discovery baseline 来自 Discovery fit 中位数；Validation baseline 来自完整 Normal Train 中位数 | 消除“对应正常分区中位数”的潜在歧义 | 否 |
| 将算法4-1第6—11步改为与 runner 相同的拟合和评价顺序 | runner 会先构造 discovery/full 两套模型，再计算 Discovery 与 Validation 指标 | 否 |
| 补充 Ridge 的中心化充分统计量、正规方程和截距定义 | 说明正式实现为自定义直接线性求解，不是 sklearn estimator；截距不受惩罚 | 否 |
| 明确 DUAL 原生首个 endpoint 为 local index 30；统一评价 mask 从 local index 63 开始 | 区分方法特征合法性与 REC/DUAL 公共评价对齐 | 否 |
| 将式（4-13）的适用域明确为候选目标集合 | 避免把未通过候选规则的变量误解为参加八项 skill 门控 | 否 |
| 重画图4-1源文件中的 Normal-only 冻结与合法 endpoint scoring 两个阶段 | 避免 Normal calibration 与 Test-driven Gate 的流程歧义 | 否 |
| 校正公式注册表的 Ridge、Cross source 和参数量证据位置 | 使用正式 runner/helper及冻结参数明细作为直接来源 | 否 |

上述修订均为表达和证据定位修正，没有源码与冻结方法冲突。

## 3. 关键实现核验

### 3.1 特征、目标与边界

- Self lags 固定为 `[1,5,10,20,30]`，不含 lag 0、未来值或目标当前观测。
- Cross lags 固定为 `[0,1,5,10,20,30]`。lag 0 仅允许读取其他 source 的当前状态。
- Cross source 只来自同一物理 process/stage 的其他可用非恒定变量，target 严格排除。
- 离散非恒定变量可以作为 Cross source；候选 target 必须为连续变量。
- `MAX_LAG=30` 使 DUAL 原生合法 endpoint 从每段 local index 30 开始。SWaT 的两个 hard segments 和 HAI 的五个 Test 文件分别评分，不跨边界。与 64 点 REC 对齐时使用 local index 63 及之后的公共 mask。

### 3.2 Ridge 与 baseline

正式 DUAL helper 先累计中心化充分统计量，再通过 `np.linalg.solve` 求解带 `alpha=1` 的 Ridge 系数。截距由 `y_mean-x_mean@coef` 得到，不进入惩罚项。因此式（4-2）与实现一致，不存在 sklearn solver 语义差异。

Discovery 模型和 baseline 均只使用 Discovery fit。模型在 Discovery select 上评价，baseline 不会在 select 上重新计算。Validation 使用由完整 Normal Train 拟合的模型和完整 Normal Train target median baseline，并在 Normal Validation full、first half、second half 上评价。

### 3.3 Skill、稳定性与标定

- 当 `MAE_base<=0` 时，`skill` 明确返回 `0.0`；否则精确计算 `1-MAE_model/MAE_base`，没有额外 epsilon。
- constant target 在候选规则中已经排除；lagged design 对非有限输入执行 hard fail。skill 函数本身没有额外 NaN fallback。
- 正 skill 条件确为八项：Discovery Self/Cross 两项，Validation full/first/second 的 Self/Cross 六项。
- 稳定性统计来自 Normal Validation 各连续段前、后半部分的原始 Self/Cross disagreement。对称比例使用双向比值的最大值，分母下限为 `1e-12`；median 和 q95 上限分别为2和3。
- 每个可靠 target 的 `mu_j` 与 MAD scale 来自完整 Normal Validation disagreement，`scale=max(1.4826*MAD,1e-6)`。

### 3.4 参数量与复杂度

冻结清单复算结果如下：

| 数据集 | 可靠目标数 | Cross source count总和 | Self系数与截距 | Cross系数与截距 | DUAL learned coefficients |
|---|---:|---:|---:|---:|---:|
| SWaT | 15 | 111 | 90 | 681 | 771 |
| HAI21 | 54 | 1,110 | 324 | 6,714 | 7,038 |

式（4-14）逐 target 计入 `5+1` 个 Self 参数和 `6m_j+1` 个 Cross 参数，与冻结参数明细完全一致。复杂度文字已限定为本实现的中心化充分统计量和直接线性求解：单个 `p` 维 Ridge 系统可估计为 `O(Np^2+p^3)`；推理为所有可靠 target 线性预测代价的和。该表述是论文级结构估计，没有声称 sklearn 某一 solver 的精确运行复杂度。

## 4. Final review回答

| 问题 | 结论 | 依据 |
|---|---|---|
| A. 全部核心公式是否与源码一致？ | 是 | 公式（4-1）至（4-14）均为 `MATCH`；见公式源码审计表 |
| B. Algorithm 4-1是否真实描述执行流程？ | 是 | 第6—11步已按正式 runner 的两套模型拟合、两套 baseline 和四个评价分区修正；22步均为 `MATCH` |
| C. Discovery/Validation/Calibration split是否无歧义？ | 是 | Discovery fit/select、完整 Normal Train、Normal Validation full/halves 与 full-validation calibration 已分别定义 |
| D. 是否存在 observed-target residual 误入 DUAL score？ | 否 | observed target 仅用于正常 skill 审计；DUAL score 始终为 `abs(pred_self-pred_cross)` 的稳健标准化 |
| E. 是否存在 Test-driven Gate？ | 否 | target CSV和SHA在首次 Attack/Test feature access之前物化；Gate和标定只读取 Normal Train/Validation |
| F. 参数量和复杂度描述是否可靠？ | 是 | 15/54 targets与771/7,038 coefficients已由冻结清单复算；复杂度已限定求解方式与近似层级 |
| G. 是否可以进入Chapter 5？ | 是 | 未发现源码/冻结方法冲突，公式、算法、流程图与证据位置均已冻结 |

## 5. 最终状态

`THESIS_CHAPTER4_CORRECTNESS_FROZEN`

本轮仅修复文字、数学适用域、伪代码顺序、流程图分阶段表达和源码引用位置。

NO TRAINING；NO INFERENCE；NO EXPERIMENT；NO METHOD CHANGE；NO NEW CLAIM；NO NEW LITERATURE SEARCH；NO GIT ADD；NO COMMIT；NO PUSH。
