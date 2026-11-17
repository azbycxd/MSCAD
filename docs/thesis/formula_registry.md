# Formula registry

仅登记可由代码、配置或阶段文档直接支持的公式。`X ∈ R^{T×C}` 为一个窗口，`X_hat` 为重构，`R(·)` 为代码选择的 corr/cov/shrink-cov 关系矩阵。状态范围：`ACTIVE`、`LEGACY`、`ABLATION_ONLY`、`PROPOSED`、`REJECTED_AS_PRIMARY`。

## F1 — Reconstruction loss and reconstruction score

训练损失使用 PyTorch 默认 `SmoothL1Loss(reduction="mean")`：

```text
e = X_hat - X
smooth_l1(e) = 0.5*e^2                 if |e| < 1
               |e| - 0.5              otherwise
L_rec = mean(smooth_l1(e))
```

实际异常分数并不等于训练损失；V1/A12 M1 的 full-window 分数为：

```text
s_rec = mean(|X_hat - X|, axes=(time, channel))
```

- 代码：`losses/recon.py::ReconstructionLoss`；`models/mscad.py::score_components`；`scripts/swat2026/train_eval_rec_baseline.py::reconstruction_score`。
- 首次可追溯实验：Legacy SWaT main/rec-only；当前冻结证据：HAI Tiny、A12 M1。
- 目的：检测点值、局部形态或幅值重构偏差。
- 后来发现的问题：分数受 operating-regime shift、scaler extrapolation、通道漂移和阈值影响；HAI 也显示 seed 与误报敏感性。
- 当前状态：`ACTIVE`（最终真实基线仍使用 reconstruction；须明确区分 SmoothL1 训练损失与 MAE 检测分数）。

## F2 — Original joint loss

```text
L = lambda_rec * L_rec + lambda_s * L_short + lambda_l * L_long
```

- 代码：`models/mscad.py::MSCAD.loss`、`train.py::train_one_epoch`。
- 配置：`configs/swat_main.yaml` 为 `1.0/0.3/0.3`；`swat_improve_v4_struct_finetune.yaml` 为 `1.0/0.02/0.01` 并 warmup。
- 首次使用：Legacy SWaT MSCAD main；后续 R1–R6、weak struct finetune 与 ablation。
- 目的：同时保持数值重构与短/长尺度关系结构。
- 后来发现的问题：强结构项与重构目标竞争；`w/o struct` 与 rec-only 结果促使 strong joint 降级，弱微调也未形成有效最终 benchmark（Legacy 数据已失效）。
- 当前状态：`LEGACY`；弱权重形式仅作历史/消融说明。

## F3 — Original reconstruction-coupled structure discrepancy

关系矩阵由 `build_structure` 定义；当前代码的逐窗口结构距离为：

```text
R_X = relation(X)
R_hat = relation(X_hat)
L_struct_or_s_struct = sqrt(sum((R_X - R_hat)^2) + 1e-8)
```

若 checkpoint 含筛选 pair，则只对这些 `(i,j)` 元素求和；否则对完整矩阵求 Frobenius 型距离。`L_short` 在末尾短窗口计算，`L_long` 在完整窗口计算。

- 代码：`utils/cov.py::build_structure/structure_distance_batch`、`losses/structure.py`、`models/mscad.py`。
- 首次使用：Legacy MSCAD main；相关 ablation、R1–R6 与融合。
- 目的：让重构保持输入通道关系，并将关系差异用于检测。
- 后来发现的问题：`R(X_hat)` 由 reconstruction 产生，结构分数与重构质量天然耦合；可能检测“模型没重构好”而非独立协同异常。受控比较促成 V2。
- 当前状态：`REJECTED_AS_PRIMARY`；保留为 V1 定义及消融基线。

## F4 — Original score fusion

代码先使用训练分数统计进行逐分量标准化：

```text
z_k(s) = (s - mean_train,k) / (std_train,k + 1e-8)
S = alpha*z_rec + beta*z_short + gamma*z_long
```

- 代码：`utils/metrics.py::z_norm/fuse_scores`、`evaluate.py::run_eval`。
- 首次使用：Legacy MSCAD main；后续 stage4 fusion 与 R5/R6。
- 设计目的：把量纲不同的 rec/short/long 分数组合。
- 是否标准化：**是**，使用训练分数均值和标准差；并非直接对原始三分数相加。
- 后来发现的问题：权重、平滑和阈值搜索敏感；部分搜索使用失效 Legacy 测试构造，因此最优权重不能作为最终性能证据。
- 当前状态：`LEGACY`。

## F5 — Independent relation score

对每条关系的当前相关系数 `rho_j(t)` 做 Fisher-z，再与 normal-only 参考比较：

```text
z_j(t) = arctanh(clip(rho_j(t), -0.999999, 0.999999))
mu_j   = median_train(z_j)
sigma_j = max(1.4826 * median_train(|z_j - mu_j|), scale_floor)
d_j(t) = |z_j(t) - mu_j| / sigma_j
S_ind(t) = max_j d_j(t)              # secondary: top-3 mean
```

- 代码：`data/swat2026_structure.py::fisher_z/fit_robust_reference/relation_deviation/aggregate_deviation`；HAI 对应 `utils/independent_structure_score.py`。
- 首次使用：Legacy normal-only 4.2/4.3 controlled；正式隔离版本见 A12 M2 与 HAI Structure V1。
- 目的：把协同关系异常从 `X_hat` 中解耦，直接衡量 current relation 相对正常参考的偏离。
- 后来发现的问题：zero-lag pairwise Pearson 在 A12 无 strict relation；HAI 真实 AUROC 0.5280、added FP/added TP 14.97，且 global max 容易受重复目标和正常非平稳性影响。
- 当前状态：`ABLATION_ONLY`（有效诊断/机制公式，不是最终主分数）。

## F6 — Lagged and state-conditioned relation representations

代码实际使用的是“改变关系的取样/适用窗口”，偏离仍使用 F5：

```text
rho_ij,l(t) = corr(window(x_i, t), lagged_window(x_j, t, l))
rho_ij,l^delta(t) = corr(delta window(x_i, t), delta lagged_window(x_j, t, l))

applicable_ij,s(t) = 1 only when the selected state channel is stable at state s
S_context(t) = max over applicable relations of standardized Fisher-z deviation
```

HAI Structure V2 另有可追溯的有向 conditional-dependency 表示：self-history Ridge 与 self+other-channel lagged Ridge 的误差比较，rolling window=16，并按 target group max 后 top-2 mean 聚合。

- 代码：`data/swat2026_contextual_structure.py`；`utils/lagged_conditional_dependency.py`；`scripts/hai_structure_v2a_validate_lagged_dependency.py`。
- 首次使用：A12 M2S（lag/state）；HAI V2-A（lagged conditional dependency）。
- 目的：处理同步 Pearson 遗漏的物理时延和 operating-state 混合。
- 后来发现的问题：A12 仅 narrow lagged relation，未冻结 primary；HAI V2 controlled 有效但 A1S 正常稳定性门禁失败，A1M/A1C 没有解决漂移。
- 当前状态：`ABLATION_ONLY`。

## F7 — Conditional cross-channel score

最终 V3 公式尚未冻结，因此本注册表**不把原型公式宣称为论文方法**。当前 A12 M2P 原型可追溯的操作是：Ridge 预测目标通道，计算相对 normal residual reference 的稳健标准化绝对残差，并比较 cross-channel 与 self-history 两套分数；target 聚合使用 max 或 top-3 mean。

```text
prototype only:
e_cross,j(t) = |y_j(t) - y_hat_cross,j(t)| / robust_scale_cross,j
e_self,j(t)  = |y_j(t) - y_hat_self,j(t)| / robust_scale_self,j
```

- 原型代码：`data/swat2026_conditional_prediction.py`、`scripts/swat2026/analyze_conditional_prediction.py`。
- 原型实验：A12 M2P。
- 候选目的：构造真正由其他通道条件信息贡献、且独立于自历史与重构误差的 collaboration residual。
- 已知问题：M2P 得到 `cross_score_not_distinct_from_self_history`，`primary_frozen=false`；“cross residual”尚不能等同于“independent collaboration residual”。
- 当前状态：`PROPOSED`。

## 🔴 代码—文档冲突与过期陈述

1. 🔴 `docs/研究内容_研究方案_研究进展_综合整理.md`、`实验思路与结果分析.md` 等早期材料把 Legacy SWaT 的 V1 指标描述为可支撑主实验；4.3D 后续内容审计已证明该数据有 overlap 和错误 chronology。这些数值只能是 historical，不能继续沿用早期结论。
2. 🔴 `configs/swat_main.yaml` 的 protocol notes 声称 train/val/test 严格分离，但其 `test_csv=Swat/merged.csv` 实际包含完整 `normal.csv` 前缀。配置意图与数据内容不一致。
3. 🔴 部分早期材料把结构分数提升 AUPR/AUROC 解释为可进入最终论文；HAI/A12 后续严格证据只支持 controlled mechanism 或 negative diagnostic，不支持真实稳定增益。
4. 🔴 仓库已有 M2P 原型代码与输出，但 V3 的最终“independent collaboration residual”仍未定义或冻结。应写“候选方向已有一次否定性原型试验”，不能写“V3 已实现”。

F1–F6 的上述数学表达与当前代码实现一致；冲突集中在证据解释、协议描述和方法成熟度，而不是加号、标准化或矩阵距离的实现。
