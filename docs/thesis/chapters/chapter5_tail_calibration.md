# 第 5 章 基于正常尾概率校准的异构双证据融合方法

第 3 章的轻量化重构分支与第 4 章的局部滞后双视角协同分支从不同机制刻画工业多变量时序异常：前者衡量观测序列与重构结果之间的偏离，后者衡量 Self View 与 Cross View 对同一目标状态的预测分歧。两条分支都遵循“分数越大越异常”的方向约定，但其生成机制、正常分布、尺度范围和上尾形态并不相同。因而，两个分支上数值相同的原始分数并不必然表示相同程度的异常极端性。

本章提出基于正常尾概率校准的异构双证据融合方法（Normal-Tail Calibrated Heterogeneous Dual-Evidence Fusion）。该方法不改变 REC 或 DUAL 的内部定义，而是使用正常验证数据建立各分支自己的经验上尾映射，将原始分数转换为相对于各自正常参考的尾部证据，再在可比较的证据尺度上执行 max 融合。需要预先说明的是，本章解决的是 score comparability，而不是物理量纲不一致；即使两个输入分数均为无量纲量，只要正常上尾分布不同，直接融合仍可能由某一分支持续主导。尾概率校准也不保证融合排序在所有数据集上优于最强单分支，其作用范围和负向边界将在第 6 章统一检验。

## 5.1 异构异常分数问题

### 5.1.1 REC 与 DUAL 的尺度差异

记分支集合为 \(\mathcal{B}=\{\mathrm{REC},\mathrm{DUAL}\}\)。在共同合法 endpoint \(t\) 上，两条输入分数统一记为

\[
S_b(t),\qquad b\in\mathcal{B},
\tag{5-1}
\]

其中，\(S_{\mathrm{REC}}(t)\) 是第 3 章重构分支的 endpoint 通道平均重构误差；\(S_{\mathrm{DUAL}}(t)\) 是第 4 章在可靠 target 集合上聚合得到的双视角协同分歧分数。式（5-1）只统一符号，不把两个分数视为同分布变量，也不改变任一分支的训练、校准或聚合方式。

REC 的分数由重构残差形成，受通道误差分布、网络拟合能力和输入状态影响；DUAL 的分数由多个 target 的稳健标准化预测分歧取最大值得到，受可靠 target 数、局部关系稳定性和极值聚合影响。即使两者都经过训练期或验证期的内部标准化，其上尾厚度和极端值频率仍可能不同。直接比较 \(S_{\mathrm{REC}}(t)\) 与 \(S_{\mathrm{DUAL}}(t)\) 的绝对大小，等价于默认两个分支的数值尺度已经具有相同异常语义，而这一默认没有由方法结构保证。

### 5.1.2 可比证据的需求

融合前需要回答的问题不是“两个分数的单位是否相同”，而是“一个当前分数相对于其分支正常分布有多极端”。因此，本文把跨分支可比性的目标定义为：对每个分支分别参照其正常上尾，把原始分数映射为具有共同经验尾概率语义的证据。该目标不要求两个变换后分数具有完全相同的整体分布，也不要求 REC 与 DUAL 统计独立；它只要求二者都能解释为“相对于自身正常参考，达到当前或更大分数的经验频率”。

正式实现先将 REC 与 DUAL 对齐到同一组合法 Validation endpoint。对每个连续 Validation segment 或文件，程序检查两个分数序列长度完全一致，并使用相同的逐段时间边界进行后续划分。因此，校准不是在两个不同时间轴上分别完成的。Test 不参与用于 Calibration-A/Threshold-B 的 Validation endpoint 对齐、经验分布建立或阈值冻结；冻结推理阶段只按预先规定的 window、segment/file 边界和 common endpoint 规则对齐 Test 分数，Test 标签不决定该掩码。

## 5.2 Robust-Z 的局限

### 5.2.1 线性稳健标准化

一种直接的异构分数处理方式是对每个分支进行稳健中心—尺度标准化。记正常参考上的中位数为 \(m_b\)，MAD 尺度为 \(a_b=\max(1.4826\operatorname{MAD}_b,10^{-6})\)，历史 Robust-Z 分数写为

\[
Z_b(x)=\max\left(0,\frac{x-m_b}{a_b}\right).
\tag{5-2}
\]

中位数与 MAD 对中心位置和局部尺度较稳健，可降低少量极端值对估计的影响。[CITATION:robust-statistics] 然而，式（5-2）是线性位置—尺度变换：它没有显式利用完整经验分布，也不能保证不同形状或不同重尾程度的正常分布在高分位区域具有相同概率意义。两个分支即使在中心区域被缩放到相近尺度，其上尾仍可能相差显著。

### 5.2.2 重尾极端值不可比

若一个分支的正常分数具有更厚的上尾或由 max 聚合产生更多极端值，则该分支可能在 Robust-Z 后仍持续主导融合分数。此时，较大的 \(Z_b\) 既可能表示真正异常，也可能只是该分支正常尾部的结构性特征。Robust-Z 因而保留为问题动机和第 6 章消融参照，不是本文最终融合方法。

**表 5-1 异常证据校准方式比较**

| 方法 | 中心尺度校准 | 尾概率语义 | 需要 Test 数据 | 保持分支内排序 | 最终采用 |
|---|---|---|---|---|---|
| Raw score | 否 | 否 | 否 | 是 | 否 |
| Robust-Z | 是，中位数/MAD | 否 | 否 | 弱单调，不逆转但可能截断为并列 | 否，仅作动机与消融参照 |
| Normal-tail calibration | 由完整正常经验上尾自适应 | 是，相对于本分支正常上尾的经验极端性 | 否 | 弱单调，不逆转但可能产生并列 | 是 |

表 5-1 中“保持分支内排序”指变换不会把更大的原始分数映射成更小证据；由于经验分布是阶梯函数，并列值或相邻秩可能映射到相同证据，因此不保证严格排序。

## 5.3 经验正常上尾概率

### 5.3.1 Calibration-A 经验分布

设正常验证数据由若干互不跨越的连续段 \(\mathcal{V}^{(g)}\) 组成，且 REC 与 DUAL 已对齐到各段的共同合法 endpoint 集合 \(\mathcal{I}^{(g)}\)。最终实现对每一段按时间顺序切分：前 \(\lfloor |\mathcal{I}^{(g)}|/2\rfloor\) 个 endpoint 属于 Calibration-A，剩余 endpoint 属于 Threshold-B。对分支 \(b\)，把所有段的 Calibration-A 分数连接为

\[
\mathcal{C}_b
=
\mathop{\mathrm{concat}}_g\left\{S_b(t):t\in\mathcal{I}^{(g)}_{A}\right\}
=\{s_b^{(1)},\ldots,s_b^{(N_b)}\}.
\tag{5-3}
\]

这里使用按段连接而不是集合并集，因为重复分数和并列值必须保留并参与经验计数。切分在每个连续段内部独立执行，因此不会跨 SWaT session gap、HAI 文件边界或 WADI 连续运行段。REC 与 DUAL 使用完全相同的 endpoint 和切分边界，正式实现因而满足 \(N_{\mathrm{REC}}=N_{\mathrm{DUAL}}\)。Calibration-A 只来自 Normal Validation；Normal Train 用于基础分支拟合，Test 不进入 \(\mathcal{C}_b\)。

还需区分基础分支的内部校准与本章的 Tail 校准。进入式（5-3）前，REC 与 DUAL 已按第 3、4 章冻结，其中 DUAL 的 target 级中位数/MAD 参考仍由完整 Normal Validation 建立；本章不重新拟合该参考。Calibration-A/Threshold-B 切分只作用于最终 \(S_{\mathrm{REC}}\) 和 \(S_{\mathrm{DUAL}}\) 序列，用于分别建立 Tail 经验映射与最终融合阈值。

对任意分支 \(b\) 和当前分数 \(x\)，经验上尾概率定义为

\[
\widehat p_{\mathrm{tail},b}(x)
=
\frac{1+\#\{s\in\mathcal{C}_b:s\ge x\}}
{N_b+1}.
\tag{5-4}
\]

式（5-4）采用包含并列值的“\(\ge\)”计数约定。源码先将 Calibration-A 数组升序排序，再调用 `searchsorted(..., side="left")` 得到第一个不小于 \(x\) 的位置，因而所有等于 \(x\) 的正常参考值都计入上尾。该约定对并列值更保守：大量正常参考与当前分数相等时，尾概率不会被低估。式（5-4）是经验正常上尾概率，不是严格统计假设检验的 p-value，也不是 \(P(Y=\mathrm{anomaly}\mid x)\) 意义下的概率校准。[CITATION:ECDF]

### 5.3.2 Threshold-B 阈值冻结

Calibration-A 与 Threshold-B 承担不同职责。Calibration-A 只建立两个分支的经验上尾映射；Threshold-B 使用已经冻结的映射产生 \(E_{\mathrm{REC}}\)、\(E_{\mathrm{DUAL}}\) 和融合分数，并据此冻结最终 q99 operating point。在第 5 章 Tail 层面，该隔离避免同一正常 endpoint 既定义经验尾分布又直接决定最终阈值；它不改变第 3、4 章已经冻结的基础分支内部校准协议。

历史 Innovation-3A 校准可行性审计曾在 Calibration-A 上计算阈值，并只用 Audit-B 检查正常稳定性；最终 Innovation-3B runner 与冻结消融协议则把后半段明确作为 Threshold-B，并在其融合分数上计算 q99。本章遵循最终方法：Calibration-A 建图，Threshold-B 定阈值。两阶段都属于 Normal Validation，任何 Test 分数或标签均不参与。

## 5.4 尾概率证据变换

### 5.4.1 负对数尾证据

为使较极端的正常上尾事件对应更大的异常证据，对式（5-4）作负十进制对数变换：

\[
E_b(x)
=
-\log_{10}\widehat p_{\mathrm{tail},b}(x),
\qquad b\in\mathcal{B}.
\tag{5-5}
\]

于是，在 endpoint \(t\) 上有
\(E_{\mathrm{REC}}(t)=E_{\mathrm{REC}}(S_{\mathrm{REC}}(t))\) 和
\(E_{\mathrm{DUAL}}(t)=E_{\mathrm{DUAL}}(S_{\mathrm{DUAL}}(t))\)。当某一原始分数在其分支正常参考中越少出现同等或更大的值时，经验上尾概率越小，证据越大。两个分支由此都转化为“相对于自身正常上尾有多极端”的尺度。

式（5-5）在每个分支内为弱单调变换：若 \(x_1\le x_2\)，则 \(E_b(x_1)\le E_b(x_2)\)。因此它不会反转该分支的原始排序，但经验阶梯映射可能引入并列。所谓共同概率语义也不意味着两个变换后的整体分布完全相同，更不意味着 REC 与 DUAL 独立；它只使给定证据值可以通过各自的正常上尾频率解释。

### 5.4.2 极值、平滑与数值稳定性

式（5-4）的分子加 1、分母取 \(N_b+1\)，使当前分数即使超过全部正常参考，尾概率仍为 \(1/(N_b+1)\)。因此

\[
0\le E_b(x)\le \log_{10}(N_b+1).
\tag{5-6}
\]

这一有限样本修正避免经验尾概率变为 0，也避免式（5-5）产生无穷大。当前冻结实现使用相同共同 endpoint 构造两个 Calibration-A 集合，所以两个分支的 \(N_b\) 相同，证据上界一致。若未来在不对齐的分支上使用不同 \(N_b\)，式（5-6）的上界将随分支变化，必须单独处理，而不能直接假定上界相同。

有限样本修正不消除尾部估计的不确定性。经验映射的分辨率受 \(N_b\) 限制，最小非零尾概率为 \(1/(N_b+1)\)，极端区域会出现离散台阶和饱和。本方法没有外推 Calibration-A 未观测到的尾分布，也没有引入参数化极值分布或额外尾阈值搜索。

## 5.5 异构双证据融合

### 5.5.1 REC/DUAL evidence 的 max 融合

经过正常上尾校准后，最终尾概率校准融合分数（Tail-Calibrated Fusion Score）定义为

\[
S_{\mathrm{TAIL}}(t)
=
\max\left(E_{\mathrm{REC}}(t),E_{\mathrm{DUAL}}(t)\right).
\tag{5-7}
\]

REC 与 DUAL 对应两类异构证据：异常可能主要表现为数值或重构偏离，也可能主要表现为跨通道协同关系破坏。采用 max 的目的，是让任一分支出现相对于自身正常上尾的高极端性时，全局分数都能够响应。式（5-7）不含可训练融合权重，也没有根据 Test 结果搜索权重、平均方式或其他聚合规则。

这里的 max 与第 4 章的 max 处于不同层级。第 4 章的 \(\max_{j\in\mathcal{R}}S_j(t)\) 是在可靠 target 之间聚合局部协同证据，属于 target-level aggregation；式（5-7）是在 REC 与 DUAL 两类分支之间融合已校准证据，属于 branch-level evidence fusion。前者需要 Unified Reliability Gate 防止不可靠 target 污染 DUAL，后者依赖正常尾映射改善跨分支可比性。

式（5-7）并不保证 \(S_{\mathrm{TAIL}}\) 的 AUROC、AUPR 或固定阈值性能必然超过每个单分支。尾概率校准改善的是跨分支分数解释的一致性，而不会自动创造两个基础分支中不存在的排序信息；若一个分支在 Test 上发生域偏移，或者两个分支都缺乏有效 ranking，max 融合无法从校准本身恢复检测能力。

### 5.5.2 检测与阈值输出

对 Threshold-B 中所有共同合法 endpoint，先按 Calibration-A 映射计算式（5-7），得到正常融合分数集合 \(\mathcal{S}_{B}^{\mathrm{TAIL}}\)。冻结检测阈值为

\[
\tau_{\mathrm{TAIL}}
=
Q_{0.99}\!\left(\mathcal{S}_{B}^{\mathrm{TAIL}}\right).
\tag{5-8}
\]

最终 runner 使用 `numpy.quantile(values, 0.99)`，未另传 `method`，因此在冻结的 NumPy 2.2.5 环境中采用其默认线性分位数规则。Test endpoint 的二值输出为

\[
\widehat y(t)
=
\mathbb{I}\!\left[S_{\mathrm{TAIL}}(t)>\tau_{\mathrm{TAIL}}\right].
\tag{5-9}
\]

比较符严格为“\(>\)”而不是“\(\ge\)”。阈值冻结后，Test 仅通过已经固定的 Calibration-A 上尾映射计算证据，再与 \(\tau_{\mathrm{TAIL}}\) 比较；Test 标签只在第 6 章评价阶段使用。本方法不使用 POT、Test-best threshold、point adjustment 或 Test-Normal adaptation。

Tail calibration 与 threshold calibration 需要明确区分：前者把 REC 与 DUAL 映射到可比较的上尾证据尺度，后者在最终融合分数上确定二值 operating point。前者成功并不意味着 Normal Validation q99 一定能稳定迁移到 Test；第 6 章将把证据可比性、score ranking 与固定阈值性能分别报告。

### 5.5.3 完整算法流程

图 5-2 给出正常校准阶段与冻结 Test 推理阶段的隔离流程，其源文件为 `docs/thesis/figures_src/tail_calibration_method_flow.mmd`。

**算法 5-1 基于正常尾概率校准的异构双证据融合**

```text
Input : aligned Normal-Validation REC/DUAL score segments V,
        aligned Test REC/DUAL scores T, quantile q=0.99
Output: Tail score S_TAIL and binary predictions y_hat

1  for each continuous Validation segment g do
2      assert REC and DUAL share the same legal endpoints
3      split g chronologically at floor(|g|/2)
4      append the front half to Calibration-A for both branches
5      append the back half to Threshold-B for both branches
6  end for
7  freeze C_REC and C_DUAL from Calibration-A only
8  define each tail lookup to sort its frozen calibration array per call,
   then count with searchsorted(side="left")
9  transform Threshold-B REC scores to E_REC by (5-4) and (5-5)
10 transform Threshold-B DUAL scores to E_DUAL by (5-4) and (5-5)
11 compute Threshold-B S_TAIL=max(E_REC,E_DUAL)
12 freeze tau_TAIL as its q99 quantile
13 for each legal Test endpoint t do
14     map S_REC(t) and S_DUAL(t) with the frozen branch references
15     set S_TAIL(t)=max(E_REC(t),E_DUAL(t))
16     set y_hat(t)=1 iff S_TAIL(t)>tau_TAIL, else 0
17 end for
18 return S_TAIL and y_hat
```

算法 5-1 的输入不包含 Test 标签。基础 REC/DUAL 模型、共同 endpoint mask、Calibration-A 映射和 Threshold-B 阈值均在 Test 推理前冻结。

## 5.6 方法性质与边界

### 5.6.1 单调性与可比性

本方法具有以下性质。

1. **Normal-only。** 经验上尾映射和最终阈值都只由 Normal Validation 的两个时间隔离子集建立，不需要 Attack/Test 标签。
2. **分支内弱单调。** 尾证据不会反转同一分支的原始分数顺序，但有限经验分布可能把多个值映射为并列证据。
3. **适应分布形态。** 与仅使用中位数和 MAD 的线性变换相比，经验计数直接利用各分支完整正常上尾的秩信息，可适应不同的偏态和重尾形态。
4. **分支模型无关。** 映射只要求输入分数方向为“越大越异常”，不要求 REC 与 DUAL 共享模型结构或原始尺度。
5. **无可训练融合参数。** Tail 模块不新增神经网络权重或 Ridge 系数；它保存的是非可训练的正常校准参考和阈值。
6. **有限样本限制。** 证据分辨率和最大值由 Calibration-A 大小决定，尾部估计无法超越有限正常参考所提供的信息。

### 5.6.2 可比性不等于检测增益

Tail calibration 只保证两类证据可以按相对于各自正常上尾的经验极端性进行比较，不保证融合分数的检测排序或 q99 operating point 在任意数据集上都优于单分支。第 6 章将分别回答：Robust-Z 是否解决正常尾部不平衡、尾概率映射是否改善正常上尾可比性，以及这种改善是否在各数据集上转化为检测性能收益。

论文对结论的使用范围保持以下约束：HAI 的结果可用于讨论单数据集内的正向互补关联；SWaT 必须同时报告 REC 强排序、DUAL/TAIL 有限及 q99 阈值迁移问题；WADI 只作为 domain-shift stress test，用于说明证据平衡不等同于检测有效。上述数值和结果表只在第 6 章出现，本章不以实验结果反向修改方法定义。

### 5.6.3 计算与存储复杂度

设某一分支 Calibration-A 大小为 \(N_b\)，待映射分数数目为 \(M\)。经验上尾查找先排序正常参考，再用二分位置计算“\(\ge x\)”的计数。若排序结果被复用，预处理复杂度为 \(O(N_b\log N_b)\)，单个分数查询为 \(O(\log N_b)\)，批量查询为 \(O(M\log N_b)\)，非可训练存储为 \(O(N_b)\)。两个分支的复杂度为上述代价之和，max 融合本身为 \(O(M)\)。

必须区分概念算法、最终 3B 评价 runner 与后续效率归档。最终 3B runner 的 `tail_evidence` 在每次调用时执行 `np.sort(calibration)`，`freeze_tail` 保存的是未排序的 `rec_a` 和 `dual_a`，并未在该 runner 内持久化排序结果。因此，3B runner 一次批量映射的实际代价包含 \(O(N_b\log N_b+M\log N_b)\)，不同 Test 文件或调用会重复排序。Efficiency-E1 为复现实测与部署状态审计另行持久化 `rec_ecdf_sorted`、`dual_ecdf_sorted` 和 threshold，并通过 `searchsorted` 直接查询；这不反向改变 3B 的方法定义。本文不把 E1 的归档优化误写成 3B runner 已实现的一次排序永久复用。

Tail 模块没有 trainable neural parameters，但完整复现或部署状态需要保存两个 Calibration-A 分数数组及一个阈值。Efficiency-E1 的 Tail artifact 正是两条已排序 ECDF 数组、阈值和 segment-size 元数据。因而，论文中的“参数量”与“状态大小”必须分开：训练参数统计可以记 Tail 为 0，完整状态则还包括非可训练 calibration storage。具体数组大小、端到端时延和显存结果统一在第 6 章效率审计中报告，不在本章估算。

### 5.6.4 适用边界

本方法依赖 Calibration-A 能够代表部署时的正常运行机制。若发生显著 normal-domain shift，经验尾映射可能把大量新的正常状态解释为极端证据；尾校准本身不能修复基础 detector 失效或正常参考过时。max 融合也无法弥补 REC 与 DUAL 同时缺乏有效排序信息的情况，并可能把任一分支的失配传递到全局分数。

此外，异常证据校准与阈值迁移是两个不同问题。Calibration-A 可以改善跨分支可比性，而 Threshold-B q99 仍可能在未来 Test 分布上产生不理想的 FPR 或 Recall。本文不将式（5-4）称为 conformal prediction，也不声称其满足交换性覆盖保证；不将式（5-5）称为异常后验概率；不把两分支证据的可比较性解释为统计独立性。

## 5.7 本章小结

本章以第 3 章 REC 分数和第 4 章 DUAL 分数为输入，针对两类分数正常分布与上尾尺度异构的问题，提出正常尾概率校准方法。该方法在共同合法 Normal Validation endpoint 上逐段进行时间隔离：Calibration-A 建立包含并列值的经验上尾概率，有限样本修正确保概率非零，负对数变换把两个分支转换为可比较的尾部极端性证据；Threshold-B 则用于冻结融合分数的 q99 operating point。最终方法采用 \(\max(E_{\mathrm{REC}},E_{\mathrm{DUAL}})\) 形成 Tail-Calibrated Fusion Score，并以严格“\(>\)”规则输出二值异常判定。

需要强调的是，正常尾概率校准改善的是 heterogeneous upper-tail evidence 的可比性，而不是保证融合在所有数据集上获得普遍增益。映射质量依赖正常参考的代表性，max 无法补偿两个分支都缺少有效 ranking 的情况，阈值迁移也仍是独立问题。第 6 章将在冻结协议下分别检验 Robust-Z 的尾部不平衡、Tail calibration 的正常证据可比性，以及这种可比性在不同数据集上能否转化为互补检测收益。
