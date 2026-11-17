# 第 4 章 局部滞后双视角协同异常检测方法

工业过程中的多变量观测通常由传感器、执行器和控制回路共同产生。一个变量的当前状态不仅与其自身历史有关，也可能受到同一物理过程内其他变量的即时状态和滞后状态约束。传统多变量重构模型能够在隐空间中隐式学习这类依赖，因此本文并不认为重构模型完全不能检测协同异常。问题在于，重构分支最终输出的是观测值与重构值之间的残差，该残差同时受到幅值偏离、模型拟合误差和变量关系变化的影响，难以单独解释为协同关系异常。

为获得与 observed-target residual 相对独立的关系证据，本章构建 Self View（自历史视角）和 Cross View（跨变量视角）。Self View 根据目标变量自身的多尺度历史状态形成自状态预期，Cross View 根据同一物理过程内其他非恒定变量的同步—滞后状态形成协同预期。两个预测器都仅在正常数据上拟合。当两种信息视角对同一目标的预测出现异常分歧时，该分歧被解释为局部预测协同关系可能偏离正常模式。为避免不可靠目标经全局最大值聚合后放大假异常，本章进一步引入完全由 Normal Train 与 Normal Validation 驱动的可靠性门控。

## 4.1 协同异常定义与动机

### 4.1.1 重构残差的关系耦合问题

设标准化后的多变量时序为 \(\mathbf{z}(t)=[z_1(t),\ldots,z_C(t)]^\top\)。重构模型以一个时间窗口为输入并输出对应重构值，其异常分数通常由 \(z_j(t)\) 与 \(\hat z_j(t)\) 的差异形成。由于重构网络同时利用多个通道，变量依赖可以通过共享特征被隐式编码。因此，关系破坏有可能引起重构误差变化，但这种变化仍通过“观测目标值—重构目标值”的残差体现，无法自然区分数值异常、未建模动态和关系不一致。[CITATION:dependency-MTSAD]

本文的目标不是否定重构证据，而是在重构分支之外构造一条显式的协同关系证据。该证据不以 \(|z_j(t)-\hat z_j(t)|\) 作为核心分数，而是让两个仅使用不同信息来源的预测器分别估计目标变量，再比较二者的预测结果。由此，双视角分支回答的是“自身历史所暗示的目标状态是否与同过程其他变量所暗示的目标状态一致”，而不是再次计算一个 observed-target residual。

### 4.1.2 双视角预测分歧

对目标变量 \(j\)，记其所属物理过程或阶段为 \(\mathcal{P}(j)\)。Self View 只读取目标 \(j\) 的历史值；Cross View 只读取 \(\mathcal{P}(j)\) 内除目标外的其他可用变量。两条路径分别输出 \(\hat z_j^{\mathrm{self}}(t)\) 与 \(\hat z_j^{\mathrm{cross}}(t)\)。在正常运行状态下，两种视角由同一正常过程生成，对目标状态的预期应具有相对一致性。当控制响应、物理耦合或变量间同步关系被破坏时，两个预测器可能给出不同的状态预期，其绝对差异由此形成 target 级协同异常原始证据。

这一设计仍属于预测关系建模，而非严格因果发现。Cross View 所描述的是 normal-only 数据上可学习的 local predictive collaboration relation。它既不声称恢复真实因果图，也不把统计可预测性等同于物理因果性。方法总流程见图 4-1，其源文件为 `docs/thesis/figures_src/dual_view_method_flow.mmd`。

## 4.2 Self View

### 4.2.1 自身历史特征

Self View 使用冻结滞后集合

\[
\mathcal{L}_{\mathrm{self}}=\{1,5,10,20,30\}.
\]

对目标变量 \(j\)，其 Self View 输入定义为

\[
\mathbf{x}^{\mathrm{self}}_j(t)
=
\big[z_j(t-1),z_j(t-5),z_j(t-10),z_j(t-20),z_j(t-30)\big]^\top .
\tag{4-1}
\]

该输入不包含目标当前值 \(z_j(t)\)，也不包含任何未来观测。最大滞后为 30，因此 DUAL 原生分数从每个连续 segment 或文件的 local index 30 开始定义；构造特征时不得跨越 SWaT 的硬 session boundary 或 HAI 的文件边界。与窗口长度为 64 的重构分支进行统一评价时，正式 runner 进一步裁剪到 local index 63 及之后的 common endpoint mask，但这一对齐不改变 DUAL 的特征定义。标准化器只由 Normal Train 拟合，Validation 与 Test 仅执行变换。

### 4.2.2 Self 预测器

Self View 采用带截距的 Ridge 回归。[CITATION:Ridge] 对任一视角 \(v\in\{\mathrm{self},\mathrm{cross}\}\)，训练设计矩阵和目标向量分别记为 \(\mathbf{X}^{v}_j\) 与 \(\mathbf{y}_j\)，冻结优化目标为

\[
(\hat b_j^{v},\hat{\boldsymbol\beta}_j^{v})
=
\arg\min_{b,\boldsymbol\beta}
\left\|\mathbf{y}_j-b\mathbf{1}-\mathbf{X}^{v}_j\boldsymbol\beta\right\|_2^2
+\alpha\left\|\boldsymbol\beta\right\|_2^2,
\qquad \alpha=1.
\tag{4-2}
\]

正式实现通过中心化充分统计量构造 \(\mathbf{X}_c^\top\mathbf{X}_c\) 和 \(\mathbf{X}_c^\top\mathbf{y}_c\)，再直接求解 \((\mathbf{X}_c^\top\mathbf{X}_c+\alpha\mathbf{I})\boldsymbol\beta=\mathbf{X}_c^\top\mathbf{y}_c\)，并令 \(b=\bar y-\bar{\mathbf{x}}^\top\boldsymbol\beta\)。因此截距不进入 Ridge 惩罚项，且该实现与式（4-2）一致；DUAL 的 Ridge 拟合没有调用 sklearn estimator。Self View 的预测为

\[
\hat z_j^{\mathrm{self}}(t)
=
\hat b_j^{\mathrm{self}}
+(\hat{\boldsymbol\beta}_j^{\mathrm{self}})^\top
\mathbf{x}^{\mathrm{self}}_j(t).
\tag{4-3}
\]

线性模型的作用是把信息视角差异与预测器容量差异分开，同时保留参数可追溯性。本文没有据此声称线性 Ridge 对所有工业过程都是最优预测器。

## 4.3 Cross View

### 4.3.1 同过程其他变量特征

记过程 \(\mathcal{P}(j)\) 内通过数据可用性检查的非恒定变量集合为 \(\mathcal{U}_{\mathcal{P}(j)}\)，则目标 \(j\) 的 Cross source 集合为

\[
\mathcal{C}_j
=
\mathcal{U}_{\mathcal{P}(j)}\setminus\{j\},
\qquad |\mathcal{C}_j|\ge 1.
\tag{4-4}
\]

Cross View 不使用全局全部变量，而只使用 same physical process/stage 中的其他非恒定变量。目标变量本身从 Cross sources 中严格排除。这样做有三个目的：减少无关变量引入的噪声，保留局部物理过程的协同语义，并控制线性模型的规模。候选目标还需满足连续、非恒定、Normal Train 可用以及存在至少一个同过程 source；离散变量可以作为 source，但不作为本章冻结 Gate 中的连续 target。

Cross View 的冻结滞后集合为 \(\mathcal{L}_{\mathrm{cross}}=\{0,1,5,10,20,30\}\)。若 \(\mathcal{C}_j=\{c_1,\ldots,c_{m_j}\}\)，则

\[
\mathbf{x}^{\mathrm{cross}}_j(t)
=
\mathop{\mathrm{concat}}_{c\in\mathcal{C}_j,\,\ell\in\mathcal{L}_{\mathrm{cross}}}
z_c(t-\ell).
\tag{4-5}
\]

其中 \(\ell=0\) 允许模型读取其他变量在当前时刻的状态，用于表达即时协同关系；目标变量当前观测 \(z_j(t)\) 始终不在 Cross 输入中。因此，Cross View 虽然允许同步 source，却仍保持 target-blind。

### 4.3.2 Cross 预测器

Cross View 使用与式（4-2）相同的 Ridge 目标和固定 \(\alpha=1\)，其输出为

\[
\hat z_j^{\mathrm{cross}}(t)
=
\hat b_j^{\mathrm{cross}}
+(\hat{\boldsymbol\beta}_j^{\mathrm{cross}})^\top
\mathbf{x}^{\mathrm{cross}}_j(t).
\tag{4-6}
\]

Self 与 Cross 模型均仅由正常训练数据拟合。Cross 预测在可靠性评估阶段会与真实目标值比较，以判断预测器是否具有基本 skill；但正式双视角异常分数不使用该 observed-target residual，而只比较式（4-3）与式（4-6）的两个预测值。

## 4.4 局部多尺度滞后建模

### 4.4.1 冻结滞后集合

工业过程变量之间可能存在物料传播时间、执行器响应延迟和控制回路滞后。若只使用 instantaneous relation，模型可能无法覆盖异步协同模式。本文采用预设的多尺度局部滞后集合，以覆盖即时与短中期协同关系。两个视角的输入边界见表 4-1。

**表 4-1 双视角信息来源与冻结滞后**

| 视角 | 输入变量 | 冻结滞后 | 是否含目标当前值 | 作用 |
|---|---|---|---|---|
| Self View | 目标变量自身 | 1, 5, 10, 20, 30 | 否 | 表达目标自身动态状态 |
| Cross View | 同过程其他非恒定变量 | 0, 1, 5, 10, 20, 30 | 否 | 表达即时及滞后的局部协同状态 |

所有 endpoint 必须在各自连续 segment 内完成滞后索引，不允许通过插值、填充或跨边界取值构造历史。Self View 不含 lag 0，避免目标当前观测通过该路径进入双视角分歧；Cross View 的 lag 0 只对应其他变量。

### 4.4.2 数据集依赖性边界

上述滞后集合是正式实验前冻结的工程化多尺度覆盖，不是根据 Test 结果搜索出的最优组合。第 6.6.2 节的消融结果表明，HAI 对较长滞后表现出明确需求，而 SWaT 的排序方向不同，说明滞后作用具有 dataset dependence。因此，本章只主张预设滞后能够提供即时与短中期关系信息，不主张该集合在所有数据集、所有采样周期或所有攻击类型上全局最优。

## 4.5 双视角预测分歧分数

### 4.5.1 Target 级分歧与正常校准

对可靠目标 \(j\)，双视角原始分歧定义为

\[
d_j(t)
=
\left|
\hat z_j^{\mathrm{self}}(t)
-
\hat z_j^{\mathrm{cross}}(t)
\right|.
\tag{4-7}
\]

式（4-7）是本章方法区别于普通预测残差的关键：它不计算 \(|z_j(t)-\hat z_j(t)|\)，也不把 Self residual 或 Cross residual 直接作为 DUAL 核心分数。正常状态下，两个仅由正常关系学习的视角应形成相对一致的目标预期；当关系结构发生变化时，二者的预测可能分离。

为消除不同目标分歧尺度的差异，仅在 Normal Validation 的 target 分歧上建立稳健参考：

\[
\begin{aligned}
\mu_j
&=\operatorname{median}_{t\in\mathcal{V}}d_j(t),\\
\operatorname{MAD}_j
&=\operatorname{median}_{t\in\mathcal{V}}|d_j(t)-\mu_j|,\\
\sigma_j
&=\max\left(1.4826\operatorname{MAD}_j,10^{-6}\right).
\end{aligned}
\tag{4-8}
\]

其中 \(\mathcal{V}\) 表示正常验证时刻。target 级标准化证据为

\[
S_j(t)
=
\max\left(0,\frac{d_j(t)-\mu_j}{\sigma_j}\right).
\tag{4-9}
\]

截断操作使低于正常中位数的分歧不产生负异常证据；尺度下限 \(10^{-6}\) 防止 MAD 接近零时出现数值不稳定。该参考一经由 Normal Validation 冻结，Test 不参与重新拟合或自适应更新。

### 4.5.2 全局 max 聚合

设可靠性门控保留的目标集合为 \(\mathcal{R}\)，全局双视角协同异常分数定义为

\[
S_{\mathrm{DUAL}}(t)
=
\max_{j\in\mathcal{R}}S_j(t).
\tag{4-10}
\]

采用 max 的原因是协同关系破坏可能只集中在少数局部目标上，均值聚合可能稀释强但稀疏的关系证据。max 同时会放大任何单个不可靠 target 的异常高分，因此它不能脱离可靠性门控单独使用。本文没有根据 Test 结果搜索其他 target 聚合方式，也不声称 max 在所有数据集上最优。

## 4.6 Unified Reliability Gate

### 4.6.1 Normal-only 候选可靠性约束

可靠性门控只使用 Normal Train 与 Normal Validation，分为候选约束、Discovery Gate、Validation Gate 和分歧稳定性约束。首先，候选 target 必须为连续、非恒定且训练期可用的变量，并至少拥有一个同过程其他非恒定 source。Discovery 阶段在每个 Normal Train 连续段内部按时间顺序划分：前 80% 为 Discovery fit，后 20% 为 Discovery select。Discovery 的 Self/Cross 模型和常数 baseline 均由 Discovery fit 拟合，其中 baseline 是 Discovery fit 目标值的中位数，再在 Discovery select 上计算 skill。Validation 阶段的 Self/Cross 模型和常数 baseline 则由完整 Normal Train 拟合，其中 baseline 是完整 Normal Train 目标值的中位数，再在完整 Normal Validation 及其逐段前、后半部分上计算 skill。SWaT 对单个正常序列执行该时间划分；HAI 对三个 Normal Train 文件分别划分并保持文件边界。记某一正常评估分区 \(A\) 上 baseline、Self 和 Cross 的平均绝对误差分别为 \(\operatorname{MAE}^{A}_{\mathrm{base},j}\)、\(\operatorname{MAE}^{A}_{\mathrm{self},j}\) 和 \(\operatorname{MAE}^{A}_{\mathrm{cross},j}\)，视角 \(v\) 的预测 skill 定义为

\[
\operatorname{Skill}^{A}_{v,j}
=
1-
\frac{\operatorname{MAE}^{A}_{v,j}}
{\operatorname{MAE}^{A}_{\mathrm{base},j}},
\qquad v\in\{\mathrm{self},\mathrm{cross}\}.
\tag{4-11}
\]

当 baseline MAE 不大于零时，正式实现保守地将 skill 置为 0，因此该候选不能通过正 skill 门禁。Discovery Gate 要求 Discovery normal select 分区上的 Self skill 与 Cross skill 均大于 0，即两个视角至少都优于简单中位数预测器。Validation Gate 进一步要求在完整 Normal Validation、其前半段和后半段上，Self skill 与 Cross skill 均大于 0。由此每个候选共需通过八项正 skill 条件，避免只在一个验证局部阶段偶然有效的 target 进入全局 max。

除预测 skill 外，门控还要求正常分歧在 Validation 两个部分之间保持稳定。对正数 \(a,b\)，正式对称比值定义为

\[
r_{\mathrm{sym}}(a,b)
=
\max\left(
\frac{a}{\max(b,10^{-12})},
\frac{b}{\max(a,10^{-12})}
\right).
\tag{4-12}
\]

令候选目标集合为 \(\mathcal{T}_{\mathrm{cand}}\)，\(m_j^{(1)},m_j^{(2)}\) 为 Validation 前、后半段分歧中位数，\(q_{j,0.95}^{(1)},q_{j,0.95}^{(2)}\) 为对应 95% 分位数。对任意 \(j\in\mathcal{T}_{\mathrm{cand}}\)，最终可靠性条件为

\[
\begin{aligned}
j\in\mathcal{R}\ \Longleftrightarrow\;&
\operatorname{Skill}^{A}_{v,j}>0,
\quad \forall v\in\{\mathrm{self},\mathrm{cross}\},\\
&A\in\{\mathrm{Discovery},\mathrm{ValFull},\mathrm{ValFirst},\mathrm{ValSecond}\},\\
&r_{\mathrm{sym}}(m_j^{(1)},m_j^{(2)})\le 2,\\
&r_{\mathrm{sym}}(q_{j,0.95}^{(1)},q_{j,0.95}^{(2)})\le 3.
\end{aligned}
\tag{4-13}
\]

**表 4-2 可靠性门控判据**

| 门控层级 | 正常数据来源 | 冻结条件 |
|---|---|---|
| Candidate | Normal Train schema/可用性 | continuous、nonconstant、train usable、至少一个同过程其他 source |
| Discovery | Discovery normal select | Self skill>0 且 Cross skill>0 |
| Validation | Validation full/first/second | 三个分区的 Self/Cross skill 全部>0 |
| Stability | Validation first/second | symmetric median ratio≤2；symmetric q95 ratio≤3 |

### 4.6.2 Gate 与 max 的关系

可靠性门控先确定 \(\mathcal{R}\)，之后式（4-10）才对该集合执行 max 聚合。Gate 的目的不是挑选在 Attack/Test 上表现最好的 target，而是排除在正常数据上缺少基本预测 skill、或分歧统计随验证阶段明显漂移的候选。target 集合及其模型在读取 Test 结果前冻结，Attack 标签既不参与 skill 计算，也不参与稳定性阈值或 target 数量的选择。

这一顺序直接对应 max 的风险：若将大量低可靠候选无约束地放入最大值聚合，只需一个候选在正常状态产生异常大的分歧，便可能主导全局分数。第 6 章将通过 Historical CoMuT、controlled SWaT 和 HAI 分层检验这一机制；本章不提前给出实验结果，也不把 controlled 证据等同于所有真实攻击。

## 4.7 算法流程与复杂度

### 4.7.1 训练、校准和推理算法

**算法 4-1 局部滞后双视角协同异常检测**

```text
Input : Normal Train Ntr, Normal Validation Nval, Test sequence Xte,
        process grouping P, Lself={1,5,10,20,30},
        Lcross={0,1,5,10,20,30}, alpha=1
Output: DUAL score sequence S_DUAL

1  Fit the scaler on Ntr; split each segment into 80% Ntr_fit / 20% Ntr_select
2  Build candidate targets: continuous, nonconstant, train-usable,
   and with at least one other same-process source
3  for each candidate target j do
4      build Self features from target history at Lself
5      build Cross features from other same-process variables at Lcross
6      fit discovery models on Ntr_fit and full models on complete Ntr
7      use the Ntr_fit median baseline; evaluate discovery models on Ntr_select
8      compute the two Discovery skills
9      compute the complete-Ntr median baseline
10     evaluate full models on Nval full, first half and second half
11     compute six Validation skills and symmetric median/q95 ratios
12     keep j only if all eight skills are positive and ratios <= 2 / 3
13     estimate mu_j and sigma_j=max(1.4826*MAD_j, 1e-6) on Nval
14 end for
15 Freeze the reliable target set R, coefficients and calibration reference
16 for each legal Test endpoint t do
17     predict each j in R from Self View and Cross View
18     compute d_j(t)=|z_hat_self_j(t)-z_hat_cross_j(t)|
19     compute S_j(t)=max(0,(d_j(t)-mu_j)/sigma_j)
20     set S_DUAL(t)=max over j in R of S_j(t)
21 end for
22 return S_DUAL
```

算法中的 Test sequence 仅在全部 target、模型系数和正常参考冻结后用于生成分数；Test 标签只在第 6 章的最终评价阶段读取，不参与训练、门控或阈值选择。

### 4.7.2 参数与计算复杂度

设可靠 target 数为 \(R\)，target \(j\) 的同过程 Cross source 数为 \(m_j\)。Self View 的特征数固定为 \(p_{\mathrm{self}}=5\)，Cross View 的特征数为 \(p_{\mathrm{cross},j}=6m_j\)。若截距也计入模型参数，双视角分支的 learned coefficients 总量为

\[
P_{\mathrm{DUAL}}
=
\sum_{j=1}^{R}
\left[(|\mathcal{L}_{\mathrm{self}}|+1)
+(m_j|\mathcal{L}_{\mathrm{cross}}|+1)\right].
\tag{4-14}
\]

本实现通过中心化充分统计量构造正规方程并采用直接线性求解。以单个具有 \(p\) 个特征的 Ridge 系统为例，其主要代价可估计为 \(O(Np^2+p^3)\)，其中 \(N\) 是正常训练样本数；逐 endpoint 推理代价为
\(O\!\left(\sum_{j\in\mathcal{R}}(5+6m_j)\right)\)，状态规模与式（4-14）同阶。由于 Cross 范围限制在同一过程，复杂度取决于局部 source 数，而不是直接按全局通道数构造稠密全连接关系。

按照冻结 target 与 source 清单计数，SWaT DUAL 包含 771 个 learned coefficients，HAI DUAL 包含 7,038 个 learned coefficients；这些数值包括每个 Self/Cross Ridge 的系数和截距。Ridge 模型虽通过闭式线性系统拟合，但参数由正常数据学习，因此“learned coefficients”比笼统使用“神经网络 trainable parameters”更准确。完整参数和时延审计统一放在第 6.4 节。

从方法结构看，GDN 通过可学习图、图消息传递和 next-point forecast error 形成异常分数；本文则显式构造 Self/Cross 两种预测视角，并以二者的预测分歧作为关系证据。Park & Shim 的依赖矩阵、重构和 dependency anomaly score 融合也与本文的双预测视角不同。上述差异只用于界定方法结构，不构成本文对相关方法绝对优越的结论，也不支持“首个 dependency anomaly detector”的表述。

## 4.8 本章小结

本章提出局部滞后双视角协同异常检测方法。该方法在同一物理过程内分别建立目标自身历史预测和其他变量协同预测，通过两种预测结果的绝对分歧形成不依赖 observed-target residual 的显式关系证据。预设的多尺度滞后集合同时覆盖即时和短中期关系，可靠性门控则仅使用 Normal Train 与 Normal Validation，从预测 skill 和正常分歧稳定性两方面限制 max 聚合的候选集合。整个 DUAL 分支由 per-target 线性 Ridge 模型构成，参数量由可靠 target、同过程 source 和滞后数量共同决定。

需要强调的是，DUAL 面向 collaboration/relation inconsistency，并不保证所有工业攻击都会表现为跨变量协同关系异常。滞后贡献具有数据集依赖性，Gate 的作用也应结合异常语义和聚合机制解释。第 6 章将分别使用 Historical CoMuT、controlled SWaT 和 HAI 验证关系机制，并保留 SWaT official real attacks 上 DUAL 结果有限这一适用边界；本章不把这些分层证据扩展为跨数据集普适性结论。
