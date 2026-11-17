# 第6章 实验与结果分析

本章在冻结的数据、方法与评价边界内集中说明实验设计与结果。本文题目为《轻量化多变量时序协同异常检测方法研究》。本轮仅给出数据集、实验设置与评价协议，不在本节提前讨论任何方法的性能优劣。

## 6.1 数据集与实验设置

### 6.1.1 数据集及研究角色

本文不把所有数据源视为同等地位的性能基准。SWaT 2015 Official Physical 与 HAI 21.03 是主要真实工业数据集；Historical CoMuT 和受控 SWaT 关系破坏实验用于逐层验证关系异常机制；WADI.A2 仅用于考察正常域偏移下的适用边界，不构成第三个主要性能数据集。各数据集的角色如表 6-1 所示。

**表 6-1 数据集及研究角色**

| Dataset | Type | Variables | Normal data | Test data | Attack/event | Role |
|---|---|---:|---|---|---|---|
| SWaT 2015 Official Physical | 真实工业控制系统 | 51 | Normal v1，共 495,000 行；正式 REC 划分为 Train `[0,396000)`、Validation `[396000,495000)` | Attack v0，共 449,919 行，按 82 s 缺口分为两个连续段 | 36 个 attack-list 事件；点标签形成 35 个连续异常区间 | **Primary real industrial dataset** |
| HAI 21.03 | 真实工业控制系统 | 79（P1/P2/P3/P4 分别为 38/22/7/12） | 3 个纯正常 Train 文件，共 921,603 行；每文件前 80% 为 Train、后 20% 为 Validation | 5 个 Official Test 文件，共 402,005 行 | 50 个官方攻击事件；其中 HIGH collaboration scenarios 共 19 次出现 | **Primary real industrial dataset**；协同机制的重要真实验证集 |
| Historical CoMuT | 合成关系异常基准 | 每 case 2–4 | 13 个 case；每 case 的 `train_no_anomaly.csv` 为 10,000 行且全部正常 | 每 case 的 `test.csv` 为 10,000 行 | Short、Medium、Large、Multiple-short 关系异常 | **Relation-anomaly mechanism benchmark**，不是真实工业攻击数据 |
| SWaT controlled relation breaking | 真实工业正常轨迹上的受控机制实验 | 沿用 SWaT 51 维源轨迹 | 使用冻结的 SWaT Normal Train/Validation 轨迹与训练期统计 | 对冻结 target、sample index、source replacement 和 donor 构造 clean/corrupt 对照 | donor manifest 已冻结；不读取 Official Attack 数据 | **Controlled benchmark**，用于隔离关系破坏机制，不等同于 Official real attacks |
| WADI.A2 | 真实工业控制系统外部数据 | 原始 127；冻结可用 123 | `WADI_14days_new.csv`；审计后为两个正常连续段 | 单一 Attack 文件，共 172,801 行 | 15 个官方攻击，14 个完成绑定；Attack 13 保留 source conflict | **DOMAIN_SHIFT_STRESS_TEST**，不进入 Primary 主性能排名 |

表中 SWaT 的 36 个 attack-list 事件与 35 个点标签连续异常区间是两种不同对象：前者保留官方攻击编号及事件身份，后者来自 Physical 点标签的连续区间。二者不得相互替代。HAI 的 HIGH19 是由冻结事件分类得到的 19 次高协同相关事件出现，不是从测试性能中再筛选的子集。

### 6.1.2 主要真实工业数据集

#### （1）SWaT 2015 Official Physical

SWaT 数据使用官方 Physical 发布包。Normal 数据采用完整的 `SWaT_Dataset_Normal_v1.xlsx`，不在官方 v1 之外额外删除启动阶段。正式重构分支将 495,000 个正常点按时间划分为 Train `[0,396000)` 与 Normal Validation `[396000,495000)`；Attack 数据采用完整的 `SWaT_Dataset_Attack_v0.xlsx`。模型输入为全部 51 个过程变量，不依据 Test 中的常量、近常量或变量类型删除通道。

Attack 时间线在 `2015-12-31 21:00:36` 与 `21:01:58` 之间存在 82 s 间隔。该间隔被冻结为 hard session boundary，不插值、不前向填充、不补零；任何 window、lag 或时间平滑均必须完全位于同一个连续段内。点级 ground truth 使用 Physical 工作簿的冻结 exact mapping，并保留原始标签；事件级 ground truth 则使用 36 个 canonical attack-list intervals。两套语义分别服务点级和事件级评价。

#### （2）HAI 21.03

HAI 21.03 包含 3 个 Normal Train 文件和 5 个 Official Test 文件。文件中实际存在 79 个过程变量，其中 P1、P2、P3、P4 分别包含 38、22、7、12 个变量。3 个 Train 文件共 921,603 行，所有冻结标签均为正常；5 个 Test 文件共 402,005 行。每个 Train 文件独立按时间取前 80% 作为 Normal Train、后 20% 作为 Normal Validation，禁止先拼接文件再切分。

HAI 的 window、lag 和平滑运算不得跨越文件边界。Official Test 中的点标签形成 50 个攻击事件，事件身份及 HIGH19、DIRECT31 等分组均沿用冻结事件目录；测试标签不参与过程、target、lag、特征或阈值选择。

### 6.1.3 机制基准与域偏移压力测试

Historical CoMuT 由历史官方 `timeeval-gutenTAG==1.0.0`、固定 commit 与未修改的官方 CoMuT 配置生成。13 个 case 均提供独立的全正常训练文件和测试文件，每个文件 10,000 行。正常通道具有高度相关的协同变化，测试异常按 Short、Medium、Large 与 Multiple-short 划分，因而用于检验纯关系异常敏感性。该数据是合成机制基准，不能被描述为真实工业攻击数据。

受控 SWaT 关系破坏实验使用真实 SWaT 正常轨迹，但通过冻结 donor 机制替换同阶段 source context，以形成 clean/corrupt 对照。target、sample index、donor endpoint、source replacement 和 donor manifest 均在评价前冻结，Official Attack 工作簿不参与构造。该实验用于在真实工业正常轨迹上隔离关系破坏机制，其结论不能直接替代 Official SWaT real-attack 评价。

WADI 使用 WADI.A2 的 14-day normal source 和单一 Attack 文件。原始 127 个过程变量中，4 个字段在合并的 Normal Train 上缺失率为 1，按冻结规则排除，形成 123 维模型 schema；Test 不参与字段排除。其余缺失点不做插补，而是作为 hard segment boundary。Normal 时间线中的 264,001 s 间隔将正常数据分为两个连续段，window 与 lag 均不得跨段。

WADI 官方表包含 15 个攻击，冻结协议成功绑定其中 14 个。Attack 13 的官方 12:07:30–12:10:52 与 CSV 点标签不重合，而 CSV 存在约早 12 h 的持续时间相近区间；本文保留该 source conflict，不实施事件特定的 12 h 修正。WADI Test 仅按预注册流程打开一次用于最终压力测试，之后不执行 Test-driven rescue、特征重选、阈值重调或方法修改。

### 6.1.4 实现环境、方法与对照方法

冻结环境记录为 Windows 10 `10.0.19045 SP0`、Python `3.10.19`、PyTorch `2.12.1+cu126` 与 PyTorch CUDA runtime `12.6`。GPU 为 NVIDIA GeForce RTX 4060（8,188 MiB）；正式记录未能可靠取得 CPU 型号，因此本文不作推测。数据划分、随机种子、模型状态、阈值和原始分数均通过项目 manifests 与输出哈希追踪。

最终方法由三个职责不同的组成部分构成：REC 是轻量重构基础分支；DUAL 是第 4 章定义的局部滞后双视角协同分支；TAIL 是第 5 章定义的正常尾概率校准与融合分支。本节仅登记模块角色，不重复方法公式。

统一实测对照包括 LSTM-AE、USAD 与 GDN，分别代表经典循环重构、对抗式双解码重构和图依赖预测。公平性约束是所有预处理参数只能由 Normal Train 拟合，Validation/Test 只能 transform。LSTM-AE、GDN 与本文方法使用 Train-only StandardScaler；USAD 因官方 sigmoid decoder 的输出范围要求，使用逐通道 Train-only MinMaxScaler `[0,1]` 且 `clip=False`。该架构兼容性差异不改变共同的无泄漏边界。

Park & Shim、LatentFlow、MAD-TCN 与 CCM-TAD 等 2026 年方法不作为本地统一实测 baseline。它们仅以原论文结果和源码协议审计进入 **Paper-Reported Recent Literature** 补充比较，并与统一实测结果分层呈现。

效率评价在同一 RTX 4060 主机上执行，batch size 固定为 1；先进行 100 个合法 endpoint 的 warm-up，再计时 1,000 个合法 endpoint，并报告参数、状态存储、时延和显存协议。具体效率数值留待 6.4 节。

## 6.2 评价协议

### 6.2.1 Primary Strict Evaluation

本文将 **Primary Strict Evaluation** 定义为主要评价协议。其核心边界如下：

1. 数据预处理、scaler 和模型参数只由 Normal Train 拟合；
2. Normal Validation 只用于 checkpoint selection、正常参考/校准以及 q99 threshold；
3. Test 只在模型、分数、校准和阈值全部冻结后用于最终评价；
4. Primary 中不允许 Test tuning、Point Adjustment、POT/SPOT 或 Test-best threshold。

不同方法保留各自已冻结的 Normal Validation endpoint score。统一主表的二值规则经正式 manifest 核对为

\[
\hat y_t=\mathbb I\bigl(s_t>Q_{0.99}(S_{\mathrm{val}})\bigr),
\]

即分数严格大于 q99 时才报警，等于阈值不触发报警。AUROC 与 AUPR 评价连续分数的排序能力；Precision、Recall、F1 与 FPR 描述冻结 q99 operating point。主排序以 AUPR 优先、AUROC 其次。F1 只表示该固定 operating point，不能单独代表完整的 ranking quality。

### 6.2.2 合法 endpoint 与边界

窗口重构模型在 endpoint \(t\) 使用 `[t-63,t]`；GDN 使用 `[t-64,t-1]` 预测 \(t\)。Historical Strict 结果保留每个正式方法当时冻结的合法 endpoint 集合，不因后续比较而覆盖。

为排除不同历史起点造成的时间线差异，E2 另行冻结 **文献兼容补充评价** 的 common endpoint mask：SWaT 为 449,791 个 endpoint，HAI21 为 401,685 个 endpoint。该 mask 通过 segment/file、local/global index 与 timestamp 的精确交集建立，不用数组长度裁剪。common-mask 结果只用于 supplementary comparison，不反向替换 Primary historical table。

所有数据处理均保留连续性边界：SWaT 不跨 82 s hard gap，HAI 不跨官方文件，WADI 不跨正常 session gap 或缺失点边界。时间平滑和 PA 同样受这些边界约束。

### 6.2.3 点级指标与事件级指标

点级评价以每个合法 endpoint 的 ground truth 与连续异常分数计算 AUROC/AUPR，并以冻结 q99 预测计算 Precision、Recall、F1 和 FPR。事件评价不把点级预测改写为整段异常，也不称为 event-F1。

对每个冻结事件，先计算其合法 endpoint 中预测为异常的比例 \(h_e\)。统一实验报告中的 event hit 判据为 \(h_e\ge 0.10\)，随后统计满足该判据的事件数。SWaT 使用 36 个 canonical attack-list event identities，即使点标签相连，ID 10/11 与 ID 27/28 仍分别评价；HAI 使用 50 个官方事件，并可按冻结的 HIGH19、DIRECT31 分组汇总。该事件统计与 point-wise 35/50 个连续标签区间是不同层次的对象。

### 6.2.4 文献兼容补充评价

为解释 score ranking、阈值迁移与经典区间评价之间的差异，E2 在 common endpoint 上定义三类回顾性指标：

- **F1-Oracle**：在 Test 上枚举精确分数边界并选择 Test-best F1 threshold，不使用 PA；
- **PA-F1@q99**：先用 Normal Validation q99 得到二值预测，再施加 classic PA；
- **PA-F1-Oracle**：使用 Test-best threshold，并施加 classic PA。

Classic PA 的规则是：若某个真实连续异常区间内至少有一个预测异常点，则把该真实异常区间内的全部点设为预测异常；正常位置的原始预测不删除。PA 不得跨越 SWaT hard gap 或 HAI file boundary。由于 PA 可能显著改变 Recall 和 F1，PA 指标不能与 Strict F1 混用。

F1-Oracle 使用 Test labels 选择 threshold，其目的仅是诊断原始分数排序与 Normal Validation 到 Test 的 threshold transfer 差异。上述三类结果全部是 retrospective、supplementary diagnostic，不是可部署 operating point，也不属于 Primary Strict Evaluation。

### 6.2.5 Paper-Reported Recent Literature

2026 年近期方法采用 **Paper-Reported Recent Literature** 层级：只引用原论文正式结果，并同时登记其数据处理、阈值、PA、模型选择和指标语义。当前外部记录中 `DIRECT` comparability count 为 0。主要原因包括 classic PA、Test-best threshold、Affiliation 指标、VUS、PF1/SF1 以及 Test-driven 参数或阈值选择；这些语义不能转换为本文 Strict 指标。

因此，近期论文结果只作 protocol-aware supplementary comparison，不与 Primary Strict Evaluation 混成单一排行榜。三类评价协议的边界汇总于表 6-2。

**表 6-2 评价协议层级**

| Protocol | Threshold | PA | Test-label tuning | Purpose | Used as Primary? |
|---|---|---|---|---|---|
| Primary Strict Evaluation | Normal Validation q99，规则为 `score > threshold` | No | No；Test 仅用于最终评价 | 主要排序与固定 operating-point 评价 | **Yes** |
| Common-endpoint q99 | 原方法冻结的 Normal Validation q99 | No | No | 在完全相同 timeline 上补充比较 | No；supplementary |
| F1-Oracle | Test-best exact threshold | No | Yes | 诊断 ranking 与 threshold transfer | No；retrospective diagnostic |
| PA-F1@q99 | Normal Validation q99 | Classic PA | PA 使用 Test ground-truth intervals | 与采用 PA 的文献作受限比较 | No；retrospective diagnostic |
| PA-F1-Oracle | Test-best exact threshold | Classic PA | Yes | 同时审计 Test-best 与 PA 的影响 | No；retrospective diagnostic |
| Paper-Reported Recent Literature | 按原论文记录 | 按原论文记录 | 按原论文记录 | 2026 方法的协议感知补充对照 | No；external `DIRECT=0` |

### 6.2.6 稳定性、消融与效率协议

三随机种子稳定性评价使用 seeds `1,42,43`，只针对含随机训练的 REC 与由相应 REC seed 形成的 TAIL。DUAL 在冻结数据、特征和 solver 下是确定性分支，不伪造三次随机运行或 `mean±0`。Primary 主表仍保留原正式 seed；三种子均值、标准差和范围仅作为稳定性补充证据。

消融实验继续使用 Strict protocol：Normal Validation q99、无 PA、无 Oracle。所有消融均标记为 `RETROSPECTIVE_ABLATION`，用于解释冻结方法组成，不允许根据消融结果反向修改 Final method。

效率评价报告 Trainable Params、serialized state size、mean/median/p95 latency 与 peak VRAM。参数量与状态存储必须分开：TAIL 的融合本身有 0 个 trainable parameters，但需要保存经验分布数组、阈值等 non-trainable calibration storage，因而不能把“无可训练参数”等同于“无状态开销”。

## 后续章节占位

- 6.3 总体检测性能比较
- 6.4 轻量化与效率分析
- 6.5 协同关系异常机制验证
- 6.6 消融实验
- 6.7 稳定性与文献兼容评价
- 6.8 适用边界与讨论

