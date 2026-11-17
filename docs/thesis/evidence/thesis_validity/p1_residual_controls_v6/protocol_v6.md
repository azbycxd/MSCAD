# Codex 任务：Thesis-VALIDITY-P1（v6，在 v5 停止报告基础上修订 S1 一致性检查的数值判据）
## 双预测分歧 vs 预测残差：关键判据有效性对照（回顾性）

**状态标签**：所有产出标记 `RETROSPECTIVE_VALIDITY_AUDIT`。本实验属于"回顾性研究阶段中预先固定的分析方案"：SWaT / HAI 21.03 的 Test 已被多次观察，方法结构也在此期间确定。不得表述为预注册确认性实验或未见 Test 的独立验证。

**时间限制**：预计 10–20 分钟，**硬停止 30 分钟**。遇到"停止条件"立即停止并报告，不得自行修补后继续。

**v2 相对 v1 的改动**：
1. 受控 SWaT 按"被替换的 source 变量"记录，S1 作为阴性对照；
2. 数学检查区分"逐目标不敏感"与"全局系统响应"；
3. 新增正常期统计（第 3 步），必须在任何 Test 指标计算之前完成并记录哈希；
4. 判定表降级为"预设描述标签"，不作为创新是否成立的判决；
5. 可选：在"仅技能门控"目标集上的敏感性报告（只在已有冻结列表时执行）。

**v3 相对 v2 的改动**（以下条款已直接改写进正文，无单独补丁）：
1. S3 复现门槛扩展到端点对齐、q99 阈值、q99 工作点指标、事件命中，以及受控 SWaT 原 6 个 case；
2. S1 阴性对照改为先审计 donor manifest，以"配对逐点一致"为主判据，删除 AUROC ≥ 0.6 的规则；
3. 数学扰动明确为"修改序列副本并重建滞后特征"（操作 B）；
4. H-κ 增加 NOT_TESTABLE 规则；
5. 输出要求逐端点绑定元数据；
6. 增加超时后的 INCOMPLETE 处理规则。

**v4 相对 v3 的改动**（依据仓库 commit 43270769 的实际代码）：
1. 仓库中**没有保存 Ridge 系数文件**，正式 runner 每次都从 Normal 数据确定性重拟合。因此"重放"改为：调用正式 runner 中的同一组函数、使用相同数据区间与超参数进行确定性重拟合，并以第一步复现校验证明与正式结果等价。
2. 受控 SWaT 按 `run_innovation2d1_swat_controlled.py` 的真实设计改写：6 个目标、逐目标配对评价、无跨目标聚合、使用其自身的 calibration 区间。
3. 补充历史披露：S2 类方法（V3R selective cross-only 残差）曾在另一种设置下做过受控验证并失败，REPORT 必须写明这一点。
4. 仅技能门控目标集可以从重放过程中直接得到（只用 Normal 数据），不再依赖已保存列表。
5. 逐点分数复现改为"有存档则逐点比对，无存档则标 NOT_AVAILABLE"。

**v5 相对 v4 的改动**（依据 commit 5c0c8c7 的 v4 停止报告；修订在任何 S1/S2 Test 指标计算之前做出）：
1. **逐点门禁的参照对象写错了。** v4 第 2 节第 2 项本意是"正式 runner 产生的逐端点分数存档"。v4 实际比对的 `literature_compatible/raw_scores/swat/ours_dual/test_raw_scores.npz` 并不是正式 runner 的输出，而是 `recover_e2_r1d_artifacts.py` 用效率目录存的系数和另一条标准化路径（float32 cache 标准化后再转 float64）**重建**出来的。正式 runner 从未保存过逐端点分数。因此按 v4 自己的条文，第 2 项应为 `NOT_AVAILABLE`，而不是 FAIL。v5 明确这一点，不放宽任何容差。
2. **新增确定性检查**：同一重放连续运行两次，Normal Validation 与 Test 上的逐端点 S3 必须逐位相同（差值 = 0）。
3. **新增与重建存档的差异诊断**（只记录，不作门禁）：报告绝对差和相对差（|Δ| / max(|存档值|, 1)）的最大值、中位数、q99，以及差异最大的 10 个端点。
4. **新增正常期诊断**：SWaT 聚合后的正常 q99 = 226462.9（robust-z 单位），HAI 只有 40.2，SWaT 明显异常，说明可能有目标的正常 MAD 接近 1e-6 下限。第 3 步须报告每个目标的 σ_j、是否触及下限，以及每个目标在正常期 max 聚合中占主导的比例。**只报告，不修改方法。**
5. **失败即停**：任一门禁 FAIL 时，必须在读取下一数据集的 Attack 数据之前退出。v4 运行中"SWaT 失败后仍进入 HAI"的执行偏差，要在新报告里继续保留。

**v6 相对 v5 的改动**（依据 v5 停止报告；修订时尚未计算任何 S1/S2 真实攻击指标，受控 S1/S2 的 AUROC 也未保存）：
1. **S1 一致性检查的判据改为"逐对分数相等"。** v5 写的是"S1 逐对相同，AUROC 恰好为 0.5"，执行时被实现成浮点 `AUROC == 0.5` 的严格断言。在精确算术下，两组完全相同的样本 AUROC 确实是 0.5；但 sklearn 用梯形积分计算，累加时可能产生 1e-16 量级的舍入误差。v5 中 FIT601 的逐对相等检查已经通过，只有这条断言失败。v6 的规则：
   - **门禁**：每个目标 clean / corrupt 的 S1 逐对最大绝对差 = 0；
   - **记录**：用 `repr()` 原样保存实际的 S1 AUROC；
   - **数值合理性**：|AUROC − 0.5| ≤ 1e-12 记为 PASS；超过则说明不只是舍入问题，标记 `S1_CONSISTENCY_FLAG` 并停止受控部分。
2. **与 v5 冻结结果的一致性**：本轮全部重新运行，不复用 v5 的中间文件。第 3 步生成的 `normal_stats.csv` 的 SHA256 必须与 v5 冻结的哈希完全一致，否则停止。若文件里含有运行时间戳等非数值字段导致哈希必然不同，则改为：除这些字段外，所有列逐值完全相等（差值 = 0），并在报告中写明比较了哪些列。这一项用来证明 v5 到 v6 之间正常期统计没有发生任何变化。

---

## 0. 绝对禁止事项

1. 不训练任何神经网络（REC、基线模型都不加载、不训练，P1 用不到它们）。Ridge 只允许按第 2 节规定的方式做确定性重放，不得改动任何代码路径、数据区间或超参数。
2. 不改 lag、窗口、Ridge alpha、门控规则、目标集合、训练/验证划分、标签、事件定义。
3. 不使用 oracle 阈值、point adjustment、POT/SPOT，不用 Test 标签选择任何东西。
4. 不覆盖、不修改正式主表和已有 outputs 目录中的任何文件。
5. 不在看到结果后追加新的评分方案、数据集、门控变体或目标集合。
6. 若 S1 或 S2 结果更好，不得把它们升级为正式方法。
7. 不重新运行门控来生成新的目标集合。

---

## 1. 输入（只读）

**正式代码（只读，只能 import 调用，不得修改）**：
- `scripts/thesis/02_collaboration/run_swat2015_final_unified.py`
- `scripts/thesis/02_collaboration/run_hai21_final_unified.py`
- `scripts/thesis/02_collaboration/run_innovation2d1_swat_controlled.py`（及其依赖 `run_innovation2b_controlled.py`、`run_innovation2d0_dual_view_comut.py`、`run_innovation2a2_joint_lagged.py`）
- 上述脚本 import 的 `w.leave_target_out_models`、`w.fit_ridge`、`evaluate_views`、`score_segment`、`robust` 等函数

**正式产物（只读）**：
- `outputs/thesis_swat_final_unified/`、`outputs/thesis_hai21_final_unified/`：`dual_targets_unified_frozen.csv`（SWaT 15 / HAI 54）、`global_metrics.csv`、`thresholds.json`、事件表、`manifest.json`
- `outputs/thesis_innovation2d1_swat_controlled/`：逐目标指标与 summary
- `outputs/thesis_innovation2b2_relation_breaking/donor_manifest.csv`
- 本地 Normal / Attack 数据缓存（按正式 config 解析的路径）
- （若存在）`outputs/thesis_evaluation/literature_compatible/` 下 DUAL 的逐端点 `test_raw_scores.npz`

**实现方式**：新建 `scripts/thesis/06_validity/run_validity_p1.py`，**import** 正式 runner 中的函数，不复制粘贴后再修改。正式 runner 的 `main()` 不得调用（它会写入正式 outputs 目录）。

**停止条件 A**：缺少数据缓存、`dual_targets_unified_frozen.csv`、donor manifest 或正式指标表 → 停止，报告缺少哪些文件。不得用其他数据或自行构造的目标列表替代。

---

## 2. 第一步：复现校验（必须先通过）

**重放定义**：Ridge 是闭式解，给定相同数据、相同特征构造和相同 alpha，结果是确定的。重放 = 用正式 runner 的同一组函数、相同的 Train / Discovery 区间、lag、alpha 和 train-only scaler，重新得到 Self / Cross 模型；**所有拟合只读 Normal 数据，并在读取任何 Attack 数据之前完成**。重放出的可靠目标列表必须与 `dual_targets_unified_frozen.csv` 完全一致（包括顺序），否则停止。

然后得到每个可靠目标在每个合法端点上的 `zhat_self`、`zhat_cross` 和观测值 `z`。

计算 S3 = |zhat_self − zhat_cross|，按原正式规则（完整 Normal Validation 上计算 median 与 max(1.4826·MAD, 1e-6)，截断为非负，对目标集合取 max）得到 S_DUAL。

**通过标准（第 1、2b、3、4、5、6 项全部满足；第 2 项按其条件执行）**：
1. **端点对齐**：重放分数的端点集合、顺序和原始时间线与正式工件完全一致（数据集、段/文件、局部索引逐一相同）。
2. **逐点分数**：只有当存档**由正式 runner（`run_swat2015_final_unified.py` / `run_hai21_final_unified.py`）的同一代码路径生成**时，才做逐点比对，容差 ≤ 1e-9。当前已知 SWaT、HAI 都没有这样的存档，本项记 `NOT_AVAILABLE`，并写明原因：`literature_compatible` 存档来自 E2-R1D 重建路径。不得为了通过检查而临时放宽容差，也不得改用重建路径来追平存档。
2b. **确定性**：重放连续运行两次，Normal Validation 与 Test 逐端点 S3 完全相同（差值 = 0）。
2c. **重建存档差异诊断**（不作门禁）：按 v5 改动第 3 条报告。
3. **校准**：重放的 Normal Validation q99 阈值与正式工件一致（容差同上）。
4. **主表指标**：与主表一致到小数点后 6 位：
   - SWaT：AUROC 0.675631，AUPR 0.167510，Precision 0.087166，Recall 0.008605，F1 0.015663，FPR 0.012455
   - HAI：AUROC 0.916334，AUPR 0.570193，Precision 0.554350，Recall 0.679446，F1 0.610556，FPR 0.012443
5. **事件命中**：SWaT 2/36，HAI 44/50，HAI HIGH19 命中比例中位数 0.841085。
6. **受控 SWaT**：原 6 个 case 的 S3 逐 case AUROC / AUPR 与冻结结果一致（中位 AUROC 0.996913，中位 AUPR 0.996931）。

注意：以上是唯一允许在第 3 步之前读取的 Test 指标，因为它们已在主表中公开。

**停止条件 B**（必须在读取下一数据集 Attack 数据之前生效）：
- 第 1–5 项任一不通过 → 停止真实攻击部分，输出差异统计（最大差、差异端点数、首个差异位置）。
- 第 6 项不通过 → 停止受控 SWaT 部分。
- 不得修改评分或校准规则来追平历史指标。

---

## 3. 第二步：正常期统计（只用 Normal Validation，必须在任何 S1/S2 Test 指标之前完成）

对每个可靠目标 j，在完整合法 Normal Validation 上计算：

**3.1 带符号残差统计**
- Var(r_self)、Var(r_cross)、Cov(r_cross, r_self)
- κ_j = Cov(r_cross, r_self) − ½·Var(r_self)（κ_j > 0 等价于 Var(r_cross − r_self) < Var(r_cross)）

**3.2 绝对分数统计（与正式校准口径一致）**
对 S1、S2、S3 三种逐目标原始分数，各自报告：median、1.4826·MAD、q95、q99，以及 q99 与 median 之比。

**3.2b 正常期主导性诊断（只报告，不改方法）**
对每个可靠目标报告：σ_j 是否等于 1e-6 下限；σ_j 的数值；在 Normal Validation 上，聚合 max 由该目标给出的端点比例；该目标单独 robust-z 分数的 q99。S1、S2、S3 都要报告。

**3.3 写入与冻结**（v6：哈希必须等于 v5 冻结的 `normal_stats.sha256`）
将以上结果写入 `normal_stats.csv`，计算 SHA256，写入 `normal_stats.sha256`，并记录完成时间戳。**这一步完成之前，不得计算任何 S1/S2 的 Test 指标。** 报告中要写明这个顺序。

**预先固定的描述性假设 H-κ**（只是理论预测，不作为创新是否成立的判据）：
κ_j > 0 的目标中，S3 的逐目标 AUROC 高于 S2 的比例，大于 κ_j ≤ 0 的目标中的这一比例。

- 若 κ_j > 0 或 κ_j ≤ 0 任一分组为空，该数据集的 H-κ 记为 `NOT_TESTABLE`，不得写成成立或失败。
- 无论是否可检验，都报告每个目标的 κ_j 以及 S3 − S2 的逐目标 AUROC 差异。
- 即使趋势成立，也只说明正常期二阶统计与检测表现之间存在关联，不代表因果关系，也不构成 AUROC / AUPR 的保证。

---

## 4. 第三步：计算三个评分（只改原始分数公式）

本节规则适用于真实攻击部分（SWaT 15 个、HAI 54 个可靠目标）；受控 SWaT 的规则见 5.2。

对每个可靠目标 j：

| ID | 每目标原始分数 | 角色 |
|---|---|---|
| S1 自历史残差 | \|z_j(t) − zhat_self_j(t)\| | 对照 |
| S2 跨变量残差 | \|z_j(t) − zhat_cross_j(t)\| | 经典虚拟传感器残差 |
| S3 双预测分歧（正式） | \|zhat_self_j(t) − zhat_cross_j(t)\| | 本文方法 |

三者共用完全相同的后处理：
- 每目标在**同一批合法 Normal Validation 端点**上计算该评分自身的 median 和 max(1.4826·MAD, 1e-6)；
- 截断为非负，对同一目标集合取 max；
- 阈值 = 该评分在同一批 Normal Validation 端点上的 q99，判定规则为严格大于；
- Test 只用于最终评价。

**目标集合说明（原文写入报告）**：
目标集合由正式双视角门控选出。8 个正技能条件对 Self 与 Cross 是对称的；但分歧稳定性比值筛选只针对 S3，可能使集合偏向 S3。因此本实验的结论仅限于"相同预测器与相同目标条件下"的比较，不代表 S2 在为其单独优化的目标集上的最佳表现。

---

## 5. 第四步：评价指标

### 5.1 真实攻击（SWaT Official、HAI 21.03）
每个评分报告：AUROC、AUPR、Precision、Recall、F1@q99、FPR@q99、官方事件命中数（原冻结规则：事件内合法端点报警比例 ≥ 0.10）。
- HAI 另报告：HIGH19 事件命中数与命中比例中位数。
- 输出逐事件命中比例表（三个评分并排），并统计 S3 命中比例高于 S2 的事件数和低于 S2 的事件数。
- 输出逐目标 AUROC 表（三个评分并排，含 κ_j），用于检验 H-κ。

### 5.2 受控 SWaT 关系破坏（按 `run_innovation2d1_swat_controlled.py` 的真实设计）

**实际设计（以代码为准，先核对再执行）**：
- 目标：6 个冻结目标 `AIT202, FIT201, DPIT301, FIT301, FIT401, FIT601`（来自 2A2 stable 列表），**不是**真实攻击部分的 15 个目标；
- 每个目标 2,000 对样本，来自 `thesis_innovation2b2_relation_breaking/donor_manifest.csv` 的 `clean_endpoint` / `donor_endpoint`；
- clean：Self 与 Cross 都取 clean 时刻；corrupt：Self 和目标观测仍取 clean 时刻，**只有 Cross 输入取 donor 时刻**；
- 校准区间：`[CALIBRATION_START, CALIBRATION_END)`（Normal Validation 前半段），不是完整 Normal Validation；
- 逐目标单独评价（`b2.balanced_metrics`），**没有跨目标取 max**。

**三个评分在受控部分的定义**（每个目标各自校准，用上述区间的 median / MAD 与 q99）：
- S1：clean = corrupt = |z(clean_t) − ẑ_self(clean_t)|；
- S2：clean = |z(clean_t) − ẑ_cross(clean_t)|，corrupt = |z(clean_t) − ẑ_cross(donor_t)|；
- S3：clean = |ẑ_self(clean_t) − ẑ_cross(clean_t)|，corrupt = |ẑ_self(clean_t) − ẑ_cross(donor_t)|。

**复现前提**：用上述方式重放得到的 S3 逐目标 AUROC / AUPR、Recall@q99、FPR@q99，必须与 `thesis_innovation2d1_swat_controlled` 的冻结结果一致到小数点后 6 位（这就是第 2 节第 6 项）。

**每个目标报告**：三个评分的 AUROC、AUPR、Recall@q99、FPR@q99；该目标的 Cross source 列表；正常期（校准区间）三个评分的 median、MAD、q95、q99。

**S1 一致性检查**（替代 v3 的阴性对照）：按代码设计，corrupt 样本没有改动目标的观测值和 Self 输入，所以 S1 在每一对样本上应该逐点相同。这一项只是确认实现正确，不是有信息量的阴性对照。门禁是每个目标逐对最大绝对差 = 0；实际 AUROC 用 `repr()` 保存，|AUROC − 0.5| ≤ 1e-12 视为舍入误差范围内（见 v6 改动第 1 条）。任一项不满足，标记 `S1_CONSISTENCY_FLAG` 并停止受控部分。

**理论预期（写入报告，供对照）**：同一对样本中，S2 与 S3 的带符号量变化相同（都等于 ẑ_cross(clean_t) − ẑ_cross(donor_t)），差别只来自各自正常基线的位置和宽度。

### 5.3 Historical CoMuT（条件纳入）
只有当冻结数据和原评分定义可以不加修改地运行时才纳入，同样报告三个评分。否则跳过，并写明原因。

### 5.4 敏感性：仅技能门控目标集（真实攻击部分）
正式 runner 在门控时会为每个候选目标算出 8 个技能值和分歧稳定性比值，重放过程中可以直接拿到这些数据（只用 Normal 数据）。
- 仅技能目标集 = 8 个技能值都 > 0 的候选目标（不看稳定性比值）；
- 在第 3 步写入 `normal_stats.csv` 之前，先把这个集合写入 `skill_only_targets.csv` 并记录哈希；
- 在该集合上按第 4 步同样的规则报告 S2、S3 的 AUROC / AUPR；
- 这一项只用来说明第 4 步目标集合的偏向，不参与第 7 节标签。

---

## 6. 第五步：数学性质检查（只用 Normal Validation，不用 Test）

在 SWaT 和 HAI 各选 1 个可靠目标 j。优先选择同时是其他可靠目标 Cross source 的变量；如果没有，在报告中写明。

**t0 的选择**：Normal Validation 中的一个合法端点，要求 t0−30 与 t0+30 都在同一连续段内，不跨 SWaT hard gap，不跨 HAI 文件边界。

**扰动操作（必须是"操作 B"）**：
- 复制一份 Normal Validation 原始（标准化后）时间序列；
- 在副本中只修改 z_j(t0) 这一个值，改为 z_j(t0) + 5 × 该目标正常 MAD；
- 用副本**重新构造所有受影响的滞后特征**，使修改在 t0+1、t0+5、t0+10、t0+20、t0+30 进入目标 j 的 Self 输入，并在 t0 进入以 j 为 source 的其他目标的 Cross 输入；
- 模型系数、scaler、门控集合、正常参考统计全部冻结，只做确定性前向预测。

**检查项**：
1. **恒等式**：对全部 Normal Validation 端点验证 S3 == |r_cross − r_self|，报告最大绝对差。
2. **逐目标当前值不敏感**：报告目标 j 的 S1、S2、S3 在 t0 处的变化；报告目标 j 的 S3 在 t0+1、t0+5、t0+10、t0+20、t0+30 处的变化。
   预期：t0 处目标 j 的 S3 不变，S1、S2 变化；之后各 lag 位置的 S3 可能变化。
3. **全局系统响应**：报告所有以 j 为 Cross source 的其他可靠目标 k 在 t0 处的 S3 变化，以及全局 S_DUAL(t0) 的变化。
   预期：由于 Cross 允许 lag 0，这些目标在 t0 处就可能响应。

逐目标结果与全局聚合结果分开报告，不得混写。

这一步只确认数学性质，不作为性能结论。

---

## 7. 预设描述标签（运行前固定；只是描述，不判定创新是否成立）

记 ΔHAI = AUPR(S3) − AUPR(S2)（HAI 21.03），
ΔCTRL = 受控 SWaT 6 个目标 AUROC 中位数(S3) − 中位数(S2)。
0.02 是预先设定的实用差距，不是显著性检验。

| 标签 | 条件 |
|---|---|
| R-UP | ΔHAI ≥ +0.02 且 ΔCTRL ≥ +0.02 |
| R-DOWN | ΔHAI ≤ −0.02 且 ΔCTRL ≤ −0.02 |
| R-EQ | 其余情况（包括两者方向不一致） |

SWaT Official 不参与标签，只作为边界证据单独报告。

Codex 只输出：命中的标签、ΔHAI、ΔCTRL、H-κ 检验的两个比例，以及逐目标、逐事件差异的统计。**不得**解释其对论文创新性的含义，也不得改写任何论文章节。最终判断由作者与导师根据完整证据做出。

---

## 8. 输出

目录：`outputs/thesis_validity/p1_residual_controls_v6/`（新建，不写入其他目录）

**完整性要求**：所有 Test 分数必须逐端点绑定以下字段：dataset、segment / file、local index、原始 label、valid mask、source hash。不允许只保存分数数组和聚合指标。

- `replay_check.json`：第一步全部 6 项复现校验结果，含 dtype 与所用容差
- `normal_stats.csv` + `normal_stats.sha256`：第二步结果，含完成时间戳
- `scores_{swat,hai}_{S1,S2,S3}.parquet`（或 csv）：聚合后的逐端点评分，附上述绑定字段
- `per_target_scores_{swat,hai}.parquet`：逐目标、逐端点的 S1 / S2 / S3 原始分数，附上述绑定字段
- `donor_manifest_audit.json`：受控 SWaT 构造方式核对结果（目标、样本数、clean/donor 配对、校准区间是否与 5.2 描述一致）
- `normal_reference_{swat,hai}.csv`：每个评分、每个目标的 median、MAD 尺度，以及聚合后的 q99 阈值
- `metrics_main.csv`：SWaT / HAI 三个评分的主指标
- `per_target_{swat,hai}.csv`：逐目标 AUROC（S1/S2/S3）+ κ_j
- `events_{swat,hai}.csv`：逐事件命中比例；HAI 含 HIGH19 标记
- `controlled_swat.csv`：6 个目标的逐项指标、Cross source 列表、正常期分数统计、S1 一致性检查结果
- `comut.csv`（若纳入）
- `skill_only_targets.csv` + 哈希、`sensitivity_skill_only.csv`：5.4 结果
- `math_checks.json`：第五步结果
- `decision.json`：标签、ΔHAI、ΔCTRL、H-κ 两个比例
- `manifest.json`：所有输入工件路径与 SHA256、源码哈希、各步骤时间戳、环境版本
- `REPORT.md`：简短事实报告，按上述顺序列出；所有异常、跳过项、停止条件、阴性对照标记都要写明

所有文件头部或字段中注明 `RETROSPECTIVE_VALIDITY_AUDIT`。

---

## 9. 超时与不完整处理

若在 30 分钟硬停止前无法完成全部项目：
- 保存已完成的原子步骤及 manifest；
- 将状态标为 `VALIDITY_P1_INCOMPLETE`，并在 `REPORT.md` 中列出已完成和未完成的步骤；
- 不得为了赶时间而缩减 case、跳过复现校验或跳过第 3 步的哈希冻结。

## 10. 历史披露（必须写入 REPORT.md）

仓库中 `docs/thesis/v3r_selective_cross_only.md` 记录：S2 类方法（V3R selective cross-only 残差：目标排除，Cross lags `[1,5,10,20,30]` 不含 lag 0，只用 Top5 predictor）在 23 个目标、46 个受控 cell 上的 median AUROC 为 0.523134，状态 `SELECTIVE_CROSS_ONLY_NOT_SUPPORTED`。双预测分歧是在这之后设计的。

REPORT 必须原样说明：
- 上述历史结果与本轮 S2 的设置不同（predictor 集合、是否含 lag 0、受控扰动方式都不同），二者不可直接比较；
- 本轮 S2 与 S3 使用同一组预测器，目的正是把"评分公式的作用"和"预测器设置变化的作用"分开。

## 11. 研究边界（重申）

- 保持 `RETROSPECTIVE_VALIDITY_AUDIT` 标签；
- 不得以本轮结果替换正式方法或主表；
- 不得把本轮称为独立留出验证；
- S3 即使表现更好，也只说明检测上有增量，不证明学术上首次提出。