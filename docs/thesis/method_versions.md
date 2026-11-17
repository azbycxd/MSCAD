# Method versions

本表按“研究假设发生实质变化”划分方法版本，不按脚本数量划分。审计日期：2026-09-14。路径默认相对于相应 worktree；Legacy SWaT 与 HAI 历史产物位于 `mscad` main worktree，A12 产物位于 `mscad-swat2026` worktree。

## V1 — Original MSCAD

### 方法定义

- Reconstruction backbone：轻量一维卷积自编码器 `BackboneAE`/`BackboneAEPlus`。
- Short structure：窗口末尾 `short_window` 的关系矩阵差异。
- Long structure：完整窗口的关系矩阵差异。
- 关系比较：`relation(X)` 与 `relation(X_hat)`，支持 `corr`、`cov`、`shrink_cov`。
- Strong joint training：从头最小化 reconstruction + short + long；典型权重 `1.0/0.3/0.3`。
- Weak struct finetune：先训练 rec-only，再以较小结构权重（典型 `1.0/0.02/0.01`）微调。
- Legacy score fusion：按训练分数均值/标准差分别 z-normalize，再计算 `alpha*rec + beta*short + gamma*long`。

### 对应资产

| 项目 | 位置 |
|---|---|
| 核心代码 | `models/mscad.py`、`models/backbone_ae.py`、`losses/recon.py`、`losses/structure.py`、`utils/cov.py`、`utils/metrics.py` |
| 训练/评价入口 | `train.py`、`evaluate.py`、`run_swat.py`、`run_stage2_grid.py`、`run_stage3_refine.py`、`stage4_fusion_refine.py` |
| 代表配置 | `configs/swat_main.yaml`、`configs/swat_improve_v3_rec_only.yaml`、`configs/swat_improve_v4_struct_finetune.yaml`、`configs/swat_main_fusion_refined.yaml` |
| 代表实验 | Legacy SWaT main、rec-only、struct-finetune、R1–R6、ablation、fusion、三随机种子 |
| 代表输出 | `outputs/swat_main`、`outputs/swat_improve_v3_rec_only`、`outputs/swat_improve_v4_struct_finetune`、`outputs/stage3_refine`、`outputs/stage4_fusion_refine`、`results/multiseed` |
| 数据来源 | Legacy `Swat/normal.csv` + `Swat/merged.csv`；后经 4.3D 审计确认 train/test 与 validation/test overlap |

### 科学状态与论文使用

V1 的代码实现、损失公式和融合公式彼此一致，可用于解释项目起点、消融与研究转折。使用 Legacy `merged.csv` 的性能结果统一为 `INVALID_FOR_FINAL`：只能作为 historical design evidence，不能进入最终 benchmark。强联合训练和弱微调可进入论文“方法演化/负结果”部分，但不能把旧 SWaT 指标写成最终效果。

## V2 — Independent Relation

### 方法定义

- Normal relation reference：仅用正常训练区间估计关系中心与稳健尺度。
- Current vs normal：异常分数比较 `relation(current)` 与正常参考，而不是比较 `relation(X)` 与 `relation(X_hat)`。
- Synchronous correlation：zero-lag Pearson/Fisher-z。
- Delta correlation：对一阶差分序列计算 Pearson/Spearman。
- Lagged relation：关系允许非零时滞；A12 M2S 使用 lagged correlation，HAI Structure V2 使用有向 lagged Ridge dependency。
- State-conditioned relation：仅在训练期定义且满足稳定状态掩码的窗口中建立条件关系参考。

### 对应资产

| 项目 | 位置 |
|---|---|
| 公共/Legacy 机制代码 | `utils/independent_structure_score.py`、`utils/lagged_conditional_dependency.py`、`scripts/stage4_2a_*` 至 `scripts/stage4_3c_*` |
| A12 代码 | `data/swat2026_structure.py`、`data/swat2026_structure_recovery.py`、`data/swat2026_contextual_structure.py`、`scripts/swat2026/analyze_rec_and_structure.py`、`recover_structure_relations.py`、`analyze_contextual_structure.py` |
| HAI 代码 | `scripts/hai_step4a_*`、`hai_step4b_*`、`hai_structure_v2a_*`、`hai_structure_v2a1*` |
| A12 配置 | `configs/swat2026/independent_structure_seed1.yaml`、`structure_recovery_seed1.yaml`、`contextual_structure_seed1.yaml` |
| HAI 配置 | `configs/hai_structure_controlled_validation.yaml`、`configs/hai_structure_v2a_lagged_dependency.yaml` |
| 代表实验/输出 | A12 M2/M2R/M2S；HAI V1 controlled/real、V2-A/A1R/A1S/A1D/A1M/A1C |
| 数据来源 | A12 2026 官方 OT + N2 标签/划分；HAI 22.04 严格文件边界与 normal-only reference |

### 科学状态与论文使用

V2 已有可运行的诊断实现，但没有形成“真实数据上稳定优于重构”的冻结主方法。A12 M2 无 strict relation；M2R/M2S 只有 narrow relation 且 `primary_frozen=false`。HAI V1 真实 AUROC 接近随机且误报代价高；HAI V2 的受控机制有效，但 A1S 正常稳定性门禁失败，A1M/A1C 为负消融。因此 V2 可进入论文的 `VALID_DIAGNOSTIC` 机制证据、负结果与局限性部分，不能作为已验证的最终真实检测器。

## V3 — Conditional Collaboration

### 候选定义

- Other-channel conditional prediction：目标通道由其他通道及其时滞历史条件预测。
- Independent collaboration residual：候选目标是构造不等同于自历史误差、也不与 reconstruction score 重复的协同残差。
- 当前状态：**PROPOSED**。尚无冻结的 V3 数学定义、协议、主模型或最终实验。

### 已有原型与边界

| 项目 | 位置/结论 |
|---|---|
| 探索性原型代码 | `data/swat2026_conditional_prediction.py`、`scripts/swat2026/analyze_conditional_prediction.py` |
| 原型配置/输出 | `configs/swat2026/conditional_prediction_seed1.yaml`、`outputs/swat2026_batch_m2p_conditional_prediction` |
| 原型数据 | A12 2026 官方 OT，复用冻结 M1 scaler/checkpoint |
| 原型科学结论 | `cross_score_not_distinct_from_self_history`，`primary_frozen=false` |
| 最终论文可用性 | 只能用于说明候选方向为何尚未成为方法；不得写成 V3 已实现或已验证 |

代码库中存在 M2P 原型并不等价于 V3 已实现：它回答了一次候选可行性问题，结果是否定性的，且没有冻结“independent collaboration residual”。后续如继续，必须另行定义并预注册；本次审计没有启动该方向。

## 当前方法冻结结论

- 可作为最终真实数据基线的是 HAI Tiny reconstruction（来自 HAI Step 3G 的冻结证据）。
- V1 是历史方案与消融对象。
- V2 是受控机制原型/诊断分支，真实迁移未通过门禁。
- V3 仍是候选研究方向。
- 因此，“完整 MSCAD 最终方法”尚未冻结为一个包含 reconstruction + collaboration 的统一检测器。
