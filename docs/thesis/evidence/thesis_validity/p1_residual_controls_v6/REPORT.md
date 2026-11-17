# Thesis-VALIDITY-P1 v6

RETROSPECTIVE_VALIDITY_AUDIT

最终状态：VALIDITY_P1_COMPLETED。本轮是回顾性研究阶段中预先固定的分析方案，不是预注册确认性实验，也不是未见Test的独立验证。

## 来源与复现门禁

v6为唯一执行协议。正式runner只读；Normal-only Ridge按正式函数重新拟合两次。所有拟合完成后才读Attack特征。主实验使用冻结15/54目标且顺序不变。E2-R1D是效率系数与Innovation3B路径重建，不是正式runner逐点存档，因此正式逐点档案项NOT_AVAILABLE；不放宽容差、不切换路径。

两次重放的Validation/Test S3要求字节完全相同。共同预测源检查固定使用本次保存的z/pred_self/pred_cross和正式Normal参考，原始、校准和聚合分歧均以1e-9检查。失败即退出下一数据集Test调用。

详细门禁及重建差异见replay_check.json、shared_source_*.json、archive_difference_*.json和top10 CSV。

## 已完成步骤

- swat two Normal replays and exact threshold/target gate

- hai two Normal replays and exact threshold/target gate

- controlled six-case S3 reproduction

- swat all S3 gates and shared-source consistency

- hai all S3 gates and shared-source consistency

- Fresh Normal statistics equal v5 excluding only run timestamp, tolerance zero

- Normal statistics/hash, skill-only list/hash and mathematical operation B

- controlled all18 target-score rows and paired consistency

- swat three score metrics, target/event tables and bound Parquet scores

- hai three score metrics, target/event tables and bound Parquet scores

- Historical CoMuT13 frozen cases and score controls

## 正常统计与冻结

完整Normal Validation逐目标残差方差/协方差采用ddof=0；每个S1/S2/S3各自median、MAD、q95/q99、sigma下限及max主导性见normal_stats.csv。仅技能列表先于正常统计哈希冻结；新的S1/S2 Test指标必须在冻结后计算。q99/median在median=0时空值，不引入epsilon。

主导比例采用冻结顺序argmax的确定性归属，另报包括并列的比例；只诊断，不调整floor、目标或校准。

新生成normal_stats.csv按v6与v5核验；仅排除normal_statistics_completed_utc运行时间戳列，所有其他列及行顺序逐值精确比较，容差0。具体比较列、哈希与差异见normal_statistics_v5_comparison.json。

目标集合由正式双视角门控选出。8 个正技能条件对 Self 与 Cross 是对称的；但分歧稳定性比值筛选只针对 S3，可能使集合偏向 S3。因此本实验的结论仅限于"相同预测器与相同目标条件下"的比较，不代表 S2 在为其单独优化的目标集上的最佳表现。

## 受控与数学检查

受控部分沿用6目标、每目标2000 frozen donor配对、原calibration区间，逐目标评价，无跨目标聚合。S1门禁是逐对最大差0，AUROC用repr原样保存，并要求与0.5差值不超过1e-12；此为实现一致性检查。S2/S3带符号变化都等于cross(clean)-cross(donor)，区别来自正常基线位置和宽度。

数学操作B只修改Normal序列副本的一个当前值并重建滞后特征；恒等式、目标当前/未来lag和其他目标/全局响应分别记录于math_checks.json。没有改变模型、参考或门控。

## 结果与描述标签

{
  "audit_label": "RETROSPECTIVE_VALIDITY_AUDIT",
  "status": "VALIDITY_P1_COMPLETED",
  "descriptive_label": "R-EQ",
  "delta_HAI": 0.014652356734228045,
  "delta_CTRL": -0.000492000000000159,
  "H_kappa": {
    "swat": {
      "status": "TESTABLE",
      "positive_count": 3,
      "nonpositive_count": 12,
      "positive_group_S3_gt_S2_fraction": 0.3333333333333333,
      "nonpositive_group_S3_gt_S2_fraction": 0.5,
      "trend_present": false,
      "interpretation": "descriptive association only; no causality or guaranteed ranking performance"
    },
    "hai": {
      "status": "TESTABLE",
      "positive_count": 18,
      "nonpositive_count": 36,
      "positive_group_S3_gt_S2_fraction": 0.7777777777777778,
      "nonpositive_group_S3_gt_S2_fraction": 0.4722222222222222,
      "trend_present": true,
      "interpretation": "descriptive association only; no causality or guaranteed ranking performance"
    }
  },
  "event_differences": {
    "swat": {
      "S3_greater_than_S2": 5,
      "S3_less_than_S2": 3,
      "equal": 28
    },
    "hai": {
      "S3_greater_than_S2": 33,
      "S3_less_than_S2": 4,
      "equal": 13
    }
  },
  "target_differences": {
    "swat": {
      "S3_greater_than_S2": 7,
      "S3_less_than_S2": 8,
      "equal": 0
    },
    "hai": {
      "S3_greater_than_S2": 31,
      "S3_less_than_S2": 23,
      "equal": 0
    }
  },
  "interpretation": "Pre-specified descriptive label only; no innovation/novelty conclusion"
}

H-κ为描述性关联，不表示因果，不保证AUROC/AUPR；任一分组空时NOT_TESTABLE。描述标签不决定创新成立，不升级S1/S2为正式方法。

## 历史披露

v4因把E2-R1D重建分数当作正式存档触发STOP_B。v4编排在SWaT失败后仍进入HAI，读取了5个Test并完成test1评分后被终止；该执行偏差继续保留。v6显式门禁防止下一数据集调用，旧代码和停止产物均保留。v5的FIT601逐对相等检查通过，但AUROC严格浮点等于0.5断言失败且实际数值未保存；v6按本次先行修订的数值判据全量重跑，不使用旧中间预测。完整性见preservation_check.json。

docs/thesis/v3r_selective_cross_only.md记录：S2类V3R selective cross-only残差采用目标排除、Cross lags [1,5,10,20,30]不含lag0、Top5 predictor；23目标、46受控cell的median AUROC为0.523134，状态SELECTIVE_CROSS_ONLY_NOT_SUPPORTED。双预测分歧是在这之后设计的。

上述历史结果与本轮 S2 的设置不同（predictor 集合、是否含 lag 0、受控扰动方式都不同），二者不可直接比较；本轮 S2 与 S3 使用同一组预测器，目的正是把"评分公式的作用"和"预测器设置变化的作用"分开。

## 未通过、异常或跳过

必需门禁无失败，条件 CoMuT 13/13 完成，技能门控敏感性分析完成。正式 runner 逐点存档比对为 NOT_AVAILABLE：已找到并审计 E2-R1D 存档，但它使用重建路径；其差异仅作诊断，未作为正式逐点相等证据。两次正式路径重放及三评分共同预测源门禁均通过。

NO TEST TUNING / NO PRIMARY OVERWRITE / NO METHOD CHANGE / NO NEURAL TRAINING / NO PUSH。审计进程不操作Git；结束后按既有授权只自动本地提交小型报告和脚本。没有修改论文章节或创新点表述。


## 已冻结结果数值摘录

仅汇总已经完成、正常统计哈希冻结后的结果；没有新增评分方案或结果驱动选择。

|dataset|score|auroc|aupr|precision|recall|f1|fpr|threshold|event_hit_count|
|---|---|---|---|---|---|---|---|---|---|
|swat|S1|0.477693|0.115830|0.084918|0.008257|0.015050|0.012298|95620.623692|2|
|swat|S2|0.866654|0.531623|0.085855|0.008623|0.015672|0.012691|228703.243284|2|
|swat|S3|0.675631|0.167510|0.087166|0.008605|0.015663|0.012455|226462.919502|2|
|hai|S1|0.559325|0.023453|0.018580|0.008718|0.011868|0.010490|1306662.834942|0|
|hai|S2|0.912564|0.555541|0.528794|0.625014|0.572892|0.012688|55.175987|40|
|hai|S3|0.916334|0.570193|0.554350|0.679446|0.610556|0.012443|40.211984|44|

受控6个目标的中位AUROC/AUPR：

|score|auroc|aupr|
|---|---|---|
|S1|0.500000|0.500000|
|S2|0.997405|0.997380|
|S3|0.996913|0.996931|

Normal-only尺度下限与max主导性（不改变校准）：

|dataset|score|q99|floor_targets|target_count|largest_dominance_target|dominance_fraction|dominant_sigma|
|---|---|---|---|---|---|---|---|
|swat|S1|95620.623692|1|15|AIT504|0.202718|0.000140|
|swat|S2|228703.243284|1|15|LIT101|0.204557|0.103085|
|swat|S3|226462.919502|1|15|FIT601|0.361170|1e-06|
|hai|S1|1306662.834942|2|54|P1_FCV02D|0.459549|1e-06|
|hai|S2|55.175987|0|54|P1_FCV02D|0.120738|0.046155|
|hai|S3|40.211984|0|54|P1_FCV02D|0.092209|0.046178|

数学性质：当前目标和全局聚合分开记录。微小恒等式误差按浮点实测保留；若正常MAD为0，不添加人为扰动下限。

|dataset|target|perturbation|identity_max_diff|own_current_S3_delta|global_S3_delta|
|---|---|---|---|---|---|
|swat|FIT101|0.493099|2.22044604925e-16|0.000000|-0.756303|
|hai|P1_B2004|0.217338|2.6645352591e-15|0.000000|409.171784|

v5与v6 Normal逐值一致性：PASS；行数=345；差异单元格=0；数值最大差=0.0。

仅排除列：normal_statistics_completed_utc

按原行列顺序精确比较的列：audit_label, dataset, target, score, main_frozen_target, skill_only_target, scope, n, variance_ddof, var_self, var_cross, cov_cross_self, kappa, s3_reference_source, median, mad, scaled_mad, scale, sigma, scale_at_floor, q95, q99, q99_over_median, individual_robust_z_q99, normal_max_dominance_fraction, normal_max_tied_fraction, dominance_tie_policy, aggregate_q99_threshold, skill_only_aggregate_q99_threshold, stage, q99_median_ratio, threshold, case_name

S1实际AUROC repr（6目标）：

|target|s1_auroc_repr|s1_pair_max_abs_difference|
|---|---|---|
|AIT202|0.5|0.000000|
|FIT201|0.5|0.000000|
|DPIT301|0.5|0.000000|
|FIT301|0.5|0.000000|
|FIT401|0.5|0.000000|
|FIT601|0.49999999999999994|0.000000|

产物校验：PASS（70/70）。
