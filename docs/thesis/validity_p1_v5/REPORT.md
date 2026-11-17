# Thesis-VALIDITY-P1 v5

RETROSPECTIVE_VALIDITY_AUDIT

最终状态：VALIDITY_P1_INCOMPLETE。本轮是回顾性研究阶段中预先固定的分析方案，不是预注册确认性实验，也不是未见Test的独立验证。

## 来源与复现门禁

v5为唯一执行协议。正式runner只读；Normal-only Ridge按正式函数重新拟合两次。所有拟合完成后才读Attack特征。主实验使用冻结15/54目标且顺序不变。E2-R1D是效率系数与Innovation3B路径重建，不是正式runner逐点存档，因此正式逐点档案项NOT_AVAILABLE；不放宽容差、不切换路径。

两次重放的Validation/Test S3要求字节完全相同。共同预测源检查固定使用本次保存的z/pred_self/pred_cross和正式Normal参考，原始、校准和聚合分歧均以1e-9检查。失败即退出下一数据集Test调用。

详细门禁及重建差异见replay_check.json、shared_source_*.json、archive_difference_*.json和top10 CSV。

## 已完成步骤

- swat two Normal replays and exact threshold/target gate

- hai two Normal replays and exact threshold/target gate

- controlled six-case S3 reproduction

- swat all S3 gates and shared-source consistency

- hai all S3 gates and shared-source consistency

- Normal statistics/hash, skill-only list/hash and mathematical operation B

## 正常统计与冻结

完整Normal Validation逐目标残差方差/协方差采用ddof=0；每个S1/S2/S3各自median、MAD、q95/q99、sigma下限及max主导性见normal_stats.csv。仅技能列表先于正常统计哈希冻结；新的S1/S2 Test指标必须在冻结后计算。q99/median在median=0时空值，不引入epsilon。

主导比例采用冻结顺序argmax的确定性归属，另报包括并列的比例；只诊断，不调整floor、目标或校准。

目标集合由正式双视角门控选出。8 个正技能条件对 Self 与 Cross 是对称的；但分歧稳定性比值筛选只针对 S3，可能使集合偏向 S3。因此本实验的结论仅限于"相同预测器与相同目标条件下"的比较，不代表 S2 在为其单独优化的目标集上的最佳表现。

## 受控与数学检查

受控部分沿用6目标、每目标2000 frozen donor配对、原calibration区间，逐目标评价，无跨目标聚合。S1理论上配对相等且AUROC=0.5，只用于构造一致性检查；本轮实际受控执行中断，详见末尾停止说明。S2/S3带符号变化都等于cross(clean)-cross(donor)，区别来自正常基线位置和宽度。

数学操作B只修改Normal序列副本的一个当前值并重建滞后特征；恒等式、目标当前/未来lag和其他目标/全局响应分别记录于math_checks.json。没有改变模型、参考或门控。

## 结果与描述标签

{
  "audit_label": "RETROSPECTIVE_VALIDITY_AUDIT",
  "status": "VALIDITY_P1_INCOMPLETE",
  "descriptive_label": "NOT_AVAILABLE",
  "delta_HAI": null,
  "delta_CTRL": null,
  "H_kappa": {
    "swat": "NOT_TESTABLE",
    "hai": "NOT_TESTABLE"
  },
  "stop_reasons": [
    "RuntimeError: S1_CONSISTENCY_FLAG: AUROC FIT601"
  ]
}

H-κ为描述性关联，不表示因果，不保证AUROC/AUPR；任一分组空时NOT_TESTABLE。描述标签不决定创新成立，不升级S1/S2为正式方法。

## 历史披露

v4因把E2-R1D重建分数当作正式存档触发STOP_B。v4编排在SWaT失败后仍进入HAI，读取了5个Test并完成test1评分后被终止；该执行偏差继续保留。v5显式门禁防止下一数据集调用，旧代码和停止产物均保留，完整性见preservation_check.json。

docs/thesis/v3r_selective_cross_only.md记录：S2类V3R selective cross-only残差采用目标排除、Cross lags [1,5,10,20,30]不含lag0、Top5 predictor；23目标、46受控cell的median AUROC为0.523134，状态SELECTIVE_CROSS_ONLY_NOT_SUPPORTED。双预测分歧是在这之后设计的。

上述历史结果与本轮 S2 的设置不同（predictor 集合、是否含 lag 0、受控扰动方式都不同），二者不可直接比较；本轮 S2 与 S3 使用同一组预测器，目的正是把"评分公式的作用"和"预测器设置变化的作用"分开。

## 未通过、异常或跳过

- RuntimeError: S1_CONSISTENCY_FLAG: AUROC FIT601

NO TEST TUNING / NO PRIMARY OVERWRITE / NO METHOD CHANGE / NO NEURAL TRAINING / NO COMMIT / NO PUSH。没有修改论文章节或创新点表述。


## 实际停止原因与未完成范围

本轮 SWaT/HAI 主实验复现门禁及共同预测源门禁均已通过，raw/calibrated/aggregate差值均为0。随后受控FIT601的S1触发辅助脚本的严格浮点相等断言：metrics[auroc] != 0.5。

到达该断言前，逐对raw S1差值检查已经通过（最大差0）。因此这不是已证实的配对样本不一致。异常没有携带实际AUROC数值，局部结果也未返回；无法仅据保存记录确认偏差幅度或数值原因，不猜测数值。

该严格断言来自复用的受控审计辅助函数。本轮没有修改断言、容差、模型或校准后继续，也没有补跑。该异常属于审计执行未完成，不能解释为方法失败。

前5个目标的控制指标在函数局部内存中计算过，但未返回/持久化；本报告不重构或填补这些数值。controlled_swat.csv保留6个目标通过的正式S3复现数值，并明确登记S1/S2的未持久化/未执行状态。

真实攻击S1/S2、逐目标/逐事件比较、技能敏感性、CoMuT Test、H-κ关联与描述标签均未完成。metrics_main.csv的S3是复现结果，不能作为三评分完整对照表。

Normal统计和skill列表已在部分控制指标计算前冻结，math_checks.json已经完成。normal_stats.csv不是占位文件。

## Normal-only尺度诊断摘录

|dataset|score|sigma=1e-6目标数|目标数|max主要来源|主导比例|主导目标sigma|
|---|---|---|---|---|---|---|
|swat|S1|1|15|AIT504|0.202718|0.000140382188319|
|swat|S2|1|15|LIT101|0.204557|0.103084699224|
|swat|S3|1|15|FIT601|0.361170|1e-06|
|hai|S1|2|54|P1_FCV02D|0.459549|1e-06|
|hai|S2|0|54|P1_FCV02D|0.120738|0.046154907369|
|hai|S3|0|54|P1_FCV02D|0.092209|0.0461784470643|

## 已完成数学检查摘录

|dataset|target|identity最大误差|当前目标S3变化|全局S3变化|
|---|---|---|---|---|
|swat|FIT101|2.22044604925e-16|0|-0.756302897886|
|hai|P1_B2004|2.6645352591e-15|0|409.171783834|

已完成工件检查：21/21。这些检查不改变INCOMPLETE状态。
