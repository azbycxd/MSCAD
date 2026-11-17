# Thesis-VALIDITY-P1 v4 停止报告

RETROSPECTIVE_VALIDITY_AUDIT

最终状态：**VALIDITY_P1_INCOMPLETE**。停止原因：**STOP_B_S3_POINTWISE_REPRODUCTION_MISMATCH**。本报告只保存已完成证据，不补跑、不修评分追平结果。

这是回顾性研究阶段中预先固定的分析方案。SWaT/HAI Test 曾被多次观察；本轮不是预注册确认性实验，也不是未见 Test 的独立验证。

## 版本与来源

正式 source worktree 为 `thesis-clean@7607b5fc44aaf72f2f5b6d4ab7ddd6e6ebb6fd2e`。最终 runner 在该 worktree 原为未跟踪文件；其 14 个关键源码/config 与发布快照 `43270769ce2c9353b762ab79833ba01167beb570` 的 SHA256 全部相同。17 项必要工件存在，Normal 源文件、冻结目标和存档哈希符合已有 manifest。没有把 source HEAD 冒充正式 runner 的生产 commit。

正式 manifest 没有记录生产 Git commit 或 runner hash，因此历史 run-to-commit 无法独立证明。本轮以源码内容绑定及 S3 重放检查其等价性；严格分数门禁未通过，不能宣布该链完全闭合。

v4 “没有保存 Ridge 系数”这一背景已过时：效率目录存在 `swat_dual_coefficients.npz` / `hai21_dual_coefficients.npz`；E2-R1D 使用了这些系数（`ridge_refit=false`）。本轮仍按用户授权，使用正式 Normal 数据和正式函数确定性重新拟合，没有用存档系数替换以追平结果。

## S3 门禁实测

|检查|结果|
|---|---|
|SWaT 15个目标及顺序|PASS|
|HAI 54个目标及顺序|PASS|
|SWaT Normal q99|PASS，226462.91950212093，差0|
|HAI Normal q99|PASS，40.21198385444111|
|SWaT 449793端点文件/索引/标签/时间绑定|PASS|
|SWaT 主指标六位一致|PASS|
|SWaT 事件|PASS，2/36|
|SWaT float64逐点分数|**FAIL**，容差1e-9|
|6个受控case S3 AUROC/AUPR/Recall/FPR|PASS|
|HAI Test完整S3门禁|中断，未形成最终结果|

SWaT 最大绝对差：**0.8635878833010793**；超差端点：**449635/449793**；第一个超差在 archive position 0，即 segment/file `0`、local/global index `63`。受控 S3 median AUROC=0.996912750，median AUPR=0.996930592。

已证实的实现来源差异：正式 SWaT runner 在标准化前转 float64；E2-R1D/Innovation3B 路径对 float32 cache 标准化后才转 float64。正式 Normal reference 来自 group 多输出预测，而 E2-R1D 从保存的逐目标系数重建 reference。这些是候选差异来源，未通过额外试验归因其各自贡献；没有将其改成相同路径，也没有放宽容差。

## 停止控制缺陷与实际执行范围

本轮编排脚本在 SWaT FAIL 后没有立即跳出 real replay 循环，仍进入 HAI S3。监控读到失败后终止了命令行已核实的本轮进程。HAI helper 会先载入全部5个 Test 文件；日志显示在终止前完成了 test1 的43138个端点评分，没有输出 HAI最终主指标。**这不符合v4“失败后立即停止真实攻击部分”的要求，作为执行偏差保留，不掩饰为严格完成。**

所有 Ridge 拟合都在 2026-11-17T09:51:10.436496+00:00 完成，先于 S3 Attack 特征读取。没有计算新的 S1/S2 Test指标，没有进行评分/参数/目标调优。原执行脚本保存在 executed_runner.py.txt；后续只修正新编排脚本的失败即停逻辑，不重放实验，不改变正式函数、公式或阈值。

## 正常统计、比较及数学检查

正常统计及仅技能列表尚未写盘、哈希冻结。normal_stats.csv 是**状态登记文件**，不是数值结果，不生成虚假的正常统计冻结哈希。S1/S2真实攻击指标、受控S1/S2、敏感性、H-κ、CoMuT Test及操作B数学检查均跳过。metrics_main.csv 只含已完成的SWaT S3门禁数值，其余明确空值/跳过；controlled_swat.csv 只含6个case的S3门禁数值，S1/S2不填。

描述标签：NOT_AVAILABLE；ΔHAI、ΔCTRL以及H-κ两组比例均不可计算，不将缺失结果归为R-EQ。

目标集合由正式双视角门控选出。8 个正技能条件对 Self 与 Cross 是对称的；但分歧稳定性比值筛选只针对 S3，可能使集合偏向 S3。因此本实验的结论仅限于"相同预测器与相同目标条件下"的比较，不代表 S2 在为其单独优化的目标集上的最佳表现。

受控理论说明：Self与目标观测均取clean，仅Cross输入取donor。S1逐对相同应得AUROC=0.5，只是实现一致性检查，本轮未执行该检查。S2与S3带符号量变化都等于cross(clean)-cross(donor)，区别只来自正常基线的位置与宽度。

## 历史披露

`docs/thesis/v3r_selective_cross_only.md`记录：S2类V3R selective cross-only残差采用目标排除，Cross lags `[1,5,10,20,30]`不含lag0，只用Top5 predictor；23个目标、46个受控cell的median AUROC为0.523134，状态`SELECTIVE_CROSS_ONLY_NOT_SUPPORTED`。双预测分歧是在这之后设计的。

上述历史结果与本轮 S2 的设置不同（predictor 集合、是否含 lag 0、受控扰动方式都不同），二者不可直接比较；本轮 S2 与 S3 使用同一组预测器，目的正是把"评分公式的作用"和"预测器设置变化的作用"分开。

## 工件与边界

详情：replay_check.json、provenance_report.json、donor_manifest_audit.json、checks.csv、stopped_manifest.json。manifest.json保留进程停止前的原始时间戳和输入哈希；stopped_manifest.json增加最终状态与执行脚本哈希。

NO TEST TUNING；NO METHOD CHANGE；NO PRIMARY OVERWRITE；NO NEURAL TRAINING。未修改论文章节或创新点表述。用户追加授权只允许发布本轮小型报告和脚本；数据、模型、逐点分数不上传。
