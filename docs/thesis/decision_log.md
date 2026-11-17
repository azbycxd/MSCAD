# Decision log

本日志只连接已有证据，不改写历史。每条决策给出触发证据、结论和当前边界。

## D1 — 为什么强结构联合训练被降级

- 证据：Legacy ablation 中 `w/o struct`/rec-only 的 F1 与 AUPR 不弱于 strong joint；`configs/swat_main.yaml` 使用 `lambda_s=lambda_l=0.3`。
- 判断：从训练开始施加强关系矩阵约束会与轻量 reconstruction backbone 竞争。
- 决策：strong joint 保留为 V1 历史/消融，不再作为主训练策略。
- 限制：Legacy 指标已被数据 overlap 审计失效，能支持“为何转向”的内部证据，不能支持最终性能排名。

## D2 — 为什么采用弱结构微调

- 证据：rec-only 后以 `lambda_s=.02, lambda_l=.01`、warmup=4 的 struct-finetune，在早期实验中比 strong joint 更稳；R1–R6 延续此策略。
- 判断：先让 backbone 学会数值重构，再以弱权重接入结构，较少破坏主干。
- 决策：作为 V1 的改良路径与实验转折保留。
- 限制：相关 Legacy 实际指标均 `INVALID_FOR_FINAL`；它不是当前冻结最终方案。

## D3 — 为什么 `relation(X)-relation(X_hat)` 有重构耦合问题

- 证据：`utils/cov.py::structure_distance_batch` 的两端分别来自原始窗口和同一模型的重构窗口；结构误差会同时受异常关系与 reconstruction failure 影响。
- 判断：该分数不具备独立证据属性，可能只是另一种表达的重构质量。
- 决策：F3 降为 `REJECTED_AS_PRIMARY`，仍保留作 V1 定义与 ablation。

## D4 — 为什么提出 independent relation score

- 证据：4.3A/4.3B 的受控关系破坏试验显示直接比较 current relation 与 normal-only reference 能响应关系变化，并可与 reconstruction-coupled score 对照。
- 判断：去掉 `X_hat` 可使结构分支至少在定义上不依赖重构器。
- 决策：建立 V2/F5；所有关系、中心、尺度和阈值只能由正常 train/validation 冻结。
- 限制：定义上的独立不等于真实数据上的互补；HAI V1 real transfer 很弱。

## D5 — 为什么简单同步 Pearson 不足

- 证据：A12 M2 strict relation count=0；HAI V1 real AUROC=.5280，结构新增一个 TP 平均伴随约 14.97 个 FP；preserving-control audit 也指出 Pearson 对某些共同时间置换不敏感。
- 判断：zero-lag pairwise linear association 不能稳定区分工况变化与协同异常，也遗漏物理时延。
- 决策：同步 Pearson 保留为机制/失败基线，不作为主分数。

## D6 — 为什么增加 lagged/state-conditioned 分析

- 证据：工业通道具有响应时延和 operating-state 混合；A12 M2S 与 HAI V2-A 分别实现 lag/state correlation 和 lagged conditional Ridge dependency。
- 判断：上下文条件能减少把时延或工况切换误判为关系破坏的概念偏差。
- 决策：推进 F6 诊断。
- 限制：A12 仍只有 narrow lagged relation；HAI A1S 稳定性门禁失败，不能升级为最终 detector。

## D7 — 为什么 pairwise 关系没有继续放宽阈值

- 证据：4.2C strict/diverse/expanded pool 与 4.3B 聚合对比显示，扩大候选池会引入噪声、稀释定位信号或增加误报；A12 M2 明确记录 `thresholds_relaxed_after_test=false`。
- 判断：为得到关系数量而放宽门槛会构成 test-driven selection，并破坏可复现性。
- 决策：保留“未找到足够 strict relations”的负结论，不追阈值直到出现好看结果。

## D8 — 为什么下一候选方向是 conditional cross-channel prediction

- 证据：相关性只能描述对称线性共变；HAI V2 的有向 lagged dependency 说明 self-history 与 other-channel 条件信息可分开建模。
- 判断：理想 collaboration signal 应衡量“加入其他通道后才获得的预测贡献”，而不是原始相关系数或普通重构误差。
- 决策：V3/F7 记为 `PROPOSED`。
- 当前证据：A12 M2P 原型得到 `cross_score_not_distinct_from_self_history`，因此不能声称 V3 已实现；新的 independent residual 尚待定义和预注册。

## D9 — 为什么 SWaT 2015 将重新作为正式主 benchmark

- 证据：Legacy `merged.csv` 不合法；A12 是单日、31 通道且当前 M1/关系分支主要提供负诊断；HAI 可支持严格真实基线但 structure 主分支未过稳定门禁。
- 判断：要验证冻结方法的工业水处理泛化，需要具有合法 chronology、官方 Physical 攻击段和可审计 split 的独立 SWaT 数据版本。
- 决策：SWaT 2015 official Physical 登记为 `TODO / awaiting audit`，拟承担未来正式 benchmark。
- 门禁：必须先做来源、文件、标签、时间线、train/validation/test isolation 审计；不得复用 Legacy 指标或假定文件名。

## 决策后的当前研究状态

HAI Tiny reconstruction 是唯一冻结的真实数据主基线；V2 是受控机制原型；V3 是候选。项目可以按受限主张撰写论文，但如果要把“协同关系检测器”写成最终主方法，仍需合法外部 benchmark 和新的冻结证据。
