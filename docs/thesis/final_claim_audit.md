# 最终主张审计

## 审计结论

- 论文题目固定为：**轻量化多变量时序协同异常检测方法研究**。
- 贡献结构固定为：两个核心方法创新 + 一个轻量化设计贡献。
- 前置状态已核对：`THESIS_CONTRIBUTION_STRUCTURE_FROZEN`、`ABLATION_FINAL_PACKAGE_READY`、`THESIS_NARRATIVE_STRUCTURE_READY`、`THESIS_MULTI_SEED_STABILITY_COMPLETED`。
- C1–C8 均能在冻结实验、结果表和消融报告中找到对应证据；没有新增主张，也没有用补充诊断覆盖严格主结果。
- 当前不存在尚未处理的、不受证据支持的 HIGH-RISK 主张。C5、C6、C7 仍属于答辩高风险项，只有在下表边界内才可使用。

## C1–C8 主张矩阵

| Claim | Evidence | Strength | Boundary | Thesis-safe wording | Defense-safe wording | Forbidden wording | Risk level |
|---|---|---|---|---|---|---|---|
| **C1 双视角预测分歧能够识别跨通道关系破坏** | Historical CoMuT D0 median AUROC `0.995989`；SWaT controlled D0 median AUROC `0.996913`；HAI21 DUAL AUROC/AUPR/F1=`0.916334/0.570193/0.610556` | `STRONG_MULTI_LEVEL` | 合成机制、真实轨迹控制实验和一套独立真实攻击构成分层证据；SWaT official real-attack DUAL 结果有限，机制实验不等同于工业榜单 | 双视角协同检测对关系破坏的响应获得了合成、受控真实轨迹和 HAI 真实攻击的分层支持 | 我的协同分支比较 Self 与 Cross 的预测分歧，使关系变化可以形成独立证据；效果取决于攻击是否破坏可学习关系 | 在所有数据集、所有攻击类型上都有效；已证明普适性 | LOW |
| **C2 多尺度滞后的贡献具有数据集依赖性** | HAI D0→D1 AUPR `0.570193→0.426064`；SWaT D0→D1 AUPR `0.167510→0.565012`；SWaT event hit 均约为 `2/36` | `STRONG_DATASET_DEPENDENT` | 较长滞后对 HAI 有正向贡献；在 SWaT，短滞后改善排序却未改善事件覆盖 | 较长滞后在 HAI 中提供正向贡献，滞后作用随数据语义和事件结构变化 | 滞后不是越多越好；HAI 支持多尺度滞后，SWaT 则提示需要未来研究自适应滞后 | 长滞后普遍必要；长滞后总能提高性能 | MEDIUM |
| **C3 Normal-only Reliability Gate 抑制不可靠 target 污染 max aggregation** | SWaT controlled D0→D2 AUROC `0.996913→0.630637`，AUPR `0.996931→0.550076`；D3-D 复现 D0 | `STRONG_CONTROLLED` | 真实 SWaT/HAI 上 D2 与 D0 接近；Gate 是正常数据可靠性约束，不是 Test 性能优化器 | 受控关系破坏实验支持 Gate 对不可靠候选经 max 聚合造成污染的抑制作用 | max 对单个异常大的不可靠 target 敏感，Gate 用 Normal-only 稳定性与预测可靠性先约束候选 | Gate 保证每个真实数据集都提升；Gate 根据 Test 结果筛选 target | MEDIUM |
| **C4 正常尾概率校准改善 REC 与 DUAL 的尾部可比性** | Robust-Z 在 SWaT/HAI/WADI 的 Normal top-1% 均为 DUAL `100%`；Tail 后 REC/DUAL=`65.19/34.81`、`59.15/40.85`、`51.78/48.22`，q99 balance≈1 | `STRONG_CROSS_DATASET_CALIBRATION` | 证明的是正常尾部证据尺度可比，不是 Test 排序或 F1 必然提高 | 经验正常尾概率映射显著改善了两类重尾分数的证据可比性 | Tail calibration 先解决“不同分支的极端分数能否比较”，再讨论互补性；它不强迫 Test 指标变好 | 尾校准在所有数据集都提高检测性能；贡献比例就是训练权重 | LOW |
| **C5 Tail-calibrated fusion 在 HAI 上体现互补性** | HAI 正式 seed：最佳单分支 AUPR `0.570193`，Tail AUPR `0.617897`，差值 `+0.047703`；三 seed Tail AUPR `0.604836±0.019403`，HIGH19 均为 `19/19` | `MODERATE_SINGLE_DATASET` | 多 seed 补强 HAI 内部重复性，但仍是单数据集正向关联；SWaT 为反例，不构成跨数据集普遍因果 | HAI 结果显示校准后的 REC 与 DUAL 证据具有正向互补关联，三 seed 排序和 HIGH19 覆盖提供补充重复性证据 | 我只把它表述为 HAI 上的互补证据；SWaT 明确说明这种收益不是普遍的 | 融合已被证明普遍优于所有单分支；Tail 是跨数据集最优 | HIGH |
| **C6 Depthwise-separable REC 支持参数高效的轻量化框架** | R0 参数为 SWaT/HAI `11,730/15,482`；dense R2 是 R0 的 `2.192/2.126` 倍且 AUROC/AUPR 未改善；三 seed REC AUROC/AUPR：SWaT `0.829704±0.008863`/`0.735474±0.016188`，HAI `0.851566±0.004793`/`0.534387±0.020171` | `STRONG_EFFICIENCY_WITH_ARCHITECTURE_BOUNDARY` | 三 seed 只支持 R0 ranking variation 相对有限；q99 operating point 更敏感；R1 仍是 retrospective；固定 k=3 无最优性证据；GDN 参数少于 Ours-FULL | Depthwise-separable 设计有效控制参数规模，冻结 R0 的 ranking evidence 在当前三 seed 下相对稳定 | 本文并不声称当前 k=3 配置是最优结构；贡献是参数高效框架及完整效率审计 | R0 或 k=3 最优；每个组件都必要；Ours 对所有基线都更小、更快 | HIGH |
| **C7 SWaT official real attacks 是 DUAL 与 TAIL 的明确边界** | 正式 seed：REC AUPR `0.743353`，DUAL/TAIL AUPR=`0.167510/0.206747`，REC strict q99 F1=`0.258067`；三 seed SWaT TAIL AUPR std=`0.151947`、range=`0.168898–0.448953` | `STRONG_BOUNDARY` | 记录 `SEED_VARIABILITY_OBSERVED`；必须同时报告 REC 强排序、q99 高 FPR/低 F1、DUAL/TAIL 负向结果；Oracle 为 Test-best 诊断，不是主结果 | SWaT 表明协同证据与尾校准收益依赖异常语义和随机初始化，同时 REC 暴露 Normal Validation→Test 阈值迁移问题 | SWaT 上 REC 排序仍强，但 DUAL/TAIL 不占优且 Tail 存在 seed 波动；稳定性表不替代正式 seed 主表 | 隐藏负向或 seed 波动；用均值或 Oracle 代替 strict；声称 DUAL/TAIL 在 SWaT 优于 REC | HIGH |
| **C8 WADI.A2 揭示 normal-domain shift 下“证据平衡≠检测有效”** | F0/F1/F3/F4 AUPR 均约 `0.052–0.055`；Tail REC/DUAL share=`51.78/48.22` | `STRESS_TEST_ONLY` | WADI 是 `DOMAIN_SHIFT_STRESS_TEST`，不进入 Primary 主性能表，也不用于重选 gate、lag、alpha、score 或 fusion | WADI 用于揭示域偏移条件下证据平衡与检测有效性之间的边界 | WADI 的平衡结果说明校准完成了尺度任务，但极低 AUPR 表明它没有解决域偏移 | WADI 证明成功跨数据集泛化；把 WADI 纳入主性能排名 | MEDIUM |

## 全局写作门禁

1. 严格主结果只能来自 Historical Strict：Normal Validation q99、无 Test tuning、无 PA。
2. Oracle、PA-Oracle 与 common-mask 只作为回顾性或文献兼容诊断，不能替代主结果。
3. WADI 只能标为 domain-shift stress test；2026 外部方法 `DIRECT=0`，不能据此声明 SOTA。
4. 轻量化贡献使用“参数高效”“性能—效率权衡”“可解释且可审计”，不使用“结构最优”。
5. 正向结论必须同时写明数据集、协议、指标和边界；禁止“全面优于”“普遍有效”“绝对证明”。
6. 三 seed 结果只作为 Stability/Robustness supplementary evidence；历史 Primary 继续使用正式 seed，不以均值覆盖。

## 最终状态

`FINAL_CLAIMS_DEFENSIBLE`
