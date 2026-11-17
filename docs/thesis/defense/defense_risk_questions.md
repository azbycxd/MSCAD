# 答辩高风险问题与安全答法

使用原则：先给事实，再解释研究边界，最后给后续方向。不要先辩解，也不要用 controlled、Oracle 或 PA 覆盖严格主结果。

## Q1. 你的核心创新到底是什么？为什么不是三个算法创新？

**建议回答：**

两个核心方法创新分别是：第一，局部滞后双视角协同一致性检测，用 Self/Cross 预测分歧形成独立关系证据；第二，正常尾概率校准，把 REC 与 DUAL 的异构分数映射到可比较的概率证据尺度。轻量 REC、线性 Ridge 和非可训练 Tail 共同构成轻量化设计贡献。REC 的 R1 消融说明固定 k=3 没有最优性证据，所以我没有把某个卷积核配置包装成第三个算法创新。

**不能说：** 本文有三个已经被充分验证且彼此独立的算法创新。

## Q2. DUAL 与普通 reconstruction residual 有什么本质区别？

**建议回答：**

我的协同异常分支不使用 observed-target residual，而是比较 Self 历史预测与 Cross 同过程变量预测之间的分歧，因此关系变化本身可以形成独立异常证据。Historical CoMuT、SWaT controlled 和 HAI 分别提供合成、受控真实轨迹和独立真实攻击证据；它仍然只适用于会破坏可学习关系的异常。

**不能说：** DUAL 能替代 reconstruction，或所有攻击都会产生关系分歧。

## Q3. 为什么选择 Ridge，而不是更复杂的神经网络？

**建议回答：**

本研究需要把“信息视角差异”与“模型容量差异”分开。Ridge 在 Normal-only 条件下可确定地拟合 Self/Cross 条件关系，参数少、可解释、易审计，能把主要变量控制在输入视角而不是预测器复杂度。现有实验验证的是冻结 Ridge 实现，不支持推断神经网络一定更差；非线性预测器属于后续研究。

**不能说：** Ridge 是所有关系预测任务的最优模型。

## Q4. 为什么需要 lag？为什么这些 lag 不是调出来的？

**建议回答：**

工业过程具有控制响应延迟，同步输入无法覆盖全部动态依赖。本文在正式评价前冻结局部多尺度 lag，未根据 Test 结果搜索。消融显示 HAI 中完整 lag AUPR 0.570193，short-lag 降至 0.426064；SWaT 的方向相反，而且事件命中都约为 2/36。因此 lag 有机制必要性，但具体尺度具有数据集依赖性，未来应做 Normal-only 自适应选择。

**不能说：** 长 lag 在所有数据集都必要，或当前 lag 集合全局最优。

## Q5. 为什么全局聚合用 max？它不会被噪声支配吗？

**建议回答：**

关系破坏可能集中在少数 target，mean 会稀释局部强证据，所以冻结方案使用 max。它确实会放大不可靠候选，这正是 Unified Reliability Gate 的作用：先用正常数据限制候选可靠性，再做 max。Controlled No-Gate 的 AUROC/AUPR 从 0.996913/0.996931 降至 0.630637/0.550076，支持这个机制。本文没有做 Test-driven aggregation search，也不声称 max 在所有数据上最优。

**不能说：** max 没有噪声风险，或已经证明优于所有聚合方式。

## Q6. Gate 是不是根据测试结果筛 target？

**建议回答：**

不是。Gate 的发现、验证和稳定性判断全部来自正常数据，Test 标签不参与 target selection。它的定位是 normal-only reliability constraint，而不是 Test optimizer。真实 SWaT/HAI 上 No-Gate 与 Final 接近，所以论文只把它的强作用界定在 controlled relation-breaking 机制证据中。

**不能说：** Gate 保证任何真实测试集都提升。

## Q7. Tail calibration 不就是一种归一化吗？为什么能算核心创新？

**建议回答：**

关键不是把均值和方差变得相近，而是让不同重尾分支的极端分数具有统一的正常上尾概率语义。Robust-Z 后，三套数据 Normal top-1% 仍是 DUAL 100% 支配；经验尾概率校准后，REC/DUAL 贡献变为约 65/35、59/41 和 52/48。它直接解决跨分支尾部证据不可比问题。HAI 的 AUPR 增量是附加的单数据集正向关联，不是这项贡献成立的唯一条件。

**不能说：** share 越接近 50/50，Test 性能就一定越好。

## Q8. Tail 在 SWaT 上明显变差，为什么仍保留？

**建议回答：**

Tail 的核心目标是证据可比性，不是保证所有数据集的排序都超过最强单分支。SWaT 上 REC AUPR 0.743353，而 Tail 只有 0.206747，这是明确的负向边界；HAI 上 Tail 从最佳单分支 0.570193 提升到 0.617897。论文同时呈现两者，因此结论只是：校准改善可比性，并在 HAI 体现互补性，不是 universal improvement。

**不能说：** SWaT 结果不重要，或可以只报告 HAI。

## Q9. SWaT 上 DUAL 很差，协同异常研究是否失败？

**建议回答：**

没有足够证据支持“所有 SWaT 攻击都是纯协同关系异常”。SWaT official 中许多攻击更偏向局部幅值、标签边界和运行状态变化，冻结 DUAL 对这些异常的覆盖有限；DUAL AUPR 为 0.167510。另一方面，Historical CoMuT、SWaT controlled 和 HAI 的关系机制证据分别为 0.995989、0.996913 和 HAI AUROC 0.916334。结论是机制在适合的关系异常上成立，但收益受异常语义影响，不具普适性。

**不能说：** controlled 等价于真实攻击，或 SWaT 负向结果应删除。

## Q10. SWaT REC 的 F1 很低，为什么还说 ranking 强？

**建议回答：**

AUROC/AUPR 评价 score ranking，固定 q99 F1 评价特定 operating point。SWaT REC AUROC/AUPR 为 0.834887/0.743353，但 Validation q99 转移到 Test 后 F1 0.258067、FPR 0.718133，说明主要问题是阈值迁移。Test-best Oracle F1 0.768959 只说明分数仍有可分信息，它使用 Test 标签，绝不能替代 strict q99 主结果。

**不能说：** Oracle 是可部署结果，或用 0.768959 作为主 F1。

## Q11. R1 更小、结果更好，为什么不替换 Final REC？

**建议回答：**

R0 是预先冻结并用于全部正式实验的 reference architecture。R1 是在正式 Test 结果已经暴露后完成的 single-seed retrospective ablation；若据此替换主方法并重做主表，就是 test-informed architecture selection。论文如实报告 R1，用它修正“固定 k=3 最优”的表述，但不反向修改冻结方法。后续应在预注册、多种子和独立数据上验证 R1。

**必须主动说：** 本文并不声称当前 k=3 配置是最优结构。

**不能说：** R1 只是偶然所以可以忽略，或 R0 总体仍优于 R1。

## Q12. 轻量化贡献是否成立？GDN 参数不是更少吗？

**建议回答：**

轻量化是 performance–efficiency design，不是“所有对手中参数最少”。Final REC 在 SWaT/HAI 为 11,730/15,482 个参数；dense R2 增到约 2.19/2.13 倍但没有 AUROC/AUPR 收益，支持 depthwise-separable 的参数效率。GDN 参数确实更少，因此论文只主张万级参数、组件级效率可追溯和相对 dense 的参数效率，不主张普遍最小或最快。

**不能说：** Ours 对所有强基线都更轻、更快。

## Q13. 为什么 WADI 贡献接近 50/50，检测仍接近随机？

**建议回答：**

贡献平衡回答“两个分支在正常尾部是否可比较”，不回答“测试分布是否与正常参考一致”。WADI Tail share 为 51.78/48.22，但各版本 AUPR 约 0.052–0.055，说明 normal-domain shift 和缺失值/运行模式变化没有被校准本身解决。因此 WADI 是 DOMAIN_SHIFT_STRESS_TEST，而非成功泛化证据。

**不能说：** 52/48 证明跨数据集泛化成功。

## Q14. 为什么没有与 2026 最新方法做一个统一排名？

**建议回答：**

我审计了 Park & Shim、LatentFlow、MAD-TCN 和 CCM-TAD 的第一方论文与官方代码。它们涉及 Affiliation、VUS、SF1、Test-best、Test-driven selection、不同后处理和数据预处理；当前外部 DIRECT 记录数为 0。论文保留原指标语义，只做 PARTIAL 或 NOT_DIRECT 的协议感知比较，不把不同指标换算成本文的 point-wise q99 结果。

**不能说：** 没有 DIRECT 排名就说明本文优于这些方法；不能把 Affiliation-F1、VUS、SF1 换算成普通 F1/AUROC/AUPR。

## Q15. 你的证据链为什么可信？

**建议回答：**

证据按角色分层：CoMuT 验证纯关系机制，SWaT controlled 验证真实正常轨迹上的受控破坏，HAI 验证独立真实攻击，SWaT official 验证严格 benchmark 并暴露失败边界，WADI 只验证域偏移压力。主结果使用 frozen Validation q99、无 PA、无 Test tuning；Oracle/PA 与 2026 异构指标全部单列为补充，不覆盖 strict 主表。

**不能说：** 任意一个层级可以单独证明普适性。

## Q16. 论文最重要的局限和下一步是什么？

**建议回答：**

局限包括：lag 作用具有数据集依赖性；Gate 的强证据主要来自 controlled 场景；Tail 正向互补目前主要来自 HAI；SWaT 存在明显的 Normal Validation→Test 阈值迁移；WADI 暴露域偏移；R1 尚缺独立多种子验证。对应后续方向是 Normal-only 自适应 lag、R1 独立验证、无 Test 标签的阈值迁移校准和域适配。这些都是 proposed future work，不是本文已完成的方法。

**不能说：** 这些问题已经由当前实验解决。

## Q17. 你的结果会不会只是随机种子碰巧好？

**建议回答：**

我在冻结方法和协议后补充了 seed 1、42、43 的三随机种子实验，没有选择 best seed，也没有用均值覆盖正式主表。REC 在 SWaT/HAI 上的 AUROC/AUPR 标准差较小；HAI TAIL 的 AUPR 为 0.6048±0.0194，三个 seed 对 HIGH19 均为 19/19。与此同时，我不回避 SWaT TAIL 的明显 seed 波动，以及 HAI 中 q99 F1/FPR 比 AUROC/AUPR 更敏感的事实。因此论文只主张核心 ranking evidence 在当前三 seed 下相对稳定，不声称所有 operating point 都对随机种子不敏感。

**必须主动说：** SWaT TAIL AUPR 的三 seed 范围为 0.168898–0.448953，记录为 `SEED_VARIABILITY_OBSERVED`；F1/FPR 波动与既有 Validation→Test threshold transfer 问题一致。

**不能说：** 模型整体非常稳定；多 seed 已证明所有数据集和所有指标均不受初始化影响；三 seed 均值可以替代历史正式 seed。

## 30 秒兜底答法

当问题超出冻结证据时，使用：

> 现有冻结证据能够支持的是【具体数据集、具体协议、具体指标】下的结论；它还不足以支持更强的普遍性判断。论文已经把这一点列为边界，后续需要通过预注册的独立数据或多种子实验验证，我不会根据当前 Test 结果反向扩大主张。
