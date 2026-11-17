# 答辩关键数字

本表共 **20 组**，是答辩真正需要记忆的数字。优先记方向和边界，不必背主表全部小数。

| # | 数字 | 用途 | 必须同时说的边界 |
|---:|---|---|---|
| 1 | Historical CoMuT D0 median AUROC **0.995989** | 合成关系异常机制证据 | 不是工业 Test leaderboard |
| 2 | SWaT controlled D0 median AUROC/AUPR **0.996913/0.996931** | 真实正常轨迹上的受控关系破坏 | controlled 不等同于官方真实攻击 |
| 3 | Controlled No-Gate AUROC/AUPR **0.630637/0.550076** | Gate 抑制不可靠 target 污染 max | Gate 不保证每个真实数据集提升 |
| 4 | HAI DUAL AUROC/AUPR/F1 **0.916334/0.570193/0.610556** | 独立真实攻击的正向协同证据 | 不推广为所有数据集普适 |
| 5 | HAI short-lag AUPR **0.426064**，Final **0.570193** | HAI 中较长 lag 有贡献 | lag 作用是 dataset-dependent |
| 6 | SWaT short-lag AUPR **0.565012**，Final **0.167510** | SWaT 与 HAI 的 lag 方向相反 | 两者 event hit 都约 **2/36** |
| 7 | Robust-Z Normal top-1% DUAL share：三数据集均 **100%** | 证明线性稳健标准化仍被重尾分支支配 | 这是正常尾部审计，不是 Test 指标 |
| 8 | Tail REC/DUAL share：SWaT **65.19/34.81**，HAI **59.15/40.85**，WADI **51.78/48.22** | 证明尾概率校准改善证据可比性 | share 不是训练权重，也不保证性能增益 |
| 9 | HAI Tail 正式 seed AUPR **0.617897**，相对最佳单分支 **+0.047703** | Tail 的单数据集互补性证据 | HAI 限定，`MODERATE_SINGLE_DATASET` |
| 10 | HAI Tail 三 seed AUPR **0.605±0.019** | HAI ranking 重复性 | 不升级为跨数据集普遍结论 |
| 11 | HAI Tail HIGH19 **19/19 × 3 seeds** | 高影响事件覆盖的重复性 | 只针对 HAI HIGH19 |
| 12 | SWaT REC AUROC/AUPR **0.834887/0.743353** | 强 reconstruction ranking | operating point 迁移仍差 |
| 13 | SWaT REC 三 seed AUPR **0.735±0.016** | REC ranking variation 相对有限 | 三 seed 均值不覆盖正式 seed |
| 14 | SWaT REC strict q99 F1/FPR **0.258067/0.718133** | 阈值迁移失败的核心证据 | 不能隐藏高 FPR |
| 15 | SWaT REC Oracle F1 **0.768959** | 说明 score 尚有可分信息 | Test-best 诊断，绝不能替代 strict |
| 16 | SWaT DUAL/TAIL AUPR **0.167510/0.206747** | 协同与融合在 SWaT 的负向边界 | 两者均低于 REC **0.743353**；TAIL 有 seed 波动 |
| 17 | Final REC Params：SWaT/HAI **11,730/15,482** | 万级参数的轻量化证据 | 不代表对所有基线参数最少 |
| 18 | Dense R2/R0 Params：SWaT/HAI **2.192×/2.126×** | dense 明显增参且 AUROC/AUPR 未改善 | 只支持参数效率，不证明 R0/k=3 最优 |
| 19 | WADI 各版本 AUPR 约 **0.052–0.055** | domain-shift stress-test 失败边界 | 不能写成成功泛化 |
| 20 | 2026 外部可直接比较记录 **DIRECT=0** | 解释为何不做“最新方法统一排名” | 协议不同，不等于本文优于它们 |

## 三句数字串

1. **机制链：** 0.995989 → 0.996913 → HAI 0.916334/0.570193/0.610556。
2. **校准链：** 100% DUAL → 65/35、59/41、52/48 → HAI AUPR +0.047703。
3. **边界链：** SWaT REC AUPR 0.743353，但 strict F1 0.258067、FPR 0.718133；DUAL/TAIL AUPR 0.167510/0.206747。
4. **稳定性链：** SWaT REC AUPR 0.735±0.016；HAI Tail AUPR 0.605±0.019，HIGH19 为 19/19×3；不回避 SWaT Tail 与 q99 operating point 的波动。
