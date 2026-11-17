# 论文最终表格整合顺序

目标是让主性能、效率、消融和文献协议各自承担单一证据角色，避免把严格指标、回顾性 Oracle/PA 和外部论文异构指标混入同一主表。

| 顺序 | 建议表名 | 建议位置 | 主要来源 | 保留内容 | 避免重复 |
|---:|---|---|---|---|---|
| 1 | 主性能比较 | 6.2.1 | final_comparison_tables Table 1 | SWaT/HAI Historical Strict；AUROC、AUPR、q99 指标 | 不加入 Oracle、PA 或外部 paper-reported 值 |
| 2 | 参数规模与推理效率 | 6.2.2 | final_comparison_tables Table 3；efficiency E1 | Params、state bytes、batch=1 latency、VRAM | 不重复在每个消融表列完整效率字段 |
| 3 | 轻量化框架结构与效率分析 | 6.4.1 | Abl-1 | R0/R1/R2 的 Params、AUROC、AUPR、F1、FPR、latency | 不称为第一算法创新；与表 2 只保留必要交叉字段 |
| 4 | 双视角真实攻击消融 | 6.4.2 | Abl-2 | SWaT/HAI D0、D1、D2；targets、ranking、q99、event hits | 不把 Controlled 指标混入工业主性能行 |
| 5 | 关系机制验证 | 6.4.2 后半 | Abl-3 | Historical CoMuT 与 SWaT controlled 的 AUROC/AUPR/Recall/FPR | 明确为 mechanism evidence，不参与 baseline 排名 |
| 6 | 尾概率校准与融合消融 | 6.4.3 | Abl-4；normal-tail balance | REC/DUAL/Robust-Z/Tail 检测指标及正常尾贡献 | WADI 只作 stress-test 补充，不进入主结论 |
| 7 | 文献兼容评价 | 6.5.1 或附录 | final_comparison_tables Table 2 | common-mask、F1-Oracle、PA-F1 | 与主性能表分离并标为 retrospective diagnostic |
| 8 | 2026 文献协议比较 | 6.5.2 或附录 | final_comparison_tables Table 4；E3 | 第一方报告值、原指标语义、comparability | 不换算 Affiliation/VUS/SF1，不填 NOT_EXTRACTED |

## 表格编号与 Ablation LaTeX 映射

- Abl-1：Lightweight framework structural and efficiency analysis。
- Abl-2：Core method innovation 1 real-attack ablation。
- Abl-3：Core method innovation 1 mechanism ablation。
- Abl-4：Core method innovation 2 calibration and fusion ablation。

## 合并与删重规则

1. 主性能表是唯一 strict leaderboard。其他表不得复制全部 baseline 行。
2. 效率表只保留一次完整 Params/state/latency/VRAM；Abl-1 仅保留解释轻量化结论所需字段。
3. 双视角真实攻击与关系机制分为两张表，因为证据语义分别是工业性能和受控机制。
4. 融合表中可将 Normal-tail share 作为独立列或紧邻子表；不得把 share 解释为训练权重。
5. Literature-compatible 与 2026 paper-reported 保持分表，前者是本地回顾性诊断，后者是不同协议的第一方报告。
6. paper_table_layout_draft 中旧 TBD 行不得直接沿用；最终数值统一来自 Abl-1 至 Abl-4 冻结表。

