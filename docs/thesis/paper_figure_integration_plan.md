# 论文消融图整合方案

三张图均来自冻结最终 CSV，并保留 PDF 矢量版与 300 dpi PNG。建议统一放在第 6 章“实验与分析”的消融小节，不分散到方法章节，以避免把回顾性消融误写为方法设计依据。

## Figure Abl-1

- 文件：fig_ablation_lightweight_rec.pdf / png
- 建议位置：6.4.1 轻量化重构分支的结构与效率分析
- 中文图题：轻量化重构分支的 AUPR—参数量与 AUPR—推理时延关系
- 英文图题：AUPR–Parameter and AUPR–Latency Trade-offs of the Lightweight Reconstruction Branch
- 正文引用句：如图 Abl-1 所示，dense R2 在两个数据集上显著增加参数量但未提高 AUPR，而 R1 的回顾性结果表明固定时间卷积核宽度不具备跨数据集已验证的最优性。
- 图后分析要点：
  1. R0 保持万级参数规模。
  2. R2 的参数为 R0 的约 2.19 倍和 2.13 倍，AUPR 均下降。
  3. R1 参数更少且 AUROC/AUPR 更高，但属于 single-seed retrospective finding。
  4. 时延方向并不与参数量单调一致，不能单凭一次硬件测量宣称结构优越。

## Figure Abl-2

- 文件：fig_ablation_dual_view.pdf / png
- 建议位置：6.4.2 滞后双视角协同检测机制消融分析
- 中文图题：真实攻击与受控关系破坏场景下的双视角组件消融
- 英文图题：Dual-View Component Ablation on Real Attacks and Controlled Relation Breaking
- 正文引用句：图 Abl-2 同时展示了真实攻击与受控关系破坏结果，HAI 上较长滞后具有正向贡献，而 Controlled SWaT 中移除 Reliability Gate 会使关系机制指标明显退化。
- 图后分析要点：
  1. HAI D1 AUPR 从 0.570193 降至 0.426064。
  2. HAI D2 与 D0 接近，说明真实测试上的 Gate 差异不能单独证明机制。
  3. Controlled SWaT D2 AUROC/AUPR 降至 0.630637/0.550076，揭示不可靠 target 对 max aggregation 的污染。
  4. SWaT 真实攻击 D1 排序提高但事件命中不变，必须保留数据集依赖性。

## Figure Abl-3

- 文件：fig_ablation_tail_calibration.pdf / png
- 建议位置：6.4.3 异构异常证据尾概率校准消融分析
- 中文图题：正常尾部证据贡献及尾概率校准后的检测结果
- 英文图题：Normal-Tail Evidence Contribution and Detection Consequences after Tail Calibration
- 正文引用句：图 Abl-3 表明，尾概率校准消除了 Robust-Z 在三套数据上的 DUAL 单分支完全支配，并同时呈现 HAI 的互补收益与 SWaT 未保持 REC 排序优势的边界。
- 图后分析要点：
  1. Robust-Z 的 Normal top-1% 在三套数据上均为 100% DUAL。
  2. Tail-Calibrated 后 REC/DUAL 分别约为 65/35、59/41 和 52/48。
  3. HAI Tail AUPR 0.617897，高于最佳单分支 0.047703。
  4. SWaT Tail AUPR 0.206747，明显低于 REC 0.743353；图中不得隐藏该负向结果。

## 排版规则

- 正文优先引用 PDF；电子附件或预览使用 PNG。
- 图题只描述比较对象，不使用“最优”“全面优于”或“有效性证明”。
- Abl-1 属于轻量化设计贡献；Abl-2 对应核心方法创新 1；Abl-3 对应核心方法创新 2。
- 图后先陈述数值事实，再给机制解释，最后保留适用边界。

