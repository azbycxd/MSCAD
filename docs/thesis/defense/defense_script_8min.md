# 8 分钟答辩压缩讲稿

论文题目：**轻量化多变量时序协同异常检测方法研究**

## 0:00–0:50 背景与问题

各位老师好。工业多变量时序异常不仅表现为单通道数值偏离，也可能表现为变量之间协同关系被破坏。传统 reconstruction score 擅长检测数值异常，但不能稳定提供独立的关系证据；同时，重构分数与关系分数的尺度和尾部形态不同，直接融合可能被一个分支支配。我的工作围绕这两个问题，并兼顾轻量部署。

## 0:50–1:30 总体方案

系统由轻量 REC、双视角 DUAL 和 Tail calibration 组成。所有模型、关系参考、可靠性筛选和阈值只使用正常训练或验证数据；论文主表统一使用 Validation q99、无 Test tuning、无 Point Adjustment。

## 1:30–3:20 核心创新 1：双视角协同一致性

对每个 target，我用自身历史建立 Self View，用同过程其他通道的同步—滞后信息建立 Cross View，并用正常数据拟合 Ridge 预测器。

> 我的协同异常分支不使用 observed-target residual，而是比较 Self 历史预测与 Cross 同过程变量预测之间的分歧，因此关系变化本身可以形成独立异常证据。

多尺度 lag 用于表达工业响应延迟，但其作用并不普适：HAI 的完整 lag AUPR 为 0.570193，短 lag 降为 0.426064；SWaT 上短 lag 排序反而更好，事件命中仍约 2/36。因此结论是 dataset-dependent。

全局分数用 max 保留稀疏强关系破坏，但 max 容易被不可靠 target 污染，所以加入 Normal-only Reliability Gate。SWaT controlled 移除 Gate 后 AUROC 从 0.996913 降到 0.630637，支持这个机制；真实测试上不保证同幅度收益。

分层证据包括 Historical CoMuT AUROC 0.995989、SWaT controlled 0.996913，以及 HAI DUAL AUROC/AUPR/F1 为 0.916334/0.570193/0.610556。

## 3:20–4:30 核心创新 2：正常尾概率校准

Robust-Z 在 SWaT、HAI、WADI 的正常 top-1% 区域都由 DUAL 100% 支配，说明中心和尺度标准化没有解决重尾可比性。本文用正常经验上尾概率把 REC 和 DUAL 转换到统一证据尺度。校准后，三套数据的 REC/DUAL 贡献约为 65/35、59/41 和 52/48。

这个贡献首先是 comparability。HAI Tail AUPR 为 0.617897，比最佳单分支增加 0.047703；但 SWaT Tail AUPR 只有 0.206747，低于 REC 0.743353，所以不能宣称普遍提升。

## 4:30–5:15 轻量化设计

REC 使用 depthwise-separable temporal convolution，DUAL 使用 Ridge，Tail 不增加可训练参数。REC 在 SWaT/HAI 为 11,730/15,482 个参数。Dense variant 约为 2.19/2.13 倍参数且没有稳定收益。

同时，R1 的 k=1 回顾性消融更小且观测排序更好。因此我明确说明：**本文并不声称当前 k=3 配置是最优结构。** R1 发生在主方法冻结和 Test exposure 之后，不能反向替换 Final；轻量化贡献是参数高效框架，而不是固定 kernel 最优。

## 5:15–7:05 主要结果与边界

SWaT 上 REC AUPR 为 0.743353，排序较强；但严格 q99 F1 仅 0.258067、FPR 0.718133，说明阈值迁移失败。Oracle F1 0.768959 只作 Test-best 诊断，不能替代主结果。DUAL AUPR 0.167510、Tail 0.206747，必须如实作为适用边界。

HAI 上 DUAL 与 Tail 分别提供关系证据和互补收益，是本文真实攻击中的正向支持。WADI 各版本 AUPR 约 0.052–0.055，即使证据贡献接近 52/48，仍说明 comparability 不等于 domain-shift detection。

2026 文献方面，我审计了 Park & Shim、LatentFlow、MAD-TCN 和 CCM-TAD。由于指标、PA、Test-best、选择方式和数据预处理不同，外部 DIRECT 比较数为 0；因此只做 protocol-aware 补充比较，不宣称超过所有最新方法。

## 7:05–8:00 总结

本文的两个核心创新分别是：以局部滞后 Self/Cross 预测分歧形成独立协同异常证据，以及以正常尾概率统一异构证据语义；轻量化贡献是由 depthwise-separable REC、线性 DUAL 和非可训练 Tail 组成的参数高效框架。

结论边界是：lag 依赖数据集，Gate 的强证据来自 controlled 场景，Tail 的正向互补主要来自 HAI，SWaT 有严重阈值迁移，WADI 有域偏移，k=3 不具最优性证据。后续将研究自适应 lag、R1 独立验证、Normal-only 阈值迁移和域适配。

我的汇报结束，谢谢各位老师。
