# 6.3 总体检测性能比较

本节仅比较统一实测的 LSTM-AE、USAD、GDN 与本文三个正式分支 REC、DUAL、TAIL。表 6-3 的全部数值来自冻结的 Historical Strict / Primary 结果：Normal Validation q99、无 Test tuning、无 Point Adjustment。common-mask、Oracle、PA、多随机种子均值和 2026 paper-reported 结果不进入本表。

**表 6-3 SWaT 与 HAI21 上的 Primary Strict 检测结果**

| Dataset | Method | AUROC | AUPR | Precision | Recall | F1 | FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| SWaT | LSTM-AE | 0.818242 | 0.706043 | 0.147874 | 0.912250 | 0.254495 | 0.726609 |
| SWaT | USAD | 0.803816 | 0.698928 | 0.148539 | 0.904762 | 0.255184 | 0.716859 |
| SWaT | GDN | 0.827665 | 0.728194 | 0.793843 | 0.659087 | 0.720216 | 0.023658 |
| SWaT | Ours-REC | 0.834887 | 0.743353 | 0.150140 | 0.917870 | 0.258067 | 0.718133 |
| SWaT | Ours-DUAL | 0.675631 | 0.167510 | 0.087166 | 0.008605 | 0.015663 | 0.012455 |
| SWaT | Ours-TAIL | 0.708375 | 0.206747 | 0.152452 | 0.901228 | 0.260789 | 0.692531 |
| HAI21 | LSTM-AE | 0.907084 | 0.577510 | 0.399575 | 0.713982 | 0.512393 | 0.024441 |
| HAI21 | USAD | 0.856873 | 0.494578 | 0.300978 | 0.595060 | 0.399760 | 0.031484 |
| HAI21 | GDN | 0.877643 | 0.497775 | 0.537723 | 0.588689 | 0.562053 | 0.011529 |
| HAI21 | Ours-REC | 0.852899 | 0.536294 | 0.459109 | 0.601095 | 0.520594 | 0.016133 |
| HAI21 | Ours-DUAL | 0.916334 | 0.570193 | 0.554350 | 0.679446 | 0.610556 | 0.012443 |
| HAI21 | Ours-TAIL | 0.901850 | 0.617897 | 0.555963 | 0.693976 | 0.617350 | 0.012627 |

注：AUROC 与 AUPR 为连续异常分数的排序指标；Precision、Recall、F1 与 FPR 均对应各方法冻结的 Normal Validation q99 operating point。表中未用加粗暗示“综合最优”，不同指标应按其统计语义分别解释。

## 6.3.1 SWaT 总体结果

从连续分数排序看，Ours-REC 的 AUROC 和 AUPR 分别为 0.834887 和 0.743353，在当前统一实测方法中均为最高值。这说明轻量重构分支能够在 SWaT Official Physical Test 上形成较强的异常排序信息。这里的结论仅针对 raw score ranking，不代表固定阈值下的全部检测指标都占优。

在冻结的 Validation-q99 operating point 上，Ours-REC 的 F1 为 0.258067，FPR 为 0.718133；GDN 的 F1 为 0.720216，FPR 为 0.023658。二者的差异表明，连续分数的 ranking quality 与 Normal Validation q99 向 Test 的 operating-point transfer 是两个不同问题。Ours-REC 保留了较高排序能力，但该阈值在 SWaT Test 上产生了较高误报率；GDN 在当前 q99 operating point 上表现出更低的误报率和更高的 F1。

协同分支与融合分支必须作为 SWaT 的适用边界主动报告。Ours-DUAL 的 AUPR 为 0.167510，Ours-TAIL 的 AUPR 为 0.206747，均未保持 Ours-REC 的排序优势。TAIL 的 F1 为 0.260789，但 FPR 仍为 0.692531，因此不能仅凭单一 operating-point 指标把融合解释为正向收益。

综合而言，SWaT 支持“REC 排序较强但 q99 阈值迁移较差”的受限结论；DUAL 与 TAIL 在 Official real attacks 上的结果有限，说明该数据集中的攻击并非都表现为可由当前双视角分支捕获的显式协同关系破坏。该现象将在 6.5 节通过机制实验进一步分析。

## 6.3.2 HAI21 总体结果

HAI21 上，Ours-DUAL 的 AUROC 为 0.916334，在表 6-3 中最高；其 AUPR 为 0.570193，F1 为 0.610556。这表明显式 Self/Cross 协同关系证据在 HAI 真实攻击上具有较强的排序能力，并能在冻结 q99 operating point 上形成有效检测输出。REC 在此处承担基础数值与重构异常证据的角色，其 AUROC/AUPR 为 0.852899/0.536294；REC 不是 HAI 上最强的正式分支，但它为后续异构证据融合提供了独立输入。

在同一 HAI Strict 协议下，本地复现 GDN 的 AUROC、AUPR 和 F1 分别为 0.877643、0.497775 和 0.562053；Ours-DUAL 对应为 0.916334、0.570193 和 0.610556。由冻结数值作确定性差值得到，DUAL 分别高出 0.038691、0.072419 和 0.048503（derived arithmetic）。该比较只适用于 HAI21 的本地统一 Strict 协议，不能扩展为本文关系建模方法在所有数据集或所有协议下全面优于 GDN。

Ours-TAIL 的 AUPR 为 0.617897，F1 为 0.617350。相对本文 AUPR 较高的单分支 Ours-DUAL，TAIL 的 AUPR 增加 0.047703（derived arithmetic）。该结果说明，在 HAI21 上，REC 与 DUAL 的互补证据经正常尾概率校准后能够转化为当前协议下的检测收益。

这一收益仍需按指标限定。Ours-DUAL 的 AUROC 0.916334 高于 Ours-TAIL 的 0.901850，GDN 的 FPR 0.011529 低于 DUAL 和 TAIL。因而，HAI 结果支持的是 DUAL 排序证据与 TAIL 互补收益，而不是 TAIL 在所有指标上均为最高。

## 6.3.3 跨数据集结果分析

两个主要真实工业数据集呈现不同的证据结构。SWaT 上，REC 的 AUROC/AUPR 较强，DUAL 较弱，TAIL 未产生正向的融合 ranking 收益；HAI21 上，DUAL 提供了较强协同关系证据，TAIL 则在 AUPR 与冻结 q99 F1 上把 REC/DUAL 的互补性转化为收益。相同方法组件在两个数据集上的作用方向并不一致。

这种差异说明协同异常分支的收益依赖具体攻击机制、可学习的跨通道关系以及训练与测试分布。REC、DUAL 和 TAIL 的职责应分别理解为基础数值/重构证据、协同关系证据以及异构证据校准融合；某一分支在一个数据集上较弱，不等同于方法定义错误，也不能被另一个数据集上的正向结果掩盖。

因此，本节不形成“本文整体方法在所有数据集均占优”的结论。SWaT 的负向边界与 HAI21 的正向协同证据共同限定了方法的 dataset-dependent applicability。关系异常机制、受控关系破坏和事件层面的进一步解释留待 6.5 节，稳定性及文献兼容诊断留待 6.7 节。
