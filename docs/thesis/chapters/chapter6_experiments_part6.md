# 6.7 稳定性、文献兼容评价与近期方法对照

本节在主实验与消融实验均已冻结后，对结论稳定性、文献常用评价口径和2026年近期方法进行补充分析。三类证据承担不同作用：多随机种子结果用于检查随机初始化敏感性；Oracle与点调整（Point Adjustment，PA）用于诊断阈值迁移和区间评价效应；外部论文结果用于说明近期研究位置及协议差异。它们均不替代第6.3节Primary Strict主表，也不用于反向修改方法或实验协议。

## 6.7.1 多随机种子稳定性

稳定性实验固定随机种子1、42和43，只重复具有随机训练过程的REC与TAIL。DUAL在冻结数据、特征集合和确定性求解器下是确定性分支，因此不构造没有实际重复运行依据的“均值±0”。表6-10为补充性统计，正式主表仍保留冻结正式种子的单次结果。

表6-10 三随机种子稳定性结果（均值±样本标准差）

| 数据集 | 方法 | AUROC | AUPR | F1 | FPR |
|---|---|---:|---:|---:|---:|
| SWaT | REC | 0.829704 ± 0.008863 | 0.735474 ± 0.016188 | 0.260023 ± 0.005344 | 0.704951 ± 0.029769 |
| SWaT | TAIL | 0.722584 ± 0.081623 | 0.274866 ± 0.151947 | 0.263162 ± 0.007548 | 0.687025 ± 0.035857 |
| HAI21 | REC | 0.851566 ± 0.004793 | 0.534387 ± 0.020171 | 0.447195 ± 0.135943 | 0.032246 ± 0.026694 |
| HAI21 | TAIL | 0.898306 ± 0.003205 | 0.604836 ± 0.019403 | 0.546487 ± 0.111776 | 0.022642 ± 0.015476 |

REC的排序指标相对稳定：SWaT和HAI21的AUROC标准差分别为0.008863和0.004793。HAI21上的TAIL也保持较稳定的排序表现，其AUROC和AUPR标准差分别为0.003205和0.019403。但固定Normal Validation q99工作点的F1与FPR更易受种子影响，尤其HAI21 REC与TAIL的F1标准差分别为0.135943和0.111776。因此，需要区分排序稳定性与工作点稳定性：AUROC/AUPR的稳定并不自动保证q99阈值下的F1/FPR同样稳定。

SWaT TAIL必须保留为明确的负向稳定性证据。其AUPR为0.274866 ± 0.151947，三种子范围为0.168898—0.448953，审计状态为`SEED_VARIABILITY_OBSERVED`。这一波动说明尾概率融合在SWaT上的排序结果对随机训练误差具有较强敏感性，不能概括为“所有数据集均稳定”。

事件级结果与上述判断基本一致。SWaT REC和TAIL的事件命中分别为28.666667±0.577350/36和28.333333±0.577350/36；HAI21 REC和TAIL分别为46.333333±1.527525/50和47.000000±1.732051/50。HAI21 TAIL在三个种子下均命中HIGH19的19/19个事件，说明其对该类高相关攻击的覆盖具有重复性。综合而言，REC排序和HAI21 TAIL排序较稳定，但q99工作点与SWaT TAIL仍存在不可忽略的种子敏感性。

## 6.7.2 文献兼容评价与阈值迁移诊断

为与文献中常见但协议强度不同的评价方式对照，本研究在统一common endpoint mask上补充q99、Oracle、PA@q99和PA-Oracle四种F1。SWaT与HAI21的共同时间线分别包含449,791和401,685个合法端点。该表只服务于回顾性诊断，不与Historical Strict的不同端点集合混合计算或排名。

表6-11 共同端点集合上的文献兼容F1对照

| 数据集 | 方法 | F1-q99-common | F1-Oracle | PA-F1@q99 | PA-F1-Oracle |
|---|---|---:|---:|---:|---:|
| SWaT | LSTM-AE | 0.254495 | 0.762499 | 0.266571 | 0.884087 |
| SWaT | USAD | 0.255184 | 0.741845 | 0.269202 | 0.805217 |
| SWaT | GDN | 0.720216 | 0.772086 | 0.870736 | 0.900728 |
| SWaT | Ours-REC | 0.258068 | 0.768959 | 0.272041 | 0.887275 |
| SWaT | Ours-DUAL | 0.015663 | 0.406974 | 0.187814 | 0.809482 |
| SWaT | Ours-TAIL | 0.260790 | 0.339804 | 0.274334 | 0.398977 |
| HAI21 | LSTM-AE | 0.512393 | 0.637348 | 0.644794 | 0.883002 |
| HAI21 | USAD | 0.399760 | 0.579345 | 0.572777 | 0.873148 |
| HAI21 | GDN | 0.562053 | 0.590018 | 0.762172 | 0.840044 |
| HAI21 | Ours-REC | 0.520594 | 0.639274 | 0.730542 | 0.908513 |
| HAI21 | Ours-DUAL | 0.610556 | 0.654843 | 0.785479 | 0.842294 |
| HAI21 | Ours-TAIL | 0.617350 | 0.686865 | 0.768181 | 0.919477 |

Oracle阈值通过Test标签选择，PA又使用真实异常区间传播区间内命中，二者都不具备部署可用性，因而不能替代Primary Strict中的Normal Validation q99结果。以SWaT Ours-REC为例，F1由q99的0.258068提高到Oracle的0.768959，同时Oracle FPR降至0.002202。其Oracle F1和PA-Oracle F1均列第2，仅次于GDN。这表明其原始分数具有较强排序能力，但Normal Validation到Test的阈值迁移明显失配。

PA并不能解释SWaT Ours-REC在q99工作点的主要问题：PA-F1@q99仅由0.258068升至0.272041。传统PA可以填充已经命中的异常区间，却不能删除大量正常点误报，因此该数据集的主要限制仍是阈值迁移，而不是“缺少PA”。

HAI21则呈现不同侧面。Ours-DUAL在共同端点上的AUROC为0.916334，而Ours-TAIL的AUPR为0.617897；q99 F1、Oracle F1和PA类F1的领先方法并不相同。Ours-TAIL的F1由0.617350提高到Oracle的0.686865，PA-Oracle达到0.919477。这些结果说明不同指标强调排序、阈值迁移或事件覆盖的不同方面，不支持单一方法在所有文献兼容口径下普遍领先。

## 6.7.3 2026年近期方法的协议感知对照

E3审计对Park与Shim的依赖关系插件、LatentFlow、MAD-TCN和CCM-TAD进行官方来源优先的结果登记。本节不声称完成这些方法的本地复现，也不把它们与本研究的Strict结果组成统一排行榜。冻结协议注册表中外部结果的DIRECT可比记录数为0；其余结果因指标语义、阈值选择、测试驱动模型选择、预处理或数据版本差异，只能标记为PARTIAL或NOT_DIRECT。

表6-12 2026年近期方法的协议感知结果摘录

| 方法 | 发表场所 | 数据集 | 报告指标 | 报告值 | 关键协议标记 | 可比性 |
|---|---|---|---|---|---|---|
| Park-Shim Plug-in | Expert Systems with Applications | SWaT | F1提升范围 | 0.028–0.050 | 仅恢复publisher文字总结；Table 3绝对值未提取 | NOT_DIRECT |
| Park-Shim Plug-in | Expert Systems with Applications | WADI | 最大F1提升 | 0.417 | Abstract claim；未知对应base model | NOT_DIRECT |
| LatentFlow | KDD 2026 | SWaT | Aff-P/R/F1；VUS-ROC/PR | Aff-Precision: 65.01 ± 0.74；Aff-Recall: 93.59 ± 2.28；Aff-F1: 76.70 ± 0.50；VUS-ROC: 78.47；VUS-PR: 59.45 | Affiliation指标；SPOT；3次重复 | NOT_DIRECT |
| LatentFlow | KDD 2026 | WADI | Aff-P/R/F1；VUS-ROC/PR | Aff-Precision: 63.72 ± 0.30；Aff-Recall: 95.71 ± 0.73；Aff-F1: 76.51 ± 0.13；VUS-ROC: 47.69；VUS-PR: 6.15 | Affiliation指标；SPOT；3次重复 | NOT_DIRECT |
| MAD-TCN | Neurocomputing | 跨数据集摘要 | 参数量；训练/推理速度提升 | 约0.026 M；约34% | 摘要级效率声明 | NOT_DIRECT |
| MAD-TCN | Neurocomputing | SWaT | AUC/F1/Precision/Recall | NOT_EXTRACTED | 官方可追溯主性能表未恢复 | PARTIAL |
| CCM-TAD | ICML 2026 | SWaT | PF1；SF1 | 0.882；0.883 | Test-best阈值与Test驱动选择 | PARTIAL/NOT_DIRECT |
| CCM-TAD | ICML 2026 | WADI | PF1；SF1 | 0.739；0.761 | Test-best阈值与Test驱动选择 | PARTIAL/NOT_DIRECT |
| CCM-TAD | ICML 2026 | SWaT | PR-AUC；VUS-PR/ROC | 0.911；0.779/0.758 | checkpoint按Test PF1/SF1选择；range-aware指标 | PARTIAL/NOT_DIRECT |
| CCM-TAD | ICML 2026 | WADI | PR-AUC；VUS-PR/ROC | 0.72；0.776/0.822 | checkpoint按Test PF1/SF1选择；range-aware指标 | PARTIAL/NOT_DIRECT |
| CCM-TAD | ICML 2026 | SWaT | Range-F1；Affiliation-F1 | 0.681；0.781 | 指标语义不同；不能转成普通point F1 | NOT_DIRECT |

Park-Shim的官方publisher文字仅支持SWaT F1提升0.028—0.050和WADI最大提升0.417，不能由增量反推出任一base model的绝对F1。LatentFlow的Affiliation-F1也不是普通点级F1，例如SWaT的76.70±0.50只能保留为百分数形式的Affiliation指标，不能换算成0.767后与本研究F1直接比较。MAD-TCN目前只恢复了约0.026 M参数与约34%的训练/推理速度提升声明，SWaT主性能表仍为NOT_EXTRACTED，因此正文不补写或猜测其AUC、F1、Precision与Recall。

CCM-TAD的冻结注册表保留20条正式结果。本节分别呈现PF1、SF1、PR-AUC、VUS、Range-F1和Affiliation-F1，而不把它们统称为同一“F1”。其官方协议包含按Test PF1/SF1选择checkpoint、利用Test标签优化alpha与阈值，以及会使用未来点的centered moving average。上述差异并不否定该方法的论文报告结果，但使其不能作为与本研究Normal Validation q99、无PA、无Test-best协议的DIRECT行。

## 6.7.4 本节小结

本节得到三点结论。第一，多随机种子结果支持REC排序和HAI21 TAIL排序具有一定重复性，同时SWaT TAIL的AUPR波动和q99工作点敏感性必须保留为稳定性边界。第二，文献兼容评价说明Oracle与PA能够显著改变F1，但它们依赖Test标签或真实区间，只能用于诊断；SWaT Ours-REC的主要限制是Normal Validation到Test的阈值迁移，而不是缺少PA。第三，2026年近期工作采用的指标、阈值、模型选择和预处理差异显著，协议审计中没有可直接并入Primary Strict主表的DIRECT外部结果。

因此，本研究最终坚持“主协议结论、补充诊断和外部文献对照”三层分离：正式结论以冻结Strict结果为准，多种子与Oracle/PA用于揭示稳定性及阈值边界，近期方法则按其原始指标与协议标记呈现，不进行不受支持的统一排名。
