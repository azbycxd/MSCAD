# 6.5 协同关系异常机制验证

本节不再进行普通性能排名，而是回答一个机制问题：DUAL是否真正对跨变量协同关系的破坏敏感。为避免把不同实验语义混为一谈，证据按“合成机制—真实轨迹受控破坏—真实攻击—适用边界”组织。前三层逐步增强场景真实性，第四层主动保留负向结果。所有数值均来自冻结产物，本节未重新训练、推理或计算实验结果。

## 6.5.1 Historical CoMuT纯关系异常验证

Historical CoMuT用于隔离纯关系异常，而不是比较工业数据集上的综合检测性能。DUAL分别从目标自身历史构造Self View，从其他通道构造Cross View，并以两类预测的分歧形成异常分数。核心分数不包含观测目标值的重构残差，因此能够单独检验预测关系变化是否产生可区分证据。

在13个冻结case上，DUAL的median AUROC、median AUPR、q99 median recall和median FPR分别为0.995989、0.686927、0.937500和0.010313，13/13个case的AUROC均不低于0.80。按异常持续时间与组合形式划分，类别median AUROC/AUPR分别为：Short为0.995730/0.403800，Medium为0.995043/0.506890，Large为0.996165/0.751877，Multiple-short为0.997180/0.756346。四类异常均取得较高AUROC，说明DUAL能够对纯关系变化形成高区分度异常分数；AUPR随异常长度和组合形式变化，提示有限异常占比下的精确排序仍受具体case影响。

该结果为双视角预测分歧提供了独立的关系异常证据，但CoMuT属于合成机制基准，不能据此推断真实工业攻击一定破坏可学习协同关系，也不能将其数值与后续工业数据结果组成无条件排行榜。

## 6.5.2 真实SWaT轨迹上的受控关系破坏

第二层实验使用真实SWaT正常运行轨迹，并冻结6个target、source、donor和样本索引，在保持目标通道不被直接替换的条件下破坏局部输入关系。Official Attack数据不参与构造或选择。该实验比纯合成CoMuT更接近工业信号分布，但异常仍是人为控制的relation breaking，不等价于真实攻击。

DUAL在6个target上的median AUROC为0.996913，median AUPR为0.996931，median corruption recall为0.872750，median clean FPR为0.009500。逐target结果如下。

| Target | Stage | AUROC | AUPR |
|---|---|---:|---:|
| AIT202 | P2 | 0.710757 | 0.709433 |
| FIT201 | P2 | 0.999887 | 0.999891 |
| DPIT301 | P3 | 0.998018 | 0.997909 |
| FIT301 | P3 | 0.995807 | 0.995952 |
| FIT401 | P4 | 0.512903 | 0.519511 |
| FIT601 | P6 | 0.999999 | 0.999999 |

其中4/6个target的AUROC不低于0.80。FIT201、DPIT301、FIT301和FIT601接近完全区分，AIT202为中等响应，FIT401的AUROC/AUPR仅为0.512903/0.519511。因此，高median不能替代target级完整报告。这一层证据说明，冻结的双视角分歧可以在真实工业正常轨迹上响应受控关系破坏，但不能用0.996913的受控AUROC替代Official SWaT真实攻击的结果。

Reliability Gate的作用也能在该场景中得到机制解释。移除Gate后，D2 No-Gate的median AUROC和median AUPR降至0.630637和0.550076，median recall仅0.008000，而Final D0保持0.996913/0.996931。原因不是Gate根据Test性能优化target，而是未通过正常期可靠性约束的target进入max aggregation后，可能以较大的正常分歧污染全局分数。完整Gate消融留至第6.6节，本节仅引用其机制含义。

## 6.5.3 HAI21真实攻击中的协同异常证据

HAI21提供本研究最重要的真实攻击证据。在Strict协议下，Ours-DUAL的AUROC、AUPR、F1和FPR分别为0.916334、0.570193、0.610556和0.012443。按事件内预测异常比例不低于0.10定义命中时，DUAL命中44/50个official event。预冻结HIGH19分组的median hit为0.841085，其中9/19个事件满足DUAL hit至少比REC高0.10。

事件互补性分析进一步显示，50个事件中REC共有5个near-miss，DUAL恢复其中3个，分别为T2-E06、T2-E14和T5-E09；在HIGH19中只有1个REC near-miss，并由DUAL在T2-E06上恢复。该结果支持“部分REC低命中事件可由协同证据补充”，但不支持“DUAL恢复全部REC漏检”。HIGH19沿用实验前冻结的高协同场景分组，没有根据当前结果重新定义。

作为简短对照，本地GDN在同一HAI Strict评价下的AUROC/AUPR/F1为0.877643/0.497775/0.562053，低于DUAL的对应指标。这一对照仅说明当前严格协议下显式双视角关系证据的表现，不在本节重复第6.3节的完整baseline排名。

多尺度lag对HAI关系证据也有正向作用。D1 Short-lag的AUROC/AUPR/F1为0.882610/0.426064/0.556056，HIGH19 median hit为0.559055；Final对应值为0.916334/0.570193/0.610556和0.841085。因此，在HAI21上保留多尺度局部滞后提高了协同异常证据质量。该结论具有数据集依赖性，详细消融仍在第6.6节展开。

## 6.5.4 SWaT Official攻击上的适用边界

Official SWaT真实攻击给出了必须保留的负向边界。DUAL的AUROC、AUPR、F1和FPR分别为0.675631、0.167510、0.015663和0.012455，按0.10事件命中阈值仅命中2/36个事件。历史冻结的contextual candidate为事件13、14和29，但三者的DUAL hit rate均为0，均未达到命中阈值。这表明Official SWaT攻击对显式协同关系异常的支持有限，受控实验的高AUROC不能替代真实攻击证据。

正常关系分布也呈现不同程度的迁移。SWaT Test-normal相对Validation的DUAL disagreement ratio中位数、q95和最大值分别为2.509313、12.827940和14.264853；HAI21对应值为1.000395、1.223965和1.312587。HAI的正常关系regime相对稳定，而SWaT的多个target在Test正常段出现明显变化，这会削弱由Normal Validation冻结的DUAL reference的迁移性。

不过，不能把SWaT结果完全归因于relation-domain shift。另一项同样重要的可能原因是Official SWaT攻击语义并不都表现为可学习跨变量关系的破坏。现有证据只能说明SWaT结果有限可能同时受到攻击语义与正常关系域偏移影响，不能对两者作未经实验分离的因果排序。

## 6.5.5 分层证据总结

表6-5按照证据语义汇总协同关系异常机制的验证链，而不是对四类场景作性能排名。

表6-5 协同关系异常机制的分层验证

| 证据层级 | 数据集 | 场景 | 主要结果 | 解释 | 边界 |
|---|---|---|---|---|---|
| Mechanism evidence | Historical CoMuT | 纯关系异常 | median AUROC/AUPR=0.995989/0.686927；13/13 AUROC≥0.80 | DUAL对纯关系变化形成高区分度分数 | 合成机制基准，不等同于工业Test性能 |
| Controlled industrial-trajectory evidence | SWaT | 真实正常轨迹上的受控关系破坏 | median AUROC/AUPR=0.996913/0.996931；4/6 target AUROC≥0.80 | 将机制证据推进到真实工业轨迹 | 异常由冻结donor构造，不是真实攻击 |
| Real-attack evidence | HAI21 | Official real attacks | AUROC/AUPR/F1=0.916334/0.570193/0.610556；44/50事件命中 | 支持高协同真实攻击中的关系证据 | 单数据集正向证据，不外推为普适性 |
| Applicability boundary | SWaT Official | Official real attacks | AUROC/AUPR/F1=0.675631/0.167510/0.015663；2/36事件命中 | 暴露关系证据在攻击语义和域偏移下的限制 | 不隐藏负结果，也不由受控结果替代 |

四层证据共同构成“机制—受控—真实—边界”的完整链条。能够支持的结论是：DUAL对跨通道关系破坏的响应获得了合成、真实轨迹受控实验和HAI真实攻击的分层支持；其有效性依赖异常是否破坏可学习关系以及正常关系reference能否迁移。该证据链不支持在所有数据集或所有攻击类型上普遍有效的表述。
