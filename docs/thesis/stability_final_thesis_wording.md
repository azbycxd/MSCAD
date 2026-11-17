# 三随机种子稳定性论文措辞

## 1. 论文可直接使用的正文

为分析随机初始化对结果的影响，本文进一步进行了三随机种子实验。该实验固定数据划分、模型结构、训练协议、正常验证集阈值和确定性的 DUAL 分支，仅改变 REC 的随机初始化，并将三种子结果作为稳定性补充证据，不覆盖正式主表中的历史 seed 结果。SWaT 上 REC 的 AUROC/AUPR 为 `0.829704±0.008863`/`0.735474±0.016188`，HAI 上 REC 的 AUROC/AUPR 为 `0.851566±0.004793`/`0.534387±0.020171`，表明 REC 的排序指标在当前三种子范围内波动相对有限。HAI 上 TAIL 的 AUROC/AUPR 为 `0.898306±0.003205`/`0.604836±0.019403`，且三个 seed 对 HIGH19 均实现 `19/19` 事件命中，说明其 HAI 排序证据和高影响事件覆盖具有较好重复性。与此同时，固定 Normal Validation q99 对应的 operating-point 指标对 seed 更敏感：HAI REC 的 F1/FPR 标准差分别为 `0.135943`/`0.026694`，HAI TAIL 分别为 `0.111776`/`0.015476`。此外，SWaT TAIL 的 AUPR 标准差为 `0.151947`，范围为 `0.168898–0.448953`，因此记录为 `SEED_VARIABILITY_OBSERVED`。综合而言，现有结果支持“核心 ranking evidence 相对稳定”的受限表述，但不支持“所有 operating point 均对随机种子不敏感”的结论。

## 2. Table Stability：三 seed mean±sample std

| Dataset | Method | AUROC mean±std | AUPR mean±std | F1 mean±std | FPR mean±std |
|---|---|---:|---:|---:|---:|
| SWaT | REC | 0.829704±0.008863 | 0.735474±0.016188 | 0.260023±0.005344 | 0.704951±0.029769 |
| SWaT | TAIL | 0.722584±0.081623 | 0.274866±0.151947 | 0.263162±0.007548 | 0.687025±0.035857 |
| HAI21 | REC | 0.851566±0.004793 | 0.534387±0.020171 | 0.447195±0.135943 | 0.032246±0.026694 |
| HAI21 | TAIL | 0.898306±0.003205 | 0.604836±0.019403 | 0.546487±0.111776 | 0.022642±0.015476 |

注：DUAL 在冻结数据、特征与 solver 下为确定性分支，不将同一确定性结果伪写为 `mean±0`。表中三 seed 为 `1/42/43`，其中 seed 1 是原正式 seed。

## 3. 论文解释边界

- **Ranking stability：** SWaT/HAI 的 REC 以及 HAI 的 TAIL 在 AUROC/AUPR 上表现出相对有限的三 seed 波动。
- **Threshold stability：** HAI 的 F1/FPR 波动更明显，与既有 Normal Validation→Test threshold transfer 结论一致。
- **明确负向结果：** SWaT TAIL 的 AUPR 对 seed 明显敏感，必须保留 `SEED_VARIABILITY_OBSERVED`。
- **Primary 不变：** 正式主表继续使用历史正式 seed；三 seed mean±std 只进入 Stability/Robustness supplementary table。
- **主张强度不变：** 本表补充 C5、C6、C7 的重复性或边界证据，不把 HAI 单数据集互补性升级为跨数据集普遍结论。

## 4. 禁止措辞

- 禁止写“模型整体非常稳定”。
- 禁止用三 seed 均值替换历史 Primary 数值。
- 禁止把 HAI 的结果扩展为所有数据集均稳定。
- 禁止忽略 SWaT TAIL 与 q99 operating-point 的 seed 波动。

数据来源：`outputs/thesis_stability/stability_summary.csv`、`outputs/thesis_stability/stability_event_summary.csv`。

