# 未决资产登记

本文只记录无法完全确认、容易误用或需要后续复核的资产；不删除、不移动、不重命名，也不根据目录名猜测实验结论。

## 扫描口径

- 主工作区：`D:/workspace/python/mscad`
- SWaT 2026 工作区：`D:/workspace/python/mscad-swat2026`
- 顶层 output 目录：主工作区 117 个，SWaT 2026 工作区 9 个，共 126 个。
- checkpoint 文件：主工作区 `outputs/` 141 个、`results/` 6 个，SWaT 2026 工作区 `outputs/` 2 个，共 149 个。
- `reports/`：两个工作区均未发现该目录；现有报告主要位于 `docs/`、`outputs/*/*.md` 与 `results/`。

## U1 历史 ablation 重复命名

| 资产 | 已知情况 | 未决点 | 使用限制 |
|---|---|---|---|
| `outputs/ablation_corr` | 早期 correlation ablation | 与后续 `ablation_main_corr` 的精确运行差异未在单一 manifest 中闭合 | 不单独作为论文主结果 |
| `outputs/ablation_cov` | 早期 covariance ablation | 与后续 `ablation_main_cov` 的精确运行差异未完全闭合 | 不单独作为论文主结果 |
| `outputs/ablation_full` | 早期 full ablation | 与后续 `ablation_main_full` 的精确运行差异未完全闭合 | 不单独作为论文主结果 |

这些目录视为“可追溯到研究阶段、但无法仅凭目录名确定最终协议”的历史资产，不认定为垃圾文件。

## U2 smoke、failed 与 retry 输出

扫描到 33 个顶层目录名带 `smoke`、`failed`、`retry` 或同类标记，主要包括：

- HAI Step 3C/3D/4 的 smoke、failed、retry、retry2 运行；
- Stage 4.2B/4.2C/4.3B/4.3D/4.4/5 的 smoke 或失败运行；
- multiseed smoke、quickcheck、smoke review；
- SWaT 2026 各阶段目录中的嵌套 dry-run/smoke 资产。

它们通常可由相邻正式目录、日志或脚本判断用途，但不是正式结论来源。后续引用必须指向正式输出，不得把 smoke/failed/retry 的指标抄入论文结果表。

## U3 旧 SWaT checkpoint 的科学有效性

旧 SWaT checkpoint 多数可以按目录、配置或同目录结果文件识别实验家族，但其数据协议使用 `Swat/normal.csv` 与 `Swat/merged.csv`，已确认存在 train/test 或 validation/test overlap。因此：

- 它们不是“来源不明 checkpoint”；
- 它们是“来源大体可追溯、但不具备最终科学有效性”的历史 checkpoint；
- 必须保留，禁止作为最终 benchmark 证据。

## U4 SWaT 2026 checkpoint 的元数据层级

SWaT 2026 共发现 2 个 checkpoint 文件。checkpoint 所在的直接子目录未各自复制完整 manifest，但其父实验目录中存在 config、manifest、SHA 或阶段报告，可追溯到 M1 正式/验证运行。当前不判为孤儿 checkpoint；后续若归档，应补充统一的 checkpoint-to-run 索引，而不是移动文件。

## U5 文档引用与实际目录命名不完全一致

历史证据文档中存在少量简称或早期名称，与磁盘上的最终目录名不完全一致，例如：

- `stage4_2c_local_relation_candidate_space` 与实际候选输出目录命名；
- `stage4_3b_structure_score_comparison` 与实际 controlled score comparison 目录命名。

这些引用可以依靠上下文定位，但不应继续复制为新的规范路径。新文档以当前磁盘路径为准。

## U6 数据镜像

数据审计曾确认 `Swat/*.csv` 与 `dataset/dataset/SWat/*.csv` 存在字节一致的镜像。它们不是两套独立数据版本。本文不删除任何副本；所有实验注册必须同时记录数据路径和协议版本，避免把镜像误计为不同 benchmark。

## U7 输出物理归属与分支归属分离

HAI 与大部分 legacy 输出物理上保留在主工作区 `D:/workspace/python/mscad/outputs`，而 A12 2026 输出位于 `D:/workspace/python/mscad-swat2026/outputs`。这是一种历史形成的物理布局，不代表它们属于同一数据协议，也不应据此合并结果表。

## 结论

- 无法完整追溯的 output：3 个早期 ablation 家族需要进一步闭合其精确运行参数；33 个 smoke/failed/retry 顶层目录明确为非正式运行，不属于未知来源。
- 无法追溯的 checkpoint：未发现已证实完全孤立的 checkpoint；旧 SWaT checkpoint 的主要问题是协议无效，不是来源未知。
- 本次处理：仅建立逻辑索引，没有移动、重命名或删除任何上述资产。
