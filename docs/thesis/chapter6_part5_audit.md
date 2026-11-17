# Chapter 6 Part 5 Audit

## Gate verification

| Gate | Result | Evidence |
|---|---|---|
| `THESIS_CHAPTER6_PART1_READY` | PASS | `docs/thesis/chapter6_part1_audit.md` |
| `THESIS_CHAPTER6_PART2_READY` | PASS | `docs/thesis/chapter6_part2_audit.md` |
| `THESIS_CHAPTER6_PART3_READY` | PASS | `docs/thesis/chapter6_part3_audit.md` |
| `THESIS_CHAPTER6_PART4_READY` | PASS | `docs/thesis/chapter6_part4_audit.md` |
| `ABLATION_A1_INNOVATION2_COMPLETED` | PASS | `docs/thesis/ablation_a1_innovation2_results.md` |
| `ABLATION_A2_INNOVATION3_COMPLETED` | PASS | `docs/thesis/ablation_a2_innovation3_results.md` |
| `ABLATION_A3_INNOVATION1_COMPLETED` | PASS | `docs/thesis/ablation_a3_innovation1_final_findings.md` |
| `ABLATION_FINAL_PACKAGE_READY` | PASS | `outputs/thesis_ablation/ablation_final_summary.json` |
| `THESIS_EXPERIMENTAL_PHASE_CLOSED` | PASS | `docs/thesis/experimental_phase_freeze.md` |

本轮范围仅为第6.6节写作、图表引用与结果追踪。没有修改第6.1—6.5节，没有训练、推理、新消融、结果重算或方法变更。

## Frozen source verification

| Frozen source | SHA256 |
|---|---|
| `innovation1_cross_dataset_ablation.csv` | `c4e67f14469df91814c5d26c37e298b28b866d2dce71cc26f331ee0262728246` |
| `innovation2_real_ablation.csv` | `242f721bff61c3c39b43c885fadb47b6b5311f5711fe5072888e0ea88213dbb7` |
| `innovation2_mechanism_ablation.csv` | `1828b40164099ec8b12c0adb1e42e578e8982803cccdc643ec1ea3fa3cb24fa1` |
| `gate_stage_details.csv` | `b52cfc7b161ae2a358a7af6bed09307517d4c35c699c2b01e54fcf5e92b0555e` |
| `target_selection_summary.csv` | `c886e2f504aceb978b4a1f9309606a84851891e6eaea50a7dc746bced0cfdb79` |
| `fusion_detection_ablation.csv` | `fce442bb3fde4e200dd20c43120d9300055451e86ec3f81187d04714031f180c` |
| `normal_tail_balance.csv` | `925d624211d3c5ec60f6417a4e8b1aeb53ddbd2650bd15dd5d75c007e626114a` |
| `fusion_event_summary.csv` | `20ae5f22febc4f4c781c2a7a8aae0664332f98022b3cea923073160ac32c7e1a` |

三张冻结消融图均存在且已按相对路径插入正文：

- `fig_ablation_lightweight_rec.png`: `7b74c0b3c2e5ab20a2adc067e75b94a021b4c205c1c164eafaeb4b9a981d1951`
- `fig_ablation_dual_view.png`: `d2fc1e715e85d2cd6da39752e5aeb912be8ed03a1c19bedb74e673cced9ed4b1`
- `fig_ablation_tail_calibration.png`: `270145e46fe0680a0b00e00951ff0282ad92b79c6f10eb625b25acba519374a4`

## Artifact verification

- 正文仅包含`6.6`及`6.6.1—6.6.4`四个二级小节：PASS。
- 表6-6、表6-7、表6-8、表6-9共25个数据行逐行对照冻结CSV：25/25 PASS。
- 图6-1、图6-2、图6-3分别引用三张冻结图：3/3 PASS。
- `chapter6_6_result_trace.csv`数据行：152。
- 结果追踪字段：`subsection, ablation_id, dataset, variant, metric, value, source_csv, source_row, claim_id, notes`：PASS。
- Spreadsheet validation：artifact-tool CSV import、recalculate、`A1:J153`区域检查与错误标记扫描均PASS；未发现公式错误标记。
- R2/R0参数比沿用冻结A3包中的既有比值，并在追踪表中同时指向对应R0与R2源行；未生成新的检测指标。

## Semantic self-audit

| Check | Requirement | Result |
|---|---|---|
| A | 不声称R0或`k=3`为最优结构 | PASS：正文明确否定该表述 |
| B | 不因R1观测更好而替换Final | PASS：明确记录Test暴露后的回顾性方法学边界 |
| C | 不声称long-lag普遍更优 | PASS：HAI为正向贡献，SWaT保留反向结果，结论限定为dataset-dependent |
| D | 不把Gate写成Test selector | PASS：Gate明确为normal-only reliability constraint |
| E | 不加入不存在的F2 | PASS：只比较F0、F1、F3、F4，并明确不设置F2 |
| F | 不声称Tail普遍提升检测 | PASS：仅HAI出现正向互补，SWaT保留负向边界 |
| G | 不隐藏SWaT negative result | PASS：独立段落报告REC、DUAL、TAIL AUPR及TAIL FPR |
| H | 不把WADI升为Primary | PASS：仅作为domain-shift stress test与normal-tail balance证据 |
| I | 不把controlled与real attack直接排名 | PASS：正文和图6-2说明两种证据语义不同 |
| J | 所有结果数字来自最终CSV | PASS：四表逐行验证，正文指标均有源行追踪；滞后集合只作为冻结协议元数据 |

正文未使用“消融证明”措辞，结论使用“支持”“结果表明”和“在当前冻结实验条件下”等限定表达。

## Frozen chapter preservation

| Existing chapter artifact | SHA256 after Part 5 authoring |
|---|---|
| `chapter6_experiments_part1.md` | `4908e2ae236984029e03df8a823b1a8bf95dc5d2f80e0df6a2876262139563f0` |
| `chapter6_experiments_part2.md` | `da2943d7532073db1c445aab0ea972d5998e66890d383c8e18c6991b3bfb55aa` |
| `chapter6_experiments_part3.md` | `4502f9ded85a6c8c044168c197fbde005fc34f2f7f47fcd0cdafb2bbd26a1b0a` |
| `chapter6_experiments_part4.md` | `4d53f7d98aa70be7a5d876715aaa23a6dcf3b996e7590d0a25ded46b7a4dd213` |

第6.1—6.5节文件哈希与写作前一致。

## Output hashes

- `chapter6_experiments_part5.md`: `7a19928319d478cb67c07f88f4545a942d634135496208b37496867dea84f013`
- `chapter6_6_result_trace.csv`: `9ec69e91b25db5f75250286963079c151e5f5060f8c70c460d1eb2760ef89930`

## Prohibited-action audit

- Training: NONE
- Inference: NONE
- New ablation: NONE
- Result recalculation: NONE
- Method change: NONE
- Claim inflation: NONE
- Git commit: NONE
- Git push: NONE

## Final status

`THESIS_CHAPTER6_PART5_READY`
