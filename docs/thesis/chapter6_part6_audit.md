# Chapter 6 Part 6 Audit

## Gate verification

| Gate | Result | Evidence |
|---|---|---|
| `THESIS_CHAPTER6_PART1_READY` | PASS | `docs/thesis/chapter6_part1_audit.md` |
| `THESIS_CHAPTER6_PART2_READY` | PASS | `docs/thesis/chapter6_part2_audit.md` |
| `THESIS_CHAPTER6_PART3_READY` | PASS | `docs/thesis/chapter6_part3_audit.md` |
| `THESIS_CHAPTER6_PART4_READY` | PASS | `docs/thesis/chapter6_part4_audit.md` |
| `THESIS_CHAPTER6_PART5_READY` | PASS | `docs/thesis/chapter6_part5_audit.md` |
| `THESIS_MULTI_SEED_STABILITY_COMPLETED` | PASS | `docs/thesis/stability_multiseed_results.md` |
| `LITERATURE_COMPATIBLE_EVAL_COMPLETED` | PASS | `docs/thesis/evaluation_e2_r2_complete_literature_compatible.md` |
| `E3_2026_LITERATURE_AUDIT_PARTIAL_FINAL` | PASS | `docs/thesis/evaluation_e3_2026_literature_comparison.md` |
| `THESIS_EXPERIMENTAL_PHASE_CLOSED` | PASS | `docs/thesis/experimental_phase_freeze.md` |

本轮仅撰写第6.7节并建立结果追踪表，没有修改第6.1—6.6节、Primary Strict主表、方法定义或冻结实验结果。

## Frozen source verification

| Frozen source | SHA256 |
|---|---|
| `outputs/thesis_stability/stability_summary.csv` | `1f0019729f23896a8661aa4708f30e983cdbb62a781633dd704ea25d30044f2c` |
| `outputs/thesis_stability/stability_event_summary.csv` | `2360f09cfac7287a1f46a476d525c461bf445b1cffc4700c5952468605bb828f` |
| `outputs/thesis_evaluation/literature_compatible/final/table_b_common_mask_strict.csv` | `3eb20a68fe0432fda50edf864d6cc44e6d0f3c4c1dec2907411c06f96e108061` |
| `outputs/thesis_evaluation/literature_compatible/final/table_c_literature_compatible.csv` | `1c1cc0da1f907cb7739f6f72794a1db747fb33cb849d602ed81327c3401c5629` |
| `outputs/thesis_evaluation/literature_compatible/final/threshold_transfer_analysis.csv` | `52f9b13c04a0e6fd0ac86b8e74d494d1fbd3a449d872176d8575d8cc0096b727` |
| `outputs/thesis_evaluation/literature_compatible/final/swat_literature_compatible.csv` | `c8174ca70f7d4ce3cbfa533c89490d408ec5756a321e0cf92059114a01c6358e` |
| `outputs/thesis_evaluation/literature_compatible/common_masks/common_mask_manifest.json` | `ec31eb602bc3d6aaf6497abaf67a3bae70405fb3269716efe8a4ce1bea162d16` |
| `outputs/thesis_evaluation/literature_2026/reported_results_2026.csv` | `e532d50a40480a32cfb4c10c30ca0a20ae585fe8f57c41316f6a9b343b5b87bb` |
| `outputs/thesis_evaluation/literature_2026/protocol_compatibility_2026.csv` | `f87d0973d21c97ec2b39869e214d0e3ae43426acc5e1a3f3796cb6053d71bee1` |
| `outputs/thesis_tables/table_4_recent_2026_literature.csv` | `fbd38af1b5fe595cfbf7dbda319e739e780adeab0c6d0119c73b9c3dbb098f78` |
| `outputs/thesis_evaluation/literature_2026/restricted_source_recovery_log.csv` | `3850ea43d25bd1feb438cd27d3a999dea1538fdc6ff7614de049086d0e47ee29` |

## Artifact verification

- 正文仅包含`6.7`及`6.7.1—6.7.4`四个二级小节：PASS。
- 表6-10：4个多随机种子结果行，均来自`stability_summary.csv`：PASS。
- 表6-11：12个common-mask文献兼容结果行，均来自冻结Table C：PASS。
- 表6-12：11个近期方法协议感知摘录行；所有数值均来自冻结E3注册表或最终近期文献表：PASS。
- `chapter6_7_result_trace.csv`数据行：128。
- 追踪字段：`subsection,result_family,dataset,method,metric,value,source_artifact,comparability,notes`：PASS。
- Spreadsheet validation：artifact-tool CSV import、recalculate、`A1:I129`区域检查与错误标记扫描均PASS；预览渲染成功。
- E3 external DIRECT comparability rows：0；未构造统一最新方法排名。

## Semantic self-audit

| Check | Requirement | Result |
|---|---|---|
| A | mean±std不替代Primary | PASS：表6-10明确为补充统计，正式主表保留正式种子 |
| B | SWaT TAIL variability可见 | PASS：正文报告AUPR std=0.151947与范围0.168898—0.448953，标记`SEED_VARIABILITY_OBSERVED` |
| C | Oracle不可部署 | PASS：明确其使用Test标签，仅作诊断 |
| D | PA不混入Strict | PASS：PA结果只出现在文献兼容表，未替代Primary q99 |
| E | Affiliation-F1不作为point F1 | PASS：保留百分数和原指标名，明确禁止换算 |
| F | 不发明MAD-TCN性能 | PASS：SWaT AUC/F1/P/R保持`NOT_EXTRACTED` |
| G | 2026 external无DIRECT行 | PASS：冻结协议注册表DIRECT计数为0 |
| H | 不建立统一最新方法排名 | PASS：只按原指标与协议标记摘录 |
| I | 所有数字来自冻结E2/E3/Stability产物 | PASS：128条数值/状态追踪均指向精确源文件与源行/字段 |

## Frozen chapter preservation

| Existing chapter artifact | SHA256 after Part 6 authoring |
|---|---|
| `chapter6_experiments_part1.md` | `4908e2ae236984029e03df8a823b1a8bf95dc5d2f80e0df6a2876262139563f0` |
| `chapter6_experiments_part2.md` | `da2943d7532073db1c445aab0ea972d5998e66890d383c8e18c6991b3bfb55aa` |
| `chapter6_experiments_part3.md` | `4502f9ded85a6c8c044168c197fbde005fc34f2f7f47fcd0cdafb2bbd26a1b0a` |
| `chapter6_experiments_part4.md` | `4d53f7d98aa70be7a5d876715aaa23a6dcf3b996e7590d0a25ded46b7a4dd213` |
| `chapter6_experiments_part5.md` | `7a19928319d478cb67c07f88f4545a942d634135496208b37496867dea84f013` |

第6.1—6.6节文件哈希与写作前一致。

## Output hashes

- `chapter6_experiments_part6.md`: `a18e6c5f1a0886be402dd9c6c312c5c5f8e291da3e30991c08ddbb1a0719390c`
- `chapter6_7_result_trace.csv`: `423b383b7335806e2b109d3e0f2cc1651d195c6c57891ecd116754a8b812ec66`

## Prohibited-action audit

- Training: NONE
- Inference: NONE
- New experiment: NONE
- Result recalculation: NONE
- Metric conversion: NONE
- New literature search: NONE
- Method change: NONE
- Claim inflation: NONE
- Git commit: NONE
- Git push: NONE

## Final status

`THESIS_CHAPTER6_PART6_READY`
