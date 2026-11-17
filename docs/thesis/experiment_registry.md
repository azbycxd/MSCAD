# Experiment registry

审计日期：2026-09-14。这里的“实验数”按可解释的实验族计数；同一协议的 seed、smoke、dry-run 和 failed retry 不重复提升科学证据等级。共登记 **67** 个实验族：`VALID_FINAL_CANDIDATE=1`、`VALID_DIAGNOSTIC=41`、`EXPLORATORY=5`、`INVALID_FOR_FINAL=19`、`TODO=1`。S15-01/S15-02 分别是源审计和协议裁决；S15-A0/S15-A0+/S15-A0++/V3-A 是中期单种子或诊断实验；最终 SWaT 2015 multi-seed benchmark S15-03 仍未运行。

状态只使用：`VALID_FINAL_CANDIDATE`、`VALID_DIAGNOSTIC`、`EXPLORATORY`、`INVALID_FOR_FINAL`、`TODO`。

为避免把固定字段在 61 行中机械重复，各分表标题定义该表每一行继承的 `Dataset` 与 `Dataset version`；它们与行内的 ID 一起构成每个实验的必填数据身份。其余必填字段均在行内列出。

## A. Legacy SWaT

固定字段：`Dataset=SWaT`；`Dataset version=Legacy Swat/normal.csv + Swat/merged.csv`。4.3D 已确认 `merged.csv` 以全部 normal 数据为前缀，造成 train/test 与 validation/test overlap，且 attack suffix 时间不连续。凡用该测试构造报告真实性能的实验均为 `INVALID_FOR_FINAL`。

| ID | Name | Method version | Config | Script | Output directory | Checkpoint | Purpose | Metrics | Protocol status | Scientific status | Final thesis usability |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L01 | MSCAD main joint | V1 | `configs/swat_main.yaml` | `train.py`, `evaluate.py` | `outputs/swat_main` | `best.pt` | 强 joint baseline | P/R/F1/AUPR/AUROC | overlap | historical only | `INVALID_FOR_FINAL` |
| L02 | Reconstruction only | V1 | `configs/swat_improve_v3_rec_only.yaml` | `train.py`, `evaluate.py` | `outputs/swat_improve_v3_rec_only` | `best.pt` | 重构下界 | 同上 | overlap | historical only | `INVALID_FOR_FINAL` |
| L03 | Weak structure finetune | V1 | `configs/swat_improve_v4_struct_finetune.yaml` | `train.py`, `evaluate.py` | `outputs/swat_improve_v4_struct_finetune` | `best.pt` | 两阶段弱结构 | 同上 | overlap | historical trend | `INVALID_FOR_FINAL` |
| L04 | Fusion-refined main | V1 | `configs/swat_main_fusion_refined.yaml` | `evaluate.py` | `outputs/swat_main` | `best.pt` | 固化融合参数 | 同上 | overlap/post-search | historical only | `INVALID_FOR_FINAL` |
| L05 | Stage2 grid A0–C2 | V1 | output-resolved configs | `run_stage2_grid.py` | `outputs/stage2_grid` | 9 checkpoints | 结构权重/窗口候选 | F1/AUPR/AUROC | overlap | parameter exploration | `INVALID_FOR_FINAL` |
| L06 | Stage3 R1–R6 | V1 | generated R configs | `run_stage3_refine.py` | `outputs/stage3_refine` | 6 checkpoints | 弱微调候选 | F1/AUPR/AUROC | overlap | R5/R6 historical | `INVALID_FOR_FINAL` |
| L07 | Stage4 train-light | V1 | generated L configs | `stage4_train_light.py` | `outputs/stage4_train_light` | 6 checkpoints | 冻结/解冻轻调 | F1/AUPR/AUROC | overlap | historical | `INVALID_FOR_FINAL` |
| L08 | R5/R6 fusion grid | V1 | search grid in summary | `stage4_fusion_refine.py` | `outputs/stage4_fusion_refine` | reuse R5/R6 | score fusion sensitivity | F1/AUPR/AUROC | test-driven grid + overlap | design evidence only | `INVALID_FOR_FINAL` |
| L09 | Stage5 score-only refine | V1 | summary-embedded | `stage5_score_only_refine.py` | `outputs/stage5_score_only` | reuse R5/R6 | score-only search | F1/AUPR/AUROC | overlap | historical | `INVALID_FOR_FINAL` |
| L10 | Multiseed backbone | V1 | `results/multiseed/seed_*/backbone/config.yaml` | `scripts/run_multiseed_swat.py` | `results/multiseed/seed_*/backbone` | 3 checkpoints | seed stability | mean/std metrics | overlap | stable but invalid data | `INVALID_FOR_FINAL` |
| L11 | Multiseed FT | V1 | `results/multiseed/seed_*/ft/config.yaml` | same | `results/multiseed/seed_*/ft` | 3 checkpoints | weak-FT stability | mean/std metrics | overlap | stable but invalid data | `INVALID_FOR_FINAL` |
| L12 | Multiseed score fusion | V1 | `results/multiseed/seed_*/score/config.yaml` | same | `results/multiseed/seed_*/score` | reuse FT | fusion stability | mean/std metrics | overlap | unstable/invalid data | `INVALID_FOR_FINAL` |
| L13 | Ablation-main family | V1 | output-local configs | `run_ablation.py` | `outputs/ablation_main_*` | per variant | short/long/corr/cov/threshold | F1/AUPR/AUROC | overlap | historical ablation | `INVALID_FOR_FINAL` |
| L14 | Early ablation family | V1 | unresolved predecessors | `run_ablation.py` | `outputs/ablation_*` | per variant | early module ablation | F1/AUPR/AUROC | overlap | duplicate/older family | `INVALID_FOR_FINAL` |
| L15 | AE/LSTM-AE/PCA/IF | baseline | `configs/swat_main.yaml` context | `run_baselines.py` | `outputs/baselines_main`, `outputs/paper_tables` | AE/LSTM checkpoints | comparison baseline | F1/AUPR/AUROC/cost | overlap | historical comparison | `INVALID_FOR_FINAL` |
| L16 | MTAD SWaT comparison | baseline | `configs/mtad_swat.yaml` | `run_mtad_compare.py` | `outputs/mtad_swat_compare_v2`, `v3_wf` | model checkpoints | strong baseline | F1/AUPR/AUROC | overlap | historical comparison | `INVALID_FOR_FINAL` |
| L17 | CATCH SWaT comparison | baseline | `configs/catch_swat.yaml` | `run_catch_compare.py` | `outputs/catch_swat_compare` | core/full | strong baseline | F1/AUPR/AUROC | overlap | historical comparison | `INVALID_FOR_FINAL` |
| L18 | Normal relation stability | V2 | `configs/swat_improve_v4_struct_finetune.yaml` | `scripts/stage4_2a_analyze_relation_stability.py` | `outputs/stage4_2a_relation_stability` | reuse FT | normal-only relation repeatability | corr/sign/IQR | normal-only | controlled evidence | `VALID_DIAGNOSTIC` |
| L19 | Relation validity diagnosis | V2 | same | `scripts/stage4_2b_diagnose_relation_validity.py` | `outputs/stage4_2b_relation_validity` | reuse FT | strict relation diagnosis | stability/gap | normal-only | controlled evidence | `VALID_DIAGNOSTIC` |
| L20 | Local candidate space | V2 | same | `scripts/stage4_2c_analyze_local_relation_candidates.py` | `outputs/stage4_2c_local_relation_candidates` | reuse FT | strict/diverse/expanded pools | counts/stability | normal-only | widening adds noise risk | `VALID_DIAGNOSTIC` |
| L21 | Controlled relation sensitivity | V2 | script constants | `scripts/stage4_3a_eval_candidate_relation_sensitivity.py` | `outputs/stage4_3a_candidate_relation_sensitivity` | reuse FT | block/shift/swap injections | AUROC/AUPR/coverage | controlled normal windows | mechanism evidence only | `VALID_DIAGNOSTIC` |
| L22 | Controlled score comparison | V2 | script constants | `scripts/stage4_3b_compare_structure_scores.py` | `outputs/stage4_3b_controlled_score_comparison` | reuse FT | independent vs reconstruction-coupled | score metrics/correlation | controlled | mechanism evidence only | `VALID_DIAGNOSTIC` |
| L23 | Real SWaT single score | V2 | same legacy data | `scripts/stage4_3c_eval_independent_on_real_swat.py` | `outputs/stage4_3c_real_swat_single_score` | reuse FT | real transfer attempt | point/event metrics | overlap + broken chronology | invalid real evidence | `INVALID_FOR_FINAL` |
| L24 | Legacy data audit | audit | none | `scripts/stage4_3d_audit_swat_data_pipeline.py` | `outputs/stage4_3d_swat_data_audit` | — | detect overlap/chronology | hashes/counts/gaps | read-only audit | authoritative invalidation evidence | `VALID_DIAGNOSTIC` |
| L25 | Raw source locator | audit | none | `scripts/stage4_3e1_locate_raw_swat_sources.py` | `outputs/stage4_3e1_raw_source_locator` | — | locate legal raw source | inventory | read-only | source still unresolved | `VALID_DIAGNOSTIC` |
| L26 | Correlation synthetic suite | V2 | summary-embedded | `run_correlation_suite.py` | `outputs/correlation_suite` | — | synthetic relation stress tests | Q1–Q4 metrics | synthetic/exploratory | not benchmark evidence | `EXPLORATORY` |
| L27 | Protocol repeat | V1 | summary-embedded | `stage4_protocol_repeat_summary.py` | `outputs/stage4_protocol_repeat` | reuse legacy | repeat old protocol | F1/AUPR/AUROC | overlap | cannot repair data flaw | `INVALID_FOR_FINAL` |

## B. SWaT A12 2026 official OT

固定字段：`Dataset=SWaT`；行内 `Dataset version=A12 2026 official OT`。

| ID | Name | Dataset version | Method | Config | Script | Output | Checkpoint | Purpose / metrics | Protocol status | Scientific status | Final thesis usability |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A01 | N0 compatibility audit | A12 2026 | loader audit | `configs/swat2026/data_audit.yaml` | `audit_and_smoke.py` | `outputs/swat2026_compatibility_audit` | — | schema/time/readability | read-only | source semantics initially unresolved | `VALID_DIAGNOSTIC` |
| A02 | N1 source alignment | A12 2026 | source audit | CLI | `inventory_and_align_sources.py` | `outputs/swat2026_n1_source_alignment` | — | inventory/session alignment | read-only | missing semantics identified | `VALID_DIAGNOSTIC` |
| A03 | N1R info-sheet review | A12 2026 | document audit | CLI | `extract_and_review_info_sheet.py` | `outputs/swat2026_n1r_info_sheet_review` | — | schedule/image transcription | human-reviewed | enabled N2 | `VALID_DIAGNOSTIC` |
| A04 | N2 label and split | A12 2026 | protocol construction | CLI | `build_labels_and_split.py` | `outputs/swat2026_n2_label_split` | — | labels/split/leakage audit | valid, chronological | official protocol artifact | `VALID_DIAGNOSTIC` |
| A05 | M1 rec baseline | A12 2026 | V1 rec-only | `rec_baseline_seed1.yaml` | `train_eval_rec_baseline.py` | `outputs/swat2026_batch_m1_rec_baseline` | `checkpoints/best_seed1.pt` | F1 .2638/AUPR .1906/AUROC .4386 | valid single-seed | weak baseline; regime drift | `VALID_DIAGNOSTIC` |
| A06 | M2 independent synchronous | A12 2026 | V2 | `independent_structure_seed1.yaml` | `analyze_rec_and_structure.py` | `outputs/swat2026_batch_m2_structure` | reuse M1 | relation count/controlled/real | valid | zero strict relations | `VALID_DIAGNOSTIC` |
| A07 | M2R reproducible relation | A12 2026 | V2 | `structure_recovery_seed1.yaml` | `recover_structure_relations.py` | `outputs/swat2026_batch_m2r_structure_recovery` | reuse M1 + SHA | representations/controlled | valid | narrow only; not frozen | `VALID_DIAGNOSTIC` |
| A08 | M2S lag/state conditioned | A12 2026 | V2 | `contextual_structure_seed1.yaml` | `analyze_contextual_structure.py` | `outputs/swat2026_batch_m2s_contextual_structure` | reuse M1 + SHA | lag/state relations | valid | narrow lagged only; not frozen | `VALID_DIAGNOSTIC` |
| A09 | M2P conditional prototype | A12 2026 | V3 candidate prototype | `conditional_prediction_seed1.yaml` | `analyze_conditional_prediction.py` | `outputs/swat2026_batch_m2p_conditional_prediction` | reuse M1 + SHA | cross/self residual diagnostics | exploratory valid isolation | cross not distinct from self; V3 not frozen | `EXPLORATORY` |

## C. HAI 22.04 strict protocol

固定字段：`Dataset=HAI`；`Dataset version=22.04`。

| ID | Name | Method | Config | Script | Output | Checkpoint | Purpose / metrics | Protocol status | Scientific status | Final thesis usability |
|---|---|---|---|---|---|---|---|---|---|---|
| H01 | Dataset audit | audit | script constants | `hai_step1_audit_dataset.py` | `outputs/hai_step1_dataset_audit` | — | file/schema/58 events | strict file boundaries | protocol evidence | `VALID_DIAGNOSTIC` |
| H02 | Dataloader validation | audit | script constants | `hai_step2_validate_dataloaders.py` | `outputs/hai_step2_dataloader_validation` | — | windows/splits/leakage | strict | protocol evidence | `VALID_DIAGNOSTIC` |
| H03 | Initial rec baseline seeds | V1 rec-only | `hai_rec_only_seed{1,2,3}.yaml` | `hai_step3a_train_rec_baseline.py` | `outputs/hai_step3a_rec_baseline_seed*` | 3 | point/event metrics | strict | baseline evolution | `VALID_DIAGNOSTIC` |
| H04 | Rec multiseed aggregation | V1 rec-only | same | `hai_step3b_aggregate_rec_multiseed.py` | `outputs/hai_step3b_rec_multiseed` | reuse H03 | mean/std/agreement | strict | moderate seed sensitivity | `VALID_DIAGNOSTIC` |
| H05 | Score alignment | V1 | `hai_rec_score_alignment.yaml` | `hai_step3c_evaluate_rec_score_alignment.py` | `outputs/hai_step3c_rec_score_alignment` | reuse H03 | endpoint/raw/top-k | strict | selected scoring direction | `VALID_DIAGNOSTIC` |
| H06 | Channel calibration | V1 | `hai_rec_channel_calibration.yaml` | `hai_step3d_evaluate_rec_channel_calibration.py` | `outputs/hai_step3d_rec_channel_calibration` | reuse | calibrated residuals | strict | no final upgrade | `VALID_DIAGNOSTIC` |
| H07 | Causal alarm policy | V1 | `hai_rec_causal_alarm_policy.yaml` | `hai_step3e_evaluate_rec_causal_alarm_policy.py` | `outputs/hai_step3e_rec_causal_alarm_policy` | reuse | point/event policy | strict | smoothing not retained | `VALID_DIAGNOSTIC` |
| H08 | State threshold | V1 | `hai_rec_state_conditioned_threshold.yaml` | `hai_step3f_evaluate_rec_state_conditioned_threshold.py` | `outputs/hai_step3f_rec_state_conditioned_threshold` | reuse | state threshold ablation | strict | gate failed | `VALID_DIAGNOSTIC` |
| H09 | Capacity ablation / Tiny | V1 rec-only | `hai_rec_capacity_ablation.yaml` | `hai_step3g_run_rec_capacity_ablation.py` | `outputs/hai_step3g_rec_capacity_ablation` | Tiny/Small/Medium seeds | Tiny macro AUROC mean .681241, cost | strict, three seeds | Tiny frozen real baseline | `VALID_FINAL_CANDIDATE` |
| H10 | Normal relation diagnosis | V2 | script constants | `hai_step4a_diagnose_normal_relations.py` | `outputs/hai_step4a_normal_relation_diagnosis` | — | normal Pearson stability | normal-only | nonstationarity observed | `VALID_DIAGNOSTIC` |
| H11 | V1 controlled validation | V2 zero-lag | `hai_structure_controlled_validation.yaml` | `hai_step4b_validate_independent_structure_score.py` | `outputs/hai_step4b_structure_controlled_validation` | — | controlled AUROC/AUPR | strict controlled | mechanism only | `VALID_DIAGNOSTIC` |
| H12 | Fair temporal audit | V2 | same | `hai_step4b_b_audit_fair_temporal_generalization.py` | `outputs/hai_step4b_fair_temporal_generalization_audit` | — | temporal transfer gates | strict | instability evidence | `VALID_DIAGNOSTIC` |
| H13 | Aggregation repair | V2 | same | `hai_step4b_a_repair_relation_aggregation.py` | `outputs/hai_step4b_relation_aggregation_repair` | — | max/group aggregation | strict | diagnostic repair | `VALID_DIAGNOSTIC` |
| H14 | Preserving controls | V2 | same | `hai_step4b_r_audit_preserving_controls.py` | `outputs/hai_step4b_preserving_control_audit` | — | invariance controls | strict | Pearson insufficiency | `VALID_DIAGNOSTIC` |
| H15 | V1 real HAI/complementarity | V2 zero-lag | frozen output config | `hai_step4cd_evaluate_structure_and_complementarity.py` | `outputs/hai_step4cd_structure_real_and_complementarity` | reuse Tiny | structure AUROC .5280; +14.97 FP/TP | strict, exploratory real | failed transfer | `VALID_DIAGNOSTIC` |
| H16 | Structure V2 lagged dependency | V2 | `hai_structure_v2a_lagged_dependency.yaml` | `hai_structure_v2a_validate_lagged_dependency.py` | `outputs/hai_structure_v2a_lagged_dependency` | frozen models manifest | controlled dependency | strict controlled | mechanism validated | `VALID_DIAGNOSTIC` |
| H17 | A1R cache/grouped aggregation | V2 | source config | `hai_structure_v2a1r_cache_backfill_and_grouped_aggregation.py` | corresponding output | reuse frozen artifacts | grouping/integrity | strict | duplicate-target repair | `VALID_DIAGNOSTIC` |
| H18 | A1S full-clean stability | V2 | source config | `hai_structure_v2a1s_full_clean_stability.py` | corresponding output | reuse | FPR stability | strict normal validation | worst FPR .0543 gate failed | `VALID_DIAGNOSTIC` |
| H19 | A1D drift attribution | V2 | source config | `hai_structure_v2a1d_normal_drift_attribution.py` | corresponding output | reuse | drift attribution | strict | limitation diagnosis | `VALID_DIAGNOSTIC` |
| H20 | A1M mode-conditioned reference | V2 | source config | `hai_structure_v2a1m_mode_conditioned_reference.py` | corresponding output | reuse | mode reference ablation | strict | negative ablation | `VALID_DIAGNOSTIC` |
| H21 | A1C causal adaptive reference | V2 | source config | `hai_structure_v2a1c_causal_adaptive_reference.py` | corresponding output | reuse | adaptive/absorption gates | strict | negative + safety failure | `VALID_DIAGNOSTIC` |

## D. SWaT 2015 official Physical

固定字段：`Dataset=SWaT`；行内 `Dataset version=SWaT 2015 official Physical`。

| ID | Name | Dataset version | Method | Config | Script | Output | Checkpoint | Purpose / metrics | Protocol status | Scientific status | Final thesis usability |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S15-01 | Official Physical source/schema/timeline/label audit | SWaT 2015 official | source audit (no method) | — | `scripts/thesis/00_audit/audit_swat2015_source.py` | `outputs/thesis_t2_swat2015_audit` (ignored) | — | provenance, 51-variable schema, 1 Hz timeline and attack-label alignment | `source=E. label_alignment_conflict`; `protocol=D. needs_official_semantics`; T3 blocked | package identity verified; label semantics unresolved; no training/metrics | `VALID_DIAGNOSTIC` |
| S15-02 | Label and benchmark protocol adjudication | SWaT 2015 official | protocol metadata (no model) | — | `scripts/thesis/00_audit/adjudicate_swat2015_protocol.py` | `outputs/thesis_t25_swat2015_protocol` (ignored) | — | exact point-label mapping, 36 event identities, chronology correction, hard gap boundary | label/event semantics adjudicated; next gate T3 protocol freeze | canonical metadata only; no training/scaler/threshold/results | `VALID_DIAGNOSTIC` |
| S15-A0 | Official rec seed1 | SWaT 2015 official | V1 lightweight reconstruction backbone, rec-only | `configs/thesis/swat2015_midterm_seed1.yaml` | `scripts/thesis/01_baseline/run_swat2015_midterm_a0.py` | `outputs/thesis_midterm_s1/a0_rec_baseline` (ignored) | `best.pt` (ignored) | endpoint/full-window MAE; P/R/F1/AUPR/AUROC/FPR | frozen S1 split/scaler/window; Validation-only selection; no PA | `PRELIMINARY_OFFICIAL_SWAT2015`, single seed | `VALID_DIAGNOSTIC` |
| S15-A0+ | Strong reconstruction scoring candidate diagnostic | SWaT 2015 official | frozen A0 reconstruction; Train-MAD channel calibration + fixed top-6 | `configs/thesis/swat2015_midterm_s2_a0plus.yaml` | `scripts/thesis/01_baseline/score_swat2015_midterm_a0plus.py` | `outputs/thesis_midterm_s2_a0plus` (ignored) | reuse S15-A0 epoch 13 | calibrated top-6/mean-51; P/R/F1/AUPR/AUROC/FPR | no retraining; Train-only calibration; Validation q99; identical A0 endpoints | ranking degraded; `ROBUST_TOPK_SCORING_NOT_SUPPORTED` | `VALID_DIAGNOSTIC` |
| S15-A0++ | Static POT reconstruction threshold | SWaT 2015 official | frozen A0 raw endpoint MAE + static POT | — | `scripts/thesis/01_baseline/evaluate_swat2015_midterm_a0pp_pot.py` | `outputs/thesis_midterm_s3_a0pp` (ignored) | reuse S15-A0 epoch 13 | P/R/F1/AUPR/AUROC/FPR; ranking invariant | Validation-only q95 tail; GPD loc=0; alpha=.01; no search/adaptation | POT close to/slightly worse than q99; `STATIC_POT_NOT_SUFFICIENT` | `VALID_DIAGNOSTIC` |
| V3-A | Incremental normal collaboration gate | SWaT 2015 official | V3 candidate diagnostic, not anomaly detector | `configs/thesis/swat2015_midterm_seed1.yaml` | `scripts/thesis/02_collaboration/run_swat2015_v3a_gate.py` | `outputs/thesis_midterm_s1/v3a_incremental_gate_float64` (ignored) | Ridge coefficients (ignored) | per-target self/full MAE, R2 and relative gain | Train+Validation Normal only; fixed lags/alpha; Attack unopened | 4 eligible targets gain >=10%; promising gate | `VALID_DIAGNOSTIC` |
| S15-03 | Official Physical formal benchmark | SWaT 2015 official | final method TBD | TBD in T3 | not implemented | not present | — | future primary point/event benchmark | protocol not yet frozen; not run | no performance evidence | `TODO` |

## E. Supplementary, outside the four thesis result tables

此表的 `Dataset` 同时给出数据家族与版本；若目录包含多个变体，则保持 `protocol variants`，不擅自合并为单一版本。`Config / Script` 中以文件扩展名区分配置与入口；未发现独立 config 时只登记脚本。

| ID | Name | Dataset | Method | Config / Script | Output | Checkpoint | Purpose / metrics | Protocol status | Scientific status | Final thesis usability |
|---|---|---|---|---|---|---|---|---|---|---|
| X01 | Semi-synthetic mechanism | synthetic | V1/V2 | `run_semisynth.py` | `outputs/semisynth` | — | relation mechanism | exploratory | auxiliary only | `EXPLORATORY` |
| X02 | SMAP family | SMAP variants | V1/baselines | `configs/smap.yaml`, MTAD/CATCH/stage6 scripts | `outputs/*smap*`, `outputs/stage6_*` | multiple | cross-dataset boundary | protocol variants | not main evidence | `EXPLORATORY` |
| X03 | C-MAPSS/N-CMAPSS family | FD001/N-CMAPSS | V1/baselines | `configs/cmapss*.yaml`, `run_cmapss*.py` | `outputs/cmapss_*` | multiple | degradation-task boundary | task differs from anomaly hypothesis | supplementary only | `EXPLORATORY` |

## Counting rule

- The 67 rows above are experiment families used in the final counts.
- Physical scan found 117 main-worktree output directories and 9 A12-worktree output directories.
- Seed children, parameter-grid children, smoke/dry-run and failed retry directories remain physically present and are indexed through their family row or `unresolved_artifacts.md`; they are not counted as separate scientific conclusions.
