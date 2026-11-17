# Result registry

本注册表按数据版本分表，禁止横向拼接不同数据版本的指标。结果状态仅使用 `VALID`、`INVALID`、`DIAGNOSTIC`、`TODO`。所有 Legacy SWaT 性能均注明 **historical only, not usable as final benchmark evidence**。

## A. Legacy SWaT (`normal.csv` / `merged.csv`)

4.3D 内容审计结论：`merged.csv = 完整 normal 前缀 + attack-only 后缀`；normal 前缀与 train/validation 直接重叠，attack 后缀有 34 个非一秒相邻时间间隔。下面的性能值只帮助理解项目历史，不能写入最终 benchmark 主表。

| Result family | Representative artifact | Recorded result | Status | Allowed use |
|---|---|---|---|---|
| MSCAD main | `outputs/swat_main/eval_metrics.json` | F1 .7604, AUPR .6698, AUROC .8786 | `INVALID` | historical architecture evidence only |
| Rec-only | `outputs/swat_improve_v3_rec_only/eval_metrics.json` | F1 .7997, AUPR .8181, AUROC .9515 | `INVALID` | historical training-strategy trend only |
| Weak struct-finetune | corresponding eval | F1 .7898, AUPR .8235, AUROC .9644 | `INVALID` | historical design turn only |
| R5/R6 fusion | `outputs/stage4_fusion_refine/stage4_fusion_refine_summary.json` | R5 F1 .7997; R6 AUPR .8385/AUROC .9611 | `INVALID` | fusion-sensitivity history only |
| Multiseed backbone/FT/score | `results/multiseed/multiseed_summary.csv` | reproducible mean/std recorded | `INVALID` | seed stability cannot repair invalid data |
| Ablations | `outputs/ablation_main_summary.json` | short/long/struct/corr/cov/threshold comparisons | `INVALID` | historical ablation narrative only |
| AE/LSTM/PCA/IF | `outputs/paper_tables/main_comparison1.csv` | legacy comparison | `INVALID` | do not rank methods formally |
| MTAD/CATCH SWaT | their comparison summaries | legacy comparison | `INVALID` | no final ranking |
| Independent relation real attempt | `outputs/stage4_3c_real_swat_single_score` | point/event results | `INVALID` | chronology invalidates real/event claims |
| Normal-only and controlled relation tests | `outputs/stage4_2a_*` through `stage4_3b_*` | stability/injection diagnostics | `DIAGNOSTIC` | controlled mechanism questions only; no real benchmark claim |
| Data audit itself | `outputs/stage4_3d_swat_data_audit/final_data_audit.md` | exact prefix=True; overlap; chronology gaps | `VALID` | authoritative reason to invalidate rows above |

## B. SWaT A12 2026 official OT

数据：A12 官方 2026-03-11 OT 会话；N1R 人工审核 attack schedule；N2 固定 chronological split。M1–M2S 均保持 labels test-only after freeze、SHA 校验和不重训约束。

| Result | Artifact | Key evidence | Status | Thesis use |
|---|---|---|---|---|
| N0 compatibility | `outputs/swat2026_compatibility_audit` | schema/time/source audit | `DIAGNOSTIC` | data provenance |
| N1/N1R | source alignment + info review outputs | official materials and human review trail | `VALID` | data/protocol section |
| N2 | `outputs/swat2026_n2_label_split` | fixed labels, split, validity mask, leakage audit | `VALID` | protocol definition |
| M1 reconstruction | `baseline_metrics.json` | Precision .18784, Recall .44263, F1 .26375, AUPR .19064, AUROC .43861; point adjustment=false | `DIAGNOSTIC` | valid negative baseline; not a strong final result |
| M2 synchronous independent relation | `batch_m2_readiness.json` | `insufficient_strict_relations`, strict count=0 | `DIAGNOSTIC` | shows simple synchronous relation is insufficient |
| M2R relation recovery | `batch_m2r_readiness.json` | only narrow relation diagnostic; primary not frozen | `DIAGNOSTIC` | negative/recovery evidence |
| M2S lag/state context | `batch_m2s_readiness.json` | only narrow lagged relation; primary not frozen | `DIAGNOSTIC` | motivates richer conditional modelling |
| M2P prototype | `batch_m2p_readiness.json` | cross score not distinct from self history; primary not frozen | `DIAGNOSTIC` | negative prototype evidence only; V3 remains proposed |

## C. HAI 22.04

HAI 使用 6 个训练序列、4 个测试序列、86 通道、58 攻击事件；保持文件边界，window=64，causal endpoint scoring。该数据版本是当前严格真实评估证据来源。

| Result | Artifact | Key evidence | Status | Thesis use |
|---|---|---|---|---|
| Dataset/loader protocol | HAI Step1/Step2 outputs | no cross-file windows; audited split | `VALID` | protocol foundation |
| Initial rec multiseed | `outputs/hai_step3b_rec_multiseed/multiseed_summary.json` | moderate seed sensitivity; q99 macro AUROC mean .69606 | `DIAGNOSTIC` | baseline evolution |
| Tiny reconstruction | `outputs/hai_step3g_rec_capacity_ablation` | frozen Tiny DSConv AE; 16,420 params; macro AUROC mean .681241 | `VALID` | current final real-HAI baseline, with limitations |
| Score/channel/policy/state/capacity ablations | HAI Step3C–3G outputs | endpoint raw top5 retained; other repairs not promoted | `DIAGNOSTIC` | design/negative ablation |
| Structure V1 controlled | `outputs/hai_step4b_structure_controlled_validation` | controlled relation corruption detectable | `DIAGNOSTIC` | controlled mechanism only |
| Structure V1 real | `structure_real_evaluation_summary.json` | AUROC .528026; F1 .07136; +14.9716 FP per added TP; 2 unique events | `DIAGNOSTIC` | failed real transfer / limitation |
| Structure V2 controlled | `outputs/hai_structure_v2a_lagged_dependency` | 12 directed lagged dependencies; controlled R1/R2/R3 valid | `DIAGNOSTIC` | controlled collaboration mechanism |
| A1S normal stability | `structure_v2a1s_summary.json` | overall FPR .009118; worst sequence .054300; gate failed | `DIAGNOSTIC` | explicit normal nonstationarity limitation |
| A1M mode reference | `structure_v2a1m_summary.json` | no reduction of train1 second-half drift | `DIAGNOSTIC` | negative ablation |
| A1C adaptive reference | A1C summary | train1 second-half FPR .113206; absorption gate failed | `DIAGNOSTIC` | negative/safety ablation |
| Rec + Structure V2 fusion | final freeze manifest | not executed because structure stability gate failed | `TODO` | must not report metrics without new authorization/recomputation |

## D. SWaT 2015 official Physical

| Result | Artifact | Key evidence | Status | Thesis use |
|---|---|---|---|---|
| S15-01 Official Physical source audit | `docs/thesis/swat2015_source_audit.md`; ignored machine outputs under `outputs/thesis_t2_swat2015_audit` | package identity verified; 51 variables and 1 Hz source confirmed; one 82 s gap; Physical/attack-list label conflicts remain | `DIAGNOSTIC` | provenance evidence only; formal benchmark remains TODO/not run and no performance metric may be stated |
| S15-02 Label/protocol adjudication | `docs/thesis/swat2015_protocol_adjudication.md`; ignored canonical metadata under `outputs/thesis_t25_swat2015_protocol` | Physical exact point mapping frozen; 41 raw records/36 events retained; IDs 37–41 chronology corrected in canonical metadata; gap is hard boundary | `DIAGNOSTIC` | protocol-definition evidence only; contains no model result |
| S15-A0 official rec seed1 | `outputs/thesis_midterm_s1/a0_rec_baseline/metrics.json` (ignored); `docs/thesis/midterm_sprint_s1_results.md` | endpoint q99: P .150140, R .917870, F1 .258067, AUPR .743353, AUROC .834887, FPR .718133; 11,730 params; no PA | `DIAGNOSTIC` | `PRELIMINARY_OFFICIAL_SWAT2015`; credible single-seed baseline, not final evidence |
| S15-A0+ calibrated top-6 | `outputs/thesis_midterm_s2_a0plus/metrics.json` (ignored); `docs/thesis/midterm_sprint_s2_a0plus.md` | fixed top-6: P .359431, R .787865, F1 .493654, AUPR .256131, AUROC .768496, FPR .194078; no retraining/PA | `DIAGNOSTIC` | `ROBUST_TOPK_SCORING_NOT_SUPPORTED`; operating point improved but ranking degraded; preliminary only |
| S15-A0++ static POT | `outputs/thesis_midterm_s3_a0pp/metrics.json` (ignored); `docs/thesis/midterm_sprint_s3_a0pp.md` | unchanged A0 score; POT threshold .168923: P .148442, R .920269, F1 .255647, FPR .729703; AUPR/AUROC invariant | `DIAGNOSTIC` | `STATIC_POT_NOT_SUFFICIENT`; static EVT does not repair threshold transfer; preliminary only |
| V3-A incremental normal gate | `outputs/thesis_midterm_s1/v3a_incremental_gate_float64/summary.json` (ignored); `docs/thesis/midterm_sprint_s1_results.md` | 37 eligible; gain >=5%/10%/20% counts 6/4/3; status `PROMISING_INCREMENTAL_CROSS_CHANNEL_SIGNAL`; Attack unopened | `DIAGNOSTIC` | supports later V3 research only; no anomaly-detection claim |
| S15-03 Official Physical formal benchmark | not present | T3 protocol, training and evaluation have not run | `TODO` | intended primary benchmark; no performance metric may be stated |

## Cross-table rule

- Do not compare Legacy SWaT numbers directly with A12, HAI or future SWaT 2015 numbers.
- Do not promote controlled-injection AUROC/AUPR to real-attack performance.
- `VALID` means provenance/protocol is credible, not necessarily high performance.
- Negative results remain useful when their protocol is valid and their claim is bounded.
