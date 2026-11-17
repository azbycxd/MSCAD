# Experiment Inventory

生成日期：2026-07-07  
扫描范围：`configs/`、`outputs/`、`docs/` 中与 SWaT 主实验、消融、融合搜索、baseline 和 R5/R6 结构微调相关的文件。  
约束说明：本次只整理清单，不修改模型、训练、评估代码。

## 1. Checkpoint 清单

| 方法/阶段 | Checkpoint 路径 | 用途 | 是否可复用 | 备注 |
|---|---|---|---|---|
| MSCAD-main | `outputs/swat_main/best.pt` | SWaT 主实验 MSCAD main checkpoint | 可复用 | 对应 `configs/swat_main.yaml`，从头训练，结构权重较强 |
| rec-only | `outputs/swat_improve_v3_rec_only/best.pt` | 纯重构/rec-only 版本 checkpoint | 可复用 | 可作为后续结构微调或分数分析的基础模型 |
| struct-finetune | `outputs/swat_improve_v4_struct_finetune/best.pt` | rec-only 后弱结构约束微调 checkpoint | 可复用 | 对应 `configs/swat_improve_v4_struct_finetune.yaml` |
| A0 backbone / rec-only grid | `outputs/stage2_grid/A0_backbone_wo_struct/best.pt` | stage3 R1-R6 结构微调的初始化 checkpoint | 可复用 | `outputs/stage3_refine/stage3_refine_summary.json` 中记录为 `init_ckpt` |
| R1 refine | `outputs/stage3_refine/R1_ws96_ss24_ls0.05_ll0.02_top20_corr0.9/best.pt` | 结构微调候选 R1 checkpoint | 可复用 | R1: window 96, short 24, top20, corr 0.9 |
| R2 refine | `outputs/stage3_refine/R2_ws96_ss24_ls0.05_ll0.03_top20_corr0.9/best.pt` | 结构微调候选 R2 checkpoint | 可复用 | R2: long loss 更高 |
| R3 refine | `outputs/stage3_refine/R3_ws80_ss20_ls0.05_ll0.02_top20_corr0.9/best.pt` | 结构微调候选 R3 checkpoint | 可复用 | R3: window 80, short 20 |
| R4 refine | `outputs/stage3_refine/R4_ws64_ss16_ls0.04_ll0.02_top20_corr0.9/best.pt` | 结构微调候选 R4 checkpoint | 可复用 | R4: window 64, short 16 |
| R5 refine | `outputs/stage3_refine/R5_ws96_ss24_ls0.05_ll0.02_top15_corr0.92/best.pt` | R5 结构微调 checkpoint | 可复用，优先 | stage3 中 best_by_f1；stage4 融合搜索主候选 |
| R6 refine | `outputs/stage3_refine/R6_ws96_ss24_ls0.05_ll0.02_top25_corr0.88/best.pt` | R6 结构微调 checkpoint | 可复用，优先 | stage3 中 best_by_aupr/best_by_auroc；stage4 融合搜索主候选 |
| ablation full | `outputs/ablation_main_full/best.pt` | 消融 full: rec + short + long checkpoint | 可复用 | 对应 ablation full 结果 |
| ablation w/o short | `outputs/ablation_main_wo_short/best.pt` | 去除 short 结构项 checkpoint | 可复用 | 用于结构项消融 |
| ablation w/o long | `outputs/ablation_main_wo_long/best.pt` | 去除 long 结构项 checkpoint | 可复用 | 用于结构项消融 |
| ablation w/o struct | `outputs/ablation_main_wo_struct/best.pt` | 去除结构项 checkpoint | 可复用 | 与 rec-only 思路接近，但属于 ablation_main 流程 |
| ablation corr | `outputs/ablation_main_corr/best.pt` | corr 结构度量 checkpoint | 可复用 | 与 full/corr 对应 |
| ablation cov | `outputs/ablation_main_cov/best.pt` | cov 结构度量 checkpoint | 可复用 | 结构度量消融 |
| ablation shrink_cov | `outputs/ablation_main_shrink_cov/best.pt` | shrink_cov 结构度量 checkpoint | 可复用 | 结构度量消融 |
| AE baseline | `outputs/baselines_main/ae_best.pt` | AE baseline checkpoint | 可复用 | 用于 SWaT 主表 baseline |
| LSTM-AE baseline | `outputs/baselines_main/lstm_ae_best.pt` | LSTM-AE baseline checkpoint | 可复用 | 用于 SWaT 主表 baseline |
| MTAD-W | `outputs/mtad_swat_compare_v3_wf/w/best.pt` | MTAD-W checkpoint | 可复用 | SWaT MTAD 对比 |
| MTAD-FULL | `outputs/mtad_swat_compare_v3_wf/full/best.pt` | MTAD-FULL checkpoint | 可复用 | SWaT MTAD 对比和融合搜索 |
| CATCH-CORE | `outputs/catch_swat_compare/core/best.pt` | CATCH core checkpoint | 可复用 | SWaT CATCH 对比 |
| CATCH-FULL | `outputs/catch_swat_compare/full/best.pt` | CATCH full checkpoint | 可复用 | SWaT CATCH 对比和融合搜索 |
| PCA baseline | 无 `.pt` checkpoint | 传统 PCA baseline | 不适用 | 结果在 summary/table 中保存 |
| IF baseline | 无 `.pt` checkpoint | Isolation Forest baseline | 不适用 | 结果在 summary/table 中保存 |

## 2. 结果文件清单

### 2.1 主实验与论文表格

| 方法/内容 | 结果文件路径 | 用途 | 是否可复用 | 备注 |
|---|---|---|---|---|
| 主实验汇总表 | `outputs/paper_tables/main_comparison.md` | 论文主对比表 Markdown | 可复用，优先 | 包含 MSCAD main、AE、LSTM-AE、IF、PCA |
| 主实验汇总表 CSV | `outputs/paper_tables/main_comparison1.csv` | 论文主对比表 CSV | 可复用 | 便于后续制表 |
| MSCAD-main train | `outputs/swat_main/train_metrics.json` | MSCAD main 训练指标 | 可复用 | 训练过程/最佳 epoch 等 |
| MSCAD-main eval | `outputs/swat_main/eval_metrics.json` | MSCAD main 测试指标 | 可复用 | 主实验结果来源之一 |
| rec-only train | `outputs/swat_improve_v3_rec_only/train_metrics.json` | rec-only 训练指标 | 可复用 | 纯重构版本 |
| rec-only eval | `outputs/swat_improve_v3_rec_only/eval_metrics.json` | rec-only 测试指标 | 可复用 | 与 struct-finetune 对照 |
| struct-finetune train | `outputs/swat_improve_v4_struct_finetune/train_metrics.json` | 弱结构微调训练指标 | 可复用 | 结构微调过程 |
| struct-finetune eval | `outputs/swat_improve_v4_struct_finetune/eval_metrics.json` | 弱结构微调测试指标 | 可复用 | AUPR/AUROC 论证重点 |

### 2.2 消融实验

| 方法/内容 | 结果文件路径 | 用途 | 是否可复用 | 备注 |
|---|---|---|---|---|
| 消融总汇总 | `outputs/ablation_main_summary.json` | ablation_main 全部变体汇总 | 可复用，优先 | 包含 metrics/profile/train_score_stats |
| 消融表 CSV | `outputs/paper_tables/ablation_table1.csv` | 论文消融表 CSV | 可复用 | 可直接用于论文表格 |
| full train | `outputs/ablation_main_full/train_metrics.json` | full: rec + short + long 训练指标 | 可复用 | 结构项完整版本 |
| full eval | `outputs/ablation_main_full/eval_metrics.json` | full: rec + short + long 测试指标 | 可复用 | 消融基准 |
| w/o short train | `outputs/ablation_main_wo_short/train_metrics.json` | 去 short 训练指标 | 可复用 | 短期结构项消融 |
| w/o short eval | `outputs/ablation_main_wo_short/eval_metrics.json` | 去 short 测试指标 | 可复用 | 短期结构项消融 |
| w/o long train | `outputs/ablation_main_wo_long/train_metrics.json` | 去 long 训练指标 | 可复用 | 长期结构项消融 |
| w/o long eval | `outputs/ablation_main_wo_long/eval_metrics.json` | 去 long 测试指标 | 可复用 | 长期结构项消融 |
| w/o struct train | `outputs/ablation_main_wo_struct/train_metrics.json` | 去结构项训练指标 | 可复用 | 结构项总体消融 |
| w/o struct eval | `outputs/ablation_main_wo_struct/eval_metrics.json` | 去结构项测试指标 | 可复用 | 与 full 对比重点 |
| corr eval | `outputs/ablation_main_corr/eval_metrics.json` | corr 结构度量结果 | 可复用 | 结构度量消融 |
| cov eval | `outputs/ablation_main_cov/eval_metrics.json` | cov 结构度量结果 | 可复用 | 结构度量消融 |
| shrink_cov eval | `outputs/ablation_main_shrink_cov/eval_metrics.json` | shrink_cov 结构度量结果 | 可复用 | 结构度量消融 |
| mean3std eval | `outputs/ablation_main_mean3std/eval_metrics.json` | mean+3std 阈值结果 | 可复用 | 阈值策略消融 |

### 2.3 R5/R6 结构微调与融合搜索

| 方法/内容 | 结果文件路径 | 用途 | 是否可复用 | 备注 |
|---|---|---|---|---|
| stage3 结构微调汇总 | `outputs/stage3_refine/stage3_refine_summary.json` | R1-R6 结构微调汇总 | 可复用，优先 | R5 为 best_by_f1，R6 为 best_by_aupr/best_by_auroc |
| R5 train | `outputs/stage3_refine/R5_ws96_ss24_ls0.05_ll0.02_top15_corr0.92/train_metrics.json` | R5 训练指标 | 可复用 | stage3 结构微调候选 |
| R5 eval | `outputs/stage3_refine/R5_ws96_ss24_ls0.05_ll0.02_top15_corr0.92/eval_metrics.json` | R5 测试指标 | 可复用 | stage3 best_by_f1 |
| R6 train | `outputs/stage3_refine/R6_ws96_ss24_ls0.05_ll0.02_top25_corr0.88/train_metrics.json` | R6 训练指标 | 可复用 | stage3 结构微调候选 |
| R6 eval | `outputs/stage3_refine/R6_ws96_ss24_ls0.05_ll0.02_top25_corr0.88/eval_metrics.json` | R6 测试指标 | 可复用 | stage3 best_by_aupr/best_by_auroc |
| stage4 融合搜索汇总 | `outputs/stage4_fusion_refine/stage4_fusion_refine_summary.json` | R5/R6 的 alpha/beta/gamma、smooth、POT 网格搜索结果 | 可复用，优先 | 包含 432 组组合和 best_overall_by_f1 |
| stage5 score-only 汇总 | `outputs/stage5_score_only/stage5_score_only_summary.json` | 仅分数层面 refine/搜索记录 | 可复用 | 可作为融合策略补充材料 |
| stage4 train light 汇总 | `outputs/stage4_train_light/stage4_train_light_summary.json` | 轻量结构微调候选汇总 | 可复用 | 与 R 系列不同批次，可作补充 |
| stage4 protocol repeat 汇总 | `outputs/stage4_protocol_repeat/stage4_protocol_repeat_summary.json` | 协议重复/复核记录 | 可复用 | 可用于检查协议一致性 |

### 2.4 MTAD 结果

| 方法/内容 | 结果文件路径 | 用途 | 是否可复用 | 备注 |
|---|---|---|---|---|
| MTAD SWaT 汇总 | `outputs/mtad_swat_compare_v3_wf/mtad_compare_summary.json` | SWaT 上 MTAD-W/MTAD-FULL/MTAD-FULL+FUSION 汇总 | 可复用，优先 | 当前主表中强 baseline 来源之一 |
| MTAD-W train | `outputs/mtad_swat_compare_v3_wf/w/train_metrics.json` | MTAD-W 训练指标 | 可复用 | SWaT |
| MTAD-W eval | `outputs/mtad_swat_compare_v3_wf/w/eval_metrics.json` | MTAD-W 测试指标 | 可复用 | SWaT |
| MTAD-FULL eval | `outputs/mtad_swat_compare_v3_wf/full/eval_metrics.json` | MTAD-FULL 测试指标 | 可复用 | SWaT |
| 旧版 MTAD 汇总 | `outputs/mtad_swat_compare_v2/mtad_compare_summary.json` | 旧版 SWaT MTAD 对比 | 谨慎复用 | 建议优先用 v3_wf |

### 2.5 CATCH 结果

| 方法/内容 | 结果文件路径 | 用途 | 是否可复用 | 备注 |
|---|---|---|---|---|
| CATCH SWaT 汇总 | `outputs/catch_swat_compare/catch_compare_summary.json` | SWaT 上 CATCH-CORE/CATCH-FULL/融合搜索汇总 | 可复用，优先 | 当前主表中强 baseline 来源之一 |
| CATCH-CORE eval | `outputs/catch_swat_compare/core/eval_metrics.json` | CATCH-CORE 测试指标 | 可复用 | SWaT |
| CATCH-FULL eval | `outputs/catch_swat_compare/full/eval_metrics.json` | CATCH-FULL 测试指标 | 可复用 | SWaT |

### 2.6 AE、LSTM-AE、PCA、IF 结果

| 方法/内容 | 结果文件路径 | 用途 | 是否可复用 | 备注 |
|---|---|---|---|---|
| baseline 汇总 | `outputs/baselines_main/baselines_summary.json` | AE、LSTM-AE、PCA、IF baseline 汇总 | 可复用 | 注意该文件可能对应另一批/较早 baseline 结果，论文主表建议优先对照 `paper_tables/main_comparison.md` |
| AE checkpoint | `outputs/baselines_main/ae_best.pt` | AE baseline 模型 | 可复用 | 深度 baseline |
| LSTM-AE checkpoint | `outputs/baselines_main/lstm_ae_best.pt` | LSTM-AE baseline 模型 | 可复用 | 深度 baseline |
| 主表 baseline | `outputs/paper_tables/main_comparison.md` | AE、LSTM-AE、PCA、IF 主表结果 | 可复用，优先 | 当前最适合论文引用 |
| 主表 baseline CSV | `outputs/paper_tables/main_comparison1.csv` | AE、LSTM-AE、PCA、IF 主表 CSV | 可复用 | 便于制表 |
| PCA checkpoint | 无 | PCA 传统方法无模型 checkpoint | 不适用 | 结果仅保存为表格/summary |
| IF checkpoint | 无 | Isolation Forest 无 `.pt` checkpoint | 不适用 | 结果仅保存为表格/summary |

### 2.7 跨数据集与补充结果

| 方法/内容 | 结果文件路径 | 用途 | 是否可复用 | 备注 |
|---|---|---|---|---|
| SMAP cross-dataset | `outputs/stage6_cross_dataset/cross_dataset_compare_summary.json` | SMAP 配置下 A0/R5 对比 | 谨慎复用 | 当前更适合作为适用边界分析 |
| SMAP multichannel 汇总 | `outputs/stage6_smap_multichannel/smap_multichannel_summary.json` | SMAP 多通道实验汇总 | 谨慎复用 | 需要确认通道和协议 |
| SMAP 5ch 汇总 | `outputs/stage6_smap_multichannel/smap_multichannel_summary_5ch.json` | SMAP 5 通道实验汇总 | 谨慎复用 | 补充材料 |
| SMAP 12ch 汇总 | `outputs/stage6_smap_12ch/smap_multichannel_summary.json` | SMAP 12 通道实验汇总 | 谨慎复用 | 补充材料 |
| C-MAPSS FD001 汇总 | `outputs/cmapss_fd001_compare/cmapss_compare_summary.json` | C-MAPSS FD001 对比 | 谨慎复用 | 退化检测边界分析 |
| C-MAPSS k20 汇总 | `outputs/cmapss_fd001_compare_k20/cmapss_compare_summary.json` | last-k=20 对比 | 谨慎复用 | 补充敏感性 |
| C-MAPSS k30 汇总 | `outputs/cmapss_fd001_compare_k30/cmapss_compare_summary.json` | last-k=30 对比 | 谨慎复用 | 补充敏感性 |
| C-MAPSS k50 汇总 | `outputs/cmapss_fd001_compare_k50/cmapss_compare_summary.json` | last-k=50 对比 | 谨慎复用 | 补充敏感性 |
| N-CMAPSS 汇总 | `outputs/cmapss_compare_ncmapss/cmapss_compare_summary.json` | N-CMAPSS 相关对比 | 谨慎复用 | 需确认协议 |

## 3. 配置文件清单

### 3.1 SWaT 主实验配置

| 方法/内容 | 配置文件路径 | 用途 | 是否可复用 | 备注 |
|---|---|---|---|---|
| MSCAD-main | `configs/swat_main.yaml` | SWaT paper_main 主实验配置 | 可复用，优先 | 输出目录 `outputs/swat_main` |
| fusion refined main | `configs/swat_main_fusion_refined.yaml` | 固化 stage4 最佳融合/阈值参数的主实验配置 | 可复用 | eval: alpha 0.6, beta 0.15, gamma 0.2, smooth 3, pot_q 0.93 |
| base SWaT | `configs/swat.yaml` | 较早/基础 SWaT 配置 | 可复用但需核对 | 可能不是最终 paper_main 协议 |
| tuned SWaT | `outputs/swat_default/swat_tuned.yaml` | 调参后输出配置 | 谨慎复用 | 属于 swat_default 流程 |

### 3.2 rec-only 与 struct-finetune 配置

| 方法/内容 | 配置文件路径 | 用途 | 是否可复用 | 备注 |
|---|---|---|---|---|
| rec-only | `configs/swat_improve_v3_rec_only.yaml` | 纯重构训练配置 | 可复用，优先 | 对应 `outputs/swat_improve_v3_rec_only` |
| struct-finetune | `configs/swat_improve_v4_struct_finetune.yaml` | rec-only 后弱结构微调配置 | 可复用，优先 | 对应 `outputs/swat_improve_v4_struct_finetune` |
| improve v1 | `configs/swat_improve_v1.yaml` | 早期改进配置 | 谨慎复用 | 历史实验 |
| improve v2 | `configs/swat_improve_v2.yaml` | 早期改进配置 | 谨慎复用 | 历史实验 |

### 3.3 消融配置

| 方法/内容 | 配置文件路径 | 用途 | 是否可复用 | 备注 |
|---|---|---|---|---|
| fast ablation | `configs/swat_ablation_fast.yaml` | 快速消融入口配置 | 可复用但需核对 | 可能不是 ablation_main 最终逐项配置 |
| full config | `outputs/ablation_main_full/config.yaml` | full 消融复现实验配置 | 可复用 | 与对应输出目录绑定 |
| w/o short config | `outputs/ablation_main_wo_short/config.yaml` | w/o short 消融配置 | 可复用 | 与对应输出目录绑定 |
| w/o long config | `outputs/ablation_main_wo_long/config.yaml` | w/o long 消融配置 | 可复用 | 与对应输出目录绑定 |
| w/o struct config | `outputs/ablation_main_wo_struct/config.yaml` | w/o struct 消融配置 | 可复用 | 与对应输出目录绑定 |
| corr config | `outputs/ablation_main_corr/config.yaml` | corr 消融配置 | 可复用 | 与对应输出目录绑定 |
| cov config | `outputs/ablation_main_cov/config.yaml` | cov 消融配置 | 可复用 | 与对应输出目录绑定 |
| shrink_cov config | `outputs/ablation_main_shrink_cov/config.yaml` | shrink_cov 消融配置 | 可复用 | 与对应输出目录绑定 |
| mean3std config | `outputs/ablation_main_mean3std/config.yaml` | mean+3std 阈值消融配置 | 可复用 | 与对应输出目录绑定 |

### 3.4 R5/R6 与融合搜索配置

| 方法/内容 | 配置文件路径 | 用途 | 是否可复用 | 备注 |
|---|---|---|---|---|
| R5 refine config | `outputs/stage3_refine/R5_ws96_ss24_ls0.05_ll0.02_top15_corr0.92_config.yaml` | R5 结构微调配置 | 可复用，优先 | stage3 best_by_f1 |
| R6 refine config | `outputs/stage3_refine/R6_ws96_ss24_ls0.05_ll0.02_top25_corr0.88_config.yaml` | R6 结构微调配置 | 可复用，优先 | stage3 best_by_aupr/best_by_auroc |
| R1 refine config | `outputs/stage3_refine/R1_ws96_ss24_ls0.05_ll0.02_top20_corr0.9_config.yaml` | R1 结构微调配置 | 可复用 | 候选 |
| R2 refine config | `outputs/stage3_refine/R2_ws96_ss24_ls0.05_ll0.03_top20_corr0.9_config.yaml` | R2 结构微调配置 | 可复用 | 候选 |
| R3 refine config | `outputs/stage3_refine/R3_ws80_ss20_ls0.05_ll0.02_top20_corr0.9_config.yaml` | R3 结构微调配置 | 可复用 | 候选 |
| R4 refine config | `outputs/stage3_refine/R4_ws64_ss16_ls0.04_ll0.02_top20_corr0.9_config.yaml` | R4 结构微调配置 | 可复用 | 候选 |
| 融合搜索网格 | `outputs/stage4_fusion_refine/stage4_fusion_refine_summary.json` | 记录 R5/R6 融合搜索网格和最优参数 | 可复用，优先 | 不是 YAML 配置，而是搜索结果 summary |
| 固化融合配置 | `configs/swat_main_fusion_refined.yaml` | 将 stage4 最优融合参数写入主配置 | 可复用 | 适合后续复跑/报告 |

### 3.5 Baseline 配置

| 方法/内容 | 配置文件路径 | 用途 | 是否可复用 | 备注 |
|---|---|---|---|---|
| MTAD SWaT | `configs/mtad_swat.yaml` | MTAD baseline 原始 SWaT 配置 | 可复用 | 当前 v3_wf 输出另有拷贝配置 |
| MTAD-W v3_wf | `outputs/mtad_swat_compare_v3_wf/mtad_w_config.yaml` | MTAD-W 运行配置 | 可复用，优先 | 与当前 SWaT MTAD 结果绑定 |
| MTAD-FULL v3_wf | `outputs/mtad_swat_compare_v3_wf/mtad_full_config.yaml` | MTAD-FULL 运行配置 | 可复用，优先 | 与当前 SWaT MTAD 结果绑定 |
| CATCH SWaT | `configs/catch_swat.yaml` | CATCH baseline 原始 SWaT 配置 | 可复用 | 当前输出另有拷贝配置 |
| CATCH core | `outputs/catch_swat_compare/catch_core_config.yaml` | CATCH core 运行配置 | 可复用，优先 | 与当前 CATCH 结果绑定 |
| CATCH full | `outputs/catch_swat_compare/catch_full_config.yaml` | CATCH full 运行配置 | 可复用，优先 | 与当前 CATCH 结果绑定 |

## 4. 推荐复用顺序

| 目标 | 推荐优先复用文件 |
|---|---|
| 论文主表 | `outputs/paper_tables/main_comparison.md`、`outputs/paper_tables/main_comparison1.csv` |
| MSCAD-main 指标 | `outputs/swat_main/eval_metrics.json`、`outputs/swat_main/best.pt`、`configs/swat_main.yaml` |
| rec-only 指标/模型 | `outputs/swat_improve_v3_rec_only/eval_metrics.json`、`outputs/swat_improve_v3_rec_only/best.pt`、`configs/swat_improve_v3_rec_only.yaml` |
| struct-finetune 指标/模型 | `outputs/swat_improve_v4_struct_finetune/eval_metrics.json`、`outputs/swat_improve_v4_struct_finetune/best.pt`、`configs/swat_improve_v4_struct_finetune.yaml` |
| R5/R6 结构微调 | `outputs/stage3_refine/stage3_refine_summary.json`、R5/R6 子目录下 `best.pt` 和 `*_config.yaml` |
| R5/R6 融合搜索 | `outputs/stage4_fusion_refine/stage4_fusion_refine_summary.json`、`configs/swat_main_fusion_refined.yaml` |
| 消融实验 | `outputs/ablation_main_summary.json`、`outputs/paper_tables/ablation_table1.csv` |
| MTAD baseline | `outputs/mtad_swat_compare_v3_wf/mtad_compare_summary.json` |
| CATCH baseline | `outputs/catch_swat_compare/catch_compare_summary.json` |
| AE/LSTM-AE/PCA/IF baseline | `outputs/paper_tables/main_comparison.md`，必要时参考 `outputs/baselines_main/baselines_summary.json` |

## 5. 注意事项

1. `outputs/stage4_fusion_refine/stage4_fusion_refine_summary.json` 是融合搜索结果，不是训练配置文件；其中包含 R5/R6 checkpoint 路径、网格范围和最佳组合。
2. `configs/swat_main_fusion_refined.yaml` 是把 stage4 最佳融合参数固化后的配置，可用于后续报告或复跑，但其输出目录仍指向 `outputs/swat_main`。
3. PCA 和 Isolation Forest 没有 `.pt` checkpoint；它们的可复用材料主要是结果表和 summary。
4. `outputs/baselines_main/baselines_summary.json` 与 `outputs/paper_tables/main_comparison.md` 中的 baseline 数值可能来自不同批次或协议，论文主表建议优先使用 `outputs/paper_tables/main_comparison.md`。
5. 跨数据集结果目前建议作为补充或适用边界分析，不建议优先作为主结论证据。
