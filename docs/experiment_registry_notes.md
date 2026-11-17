# Experiment Registry Notes

生成日期：2026-07-07

本文件说明 `configs/experiment_registry.yaml` 中仍未完全确定或需要特别解释的路径。注册表只用于管理旧实验名和论文候选命名之间的映射，不会重命名目录、移动 checkpoint，也不会触发训练。

## TODO 项

| Registry key | 字段 | TODO 内容 | 原因 |
|---|---|---|---|
| `methods.score` | `train_config` | `TODO: no standalone training config` | MSCAD-Score-candidate 不是一个独立训练出的模型，而是基于 R5/R6 或 multiseed FT checkpoint 做分数融合评估，因此没有单独训练配置。 |
| `methods.joint` | `checkpoint_pattern.multiseed` | `TODO: not part of scripts/run_multiseed_swat.py unless explicitly extended` | 当前新增的多随机种子脚本只覆盖 MSCAD-Backbone、MSCAD-FT、MSCAD-Score，不覆盖 MSCAD-Joint。 |
| `methods.joint` | `output_dir_pattern.multiseed` | `TODO: no current multiseed output pattern for MSCAD-Joint` | 同上，当前没有为 MSCAD-Joint 设计多种子输出目录。 |

## 已确认映射

| Registry key | 原始实验名 | 论文候选 alias | 当前主要 checkpoint |
|---|---|---|---|
| `backbone` | `rec-only` | `MSCAD-Backbone` | `outputs/swat_improve_v3_rec_only/best.pt` |
| `ft` | `struct-finetune` | `MSCAD-FT-candidate` | `outputs/swat_improve_v4_struct_finetune/best.pt` |
| `score` | `R5/R6 fusion` | `MSCAD-Score-candidate` | `outputs/stage3_refine/R5_ws96_ss24_ls0.05_ll0.02_top15_corr0.92/best.pt` and `outputs/stage3_refine/R6_ws96_ss24_ls0.05_ll0.02_top25_corr0.88/best.pt` |
| `joint` | `MSCAD main` | `MSCAD-Joint` | `outputs/swat_main/best.pt` |

## 使用注意

1. `paper_alias` 只是论文候选命名，不代表已经可以作为最终论文表述。
2. `MSCAD-FT-candidate` 仍需要多随机种子结果支撑。
3. `MSCAD-Score-candidate` 仍需要融合权重敏感性实验支撑。
4. `MSCAD-Joint` 已有消融证据，但如果作为最终论文方法名，仍需和 Backbone/FT/Score 的定位区分清楚。
5. 注册表中 `multiseed` 路径对应 `scripts/run_multiseed_swat.py` 的默认输出设计；只有运行该脚本后这些路径才会实际存在。
