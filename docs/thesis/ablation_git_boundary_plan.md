# Ablation Git Boundary Plan

## 目的与边界

本文件只给出版本控制规划，没有执行 `git add`、commit、push、删除、移动或清理。仓库现有策略对 `outputs/` 采用 blanket ignore，因此输出目录中的小型结果表与论文图仍需人工决定是否复制到经批准的证据路径，或以强制跟踪方式纳入。

逐文件建议及字节大小见 `outputs/thesis_ablation/ablation_git_inventory.csv`。建议动作含义如下：

- `COMMIT`：适合常规代码评审的人类可读源码、测试与论文文档。
- `REVIEW`：内容可能有论文价值，但位于 ignored output 路径或属于生成图，需要人工确认仓库策略。
- `ARCHIVE_EXTERNALLY`：应保留以支持追溯，但模型状态、原始分数矩阵或大型二进制不适合常规 Git。
- `GITIGNORE`：可由源码确定性再生的临时或重复输出。

## 建议纳入 Git 的候选

1. `scripts/thesis/03_ablation/*.py` 与必要的 CSV 构建脚本：消融执行或汇总逻辑。
2. `tests/test_ablation*.py`：冻结协议、来源绑定和无泄漏门禁。
3. `docs/thesis/ablation*.md`、本文件、最终论文措辞与 LaTeX 表格：人类可读研究记录。
4. 小型最终 summary CSV 与 manifest：仅在人工确认 ignored-output 例外或迁移到批准证据路径后纳入。
5. 三组最终 PNG/PDF：仅在确认论文 figure 属于版本化交付物后纳入；否则由脚本再生。

## 不建议进入 Git

- candidate-level score matrices、raw Test/Validation scores、large NPZ；
- model checkpoint、optimizer state、RNG/resume state；
- Ridge full-state dump、large calibration array；
- ABL1 的大型运行证据与其他 GB 级 artifact；
- 临时 debug、cache、重复 plot payload。

上述证据应保留在外部归档，并用小型清单记录相对路径、SHA256、大小、生成脚本、环境标识和恢复说明。本规划不删除任何旧产物。

## 推荐的人工执行顺序

1. 审阅 inventory 中全部 `REVIEW` 项，确定最终 summary CSV 与论文 figure 的版本控制路径。
2. 对 `ARCHIVE_EXTERNALLY` 项建立带校验和的外部归档清单。
3. 人工确认 `.gitignore` 建议，不在本轮直接修改。
4. 仅对确认后的源码、测试、文档和小型证据执行精确路径 `git add`；提交前再次检查 staged diff，防止带入 datasets、checkpoints、external clones 或大型 outputs。

