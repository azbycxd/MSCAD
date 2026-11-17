# Pre-ablation worktree review

审查日期：2026-09-23  
工作区：`D:\workspace\python\mscad-thesis`  
分支：`thesis-clean`

## Git snapshot

E3-FINAL 写入前执行：

- `git status --short`：151 个未跟踪入口。
- `git diff --stat`：空。
- `git diff --name-only`：空。
- `git diff --cached --stat`：空。
- `git diff --cached --name-only`：空。
- 未暂存 tracked 修改：0。
- 已暂存修改：0。

151 个入口按顶层路径为：`docs=55`、`scripts=52`、`tests=40`、`external=4`。E3-FINAL 新增 3 个 docs 后，预期 Git-visible 未跟踪入口为 154；`outputs/thesis_tables` 与 `outputs/thesis_registry` 受现有 ignore 规则管理，不进入 Git 状态。

## Complete A–G classification

下表使用互斥 path rules 覆盖全部 154 个 Git-visible 未提交入口。不存在未被规则覆盖的 status item。

| Class | Exact path rule / members | Count after E3-FINAL | Disposition |
|---|---|---:|---|
| A. Core method | `scripts/thesis/02_collaboration/**` | 23 | `KEEP` |
| B. Baseline | `docs/thesis/baseline_*`、`docs/thesis/efficiency_e1_final_method.md`、`scripts/thesis/01_baseline/**`、`scripts/thesis/05_efficiency/**` | 21 | `KEEP`; commit scope review required |
| C. Evaluation E2 | `docs/thesis/evaluation_e2_*`；`scripts/thesis/00_audit/evaluate_e2_r2_complete.py`、`evaluate_literature_compatible.py`、`persist_e2_r1_raw_scores.py`、`recover_e2_r1d_artifacts.py` | 8 | `KEEP` |
| D. Literature E3 | `docs/thesis/evaluation_e3_2026_literature_comparison.md`；`external/baselines/{plugin-mtsad,LatentFlow,MAD-TCN,CCM-TAD}/` | 5 | doc `KEEP`; nested repos `REQUIRES_REVIEW` before any commit |
| E. Docs / audit | 39 remaining pre-existing thesis docs + 3 E3-FINAL docs；15 remaining `scripts/thesis/00_audit/**` entries | 57 | `KEEP` |
| F. Tests | `tests/**` | 40 | `KEEP` |
| G. Temp/debug | Git-visible status items matching temp/debug rules | 0 | none |
| **Total** | all Git-visible uncommitted entries | **154** | complete coverage |

E3-FINAL 新增的三个 Class E 文档为：

- `docs/thesis/final_comparison_tables.md`
- `docs/thesis/final_experiment_findings.md`
- `docs/thesis/pre_ablation_worktree_review.md`

## Ignored/local artifacts review

这些项目不在 `git status` 中，但在文件系统检查中被发现。没有删除任何项目。

| Item | Classification | Disposition | Reason |
|---|---|---|---|
| `.pytest_cache/` | G cache | `SAFE_TO_DELETE_LATER` | pytest cache; not evidence |
| repository-wide `**/__pycache__/` | G cache | `SAFE_TO_DELETE_LATER` | generated Python bytecode cache |
| `dataset/hai/.git/lfs/tmp/` | G nested-repo temp | `REQUIRES_REVIEW` | inside nested dataset Git/LFS metadata; do not remove automatically |
| `dataset/hai/.git/lfs/logs/*.log` | G nested-repo log | `REQUIRES_REVIEW` | diagnostic log inside nested dataset repository |
| `outputs/thesis_*` | experiment artifacts | `KEEP` | frozen metrics, raw scores, manifests and evidence; ignored by design |
| `.codex_tmp` | temp | not found | no action |
| standalone debug/temp scripts | temp/debug | not found among Git-visible items | no action |

## High-risk commit boundary

`external/baselines/*` contains four nested upstream repositories. They are scientifically useful for source/protocol trace, but blindly adding their entire contents could vendor large histories, licenses, caches or generated files. They are therefore `REQUIRES_REVIEW` for commit scope, not `SAFE_TO_DELETE_LATER`.

The 151 pre-existing entries represent many completed phases that have never been committed. A future commit should be split by logical phase or explicitly approved as a consolidated research snapshot. This review does not stage, commit, push, delete, restore, clean or reset anything.

## Protocol consistency audit

- Each Dataset × Method has exactly one Historical Strict primary row in Table 1.
- Common-mask Strict keeps the frozen Validation q99 threshold and is not presented as a replacement primary result.
- Oracle, PA@q99 and PA-Oracle are isolated in the literature-compatible layer and labeled diagnostic.
- Paper-reported metrics preserve original semantics and source traces; no conversion to the local metric family was performed.
- `NOT_EXTRACTED` values remain explicit and unfilled.
- External literature has zero `DIRECT` rows.
- Ours main method, reliability gate, Self/Cross lags, Ridge alpha, Dual score and Tail calibration are frozen before ablation.

No conflicting primary result or primary protocol was found after applying these four layers. The worktree is not Git-clean because the research record is untracked, but the scientific pre-ablation state is frozen. Final status: `THESIS_PRE_ABLATION_STATE_FROZEN`.

