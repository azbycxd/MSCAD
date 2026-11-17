# Research commit boundary plan

Date: 2026-09-23  
Scope: planning only; no Git write operation was performed

## 1. Observed worktree state

Before ABL-P0 authoring:

- branch: `thesis-clean`;
- tracked diff: 0;
- staged diff: 0;
- untracked entries reported by `git status --short -uall`: 154;
- partition by top-level path: 58 `docs/`, 52 `scripts/`, 40 `tests/`, and 4 external cloned repositories.

The 154-entry inventory can be partitioned without overlap as follows:

| Future boundary | Current entries | Composition |
|---|---:|---|
| Group A — Core thesis method, data/protocol audits, and main results | 105 | 39 docs + 38 scripts + 28 tests |
| Group B — External baselines and efficiency | 29 | 11 docs + 10 scripts + 8 tests |
| Group C — E2 literature-compatible evaluation | 12 | 4 docs + 4 scripts + 4 tests |
| Group D — E3 literature audit and final frozen tables | 4 | 4 docs |
| Excluded from normal commits — external clones | 4 | 4 nested repositories |
| Total | 154 | exact pre-ABL-P0 inventory |

ABL-P0 adds five new docs; its CSV/JSON outputs remain under the ignored `outputs/` tree. Those new files belong to future Group E and are not part of the pre-existing 154 count.

## 2. Proposed logical commits

### Group A — Core thesis method + protocol + main results

Suggested message: `thesis: freeze core data protocols and final dual-tail method`

Include after review:

- data/source/protocol audits for SWaT, HAI, WADI, and CoMuT;
- collaboration studies and final unified method scripts under `scripts/thesis/00_audit/` and `scripts/thesis/02_collaboration/`, excluding E2-specific files;
- matching tests;
- core documents covering S6, HAI/WADI audits, Innovations 2A–2D, Innovation 3A/3B, and final SWaT/HAI/WADI method evidence.

Gate before commit:

- verify each script has a matching protocol/result document or is explicitly diagnostic;
- verify no dataset path, checkpoint, cache, or raw score archive is staged;
- run only the relevant deterministic tests; no training is implied by this plan.

### Group B — External baselines + efficiency

Suggested message: `baseline: add frozen unified baselines and efficiency audit`

Include after review:

- `docs/thesis/baseline_*` and `docs/thesis/efficiency_e1_final_method.md`;
- `scripts/thesis/01_baseline/` and `scripts/thesis/05_efficiency/`;
- baseline, normalization, protocol-audit, and efficiency tests.

Do not include the four cloned upstream repositories. Record their URL/commit/clean status in manifests and reproduce them externally.

### Group C — E2 literature-compatible evaluation

Suggested message: `eval: add frozen literature-compatible diagnostic layer`

Include after review:

- four E2 documents;
- `evaluate_literature_compatible.py`, `persist_e2_r1_raw_scores.py`, `recover_e2_r1d_artifacts.py`, and `evaluate_e2_r2_complete.py`;
- four matching E2 tests.

Gate: the strict primary tables must remain separate from Oracle/PA supplementary outputs.

### Group D — E3 2026 literature audit + final tables

Suggested message: `docs: freeze 2026 literature audit and pre-ablation tables`

Include after review:

- `evaluation_e3_2026_literature_comparison.md`;
- `final_comparison_tables.md`;
- `final_experiment_findings.md`;
- `pre_ablation_worktree_review.md`.

The machine-generated E3 CSV evidence currently lives in ignored `outputs/`. If Git versioning is required, curate only the small final CSV/JSON records into a dedicated reviewed path such as `docs/thesis/evidence/`; do not force-add the whole output tree.

### Group E — Future ablations

Suggested message after experiments are complete: `exp: add frozen thesis ablations`

Initial protocol-only files:

- `docs/thesis/ablation_p0_design.md`;
- `docs/thesis/rec_architecture_trace.md`;
- `docs/thesis/git_commit_boundary_plan.md`;
- `docs/thesis/git_artifact_policy.md`;
- `docs/thesis/paper_table_layout_draft.md`.

Later include reviewed ablation runners/tests and small human-readable result tables. Do not mix preliminary checkpoints or raw score matrices into this commit.

## 3. Recommended commit order

1. Review and commit Group A first because all later tables depend on its frozen method/data semantics.
2. Commit Group B independently so third-party baseline adaptations remain auditable.
3. Commit Group C only after confirming Oracle/PA is supplementary.
4. Commit Group D after its small evidence tables have an approved storage location.
5. Commit Group E after the ABL-P0 protocol and eventual ABL execution artifacts pass review.

No squashing across Groups A–E is recommended before review: the boundaries separate method evidence, baseline adaptations, evaluation diagnostics, literature evidence, and retrospective ablation.

## 4. Safety gate for every future commit

Run and inspect, but do not automate staging:

1. `git status --short -uall`;
2. explicit path allowlist for the intended group;
3. `git diff --cached --stat` and full `git diff --cached`;
4. check staged extensions and sizes;
5. confirm no `dataset/`, `outputs/`, checkpoint, cache, nested `.git`, secret, or raw score archive;
6. commit only after human confirmation; push only under separate explicit authorization.

Forbidden throughout: reset-hard, clean, force push, file deletion, or staging broad directory globs without inspection.
