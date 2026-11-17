# SWaT 2026 path audit (after governance)

The follow-up scan covers the same tracked text-file scope. The A12 configuration references are logical relative paths; shared code now resolves data, outputs, and checkpoints through `utils/project_paths.py`.

| Remaining file | Line | Path text | Classification | Why retained |
|---|---:|---|---|---|
| `AGENTS.md` | 12 | `D:\workspace\python\mscad-swat2026` | Local documentation convention | It states the approved SWaT worktree for this machine; it is not runtime source code. |
| `docs/project_paths.md` | examples | `D:\workspace\python\mscad-data` and `D:\workspace\python\mscad-checkpoints` | Documented local examples | They illustrate environment-variable use and are explicitly non-portable examples. |
| `docs/swat2026_path_audit.md` | table | pre-governance paths | Audit evidence | This file preserves the before-state required for traceability. |

No tracked runtime Python source or SWaT 2026 configuration contains a developer-specific absolute dataset or worktree path. Historical outputs, checkpoints, and their metadata were not edited.
