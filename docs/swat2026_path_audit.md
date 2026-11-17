# SWaT 2026 path audit (before governance)

Scope: tracked `*.py`, `*.yaml`, `*.yml`, `*.json`, `*.md`, `*.toml`, and `*.ini` files in the `swat2026` worktree. Historical `outputs/` and checkpoint metadata were intentionally excluded from edits because they are experiment records.

| File | Line | Current path | Type | Modify? | Recommended treatment |
|---|---:|---|---|---|---|
| `configs/swat2026/rec_baseline_seed1.yaml` | 7 | `D:/workspace/python/mscad/dataset/dataset/Swat_new/11-Mar-2026_0900_1700.csv` | YAML absolute dataset path | Yes | Replace with logical A12-relative path and resolve through `MSCAD_DATA_ROOT`. |
| `configs/swat2026/independent_structure_seed1.yaml` | 5 | same A12 CSV path | YAML absolute dataset path | Yes | Same resolver treatment. |
| `configs/swat2026/structure_recovery_seed1.yaml` | 5 | same A12 CSV path | YAML absolute dataset path | Yes | Same resolver treatment. |
| `configs/swat2026/contextual_structure_seed1.yaml` | 5 | same A12 CSV path | YAML absolute dataset path | Yes | Same resolver treatment. |
| `configs/swat2026/conditional_prediction_seed1.yaml` | 5 | same A12 CSV path | YAML absolute dataset path | Yes | Same resolver treatment. |
| `scripts/swat2026/inventory_and_align_sources.py` | 24 | `D:\workspace\python\mscad` | Python absolute worktree path | Yes | Derive the compatibility Git root from the active repository root. |
| `tests/test_dataset_isolation.py` | 45 | `D:\workspace\python\mscad` | Test fixture path | Yes | Derive the sibling main worktree from the test repository root. |
| `AGENTS.md` | 12 | `D:\workspace\python\mscad-swat2026` | Documentation / worktree convention | No | Retain as an explicit local operational convention. |

The scan found no tracked historical outputs or checkpoint metadata requiring a source edit. Existing relative output and checkpoint references already describe repository-local artifacts; this change preserves their experiment semantics while centralizing their resolution.
