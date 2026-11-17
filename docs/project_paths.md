# Project paths

## Repository worktrees

`main` and `swat2026` are separate worktrees. Code resolves repository resources from the active worktree root, not from the current shell directory or a developer-specific absolute path. SWaT runs use the `swat2026` worktree; this policy does not modify `main`.

## Source code

Repository-relative paths such as `configs/swat2026/rec_baseline_seed1.yaml` resolve from the worktree root. The shared resolver is `utils/project_paths.py`.

## External datasets

Datasets are external resources and are never added to Git. `MSCAD_DATA_ROOT` is the default root for relative dataset paths. An explicit absolute dataset path takes priority over the environment variable. If neither is supplied, the resolver uses one documented compatibility root derived from the sibling legacy worktree; it never scans disks.

### SWaT classic

Use a layout such as `swat/classic/physical` below `MSCAD_DATA_ROOT`. File names are intentionally not assumed.

### SWaT A12 2026

The future layout is `swat/a12_2026/ot` below `MSCAD_DATA_ROOT`. Existing batch configs retain the logical relative path `Swat_new/11-Mar-2026_0900_1700.csv` so that the current local A12 layout remains compatible. For that layout, set `MSCAD_DATA_ROOT` to its parent directory, or rely on the documented compatibility fallback.

### HAI

HAI data and configuration remain isolated. This resolver does not load, label, transform, or otherwise modify any dataset.

## Outputs

Relative output paths resolve to `<active worktree>/outputs/...`. New SWaT runs therefore default to the `swat2026` worktree and do not write to the main worktree. Outputs are Git-ignored.

## Checkpoints

Checkpoint paths may be absolute or repository-relative. Set `MSCAD_CHECKPOINT_ROOT` when relative checkpoint references should instead be rooted at an external checkpoint store. Existing SHA-256 validation remains in the experiment code; the resolver does not copy, move, or modify checkpoints.

## Environment variables

- `MSCAD_DATA_ROOT`: root for relative external dataset paths.
- `MSCAD_CHECKPOINT_ROOT`: optional root for relative checkpoint paths.

## Examples

```powershell
$env:MSCAD_DATA_ROOT="D:\workspace\python\mscad-data"
$env:MSCAD_CHECKPOINT_ROOT="D:\workspace\python\mscad-checkpoints"
D:\soft\anaconda3\envs\p310\python.exe scripts\swat2026\train_eval_rec_baseline.py --config configs\swat2026\rec_baseline_seed1.yaml --dry-run
```

Git does not record developer-specific absolute paths, datasets, outputs, or checkpoints.
