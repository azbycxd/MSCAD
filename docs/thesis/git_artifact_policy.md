# Thesis Git artifact policy

Date: 2026-09-23  
Status: policy draft; no file was moved, deleted, staged, committed, or pushed

## 1. Current scale and risk

Read-only inventory found:

- `dataset/`: 147 files, approximately 1.954 GB;
- `outputs/`: 512 files, approximately 2.061 GB;
- largest output: USAD latest training state, approximately 1.355 GB;
- other large model states: USAD HAI approximately 289 MB and USAD SWaT approximately 143 MB;
- cached SWaT feature arrays: approximately 101 MB and 92 MB;
- `external/baselines/`: four nested upstream repositories;
- 154 pre-ABL-P0 untracked source/document/test/external entries.

The existing `.gitignore` already excludes datasets, `outputs/`, `results/`, logs, checkpoints, common Python caches, environments, secrets, and PCAPs. It does not explicitly exclude `external/`, so the four nested repositories require deliberate handling.

## 2. Policy categories

### COMMIT

Version only small, reviewable, reproducibility-critical text/code:

- source under `models/`, `data/`, `utils/`, and reviewed thesis scripts;
- frozen configs;
- deterministic tests;
- thesis Markdown documents;
- environment/requirements files after review;
- compact tables or JSON manifests only when curated into an approved evidence path outside the blanket-ignored `outputs/` tree;
- adapter/wrapper code that is ours and contains no upstream vendored code.

Commit criteria: human-readable diff, no sensitive/local absolute credentials, no dataset rows, no model weights, no raw scores, and ordinarily below a few MB.

### GITIGNORE

Keep local and reproducible from source/manifest:

- `dataset/`, raw XLSX/CSV/GZip/ZIP data, PCAPs, and dataset adapters containing copied data;
- `outputs/`, `results/`, logs, caches, temporary files, `.pytest_cache/`, `__pycache__/`, `.codex_tmp/`;
- `*.pt`, `*.pth`, `*.ckpt`, optimizer/RNG resume states, and checkpoints;
- raw score arrays (`*.npy`, `*.npz`) and generated large score archives;
- generated figures that are duplicates or can be deterministically regenerated;
- full cloned repositories under `external/` unless deliberately represented as approved submodules.

Recommended future `.gitignore` review: add explicit `external/`, `.codex_tmp/`, and generated plot/cache patterns. This ABL-P0 task does not change `.gitignore`.

### ARCHIVE_EXTERNALLY

Preserve evidence that is too large or inappropriate for normal Git:

- final REC/LSTM-AE/USAD/GDN checkpoints and exact resume states;
- frozen raw test score packages needed to reproduce E2;
- official raw datasets and source archives;
- important output directories containing metrics, manifests, hashes, and exact run logs;
- environment export and driver/CUDA information when needed for execution reproduction.

Every external archive should have a small committed inventory containing logical ID, relative path, byte size, SHA256, creation time, producing script/commit, environment ID, and restoration instructions. An archive is not a substitute for the official upstream data license.

### REQUIRES_REVIEW

Do not decide automatically:

- small machine-generated CSV/JSON result tables currently under ignored `outputs/`;
- plots intended for the thesis versus duplicate diagnostic renders;
- locally modified third-party wrappers or patches;
- upstream repositories that might be better represented by URL+commit, lockfile, patch, or Git submodule;
- files with unexplained provenance, duplicate names, or embedded absolute machine paths;
- any artifact above ordinary source-file size even if its extension is textual.

The default decision for `REQUIRES_REVIEW` is not to stage it.

## 3. Path-specific decisions

| Path or pattern | Policy | Reason |
|---|---|---|
| `docs/thesis/*.md` | COMMIT after review | Human-readable research record |
| `scripts/thesis/**/*.py`, `scripts/thesis/**/*.mjs` | COMMIT after review | Reproducible experiment/audit logic |
| `tests/test_*.py` | COMMIT after review | Protocol invariants |
| `configs/thesis/**` | COMMIT after review | Frozen protocol inputs |
| `external/baselines/*/` | GITIGNORE; metadata COMMIT | Nested clones are upstream history, not thesis source |
| `dataset/**` | GITIGNORE + ARCHIVE/restore separately | Raw licensed or large data |
| `outputs/**/*.pt`, `*.pth`, `*.ckpt` | GITIGNORE + ARCHIVE_EXTERNALLY | Large binary states |
| `outputs/**/raw_scores/**`, `*.npy`, `*.npz` | GITIGNORE + ARCHIVE_EXTERNALLY when evidence-critical | Large derived arrays |
| small final CSV/JSON under `outputs/` | REQUIRES_REVIEW | Blanket ignored; may merit curated evidence copy |
| caches, temporary files, `.codex_tmp/` | GITIGNORE | Regenerable local state |
| duplicate/generated plots | GITIGNORE unless selected for thesis | Avoid non-source churn |

## 4. External repository rule

The current untracked external entries are:

- `external/baselines/CCM-TAD/`;
- `external/baselines/LatentFlow/`;
- `external/baselines/MAD-TCN/`;
- `external/baselines/plugin-mtsad/`.

Do not vendor or accidentally stage these directories. Commit a small audit manifest with repository URL, exact commit, branch, clean status, license, and any local patch hash. If a local compatibility patch is required, store the minimal patch or wrapper separately and keep upstream source external.

## 5. Evidence retention minimum

For every final experiment, retain outside Git:

- checkpoint/state only when required to reproduce published metrics;
- metrics and event tables;
- full manifest with data hashes and protocol;
- test score package when later evaluation layers depend on it;
- training history and timing;
- producing source commit and environment identity.

For every artifact not retained, record whether it is deterministically regenerable. No artifact is deleted under this policy draft.
