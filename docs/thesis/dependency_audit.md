# Dependency audit

Audit date: 2026-09-14. Scope: Python files tracked by Git at the frozen thesis baseline. Imports were inventoried from source; the historical `pip freeze` was used only to verify versions, not to select packages.

## CORE

| Distribution | Import | Evidence | T1 pin | Reason |
|---|---|---:|---|---|
| PyTorch | `torch` | 49 tracked files | `2.12.1+cu126` candidate | tensors, datasets, model layers and reconstruction backbone |
| NumPy | `numpy` | 109 tracked files | `2.2.5` | arrays, scaling, windowing and metrics |
| Pandas | `pandas` | 81 tracked files | `2.3.3` | tabular sensor/time metadata |
| SciPy | `scipy` | 11 tracked files | `1.15.3` | statistical diagnostics and signal utilities |
| scikit-learn | `sklearn` | 22 tracked files | `1.7.2` | baseline estimators, robust/conditional diagnostic models and metrics support |
| PyYAML | `yaml` | 13 tracked files | `6.0.3` | YAML configuration and manifests |

These versions are pinned because they are installed together in the verified Windows `p310` environment and passed the existing core tests. This is a compatibility choice, not an upgrade recommendation.

## TEST_ONLY

| Distribution | Import | Evidence | T1 pin | Reason |
|---|---|---:|---|---|
| pytest | `pytest` | 20 tracked test files | `9.0.2` | repository and environment verification only |

pytest is essential to the reproducibility gate but is not a runtime dependency of the detector.

## OPTIONAL

| Distribution | Evidence | Intended scope | T1 decision |
|---|---|---|---|
| Matplotlib | imported by five plotting/training-report scripts | figures and diagnostic plots | exclude from minimal environment until a formal reporting script is selected |
| psutil | optional profiling in three locations | memory/efficiency reporting | exclude; absent from historical `p310` |
| joblib | three HAI V2 diagnostic scripts; also a scikit-learn dependency | serialized diagnostic Ridge/PCA/KMeans assets | no direct pin; not needed for T1 smoke |
| openpyxl | indirect backend for `pandas.read_excel` | spreadsheet metadata inspection | exclude; absent from `p310` |
| python-docx | optional import in A12 source inventory | DOCX metadata inspection | exclude; absent from `p310` |
| pypdf | optional import in A12 source inventory | PDF metadata inspection | exclude; absent from `p310` |

Optional dependencies may be added later only when an authorized formal workflow proves it needs them.

## LEGACY_ONLY

| Distribution | Evidence | Reason excluded |
|---|---|---|
| h5py | `prepare_ncmapss.py` | only the inherited N-CMAPSS preparation path; not part of the thesis T1 environment |

Legacy SWaT scripts, old grid/fusion pipelines, and historical report-generation tools may depend on additional packages. They remain evidence code and do not define the clean thesis environment.

## UNUSED_OR_UNCLEAR

The historical environment contains many packages with no direct import in tracked project code, including `chromadb`, `langchain*`, `langgraph*`, `openai*`, `onnxruntime`, `kubernetes`, `mcp`, `uvicorn`, and related Web/Agent infrastructure. They are excluded. Transitive dependencies such as `filelock`, `networkx`, `sympy`, `jinja2`, and `fsspec` are left to the selected PyTorch wheel rather than declared as project-level dependencies.

## Environment-file responsibility

- `environment.yml`: Conda environment name, Python version, and pip bootstrap.
- `requirements.txt`: direct Python dependencies and the proposed PyTorch CUDA wheel channel.
- Neither file contains developer-specific paths, local build URLs, datasets, checkpoints, Jupyter tooling, or Agent tooling.
