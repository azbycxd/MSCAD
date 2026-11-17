# Thesis environment audit

Audit date: 2026-09-14 (Asia/Shanghai)

This document distinguishes the historical environment that produced or checked earlier artifacts from the proposed environment for future thesis work. No package was installed, removed, or updated during this audit.

## Historical environment

| Item | Audited value |
|---|---|
| Conda environment | `p310` |
| Interpreter | `D:/soft/anaconda3/envs/p310/python.exe` |
| Python | `3.10.19` (Anaconda build, MSC v.1929, 64-bit) |
| Operating system | Windows 10 `10.0.19045 SP0` |
| PyTorch | `2.12.1+cu126` |
| PyTorch CUDA runtime | `12.6` |
| CUDA visible to PyTorch | Yes |
| GPU | NVIDIA GeForce RTX 4060 |
| Compute capability | `8.9` |
| NVIDIA driver | `565.90` |
| Driver-advertised CUDA compatibility | `12.7` |
| NumPy | `2.2.5` |
| Pandas | `2.3.3` |
| SciPy | `1.15.3` |
| scikit-learn | `1.7.2` |
| PyYAML | `6.0.3` |
| pytest | `9.0.2` |

### Compatibility verification

The following imports succeeded in `p310`: `torch`, `numpy`, `pandas`, `sklearn`, `scipy`, `yaml`, `pytest`, `data.transforms`, `models.backbone_ae`, `losses.recon`, `utils.config_io`, `utils.metrics`, `utils.project_paths`, and `utils.seed`.

A non-training smoke check constructed `BackboneAE`, evaluated a random CPU tensor with shape `[2,16,5]`, and obtained finite output with the same shape. CUDA visibility was queried but no optimization step, checkpoint write, dataset read, or formal metric computation was performed.

Conclusion: the historical environment is still usable for the project's core imports, tests, tensor operations, and lightweight model construction. It is not a clean or complete dependency specification for the whole inherited repository.

### Known risks

1. `p310` mixes Conda and pip packages and contains many unrelated Agent, Web, database, GUI, and language-model packages.
2. `pip freeze` contains machine-specific `file:///C:/...` build references, so it is not portable and must not be used as the thesis requirements file.
3. The installed PyTorch build is `2.12.1+cu126`, but neither Conda metadata nor pip freeze records the trusted wheel index from which that exact local-version build can be recreated.
4. The NVIDIA driver advertises CUDA 12.7 while PyTorch embeds CUDA 12.6. This combination currently runs, but driver availability remains an external machine prerequisite.
5. Some inherited optional scripts import packages absent from `p310`: `psutil`, `h5py`, `python-docx`, `pypdf`, and `openpyxl`. They are not required for the Phase T1 core smoke test.
6. Historical artifacts were produced over multiple project phases; successful imports today do not retroactively prove every old experiment used an identical environment.

## Proposed thesis environment

| Item | Proposal |
|---|---|
| Conda name | `mscad-thesis` |
| Python | `3.10.19` |
| Platform | Windows x86-64 |
| GPU target | RTX 4060, compute capability 8.9 |
| Core numeric stack | Match the currently verified direct versions in `requirements.txt` |
| PyTorch candidate | `2.12.1+cu126`, matching the working historical runtime |

`environment.yml` owns the Conda environment and Python/pip bootstrap. `requirements.txt` owns direct Python packages. Optional plotting, profiling, document parsing, and legacy-dataset packages are intentionally excluded from the T1 core environment.

## Creation decision

The `mscad-thesis` Conda environment was **not created** in Phase T1. The exact trusted Windows wheel source and continued availability of `torch==2.12.1+cu126` have not been independently confirmed. The files therefore record a reproducible candidate, not an assertion that a fresh solve has already succeeded. The next environment gate must confirm that wheel source before any installation command is authorized.
