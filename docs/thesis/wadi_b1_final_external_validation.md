# WADI-B1-Final External Validation

The 47 B0 reliable targets were loaded directly and matched SHA256 `9c6cb7dc0810d22fa13f6ce0f22858fb536db0bd0d113069c85d5920efdb1f7c` before Test. All models and thresholds used frozen Normal Train/Validation only. WADI Test was then read once without an Attack13 patch.

REC parameters: 21,378; best epoch: 8. Common valid Test points: 172,738/172,801 (99.963542%).

| Method | AUROC | AUPR | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|
| REC | 0.508364 | 0.054459 | 0.059742 | 0.583943 | 0.108395 | 0.563360 |
| DUAL | 0.482409 | 0.051955 | 0.042984 | 0.409642 | 0.077805 | 0.559065 |
| FUSION | 0.506113 | 0.054201 | 0.049828 | 0.478501 | 0.090257 | 0.559323 |

| HIGH ID | Rec hit | Dual hit | Fusion hit | Dual top target/process |
|---:|---:|---:|---:|---|
| 2 | 0.000000 | 0.064298 | 0.062606 | 1_FIT_001_PV / 1 |
| 3 | 0.192993 | 0.041930 | 0.041930 | 2_LT_002_PV / 2 |
| 4 | 0.192993 | 0.041930 | 0.041930 | 2_LT_002_PV / 2 |
| 7 | 0.997139 | 0.005722 | 0.994278 | 1_AIT_005_PV / 1 |
| 15 | 1.000000 | 1.000000 | 1.000000 | 2B_AIT_003_PV / 2B |

Event hit counts at 0.10: {'REC': 11, 'DUAL': 7, 'FUSION': 8}. Innovation-2: `{'H1': False, 'H2': False, 'H3': False, 'H4': False}` -> **DUAL_VIEW_WADI_EXTERNAL_NOT_SUPPORTED**. Fusion: `{'F1': False, 'F2': True, 'F3': False, 'F4': False}` -> **NORMAL_CALIBRATED_MAX_FUSION_EXTERNAL_NOT_SUPPORTED**.

HAI remains preliminary old-gate evidence; WADI is unified-gate external evidence, so the cross-dataset table is not a strict final fair comparison. No Test-driven change, target change, search, point adjustment, fusion weighting, or Attack13 patch was performed.
