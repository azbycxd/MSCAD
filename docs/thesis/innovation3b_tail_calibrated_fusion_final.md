# Innovation-3B Final Tail-Calibrated Fusion

Empirical survival functions used Calibration-A only. The Tail-Fusion threshold used Threshold-B q99 only. All three frozen Unified target files matched their registered SHA256 before Test inference.

| Dataset | Method | AUROC | AUPR | Precision | Recall | F1 | FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| SWaT | REC | 0.834887 | 0.743353 | 0.150140 | 0.917870 | 0.258067 | 0.718133 |
| SWaT | DUAL | 0.675631 | 0.167510 | 0.087166 | 0.008605 | 0.015663 | 0.012455 |
| SWaT | OLD_FUSION | 0.681368 | 0.168979 | 0.087182 | 0.008605 | 0.015664 | 0.012453 |
| SWaT | TAIL_FUSION | 0.708375 | 0.206747 | 0.152452 | 0.901228 | 0.260789 | 0.692531 |
| HAI21 | REC | 0.852899 | 0.536294 | 0.459109 | 0.601095 | 0.520594 | 0.016133 |
| HAI21 | DUAL | 0.916334 | 0.570193 | 0.554350 | 0.679446 | 0.610556 | 0.012443 |
| HAI21 | OLD_FUSION | 0.907834 | 0.576716 | 0.559176 | 0.694423 | 0.619503 | 0.012471 |
| HAI21 | TAIL_FUSION | 0.901850 | 0.617897 | 0.555963 | 0.693976 | 0.617350 | 0.012627 |
| WADI.A2 | REC | 0.508364 | 0.054459 | 0.059742 | 0.583943 | 0.108395 | 0.563360 |
| WADI.A2 | DUAL | 0.482409 | 0.051955 | 0.042984 | 0.409642 | 0.077805 | 0.559065 |
| WADI.A2 | OLD_FUSION | 0.506113 | 0.054201 | 0.049828 | 0.478501 | 0.090257 | 0.559323 |
| WADI.A2 | TAIL_FUSION | 0.507639 | 0.054510 | 0.057556 | 0.559788 | 0.104380 | 0.561873 |

Event hit counts at 0.10: SWaT `{'REC': 29, 'DUAL': 2, 'OLD_FUSION': 2, 'TAIL_FUSION': 28}`, HAI21 `{'REC': 45, 'DUAL': 44, 'OLD_FUSION': 46, 'TAIL_FUSION': 45}`, WADI.A2 `{'REC': 11, 'DUAL': 7, 'OLD_FUSION': 8, 'TAIL_FUSION': 9}`. HAI HIGH19 medians: `{'REC': 0.6501901140684411, 'DUAL': 0.8410852713178295, 'OLD_FUSION': 0.8410852713178295, 'TAIL_FUSION': 0.8388625592417062}`.

Pre-registered gates `{'A': True, 'B': True, 'C': False, 'D': False, 'E': False}` yield **TAIL_FUSION_COMPLEMENTARY_BUT_SWAT_UNSTABLE**. WADI remains **DOMAIN_SHIFT_STRESS_TEST** and is excluded from the Innovation-3 success gate.

No target/source/lag/alpha change, AE retraining, ECDF variant, fusion weighting, Test threshold search, point adjustment, Test-Normal adaptation, or domain adaptation was performed.
