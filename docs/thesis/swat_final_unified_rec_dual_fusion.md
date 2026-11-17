# Official SWaT Final Unified Evaluation

The unified reliability gate retained 15/25 candidate targets. The target manifest SHA256 is `af9f9aac9cb8b369015a94d4b88a4729d817c219d305eb3276050c5d33ad31bf` and was written before Attack feature access. The Official A0 checkpoint and its frozen REC scores were reused after architecture, dimensions, Train/Validation protocol, scaler, checkpoint hash, epoch, and parameter-count checks passed.

| Method | AUROC | AUPR | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|
| REC | 0.834887 | 0.743353 | 0.150140 | 0.917870 | 0.258067 | 0.718133 |
| DUAL | 0.675631 | 0.167510 | 0.087166 | 0.008605 | 0.015663 | 0.012455 |
| FUSION | 0.681368 | 0.168979 | 0.087182 | 0.008605 | 0.015664 | 0.012453 |

Canonical 36-event hits at rate >=0.10: REC 29, DUAL 2, FUSION 2.

| Event | REC hit | DUAL hit | FUSION hit | DUAL top target/stage |
|---:|---:|---:|---:|---|
| 13 | 0.000000 | 0.000000 | 0.000000 | FIT601 / P6 |
| 14 | 0.023202 | 0.000000 | 0.000000 | FIT601 / P6 |
| 29 | 1.000000 | 0.000000 | 0.000000 | AIT202 / P2 |

Fusion gates `{'F1': False, 'F2': False, 'F3': False, 'F4': True}` yield **NORMAL_CALIBRATED_MAX_FUSION_SWAT_NOT_SUPPORTED**. Normal-domain-shift summary: `{'median_ratio': 2.509313113795096, 'q95_ratio': 12.827940464247012, 'max_ratio': 14.264853099943098, 'worst_target': 'AIT504'}`. Innovation-2 remains frozen; this result describes its real-benchmark applicability boundary and did not alter targets, sources, lags, alpha, thresholds, or fusion.

No point adjustment, POT, Test-best threshold, Test-driven selection, target Top-K, source/lag/alpha search, fusion-weight search, commit, or push was performed.
