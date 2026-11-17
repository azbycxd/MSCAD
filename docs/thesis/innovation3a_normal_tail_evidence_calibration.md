# Innovation-3A Normal-Tail Evidence Calibration Audit

This audit used only Normal Train and Normal Validation from Official SWaT 2015, HAI 21.03, and WADI.A2. Frozen unified target files matched their registered SHA256 values. No Attack/Test file, header, hash, feature, or label was read.

Each continuous Validation segment/file was split chronologically: front 50% Calibration-A and back 50% Audit-B. Calibration-A alone defined branch empirical survival functions and both fusion thresholds.

| Dataset | REC E>=2 | REC q95 | REC q99 | DUAL E>=2 | DUAL q95 | DUAL q99 | q99 balance | Tail FPR | Old FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SWaT | 0.012088 | 1.570155 | 2.041121 | 0.009036 | 1.511918 | 1.953434 | 1.045 | 0.010997 | 0.009076 |
| HAI21 | 0.012165 | 1.292779 | 2.094877 | 0.012361 | 1.347108 | 2.094291 | 1.000 | 0.013458 | 0.012361 |
| WADI.A2 | 0.025395 | 1.553254 | 2.235511 | 0.014637 | 1.300255 | 2.159295 | 1.035 | 0.016220 | 0.014561 |

Gates: A=True, B=True, C=True (3/3 datasets at q99 balance <=2). Final status: **NORMAL_TAIL_EVIDENCE_CALIBRATION_FEASIBLE**.

Empirical tail calibration tests cross-branch Normal-tail comparability only. It does not establish improved anomaly detection. For WADI it does not adapt Normal Train/Validation to future Test domain shift.

No parameter search, fusion-weight search, Test tuning, or point adjustment was performed.
