# CoMuT-P0 Normal-Only Training Protocol Freeze

Status: **COMUT_NORMAL_ONLY_PROTOCOL_READY**.

## Observed files

All 13 compatibility CoMuT cases contain `test.csv`, `train_no_anomaly.csv`, and `train_anomaly.csv`. Each `train_no_anomaly.csv` has 10,000 rows and its `is_anomaly` column is entirely zero. Each `test.csv` has 10,000 rows. No common-prefix fallback is needed.

Compatibility config SHA256: `99852D184C06F0CFD0EBBF085EDCD431DBE8D25DCB82FE004645766727C82699`.

## Frozen protocol

- TRAIN: the complete, corresponding per-case `train_no_anomaly.csv`.
- TEST: the complete, corresponding per-case `test.csv`.
- Scaler and every normal reference: fit only on TRAIN.
- TEST labels: unavailable to model fitting, calibration, selection, and tuning; read only for final evaluation. P0 read them solely to inventory the frozen anomaly intervals.
- Window: 64; stride: 1, for all later models.
- Primary metrics: point-wise AUROC and point-wise AUPR.
- No point adjustment, test-best threshold, or primary F1. Any later F1 threshold must be preregistered from normal TRAIN/validation.

The fallback based on the 13-case common clean chronological prefix and a 160-point guard is not activated because independent normal training files exist. In particular, normal points scattered through the full test sequence must never be concatenated into a training set.

## Reference positioning

Published Series2Graph++ and k-Means numbers remain **REFERENCE_ONLY / NOT_DIRECTLY_COMPARABLE** because this dataset is a compatibility generation, the historical GutenTAG commit and historical TimeEval image are unavailable, and S0R confirmed changed current-image k-Means output semantics. No later local result may be described as better or worse than those published numbers.

Data hashes matched before and after the audit. No data was modified or regenerated, and no model was run.
