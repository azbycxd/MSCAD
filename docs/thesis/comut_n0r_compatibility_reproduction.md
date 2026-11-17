# CoMuT N0R compatibility reproduction

No author-generated CoMuT artifact was found in the official repository history, branches, tags, or LFS. The original config is preserved at SHA256 `D9F6EB...175B`.

A compatibility copy removes exactly 13 `random-mode-jump.amplitude` keys; no other structural key changes. This is schema adaptation, not parameter tuning. Because the historical generator commit was unavailable, this is not a bitwise historical reproduction.

## Audit

- Status: `COMUT_COMPAT_BENCHMARK_READY`
- Generated cases: 13/13
- Rows: 10,000 per case
- Channels: 2, 3, or 4
- Label: `is_anomaly` with values 0/1
- NaN/Inf: 0/0
- Config vs generated: PASS

## Relation sanity

- large: `rmj-large-mode-correlation-on-2` normal correlation=0.077966, anomaly correlation=-0.331093, |corr| increased.
- medium: `rmj-medium-mode-correlation-on-2` normal correlation=-0.032218, anomaly correlation=0.032101, |corr| decreased.
- short: `rmj-short-mode-correlation-on-2` normal correlation=-0.043748, anomaly correlation=0.168654, |corr| increased.

CoMuT is a dedicated synthetic correlation-anomaly benchmark, not the primary real industrial anomaly benchmark. Published repository AUROC values are REFERENCE_ONLY and are explicitly marked `locally_reproduced=false`.

NO MODEL TRAINING; NO PARAMETER SEARCH; SEED 42.
