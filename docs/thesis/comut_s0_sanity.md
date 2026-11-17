# CoMuT-S0 Official k-Means Sanity Reproduction

Status: **COMUT_SANITY_FAILED**.

The three fixed compatibility CoMuT cases already have valid TimeEval `datasets.csv` and `test.csv` structure, so no adapter or data transformation was needed. Execution could not start because TimeEval 1.2.4's official `sopedu:5000/akita/kmeans` image failed registry resolution, while the current repository wrapper's `registry.gitlab.hpi.de/akita/i/kmeans` image returned `access forbidden`. No replacement implementation was used.

Audited protocol: unsupervised; TimeEval `Metric.ROC_AUC`; official `post_kmeans` with `ReverseWindowing`; anomaly window 40/80/160; `n_clusters=50`; `n_jobs=1`; `random_state=42`; `stride=1`; higher scores mean more anomalous, with no sign inversion.

No local AUROC was produced. Published k-Means values remain reference-only: short 0.992181, medium 0.803246, large 0.926983. This is an official execution-artifact availability failure, not evidence for or against CoMuT labels.

**COMPATIBILITY COMUT; NO OUR METHOD; NO PARAMETER SEARCH; NO THRESHOLD TUNING; NO POINT ADJUSTMENT.**

## S0R — current official GHCR image compatibility retry

Status: **COMUT_SANITY_FAILED**.

S0R replaced only the unavailable legacy image reference with the current official TimeEval image `ghcr.io/timeeval/kmeans:0.3.0` (`sha256:acd5f5f6c8a644a02fa755e0fc274d323263f2f34e43e496fc174224b03e0ec8`). The source image identifies `TimeEval/TimeEval-algorithms` and revision `701ba7dba4da86d04b1c72da0cea88a8f8715df3`. Parameters, datasets, labels, and score orientation were not changed; no sign inversion was used for the reported AUROC.

All three containers ran and produced 10,000 finite point scores for 10,000 labels. The current image performs stride-1 reverse-windowing internally in `KMeansAD._custom_reverse_windowing`; therefore applying the legacy TimeEval 1.2.4 wrapper's external `ReverseWindowing` again would produce 10,039 scores and fail metric length validation. This is an image-interface compatibility shift, not a data adaptation.

| Dataset | Local AUROC | Published AUROC | Absolute delta | Orientation |
|---|---:|---:|---:|---|
| rmj-short-mode-correlation-on-2 | 0.039159 | 0.992181 | 0.953022 | FAIL |
| rmj-medium-mode-correlation-on-2 | 0.345803 | 0.803246 | 0.457443 | FAIL |
| rmj-large-mode-correlation-on-2 | 0.863910 | 0.926983 | 0.063073 | PASS |

For audit only, negating the unchanged scores yields AUROC 0.960841, 0.654197, and 0.136090. Because the first two cases improve when negated, the frozen higher-is-more-anomalous orientation check fails. Scores were not flipped, thresholds were not searched, and the requested gate is not met.

`legacy_registry_available=false`; `registry_compatibility_adaptation=true`; `algorithm_reimplemented=false`; this is **not** an exact historical Docker image reproduction.

**OFFICIAL TIMEEVAL KMEANS; CURRENT OFFICIAL GHCR IMAGE; REGISTRY COMPATIBILITY ADAPTATION ONLY; NO SKLEARN SUBSTITUTE; NO PARAMETER CHANGE; NO DATA CHANGE; NO COMMIT; NO PUSH.**

## Final historical-pipeline conclusion

The combined S0/S0R conclusion is **HISTORICAL_PIPELINE_NOT_REPRODUCIBLE_WITH_CURRENT_IMAGE**. This conclusion concerns the unavailable historical execution artifact and changed current-image score semantics; it does **not** mean that the CoMuT compatibility dataset is invalid. Published results remain reference-only and not directly comparable to later local experiments.
