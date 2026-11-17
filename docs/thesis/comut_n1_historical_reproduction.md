# CoMuT-N1 Historical Official CoMuT Reproduction

Status: **COMUT_HISTORICAL_SEMANTICS_RECOVERED**.

## Provenance

- Generator: historical official `timeeval-gutenTAG==1.0.0`.
- Repository: `https://github.com/TimeEval/GutenTAG.git`.
- Commit: `5638dadf9b1e83699ca317ce9eb4569a6c350064`, dated 2022-05-30; clone remained clean.
- Configuration: unmodified `generation_configs/comut-datasets.yaml` from that commit.
- Configuration SHA256: `4D90D3871BFBB6C088F0AE4A3EAFE279D365FDBE4CACB4F85397908826570717`.
- Frozen random-mode-jump settings: frequency 250.0, variance 0.05, random-seed 42.
- Output: `dataset/CoMuT_historical_v1`; the earlier `CoMuT_compat` dataset was preserved.

The exact official configuration snapshot is stored in the N1 output and has the same SHA256. No configuration key, schema field, label, or generated value was modified.

## Generation and basic gates

All 13/13 cases generated successfully. Every case contains a 10,000-row `test.csv` and 10,000-row `train_no_anomaly.csv`; labels are binary, and all feature values contain zero NaN and zero Inf. Dataset hashes were unchanged during the audit. The generated dataset and audit outputs are Git ignored.

## Normal collaboration semantics

Across every channel pair in all 13 normal training files, median absolute Pearson correlation is **0.997508**, with range **0.997437–0.997582**. Every case individually exceeds the preregistered 0.80 gate.

Jump indicators were defined independently for each channel as `abs(delta) > median(abs(delta)) + 6 × max(1.4826 × MAD, 1e-12)`. Median pairwise jump-timing Jaccard agreement is **0.991304**, and normal signed jump agreement is 1.0 in every case. This confirms that channels share the same random-mode-jump sequence.

## Representative anomaly semantics

| Case | Normal signed Pearson | Anomaly signed Pearson | Absolute relation shift |
|---|---:|---:|---:|
| short-on-2 | 0.997522 | 0.061439 | 0.936083 |
| medium-on-2 | 0.997451 | 0.251982 | 0.745469 |
| large-on-2 | 0.997490 | -0.996530 | 1.994020 |

All three representative anomaly intervals clearly deviate from their normal relation reference. The short and medium intervals contain no detectable internal jump boundary, so their anomaly-interval jump-timing statistic is N/A rather than imputed. The large interval retains synchronized jump timing but reverses jump direction, which is consistent with a mode-correlation disruption.

## Comparison with the compatibility generation

Across same-name cases, median normal `|Pearson|` increased from **0.041088** in `CoMuT_compat` to **0.997505** in the historical reproduction, a median increase of **0.956429**. Normal correlation semantics were therefore substantially recovered by using the historical official configuration containing `random-seed: 42`.

This result supersedes neither the old compatibility data nor its D1 audit: both remain historical evidence of the failed compatibility route. No k-Means, AE, V2, V3, or Series2Graph++ model was run in N1.

**HISTORICAL OFFICIAL GUTENTAG 1.0.0; HISTORICAL OFFICIAL COMUT CONFIG; RANDOM-SEED 42; NO CONFIG MODIFICATION; NO MODEL TRAINING; NO COMMIT; NO PUSH.**
