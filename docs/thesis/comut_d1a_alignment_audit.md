# CoMuT-D1A Historical Score–Label Alignment Audit

Status: **RELATION_SIGNAL_AND_ALIGNMENT_OK** for the preregistered large-case gate.

This audit did not train or run any model. It reused the frozen D1 RawCorr/V2 implementation and inspected only Historical CoMuT `short/medium/large-mode-correlation-on-2`, with window 64, stride 1, and unchanged endpoint scoring.

## Recovered intervals and window regions

The exact `is_anomaly` intervals are `[3240,3280)` for short, `[680,760)` for medium, and `[160,320)` for large, with lengths 40, 80, and 160. Their full-anomaly core counts are respectively 0, 17, and 97, exactly matching the preregistered geometry. Each case has a 63-endpoint post-anomaly tail whose endpoint is normal although its causal window still overlaps the anomaly.

## RawCorr score regions

| Case | rho_ref | Far normal median | Far normal q99 | Head median | Core median | Post-tail median | Core rho median |
|---|---:|---:|---:|---:|---:|---:|---:|
| Short | 0.994656 | 0.003451 | 1.256198 | 0.674217 | N/A | 0.786902 | N/A |
| Medium | 0.994397 | 0.003569 | 1.214663 | 0.924597 | 0.773918 | 1.012938 | 0.220479 |
| Large | 0.994513 | 0.003448 | 1.261789 | 1.153418 | 1.991631 | 0.935608 | -0.997118 |

The 97 large-case core windows have median score 1.991631, well above far-normal q99 1.261789. Their median window correlation is -0.997118 versus the +0.994513 training reference. The core relation signal is therefore present and the D1 implementation is behaving as defined.

## Endpoint versus window-overlap labels

| Case | Endpoint-label AUROC | Window-overlap AUROC |
|---|---:|---:|
| Short | 0.628218 | 0.605512 |
| Medium | 0.676580 | 0.720028 |
| Large | 0.945171 | 0.869015 |

The overlap metric is **DIAGNOSTIC_ONLY**. RawCorr and V2 values are identical in these two-channel cases because their transformations preserve the same ranking. Window-overlap labels improve medium modestly but worsen short and large, so endpoint/window label alignment is not the sole or primary explanation for low short/medium AUROC.

## Why short and medium can score poorly

The relation score summarizes the complete causal window `[t-63,t]`, not the endpoint alone. A 40-point anomaly has no full-anomaly core; every anomaly-labeled endpoint is mixed with preceding normal observations. Medium has only 17 core endpoints. This causes boundary dilution and leaves many high scores in the post-anomaly tail after endpoint labels have returned to normal.

More importantly, far-normal q99 is around 1.21–1.26 despite a far-normal median near 0.0035. Historical random-mode-jump channels are globally synchronized, but a 64-point normal window that contains insufficient mode transitions can be dominated by independent observation noise, making local Pearson depart sharply from the near-one reference. These high-scoring legitimate windows degrade ranking. This is a fixed-window/local-Pearson representation limitation, not evidence of score inversion or a broken correlation calculation.

TimeEval `ReverseWindowing` distributes each window score over every covered point; D1 attaches the complete-window score only to its endpoint. The semantics differ, but this audit provides no basis for replacing the frozen D1 metric with ReverseWindowing or overlap labels.

## Adjudication

1. Yes: all 97 large full-anomaly windows produce a high relation score.
2. Large-case relation signal and endpoint alignment are sound. Low short/medium AUROC is mainly due to mixed boundary windows and unstable local Pearson in normal windows without enough mode transitions; a simple label-alignment correction does not resolve it.
3. Historical D1R is eligible to remain a valid controlled rerun under its frozen endpoint protocol. Repeating the identical 13-case run is unnecessary; any alternative window-to-point mapping would require a separately preregistered experiment.

**NO MODEL TRAINING; NO PARAMETER SEARCH; NO SCORE FLIP; DIAGNOSTIC ONLY; NO COMMIT; NO PUSH.**
