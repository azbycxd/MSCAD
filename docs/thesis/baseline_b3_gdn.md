# Baseline-B3: GDN unified evaluation

Status: `GDN_BASELINE_COMPLETED`

## Frozen protocol

This run used GDN source commit `9853899da860682669a134e4af315d036aab4eca`. It retained sensor embeddings, the embedding-derived top-5 graph, graph attention/message passing, and next-point forecasting. The frozen settings were history 64, embedding/hidden dimension 64, one output layer, output intermediate dimension 128, Adam with learning rate `1e-3` and zero weight decay, batch 32, at most 30 epochs, patience 10, minimum delta 0, and seed 1.

`StandardScaler` was fitted only on Normal Train. Validation and Test were transform-only. The anomaly score was the per-sensor absolute forecast error standardized using Normal Validation median and `IQR + 1e-2`, followed by the official four-point trailing mean and a top-1/max aggregation. The operating threshold was Normal Validation q99. No point adjustment, POT, Test-best threshold, or Test tuning was used.

## Training and resume

Both frozen batch-32 resource preflights passed. SWaT used 8,001 trainable parameters with a 26.64 MiB preflight peak allocation. HAI used 9,793 parameters with a 32.54 MiB preflight peak allocation.

SWaT trained for 19 epochs and stopped naturally after 10 validation epochs without improvement. Its best epoch was 9 with validation loss 0.11008502. The first B3 run stopped at the hard deadline after HAI epoch 11. B3-Resume restored the model, Adam state, best model, sample order, history, and Python, NumPy, Torch, and CUDA RNG states. It continued from epoch 12 and stopped naturally at epoch 13. HAI's best epoch remained 3 with validation loss 0.22986660052279786. The cumulative persisted HAI training time was 1191.03 seconds. SWaT training wall time is `N/A` because the first completed-run duration was not persisted; it was not estimated by retraining.

## Final results

| Dataset | AUROC | AUPR | Precision | Recall | F1 | FPR | Event hits |
|---|---:|---:|---:|---:|---:|---:|---:|
| SWaT | 0.827665 | 0.728194 | 0.793843 | 0.659087 | 0.720216 | 0.023658 | 15/36 |
| HAI 21.03 | 0.877643 | 0.497775 | 0.537723 | 0.588689 | 0.562053 | 0.011529 | 42/50 |

HAI HIGH19 median event hit fraction was 0.716535 and DIRECT31 median was 0.660099. The validation-q99 thresholds were 123.868588 for SWaT and 16.876451 for HAI.

## Efficiency

| Dataset | Params | State size | Mean latency | Median latency | P95 latency | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| SWaT | 8,001 | 40,050 bytes | 0.747 ms | 0.681 ms | 1.211 ms | 29.21 MiB |
| HAI 21.03 | 9,793 | 47,193 bytes | 0.738 ms | 0.696 ms | 0.967 ms | 36.22 MiB |

Latency used batch 1, 100 warm-up endpoints, and 1,000 legal timed endpoints on the RTX 4060.

## GDN versus Ours-DUAL

| Dataset | Method | AUROC | AUPR | F1 | FPR | HIGH median hit | Params | Mean latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SWaT | GDN | 0.827665 | 0.728194 | 0.720216 | 0.023658 | N/A | 8,001 | 0.747 ms |
| SWaT | Ours-DUAL | 0.675631 | 0.167510 | 0.015663 | 0.012455 | N/A | N/A | N/A |
| HAI 21.03 | GDN | 0.877643 | 0.497775 | 0.562053 | 0.011529 | 0.716535 | 9,793 | 0.738 ms |
| HAI 21.03 | Ours-DUAL | 0.916334 | 0.570193 | 0.610556 | 0.012443 | 0.841085 | N/A | N/A |

The existing Ours-DUAL final outputs do not contain a separately attributable parameter count or latency measurement, so those cells remain `N/A`; no Ours model was rerun.

## Graph diagnostics limitation

Raw learned graph edges are preserved for both datasets. HAI process prefixes were available and produced 224 within-process and 171 cross-process edges. SWaT process-level graph attribution is unavailable because the frozen diagnostic parser did not recognize channel names such as `FIT201`; all 255 SWaT edges were classified as `UNKNOWN`. Per the B3-Resume instruction, this parser was not repaired and the limitation does not affect model predictions or metrics.

No hyperparameter, top-k, dimension, history, or learning-rate search was performed. No LSTM-AE, USAD, or MSCAD comparator was rerun. No commit or push was performed.
