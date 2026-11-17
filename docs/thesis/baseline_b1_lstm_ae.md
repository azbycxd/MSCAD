# Baseline-B1: LSTM-AE Unified Evaluation

## Status

`LSTM_AE_BASELINE_COMPLETED`

The HAI 21.03 run was resumed exactly from the persisted epoch-42 training state and reached natural early stopping at epoch 62. The official HAI Test files were opened only after training stopped, the best epoch-52 model was restored, and the Normal Validation q99 threshold was frozen.

## Frozen protocol

- Model: LSTM autoencoder, hidden size 64.
- Window: 64; stride: 1; no cross-file or cross-session windows.
- Training: full-window MSE, Adam, learning rate `1e-3`, weight decay 0, batch size 128.
- Schedule: at most 100 epochs; patience 10; minimum delta 0; seed 1.
- Score: endpoint mean channel squared reconstruction error.
- Threshold: Normal Validation q99.
- Evaluation: no point adjustment, POT, or Test-selected threshold.

The resume restored model state, optimizer state, best-model state, frozen sample order, history, early-stop counter, and Python/NumPy/Torch/CUDA RNG states. Epoch 42 was verified with best epoch 39, best validation loss `0.06134573767856918`, and counter `3/10`; training continued at epoch 43.

## Training outcome

HAI stopped naturally after epoch 62 because epochs 53–62 produced ten consecutive validation epochs without improvement. The best model remained epoch 52 with validation loss `0.05937653954627936`. Training and validation losses rose sharply after epoch 52; this was not used to alter the protocol, and the pre-registered best-model restoration protected the final evaluation.

Total HAI training wall time is `N/A` because the earliest run did not persist a complete cumulative timer. The reliably recorded continuation time for epochs 24–62 is 3116.793 seconds, of which Resume-3 (epochs 43–62) used 1425.063 seconds.

## Point-wise results

| Dataset | Method | AUROC | AUPR | Precision | Recall | F1 | FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| SWaT | LSTM-AE | 0.818242 | 0.706043 | 0.147874 | 0.912250 | 0.254495 | 0.726609 |
| SWaT | Ours-REC | 0.834887 | 0.743353 | 0.150140 | 0.917870 | 0.258067 | 0.718133 |
| SWaT | Ours-DUAL | 0.675631 | 0.167510 | 0.087166 | 0.008605 | 0.015663 | 0.012455 |
| SWaT | Ours-TAIL | 0.708375 | 0.206747 | 0.152452 | 0.901228 | 0.260789 | 0.692531 |
| HAI 21.03 | LSTM-AE | 0.907084 | 0.577510 | 0.399575 | 0.713982 | 0.512393 | 0.024441 |
| HAI 21.03 | Ours-REC | 0.852899 | 0.536294 | 0.459109 | 0.601095 | 0.520594 | 0.016133 |
| HAI 21.03 | Ours-DUAL | 0.916334 | 0.570193 | 0.554350 | 0.679446 | 0.610556 | 0.012443 |
| HAI 21.03 | Ours-TAIL | 0.901850 | 0.617897 | 0.555963 | 0.693976 | 0.617350 | 0.012627 |

These rows are protocol-aligned local comparisons. Empty parameter or latency cells for existing Ours rows in the baseline CSV remain unfilled where no directly compatible frozen efficiency artifact was available; no values were inferred.

## Event results

- SWaT LSTM-AE: 29/36 events hit.
- HAI LSTM-AE: 48/50 events hit at the frozen `hit >= 0.10` criterion.
- HAI HIGH19: 19/19 hit; median predicted-anomaly fraction `0.837209`.
- HAI DIRECT31: 29/31 hit; median predicted-anomaly fraction `0.858586`.

## Efficiency

All latency measurements use batch size 1, 100 warm-up windows, and 1000 legal timed windows on an NVIDIA GeForce RTX 4060.

| Dataset | Parameters | State dict | Mean latency | Median latency | p95 latency | Peak inference VRAM |
|---|---:|---:|---:|---:|---:|---:|
| SWaT | 66,547 | 269,357 bytes | 0.983 ms | 0.537 ms | 2.975 ms | 38.295 MiB |
| HAI 21.03 | 75,535 | 305,315 bytes | 0.948 ms | 0.498 ms | 2.960 ms | 44.885 MiB |

SWaT was not retrained. Its frozen metrics, event table, and checkpoint hashes were verified unchanged before and after HAI completion; only the checkpoint was loaded for the requested efficiency benchmark.

## Reproducibility and exclusions

- Same frozen Baseline-B0 protocol; patience, minimum delta, learning rate, batch size, loss, model, window, and seed were unchanged.
- No Test-driven early stopping, score tuning, threshold tuning, point adjustment, or POT.
- No USAD or GDN execution.
- No SWaT retraining and no HAI restart.
- No commit or push was performed.
