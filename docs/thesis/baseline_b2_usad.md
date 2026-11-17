# Baseline-B2: USAD Unified Evaluation

Status: `USAD_BASELINE_COMPLETED`

## Protocol and resource gate

The implementation follows the official USAD core at commit `e25af45c8e1c32783aed94fc7e5ab85effef2b1a`: a shared encoder, two independent decoders, sigmoid decoder outputs, and the official AE1-then-AE2 optimizer schedule. The flattened window dimensions are `w_size -> floor(w_size/2) -> floor(w_size/4) -> 6400`; each decoder mirrors this path. No layer or loss term was removed.

Resource Preflight used only Normal Train data and executed one complete batch-256 step for both optimizers before any Validation or Test data was read.

| Dataset | Channels | Parameters | Estimated parameter bytes | Preflight peak allocation | Result |
|---|---:|---:|---:|---:|---|
| SWaT | 51 | 35,663,152 | 142,652,608 | 889.764 MiB | PASS |
| HAI 21.03 | 79 | 72,227,568 | 288,910,272 | 1,776.079 MiB | PASS |

The frozen batch size 256 was retained. Both datasets use per-channel Train-only MinMax scaling to `[0,1]` with `clip=False`; Validation and Test are transform-only.

## Training

Both runs used seed 1, Adam at `1e-3`, weight decay 0, at most 100 epochs, patience 10, and minimum delta 0. The two official losses were preserved. Since the official implementation reports `val_loss1` and `val_loss2` separately, the primary official reconstruction objective `val_loss1` was used as the single checkpoint and early-stopping monitor; `val_loss2` remains recorded in the training history.

| Dataset | Epochs | Best epoch | Best validation loss1 | Stop | Training time |
|---|---:|---:|---:|---|---:|
| SWaT | 11 | 1 | 0.001886346 | patience 10 | 798.571 s |
| HAI 21.03 | 11 | 1 | 0.005276292 | patience 10 | 5,020.974 s |

Every epoch persisted the current model, both optimizer states, best model, best epoch, early-stop counter, history, sample order, and Python/NumPy/Torch/CUDA RNG states. Final evaluation restored the epoch-1 best model for each dataset.

## Point-wise results

The score is the frozen endpoint mean across channels of `0.5*(x-w1)^2 + 0.5*(x-w3)^2`. Thresholds are Normal Validation q99. F1 therefore describes only this frozen operating point.

| Dataset | Method | AUROC | AUPR | Precision | Recall | F1 | FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| SWaT | USAD | 0.803816 | 0.698928 | 0.148539 | 0.904762 | 0.255184 | 0.716859 |
| SWaT | LSTM-AE | 0.818242 | 0.706043 | 0.147874 | 0.912250 | 0.254495 | 0.726609 |
| SWaT | Ours-REC | 0.834887 | 0.743353 | 0.150140 | 0.917870 | 0.258067 | 0.718133 |
| SWaT | Ours-DUAL | 0.675631 | 0.167510 | 0.087166 | 0.008605 | 0.015663 | 0.012455 |
| SWaT | Ours-TAIL | 0.708375 | 0.206747 | 0.152452 | 0.901228 | 0.260789 | 0.692531 |
| HAI 21.03 | USAD | 0.856873 | 0.494578 | 0.300978 | 0.595060 | 0.399760 | 0.031484 |
| HAI 21.03 | LSTM-AE | 0.907084 | 0.577510 | 0.399575 | 0.713982 | 0.512393 | 0.024441 |
| HAI 21.03 | Ours-REC | 0.852899 | 0.536294 | 0.459109 | 0.601095 | 0.520594 | 0.016133 |
| HAI 21.03 | Ours-DUAL | 0.916334 | 0.570193 | 0.554350 | 0.679446 | 0.610556 | 0.012443 |
| HAI 21.03 | Ours-TAIL | 0.901850 | 0.617897 | 0.555963 | 0.693976 | 0.617350 | 0.012627 |

USAD's Validation q99 thresholds are `0.019543846` for SWaT and `0.017465528` for HAI.

## Events

- SWaT: 29/36 events hit at the frozen `hit >= 0.10` rule.
- HAI: 46/50 events hit.
- HAI HIGH19: 18/19 hit; median predicted-anomaly fraction `0.629921`.
- HAI DIRECT31: 28/31 hit; median predicted-anomaly fraction `0.705882`.

No event adjustment or point adjustment was applied.

## Efficiency

Measurements use the RTX 4060, batch size 1, 100 warm-up windows, and 1000 legal timed endpoints.

| Dataset | Parameters | State dict | Mean latency | Median latency | p95 latency | Peak inference VRAM |
|---|---:|---:|---:|---:|---:|---:|
| SWaT | 35,663,152 | 142,659,821 bytes | 1.418 ms | 1.274 ms | 2.273 ms | 210.748 MiB |
| HAI 21.03 | 72,227,568 | 288,917,461 bytes | 12.384 ms | 12.708 ms | 14.191 ms | 403.276 MiB |

## Exclusions

No architecture, batch, score, threshold, or normalization change was made after Resource Preflight. There was no Test tuning, point adjustment, POT, GDN execution, or rerun of the existing MSCAD methods. No commit or push was performed.
