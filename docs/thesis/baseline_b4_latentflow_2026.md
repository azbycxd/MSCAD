# Baseline-B4 LatentFlow KDD 2026 unified evaluation

## Status

`HARD_STOP_INCOMPLETE`

The frozen protocol and official source binding passed. The official repository was clean at commit `3695f54317fcc1a3fdbfa0a7037d1cdf4369db27`.

Both resource preflights completed with the frozen batch size of 256. The SWaT model had 1,190,689 trainable parameters, an estimated 22.860352 MiB adjacency-sequence tensor, and a reported PyTorch peak allocation of 13,209.825684 MiB. The HAI model had 1,190,689 trainable parameters, an estimated 54.852539 MiB adjacency-sequence tensor, and a reported peak allocation of 20,498.176270 MiB. These allocator values are retained exactly as reported.

Formal execution then started SWaT seed 42. Epoch 1 did not reach an epoch boundary before the run substantially exceeded the 45-minute hard stop. The process was terminated without entering epoch 2. No Test data were read, no metrics were calculated, and no other seed was started.

Because the frozen runner originally persisted state only at complete epoch boundaries, no valid checkpoint exists for this interrupted epoch. The partial in-memory state was not presented as recoverable. A continuation must rerun SWaT seed 42 epoch 1 under the same frozen protocol, or first add and audit an intra-epoch exact-resume mechanism without changing the model, data, optimizer, batch size, or sample order.

## Preserved constraints

- No hyperparameter, k, window, learning-rate, batch, epoch, patience, or seed change.
- No Test tuning, SPOT/POT, point adjustment, Test-best threshold, or event adjustment.
- No Ours, GDN, USAD, or LSTM-AE rerun.
- No commit or push.

The final cross-seed summary and baseline table remain intentionally unchanged because zero seeds completed.
