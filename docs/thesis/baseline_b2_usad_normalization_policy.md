# Baseline-B2-P0: USAD Normalization Fairness Policy

Status: `USAD_NORMALIZATION_POLICY_FROZEN`

## Decision

USAD will retain its official method core: a shared encoder, two independent decoders, the official two-optimizer/two-stage loss schedule, sigmoid decoder outputs, and the frozen `0.5 / 0.5` reconstruction anomaly score. The decoder activation will not be removed or changed, and USAD will not be simplified into a conventional autoencoder.

The frozen USAD input policy is:

- Scaler: per-channel `MinMaxScaler(feature_range=(0, 1), clip=False)`.
- Fit scope: Normal Train only.
- Validation and Test: transform only; neither participates in fitting or adaptation.
- Out-of-range Validation/Test values: retained below 0 or above 1 when they exceed the Normal Train extrema; no clipping by default.
- Policy identifier: `TRAIN_ONLY_MINMAX_0_1`.

Normalization is part of each baseline's native input protocol. Fairness requires a common information boundary—fit only on Normal Train, no Validation/Test fitting, no Test adaptation, and no label leakage—not an identical mathematical transform for architectures with incompatible output ranges. Ours and LSTM-AE retain their frozen Train-only StandardScaler protocols. GDN is unchanged from Baseline-B0.

## Normal-Train-only range audit

Only the frozen Normal Train rows were read. No Validation or Test rows were opened for this audit.

| Dataset | Train rows | Channels | StandardScaler global range | MinMax global range |
|---|---:|---:|---:|---:|
| Official SWaT 2015 | 396,000 | 51 | `[-16.438761, 29.932448]` | `[0.0, 1.0000000000000002]` |
| HAI 21.03 | 737,280 | 79 | `[-31.916529, 30.680729]` | `[0.0, 1.0000000000000002]` |

HAI fit rows are the independently frozen 80% prefixes: 172,800 from `train1.csv.gz`, 181,440 from `train2.csv.gz`, and 383,040 from `train3.csv.gz`. The tiny excess over 1 is floating-point roundoff. The audit confirms the intended MinMax Train range while demonstrating why a sigmoid decoder paired with StandardScaler would face a material input/output range mismatch.

## Frozen evaluation rules

- Dataset splits, window 64, stride 1, and all segment/file boundaries remain Baseline-B0.
- USAD endpoint score remains the mean across channels of `0.5*(x-w1)^2 + 0.5*(x-w3)^2` at the reconstructed endpoint.
- Threshold remains Normal Validation q99 after the model and score are frozen.
- No point adjustment, POT, Test-best threshold, or Test-driven tuning.

## Scope

This phase freezes policy only. It performs no USAD training, model inference, Validation/Test reading, threshold calculation, or result selection. No commit or push is authorized.
