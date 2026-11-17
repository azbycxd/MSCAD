# Baseline-B0 Unified Strong-Baseline Evaluation Protocol

Status: `BASELINE_PROTOCOL_FROZEN`

This document freezes the protocol for the first seed-1 comparison of LSTM-AE, USAD, and GDN against the existing final MSCAD branches. It does not report newly trained baseline results. Historical baseline outputs in the legacy project are not reusable because their data and threshold protocols are not the final Official SWaT/HAI protocol.

## Scope

Primary datasets are Official SWaT 2015 Physical and HAI 21.03. WADI.A2 remains a domain-shift limitation and is excluded from the primary baseline table and success claims.

The existing `REC`, `DUAL`, Unified Reliability Gate, tail calibration, and fusion definitions are frozen. Baseline work may not change them or trigger lag, threshold, fusion-weight, or target searches.

## Dataset protocol

### Official SWaT 2015 Physical

- Normal Train: rows `[0, 396000)` from the frozen Normal v1 source.
- Normal Validation: rows `[396000, 495000)`.
- Test: complete official Attack v0, split at the frozen 82-second hard session boundary.
- Features: all 51 frozen physical variables.
- Labels: frozen exact Official Physical mapping; anomaly is `1`.
- Window: 64; train, validation, and test stride are all 1.
- Windows never cross the hard session boundary.
- Event set: 36 canonical physical attack-list events.

### HAI 21.03

- Train files: `train1.csv.gz`, `train2.csv.gz`, `train3.csv.gz`.
- Normal Train: first 80% of each file independently.
- Normal Validation: final 20% of each file independently.
- Test files: all five official test files.
- Features: the 79 process variables present in the frozen files; label columns are excluded.
- Primary label: `attack == 1`.
- Window: 64; train, validation, and test stride are all 1.
- Windows never cross a train, validation, or test file boundary.
- Event set: 50 frozen official events.

For LSTM-AE and GDN, `StandardScaler` is fitted only on the concatenated Normal Train rows. The existing MSCAD comparators also retain their frozen Train-only `StandardScaler`. USAD is the sole architecture-required normalization exception: its official sigmoid decoders are paired with a per-channel `MinMaxScaler(feature_range=(0, 1))`, fitted only on Normal Train. Validation and Test are transform-only for every method; neither split may participate in scaler fitting or adaptation.

Fairness is therefore defined by the shared fit boundary and absence of leakage, rather than by forcing architectures with different output ranges to use the same mathematical transform. USAD does not clip transformed Validation or Test values: values outside the Normal Train extrema may naturally map below 0 or above 1.

## Common evaluation protocol

Each method retains its native model and raw anomaly-score meaning, then produces one score at a pointwise endpoint. The common comparison mask contains only endpoints legal for every method. For reconstruction models, endpoint `t` uses `[t-63, t]`. For GDN, endpoint `t` is forecast from `[t-64, t-1]`; consequently the common mask begins at local index 64 in each continuous segment/file.

- Seed: 1 only.
- Early stopping: Normal Validation loss only; patience 10, minimum improvement 0, restore the lowest-validation-loss checkpoint.
- Threshold: empirical q99 of the method's Normal Validation endpoint score, with prediction `score > threshold`.
- Metrics: AUROC, AUPR, Precision, Recall, F1, and FPR.
- Primary ranking metrics: AUPR, then AUROC. F1 is interpreted only at the frozen validation q99.
- Event hit rate: fraction of legal event endpoints predicted anomalous.
- Event count: number of frozen events with hit rate at least 0.10.
- No point adjustment, POT, oracle threshold, Test-best threshold, or event adjustment.
- Test labels are read only after model selection, score definition, and q99 threshold are frozen.

## Baseline definitions

### LSTM-AE

- Source: Malhotra et al., *LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection* (2016), with a new minimal project-local PyTorch implementation. The historical `run_baselines.py` implementation and its POT evaluation are not reused.
- Input: `[B, 64, C]`.
- Encoder: one LSTM layer, hidden size 64; the final hidden state is the latent representation.
- Decoder: one LSTM layer, hidden size 64, conditioned on the repeated latent representation, followed by a linear projection to `C` channels.
- Loss: full-window mean squared reconstruction error.
- Optimizer: Adam, learning rate `1e-3`, weight decay `0`.
- Batch size: 128; maximum epochs: 100; seed: 1.
- Endpoint score: mean squared reconstruction error across channels at the final point of the reconstructed window.

### USAD

- Source repository: `https://github.com/manigalati/usad`, commit `e25af45c8e1c32783aed94fc7e5ab85effef2b1a`.
- Core structure: flattened 64-point window, shared encoder, two independent decoders, and the official two-phase adversarial training schedule. It must not be simplified to a conventional autoencoder.
- Dimensions: `w_size = 64*C`; encoder `w_size -> floor(w_size/2) -> floor(w_size/4) -> z_size`; each decoder mirrors this path. `hidden_size=100` and `z_size=64*100=6400`, following the official SWaT notebook rule.
- Decoder output activation: sigmoid, retained from the official implementation.
- Normalization: per-channel Train-only `MinMaxScaler(feature_range=(0, 1), clip=False)`; frozen policy identifier `TRAIN_ONLY_MINMAX_0_1`.
- Epoch loss at one-based epoch `n`: `L1=(1/n)MSE(x,w1)+(1-1/n)MSE(x,w3)` and `L2=(1/n)MSE(x,w2)-(1-1/n)MSE(x,w3)`, where `w3=D2(E(w1))`.
- Optimizers: two Adam optimizers with learning rate `1e-3` and weight decay `0`.
- Batch size: 256 as a fixed resource adaptation for the 8 GB evaluation GPU; maximum epochs: 100; seed: 1.
- Native score weights: `alpha=0.5`, `beta=0.5`.
- Endpoint score: at the final point, average across channels of `0.5*(x-w1)^2 + 0.5*(x-w3)^2`. This preserves the official two-reconstruction score while aligning it to the endpoint.

### GDN

- Source repository: `https://github.com/d-ailin/GDN`, commit `9853899da860682669a134e4af315d036aab4eca`.
- Core structure: sensor embeddings, learned top-k graph, graph attention/message passing, and next-point forecasting. The graph module may not be removed or replaced by an MLP.
- History length: 64, imposed by the unified thesis protocol instead of the official demo's 5-point setting.
- Hidden/sensor embedding dimension: 64; graph neighbors: top-k 5; output layers: 1; output intermediate dimension: 128.
- Batch size: 32; maximum epochs: 30; optimizer: Adam, learning rate `1e-3`, weight decay `0`; seed: 1.
- Native per-sensor score: absolute forecast error standardized by Normal Validation median and IQR plus `1e-2`, followed by the official four-point trailing mean.
- Endpoint score: maximum standardized/smoothed sensor score (`top-1`) at the forecast target timestamp.
- The official repository's Test-best reporting and maximum-normal threshold are disabled; the common Normal Validation q99 threshold is used.

## Existing MSCAD comparators

`Ours-REC`, `Ours-DUAL`, and `Ours-Tail` are read from `outputs/thesis_innovation3b_tail_fusion_final/cross_dataset_metrics.csv`. They are not rerun. Their frozen SWaT target SHA256 is `af9f9aac9cb8b369015a94d4b88a4729d817c219d305eb3276050c5d33ad31bf`; the HAI target SHA256 is `a78e820567eaa15dd5a0cd2a9f6491d04102c0a50732a4de7aa918f1b20be5bb`.

## Efficiency protocol

All methods must run on the same RTX 4060 GPU and software environment recorded in `efficiency_protocol.json`. Report trainable parameter count, serialized model-state size, training wall time, and inference latency. Latency uses `model.eval()`, inference mode, batch size 1, 100 warm-up endpoints, and exactly 1000 legal timed endpoints. CUDA is synchronized before and after every timed forward pass. Report mean, median, and p95 milliseconds. Peak CUDA inference memory is measured after resetting peak statistics and is reported only when measurement succeeds.

## Fairness limitations that cannot be removed without changing a method

1. LSTM-AE and USAD reconstruct windows, whereas GDN forecasts the next point. A common endpoint mask aligns evaluation timestamps but cannot make the learning objectives identical.
2. USAD retains its official sigmoid decoder and therefore uses per-channel Train-only MinMax scaling to `[0,1]`. This is a frozen architecture-compatible input protocol, not a result-driven exception. LSTM-AE, GDN, and the existing MSCAD comparators remain unchanged on Train-only StandardScaler.
3. The official GDN code uses MinMax scaling, a short demo window, randomized validation placement, Test-best reporting, and a different threshold. These are replaced by the frozen chronological split, StandardScaler, 64-point history, and validation q99. Graph learning and score semantics remain intact.
4. Training batch sizes differ because the methods have materially different memory footprints. Dataset endpoints, scaler fit rows, validation/Test strides, selection policy, and metrics remain identical.
5. HAI has no official dataset-specific configurations in the cited USAD or GDN repositories. The same pre-registered architecture and optimizer settings are applied to both datasets without Test-driven adjustment.

## Sources

- Malhotra et al. (2016): https://arxiv.org/abs/1607.00148
- USAD repository and implementation: https://github.com/manigalati/usad
- GDN repository and implementation: https://github.com/d-ailin/GDN

No baseline model was trained in B0 or in the B2-P0 normalization-policy freeze.

## B6-P0 extension: CCM-TAD ICML 2026

Audit status: `CCM_TAD_2026_PROTOCOL_REQUIRES_REVIEW`.

The official repository `https://github.com/Secure-and-Intelligent-Systems-Lab/CCM-TAD` was audited at clean commit `3a8bc8263b648a707cd0658bf4f06517fb9e52c5`. The full loader, training, model, correlation-profile clustering, PF1, and sequential SF1 evaluation chain is present. The official SWaT configuration uses a 24-point look-back, 6 clusters, `d_model=128`, one causal mixer, batch 512, 30 epochs, Adam with learning rate `0.00498096466936417`, and seed 42.

Formal B6 training is blocked pending review. The released runner saves checkpoints by Test PF1/SF1; both point and sequential thresholds are Test-label Best-F1; the sequential `alpha` is also Test-label selected. In addition, the SWaT loader uses a centered 50-point moving average that sees future points, and official clustering is initialized from full normal data including the nominal validation tail.

The model architecture and official raw mean-channel endpoint reconstruction MSE can be adapted to Train-only fitting, Normal-Validation checkpoint selection, and the common endpoint mask. However, raw-MSE evaluation would omit CCM-TAD's sequential contribution, while no fixed label-free sequential `alpha` is supplied by the executable protocol. HAI fallback and preprocessing adaptation are therefore not frozen. Full evidence is in `baseline_b6_ccmtad_2026_protocol.md` and `outputs/thesis_baselines/ccmtad_2026/`.

## B4-P0 extension: Park & Shim 2026 dependency plug-in

Audit status: `PARK_SHIM_2026_PROTOCOL_REQUIRES_REVIEW`.

The official repository `https://github.com/bae301764/plugin-mtsad` was frozen at commit `84e0d145de4c53b4d4733e3b33164e6068a3d5e9`. Its supplied implementation combines a GDN next-point forecasting base with an auxiliary autoencoder over dense `N×N` GAT attention matrices and trains the complete system jointly with one Adam optimizer.

The intended matched comparison is `ParkShim-Base` versus `ParkShim-Base+Plugin`, with identical base configuration and only the dependency plug-in contribution differing. The completed independent GDN baseline remains a separate external comparator.

The B0 dataset splits, Train-only scaling boundary, common endpoint mask, Normal Validation model selection, q99 threshold, metrics, event sets, and no-PA/no-POT/no-Test-tuning rules apply. Because the architecture has no bounded output restriction and the repository does not implement a scaler, the candidate preprocessing is Train-only StandardScaler.

Formal B4 training is blocked pending review because:

1. the repository supplies no fixed final base/dependency score weight and evaluates `a∈{0,0.1,...,1}` using Test labels;
2. its base score uses Test-set median/IQR normalization and its reported F1 paths use Test-best thresholds and point adjustment, all of which must be replaced;
3. README's SWaT example uses seed 41 while the parser default is 42, and the exact paper SWaT configuration could not be verified;
4. `GDN.forward` requires an `explanation` argument, while the shipped train/validation calls omit it;
5. nominal 50 epochs execute as at most 49 because the loop is `range(1, epochs)`.

No fusion-weight substitute is frozen in B4-P0. If the above issues are resolved, the same approved SWaT method configuration must transfer unchanged to HAI; HAI Test may not select a weight or hyperparameter. Full evidence and resource estimates are in `baseline_b4_park_shim_2026_protocol.md` and the four B4-P0 JSON manifests.

## B5-P0 extension: MAD-TCN 2026

Audit status: `MAD_TCN_2026_OFFICIAL_CODE_INCOMPLETE`.

The official repository `https://github.com/qianmo2001/MAD-TCN` was audited at clean commit `24d42ee0e0fc5a5460d437026cf4b89f05381293`. Its single model file verifies the dual local/global temporal convolution branches, grouped ConvFFN feature/cross-variable processing, adaptive per-variable gate, and sigmoid prediction head.

The repository does not include a dataset loader, preprocessing, training, validation, evaluation, configuration, target/loss, or anomaly-score implementation. Its constructor also reads an undefined external `args.alpha_value`. The paper's approximately 0.026M parameter claim is therefore reference-only; no source-derived SWaT or HAI parameter count is asserted.

No unified training adaptation is frozen. In particular, normalization, the native score, official SWaT hyperparameters, checkpoint selection, thresholding, point adjustment, POT/SPOT, and Test-driven behavior remain unauditable. HAI fallback is blocked until an evidence-backed SWaT configuration is recovered. See `baseline_b5_madtcn_2026_protocol.md` and `outputs/thesis_baselines/madtcn_2026/`.
