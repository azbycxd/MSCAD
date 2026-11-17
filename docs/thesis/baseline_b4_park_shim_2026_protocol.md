# Baseline-B4-P0: Park & Shim 2026 Dependency Plug-in Protocol Audit

## Decision

**Status: `PARK_SHIM_2026_PROTOCOL_REQUIRES_REVIEW`**

The official repository was cloned and audited without training or reading Test data. The architectural core is clear, but the formal comparison protocol cannot be frozen automatically because the repository provides no fixed final score-combination weight, evaluates a grid of weights using Test labels, and contains a blocking call-signature defect. The accessible publication record and abstract do not expose the exact paper equations or SWaT hyperparameter table, so those details cannot be inferred safely.

This status does not invalidate the method. It means a fair non-Test score-weight policy and a minimal source-compatibility repair must be explicitly approved before B4 training.

## Source identity

- Paper: Jeongwon Park and Jaewoong Shim, “A plug-in module for detecting and explaining anomalies in inter-variable dependencies in multivariate time series,” *Expert Systems with Applications*, vol. 316, article 131906, 2026, DOI `10.1016/j.eswa.2026.131906`.
- Official repository: `https://github.com/bae301764/plugin-mtsad`.
- Local clone: `external/baselines/plugin-mtsad`.
- Branch: `main`.
- HEAD: `84e0d145de4c53b4d4733e3b33164e6068a3d5e9`.
- Clone time: `2026-09-16T03:28:12+08:00`.
- Clean after audit: yes.
- Fork or secondary implementation: no.

The repository contains only three visible commits, no release/tag with a paper configuration, and no dataset-specific configuration file. File hashes are frozen in `repo_audit.json`.

## Method recovered from source

### Base model

The shipped base is GDN: learned sensor embeddings define a top-k KNN graph, an embedding-conditioned GAT performs message passing, and an output MLP forecasts the next point of every variable. The repository does not expose another base implementation.

### Dependency matrix

For each batch, GAT returns normalized attention weights for each of the K selected incoming neighbors plus a self-loop. `extract_dependency_matrix` writes those weights into a dense matrix `A` of shape `[B,N,N]`; entries for absent edges remain zero. The matrix is then flattened to `[B,N²]` for the plug-in.

This representation depends jointly on learned node embeddings, the top-k graph, input history, and attention. It is not a Pearson/covariance matrix.

### Plug-in autoencoder

For `D=N²`, the source defines:

`D → attn_hidden → 64 → attn_hidden → D`

The encoder and decoder use ReLU between linear layers and dropout 0.1 around the hidden layer. The output is linear. Parser default `attn_hidden` is 256; bottleneck 64 and dropout 0.1 are fixed constructor defaults.

The dependency loss is:

`L_dep = MSE(A_hat, A)`

The point-level dependency score returned by the evaluation code is:

`S_dep(t) = sum_ij |A_hat_ij(t) - A_ij(t)|`.

### Training mode

The implementation is **joint**, not sequential and not frozen-base plus plug-in. A single Adam optimizer receives all `model.parameters()`. The training loss is:

`L_total = MSE(y_hat, y) + lambda_attn * L_dep`.

The repository default and README example both use `lambda_attn=0.1`. This is a training-loss coefficient; it is not the final anomaly-score fusion weight.

## Score audit

### `PAPER_SCORE_FORMULA`

The accessible paper abstract confirms that dependency reconstruction deviations are combined with the base score, but the exact published equation was not available for verification. The official repository computes:

`S_a(t) = a*S_base(t) + (1-a)*S_dep(t)`,

for every `a ∈ {0,0.1,...,1}`.

`S_base` is constructed from per-sensor absolute forecast errors, but the official evaluator normalizes them using each sensor's **Test-set** median and IQR plus `1e-2`, applies a three-point trailing mean, and takes the maximum sensor score. `S_dep` is the unnormalized sum over the `N²` dependency reconstruction errors. No cross-score normalization is present before fusion.

### `PAPER_WEIGHT_SELECTION_POLICY`

No single fixed `a` is stored in the repository or README. The evaluator loops across all 11 weights and, for every weight, runs point-wise Test-best-F1 threshold selection and range-wise point-adjusted Test-best-F1 evaluation. Therefore the repository evaluation path is Test-driven. The full paper's exact reporting/selection policy could not be independently verified.

This is the main blocking issue. `lambda_attn=0.1` must not be misread as `a=0.1`, and this audit does not select a favorable score weight.

## Official hyperparameter evidence and conflicts

| Item | README SWaT example | Parser/source default | Freeze state |
|---|---:|---:|---|
| History/window | omitted | 5 | pending paper verification |
| Stride | omitted | 1 | source value recorded |
| GDN hidden/embedding | omitted | 64 | source value recorded |
| Top-k | 15 | 15 | consistent |
| Heads | omitted | 1 | source value recorded |
| Output layers / hidden | omitted | 1 / 128 | source value recorded |
| Plug-in hidden / bottleneck | omitted | 256 / 64 | source value recorded |
| Batch | omitted | 64 | source value recorded |
| Epochs | omitted | nominal 50 | conflict: loop executes at most 49 |
| Learning rate | omitted | 0.001 | source value recorded |
| Adam betas | omitted | 0.9 / 0.99 | source value recorded |
| Weight decay | omitted | 0 | source value recorded |
| Patience | omitted | 30 | source value recorded |
| Training `lambda_attn` | 0.1 | 0.1 | consistent |
| Seed | 41 | 42 | conflict; not frozen |
| Final score weight `a` | absent | Test-evaluated grid | blocking; not frozen |

Because the paper SWaT table is unavailable, there is no official configuration file, and the README/default sources conflict, the audit stops before selecting a configuration.

## Source defects and README mismatches

1. `GDN.forward(batch_x, explanation)` requires the `explanation` positional argument, while both training and validation call `model(x)`. Current HEAD therefore raises a Python `TypeError` before a forward pass unless a compatibility repair is approved.
2. README describes a generally attachable plug-in, but the supplied implementation embeds the plug-in directly inside GDN and trains base plus plug-in jointly.
3. `epochs=50` is passed to `range(1, epochs)`, giving at most 49 executed epochs.
4. The official environment requires PyTorch 2.1.1+cu118 and PyG 2.4.0 with compiled extensions. The thesis legacy `p310` environment lacks `torch_geometric`; it was inspected but not modified.

## Unified thesis adaptation design

These replacements are recorded but do not resolve the fusion-weight blocker:

- Datasets: Official SWaT 2015 and HAI 21.03. WADI remains outside B4 primary scope due the previously established Normal-domain shift.
- SWaT: Train `[0,396000)`, Validation `[396000,495000)`, Official Attack v0 Test, 51 variables, no gap crossing, 36 events.
- HAI: first/last 80/20 split within each of three Normal train files; five Test files; 79 variables; no file crossing; 50 events and HIGH19/DIRECT31 groups.
- Scaling: Train-only StandardScaler. The architecture has no bounded output that requires MinMax scaling; the official repository itself does not implement a scaler.
- Model selection: lowest Normal Validation total loss only.
- Score calibration: base-score median/IQR must come from Normal Validation rather than Test.
- Threshold: Normal Validation q99.
- Metrics: AUROC, AUPR, Precision, Recall, F1, FPR.
- Disabled: Test normalization/adaptation, Test-best threshold, point adjustment, POT, and Test-label checkpoint selection.
- Endpoint comparison: only the Baseline-B0 common valid mask. The official history should not be forced to 64 merely to match MSCAD, but its value remains pending paper/config adjudication.

The matched ablation pair is defined as `ParkShim-Base` and `ParkShim-Base+Plugin`. They must use the identical GDN data, history, architecture, optimizer and selection settings; the only methodological difference is inclusion of the dependency AE loss and dependency score. This matched Base does not replace the already completed independent GDN baseline.

For HAI, once a single SWaT configuration and non-Test score weight are approved, all settings must transfer unchanged. HAI Test may not select history, lambda, weight, or threshold.

## Resource estimate and preflight design

The plug-in parameter formula for hidden size 256 and bottleneck 64 is `513*N² + 33,344`. Under the repository's one-layer 64-to-1 PyG output MLP, the base estimate is `64*N + 1,025`.

| Dataset | Matrix | Base params | Plug-in params | Total params | FP32 params | Adam+grad parameter lower bound |
|---|---:|---:|---:|---:|---:|---:|
| SWaT | 51×51 | 4,289 | 1,367,657 | 1,371,946 | 5.234 MiB | 20.934 MiB |
| HAI21 | 79×79 | 6,081 | 3,234,977 | 3,241,058 | 12.364 MiB | 49.455 MiB |

A single dense dependency matrix is 10,404 bytes for SWaT and 24,964 bytes for HAI. At batch 64, the original plus reconstructed dense matrices alone require about 1.270 MiB and 3.047 MiB respectively. These are lower bounds: graph edges, intermediate activations, gradients, allocator overhead and PyG workspaces are excluded.

Before any future B4 training, both datasets require a one-batch Normal-Train-only forward/backward/optimizer resource preflight in an isolated compatible environment, recording actual parameters, matrix shape and peak CUDA allocation. Batch size may not be changed silently.

## Required human decisions before B4

1. Obtain/inspect the full paper or author-supplied SWaT configuration and reconcile seed, history and epoch semantics.
2. Approve a non-Test fusion-weight policy. It must not reinterpret `lambda_attn` as the score weight or select `a` using Test labels.
3. Approve the minimal `explanation` call-signature repair and document it as a compatibility patch without changing model mathematics.
4. Decide whether both base and dependency scores require Normal-Validation scale alignment before a fixed fusion weight can be meaningful; no such rule was selected here.

No model training, Test-data read, Test tuning, lambda search, point adjustment, POT, MSCAD rerun, commit, or push occurred.
