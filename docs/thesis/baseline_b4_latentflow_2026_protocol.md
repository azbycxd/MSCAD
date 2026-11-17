# Baseline-B4-P0-Replacement: LatentFlow KDD 2026 Protocol Audit

## Decision

**Status: `LATENTFLOW_2026_PROTOCOL_FROZEN`**

LatentFlow can be evaluated under the thesis unified protocol without changing its core model or anomaly-score semantics. The official Test-stream SPOT threshold and the scaler's Normal-Validation leakage are evaluation/preprocessing concerns rather than method-defining components, so they are replaced with Train-only scaling and Normal-Validation q99. No model was trained and no Test data was read in this audit.

## Repository identity

- Official repository: `https://github.com/SunboTax/LatentFlow`
- Local clone: `external/baselines/LatentFlow`
- Branch: `master`
- HEAD: `3695f54317fcc1a3fdbfa0a7037d1cdf4369db27`
- Clone time: `2026-09-16T12:58:21+08:00`
- Repository clean after audit: yes
- Fork used: no
- Paper: Lijun Sun et al., “LatentFlow: Discovering Latent Continuous Dynamics across Channels for Multivariate Time-Series Anomaly Detection,” KDD 2026, DOI `10.1145/3770855.3818034`.

## Method recovered from source

### Channel dependency representation

Each channel patch is processed by a shared gated temporal encoder: two one-channel Conv1d paths produce a feature and sigmoid gate, followed by flattening, linear projection, GELU/dropout and LayerNorm. A shared GRUCell evolves every channel's latent state across the patch sequence.

The dependency learner projects each channel latent state into query and key vectors and forms scaled dot-product logits. It masks diagonal logits, retains the top-k cross-channel neighbors per destination channel, and applies row-wise softmax. Consequently, every patch step yields a directed sparse dependency matrix `A_candidate ∈ R^(N×N)`.

### Latent dynamics

The source implements dependency evolution as:

`A_t = alpha_t*A_candidate_t + (1-alpha_t)*A_(t-1)`.

`alpha_t` is learned from the current channel latent state and previous normalized node degree. During inference, the code multiplies `alpha_t` by 0.5 to impose stronger inertia. The implementation is discrete gated evolution inspired by continuous/Ornstein-Uhlenbeck dynamics; it does not contain an ODE or SDE solver.

### Self-blind dependency

Self-dependency is removed twice: diagonal query/key logits are assigned `-1e9` before top-k, and diagonal entries are zeroed again after temporal interpolation. Rows are then renormalized. This self-blind rule applies to graph edges. The spatial aggregator still contains a residual path from the channel's own latent representation, so the complete reconstruction network is not deprived of all self information.

### Reconstruction backbone and loss

The window is divided into overlapping patches. At each patch step, the dynamic adjacency aggregates projected channel latent states. A shared decoder maps `d_model → d_model/2 → patch_size`. Overlapping patch reconstructions are averaged back onto a full-window canvas.

Training minimizes:

`L = MSE(reconstructed patches, input patches) + lambda_sparsity*mean(|A|)`.

### Anomaly score

The sole official point score is endpoint reconstruction MSE:

`S(t) = mean_c (x[t,c] - x_hat[t,c])²`.

There is no separate dependency score and no score fusion. Dependency dynamics affect the reconstruction path and training regularizer but are not directly added to the anomaly score.

## Relation to Innovation-2

LatentFlow explicitly models the evolution of a self-blind cross-channel dependency graph. Ours models Self-View and Cross-View predictive evidence and scores their disagreement. Therefore LatentFlow is a direct strong baseline for the Innovation-2 claim: both target cross-channel dependency anomalies, but use different representations and score mechanisms.

## Frozen official SWAT configuration

The official reproduction command is complete and takes priority over parser defaults:

| Parameter | Frozen value |
|---|---:|
| window_size | 128 |
| patch_size | 64 |
| patch_stride | 8 |
| patch steps | 9 |
| d_model | 128 |
| k | 15 |
| dropout | 0.3 |
| learning rate | 0.001 |
| LR schedule | type4 |
| batch | 256 |
| epochs | 30 |
| patience | 5 |
| optimizer | Adam |
| weight decay | 0.0001 |
| lambda_sparsity | 0.001 |
| official Affiliation/SPOT q | 0.016 |
| runs | seeds 42, 43, 44 |

`q=0.016` is not part of the model or raw anomaly score and is excluded from the primary thesis evaluation. No conflict was found between the SWAT reproduction scripts and README reproduction guidance.

The official loop changes the in-memory seed to 42/43/44, but its checkpoint filename continues to contain the unchanged base seed 42. The unified runner must therefore use a distinct checkpoint directory and manifest for each effective seed to prevent overwrite; this is output isolation, not a model change.

## Data and normalization adaptation

Primary datasets remain Official SWaT 2015 and HAI 21.03. WADI is supported by the repository but remains outside the main performance table because of the previously frozen Normal-domain-shift limitation.

The official preprocessing fits StandardScaler on all Normal `init_data` before its 80/20 Train/Validation split, allowing Normal Validation to influence scaler statistics. The unified protocol instead uses:

- SWaT: scaler fit on rows `[0,396000)` only; rows `[396000,495000)` and Official Attack Test are transform-only.
- HAI: scaler fit on the first 80% of each of the three Normal train files; their final 20% and all five Test files are transform-only.
- No Test adaptation or clipping.

The official centered rolling “stable” series is passed through data loaders but never used by the model or loss. Its only effective input-side consequence is trimming Normal endpoints before the split. The unified adapter therefore preserves the thesis frozen rows rather than reproducing unused trimming; this does not change LatentFlow's model inputs, loss, adjacency or score definition.

The official window 128 is retained. A legal LatentFlow endpoint begins at local index 127, so final comparisons use the intersection with the Baseline-B0 common mask (`>=64`), effectively `>=127`, without crossing SWaT gaps or HAI file boundaries.

## Model selection and leakage audit

Checkpoint selection is already based solely on Normal Validation reconstruction-plus-sparsity loss with patience 5. Validation labels and Test labels are not used.

The CLI option `--select_metric Affiliation_F1` is misleadingly described as selecting the best q, but current source only validates the option. It is not referenced by checkpoint selection, q selection, or evaluation after parsing. The reproduction script supplies a fixed `q=0.016`.

Test-driven behavior by category:

| Category | Official behavior | Unified primary protocol |
|---|---|---|
| Checkpoint | Normal Validation loss | retained |
| Threshold | SPOT initialized from Normal scores and updated while streaming Test scores | replaced by Normal Validation q99 |
| Score normalization | none | none |
| Metric labels | Test labels used to compute Affiliation/VUS only | labels used only for final metrics |
| Hyperparameter search | no search loop in current repo; origin of dataset-specific q is undocumented | q excluded; no Test search |

Affiliation-F1, VUS-ROC and VUS-PR may be reported only as supplementary metrics. The primary table uses AUROC, AUPR, Precision, Recall, F1 and FPR at Normal Validation q99, with no point adjustment, POT or Test-best threshold.

## HAI fallback

HAI has no official configuration. It therefore uses every frozen SWAT architecture, optimizer, learning-rate schedule, batch, epoch, patience and seed setting unchanged. Only `enc_in` changes from 51 to 79. HAI Test cannot select any hyperparameter or threshold.

## Resource estimate

Static instantiation of the source model, without forward/backward or data access, gives 1,190,689 parameters for both SWaT and HAI because channel modules are shared and the graph has no channel-specific parameter matrix. FP32 parameter payload is 4,762,756 bytes; an in-memory `torch.save(state_dict)` estimate is 4,772,111 bytes.

| Dataset | Input window | Adjacency sequence | Recon sequence | Batch-256 input | Batch-256 adjacency | Batch-256 recon |
|---|---:|---:|---:|---:|---:|---:|
| SWaT | 0.0249 MiB | 0.0893 MiB | 0.1121 MiB | 6.375 MiB | 22.860 MiB | 28.688 MiB |
| HAI21 | 0.0386 MiB | 0.2143 MiB | 0.1736 MiB | 9.875 MiB | 54.853 MiB | 44.438 MiB |

Parameters, gradients and Adam states alone have a lower-bound footprint of approximately 18.17 MiB. These estimates exclude intermediate convolutions, GRU states, top-k temporaries, gradients, CUDA workspace and allocator overhead.

Before training, B4 must execute one complete batch-256 forward/backward/Adam step on Normal Train for each dataset, recording actual parameters, tensor shapes and peak CUDA allocation. Any OOM must stop for protocol review; batch size must not be lowered silently.

No training, Test-data read, hyperparameter search, Test tuning, point adjustment, POT, commit or push occurred.
