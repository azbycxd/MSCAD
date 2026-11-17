# Baseline-B6-P0: CCM-TAD ICML 2026 protocol audit

## Decision

**Status: `CCM_TAD_2026_PROTOCOL_REQUIRES_REVIEW`.**

The official source is complete enough to identify the model, training loss, scores, datasets, and executable evaluation chain. It is not safe to start unified SWaT/HAI training yet: the released runner selects checkpoints with Test PF1/SF1, the point and sequential thresholds are optimized against Test labels, the sequential significance parameter `alpha` is likewise Test-label selected, and the official centered 50-point moving average consumes future observations.

No model was trained and no Test array was opened during this audit.

## Identity and provenance

- Paper: *Cluster-Aware Causal Mixer for Online Anomaly Detection in Multivariate Time Series*.
- Authors: Md Mahmuddun Nabi Murad and Yasin Yilmaz.
- Venue: ICML 2026.
- Official organization repository: `https://github.com/Secure-and-Intelligent-Systems-Lab/CCM-TAD`.
- Local path: `external/baselines/CCM-TAD`.
- Branch / commit: `main` / `3a8bc8263b648a707cd0658bf4f06517fb9e52c5`.
- Worktree: clean after clone.
- The organization repository is a GitHub fork of the author repository. A read-only `ls-remote` check found the same HEAD in both locations.
- `scripts/config/config.xlsx` SHA256: `D53E8A5609E71A431CEDCF8AB22E341D1726700286B8801DDE54F60CF838D7FE`.

The arXiv paper identifies the same code URL, describes normal-only training, reconstruction of the latest observation, Train-derived Min–Max scaling, correlation-profile clustering, causal mixing, and sequential evidence. The ICML 2026 official paper list contains the exact title. These checks do not rely on the repository README alone.

## Reproducibility chain

| Component | Status | Evidence |
|---|---|---|
| Dataset loaders | FOUND | `data_provider/`, including `Dataset_SWAT.py` |
| Experiment runner | FOUND | `exp/exp_anomaly.py` |
| Model | FOUND | `models/CCM_TAD.py`, `models/CausalLinearLayer.py` |
| Run entry point | FOUND | `run_anomaly.py` |
| SWaT commands | FOUND | `scripts/CCM_TAD/SWAT.sh` |
| Dataset configuration | FOUND | `scripts/config/config.xlsx` |
| Clustering | FOUND | `utils/feature_clustering.py` |
| PF1/SF1 evaluation | FOUND | `utils/anomaly_evaluation.py`, `utils/ccm_tad_anomaly_evaluation/` |
| HAI configuration | MISSING | No official HAI loader or config |
| Leakage-free selection policy | MISSING | Released selection is Test-label driven |

The machine-readable matrix is in `reproducibility_matrix.csv`.

## Method structure verified from source

1. **Channel dependency representation.** For `featureSimilarity=1`, the implementation computes the absolute Pearson correlation matrix, regards each channel row as its correlation profile, calculates cosine similarity between profiles, forms a normalized graph Laplacian embedding, and applies K-Means with seed 42.
2. **Cluster-specific embedding.** `MultiHeadEmbedding` creates one linear embedding per cluster. Output widths are allocated proportionally to cluster size and concatenated to `d_model`.
3. **Causal mixer.** Each mixer has two masked temporal linear layers with GELU. `CausalLinearLayer` masks weights so each position depends only on present/past positions. A two-layer embedding mixer performs channel/latent mixing, with residuals and batch normalization.
4. **Prediction target.** With `prediction_special`, the model reconstructs the final observation already contained in `[t-23,t]`; it is not next-point forecasting.
5. **Raw score.** Mean squared endpoint reconstruction error across channels.
6. **Sequential evidence.** Validation losses define an empirical upper-tail p-value. The implementation uses `beta_t=log(alpha/(p_t+epsilon))`, accumulates nonnegative evidence, resets after `delta=5` consecutive negative evidence values, then performs start/end boundary correction.

This is a direct Innovation-2 baseline because CCM-TAD explicitly models cross-channel dependency structure and its temporal evolution, whereas our method compares Self-View and Cross-View predictions and interprets their disagreement.

## Official SWaT configuration

The exact SWaT row in `config.xlsx` overrides parser defaults:

| Field | Frozen official value |
|---|---:|
| `seq_len` / `pred_len` | 24 / 1 |
| channels | 51 |
| clusters / heads | 6 |
| dependency rule | correlation-profile similarity (`featureSimilarity=1`) |
| `d_model` | 128 |
| causal mixers | 1 |
| temporal / embedding expansion | 1 / 1 |
| dropout / embedding dropout / input dropout | 0.05 / 0.05 / 0 |
| optimizer | Adam |
| learning rate | 0.00498096466936417 |
| weight decay | 0 |
| batch / epochs | 512 / 30 |
| seed | 42 |
| `delta` | 5 |

The parser declares patience 8, but the imported `EarlyStopping` object is never instantiated or called. Official effective training therefore runs 30 epochs, with the retained checkpoint chosen by Test F1.

## Leakage and protocol audit

### Confirmed Test-driven behavior

- `exp/exp_anomaly.py` evaluates Test every epoch, sets the optimization objective to negative Test F1, and saves the checkpoint when that value improves.
- PF1 sweeps thresholds over Test scores and selects the threshold with maximum Test-label F1.
- SF1 calls `get_top_best_threshold(test_score, label, ...)`, adds fixed alpha candidates, evaluates every alpha against Test labels, and keeps the maximum Test-label F1.
- For each sequential score, 1000 Test-score quantile thresholds are evaluated against Test labels. Boundary correction is applied within this selection path.
- The paper defines Best-F1 as a joint maximization over `alpha` and `h`; its hyperparameter appendix also reports F1-guided optimization of the cluster count.

### Not observed in the active PF1/SF1 path

- No Test-fitted Min–Max scaler: the loader fits on the normal training prefix for ordinary train/validation/Test instances.
- No Test-data channel clustering: the constructor obtains clusters from `train_full`, which is normal data only. It nevertheless includes the nominal validation tail, so it is not compliant with the thesis Train-only rule.
- Classic point adjustment, affiliation F1, range F1, and VUS exist as optional utilities, but the active selected-key list does not report them.
- Raw MSE is not normalized with Test statistics. Sequential p-values use nominal validation losses.

### Online-causality conflict

The loader applies `np.convolve(..., ones(50)/50, mode='same')` independently to every full split. This is a centered filter and therefore includes future samples for most positions. The model mixer is causally masked, but the input preprocessing is not online-causal. Keeping it violates the stated online setting; changing it would be a source-protocol adaptation.

## Unified thesis adaptation assessment

The model architecture can remain intact with the official window 24. Fair comparison can use the thesis common endpoint mask (`local endpoint >=64`) while each score still uses `[t-23,t]`. Train-only MinMax `[-1,1]`, Train-only clustering, Normal-Validation checkpoint selection, the official raw mean-channel MSE, and Normal-Validation q99 are coherent for a **model-only point-score ablation**.

They are not yet a complete CCM-TAD protocol. The sequential component is a stated contribution, but the released executable path has no fixed, source-backed, label-free `alpha`; choosing one now would invent a new policy. The centered moving average creates a second unresolved choice. HAI fallback is therefore not frozen. Once SWaT policy is approved, HAI must inherit every architecture, optimizer, preprocessing, alpha, and decision rule unchanged except for input dimension and Train-derived clusters.

## Publication evaluation versus thesis evaluation

| Aspect | OFFICIAL publication/repository | THESIS STRICT requirement |
|---|---|---|
| Checkpoint | Best Test PF1/SF1 | Lowest Normal Validation loss |
| Point decision | Test-best PF1 threshold | Normal Validation q99 |
| Sequential decision | Test-label search over `alpha` and `h`, with boundary correction | Unresolved; no Test tuning may be used |
| Primary reporting | PF1, SF1; affiliation/range/VUS utilities and supplements | AUROC, AUPR, Validation-q99 Precision/Recall/F1/FPR |
| Point adjustment | Not active in the audited selected-key path | Prohibited |
| Test handling | Labels repeatedly used during training/evaluation selection | Test labels opened only after model, score, checkpoint, and operating point freeze |

If formal training is later authorized, raw Test scores must be preserved unchanged. Literature-compatible metrics may then be computed only as clearly labelled supplementary results and may not replace the strict primary table.

## Static resource estimate

No forward/backward/training step was run. A static instantiation used only the repository's normal SWaT `train.npy`; `test.npz` was not opened.

- SWaT Train-derived official cluster sizes: `[5,15,10,12,6,3]`.
- SWaT parameters: 43,297; FP32 parameter bytes: 173,188; raw batch-512 input window: 2,506,752 bytes.
- HAI shape-only provisional estimate, using balanced six-cluster sizes because HAI clustering is not frozen: 47,253 parameters; 189,012 FP32 bytes; raw batch-512 input window: 3,883,008 bytes.

The HAI count is not a final model size. After protocol review, resource preflight should instantiate actual Train-derived clusters and run exactly one frozen batch; any OOM must stop without an automatic batch change.

## Required review before B6 training

1. Decide whether to benchmark the full sequential method or clearly label a raw-MSE model-only ablation.
2. If full sequential evaluation is required, preregister an evidence-backed validation-only `alpha` and operating threshold without inspecting Test labels.
3. Decide whether centered smoothing is removed, replaced causally, or retained only as a non-online reproduction; record the choice as an adaptation.
4. Confirm cluster fitting uses only Normal Train, not `train_full` including Validation.
5. Only after these decisions freeze the HAI no-Test fallback and run resource preflight.

## Sources

- Official ICML 2026 paper listing: https://icml.cc/Downloads/2026
- Paper and appendices: https://arxiv.org/abs/2506.00188
- OpenReview record: https://openreview.net/forum?id=R6JV2WOftz
- Official repository: https://github.com/Secure-and-Intelligent-Systems-Lab/CCM-TAD

**NO TRAINING. NO TEST DATA READ. NO HP SEARCH. NO SCORE INVENTION. NO PF1/SF1 TUNING. NO PA. NO POT. NO TEST TUNING. NO COMMIT. NO PUSH.**
