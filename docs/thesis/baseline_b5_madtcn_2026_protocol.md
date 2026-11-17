# Baseline-B5-P0 MAD-TCN 2026 Protocol and Reproducibility Audit

Status: `MAD_TCN_2026_OFFICIAL_CODE_INCOMPLETE`

This phase audits the official MAD-TCN repository without training, reading SWaT/HAI Test data, inventing an anomaly score, or freezing guessed hyperparameters. The repository is official, but its current public snapshot is insufficient for a defensible unified baseline run.

## Repository identity

- Official repository: <https://github.com/qianmo2001/MAD-TCN>
- Local path: `external/baselines/MAD-TCN`
- Branch: `main`
- HEAD: `24d42ee0e0fc5a5460d437026cf4b89f05381293`
- Worktree status after clone: clean
- Paper: *MAD-TCN: Time series anomaly detection via multi-scale adaptive dependency temporal convolutional network*, Neurocomputing 675 (2026), article 132954, DOI `10.1016/j.neucom.2026.132954`

The README identifies the repository as the official implementation and links the published paper. The tracked tree contains only:

```text
README.md
img/all.png
models/MAD_TCN.py
```

## Reproducibility matrix

| Component | Status | Evidence |
|---|---|---|
| Model definition | FOUND | `models/MAD_TCN.py` defines the convolutional blocks and `MAD_TCN`. |
| Dataset loader | MISSING | No loader or dataset-binding code. |
| Preprocessing | MISSING | No scaler or normalization implementation. |
| Training | MISSING | No entry point, optimizer schedule, epoch loop, checkpointing, or seed setup. |
| Validation | MISSING | No split or model-selection implementation. |
| Test/evaluation | MISSING | No metric, threshold, point-adjustment, or endpoint-alignment implementation. |
| Config/hyperparameters | MISSING | No config files, CLI parser, or reproduction script. |
| Training objective/loss | MISSING | No target construction or loss is present. |
| Anomaly score | MISSING | No score formula, orientation, or postprocessing is present. |

The machine-readable matrix is stored in `outputs/thesis_baselines/madtcn_2026/reproducibility_matrix.csv`.

## Method structure verified from source

The following claims are grounded in the released model code rather than inferred from the method name:

1. `TCN_DW_L` is a local temporal branch with three grouped, weight-normalized `Conv1d` blocks, kernel size 3, and dilation 1 at all levels.
2. `TCN_DW_G` is a global temporal branch with the same three-level channel schedule and dilations 1, 2, and 4.
3. The two branches are concatenated across the latent feature dimension and batch-normalized.
4. `ConvFFN1` uses 1×1 grouped convolutions with `groups=nvars`, preserving per-variable groups while transforming latent features.
5. After permuting variable and latent axes, `ConvFFN2` uses `groups=dmodel`, enabling cross-variable mixing within each latent feature group.
6. A learnable per-variable sigmoid gate combines the two views as `alpha*x1 + (1-alpha)*x2`.
7. The head applies `Linear(dmodel*n_window, 1)` plus sigmoid independently to each variable and returns `[B, nvars, 1]`.

The paper abstract describes local/global temporal branches, depthwise separable dilated convolution, multi-scale integration, cross-variable feedforward convolution, and adaptive gated dependency modeling. These descriptions are consistent with the released architecture. The exact training target and anomaly-score use of the output are not released and therefore remain unknown.

## Source-level blocker

`MAD_TCN.__init__` reads `args.alpha_value` at `models/MAD_TCN.py:138`, but the repository neither defines nor imports `args`. Consequently, the model cannot be instantiated directly from the released source without supplying an external global or modifying the implementation. This is not repaired in the audit.

The constructor also requires `dmodel`, `nvars`, and `n_window`; the repository provides no SWaT call site or configuration values. The paper's approximately 0.026M parameter statement is retained as a paper claim only. A source-derived SWaT/HAI parameter count is not reported because doing so would require guessed constructor arguments.

## Official SWaT hyperparameters

| Item | Evidence-backed value |
|---|---|
| Window size | Missing |
| Patch size / stride | Not represented in released source and not documented |
| `dmodel` | Missing |
| Local/global kernels | Source fixes kernel size 3 |
| Local dilations | Source fixes `[1,1,1]` |
| Global dilations | Source fixes `[1,2,4]` |
| Blocks | Three convolution levels per temporal branch |
| `ffn_ratio` | Constructor default 2; official SWaT value unverified |
| Adaptive gate initialization | Missing because `args.alpha_value` is external |
| Dropout | Constructor default 0.1; internal temporal blocks default 0.2; official SWaT value unverified |
| Learning rate | Missing |
| Batch size | Missing |
| Epochs / patience | Missing |
| Optimizer / weight decay | Missing |
| Seed | Missing |

Code defaults are not treated as the official SWaT experiment configuration. There is therefore no evidence-based HAI fallback to freeze. The intended fallback rule remains: use the verified SWaT architecture and optimizer settings unchanged and vary only the input dimension. It is blocked until the SWaT settings are recovered.

## Normalization, loss, score, and leakage

Normalization is not auditable because preprocessing code is absent. The repository does not establish Train-only fitting, Validation/Test transform-only behavior, clipping, or Test adaptation.

The model source exposes a sigmoid prediction head, but no target construction or loss. It is therefore unsafe to label the task as reconstruction or forecasting beyond saying that the source returns a per-variable prediction tensor.

The anomaly score is missing. Residual type, channel aggregation, temporal aggregation, score orientation, and postprocessing cannot be reconstructed from the repository. No substitute is introduced.

The following behaviors are `UNKNOWN`, not assumed absent:

- Test-label checkpoint or metric selection;
- Test-derived thresholding;
- Test-set score normalization;
- Test-label hyperparameter selection;
- point adjustment;
- POT/SPOT;
- Test adaptation.

Because the evaluation path is absent, the unified Normal Validation q99 protocol cannot yet be applied without inventing the native score.

## Window fairness and dataset scope

SWaT and HAI 21.03 remain the intended primary datasets. WADI is not added to the main performance table. If an official window is recovered and is method-essential, it should be preserved; unified comparisons would then use the common legal endpoint mask beginning at `max(64, official_window)` within each continuous segment/file. This remains a candidate rule, not a frozen runnable protocol.

## Relationship to Innovation-2

MAD-TCN is conceptually relevant because it explicitly combines local/global temporal structure with learned cross-variable dependency processing. The thesis method instead keeps Self View and Cross View prediction evidence separable and evaluates their disagreement. MAD-TCN is therefore relevant to the lightweight design contribution and core method innovation 1, and would be a direct lightweight dependency-aware baseline only after its official objective, anomaly score, and reproducible SWaT setup are recovered.

## Position among existing baselines

| Baseline | Thesis role |
|---|---|
| LSTM-AE | Classic reconstruction baseline |
| USAD | Adversarial reconstruction baseline |
| GDN | Graph-dependency forecasting baseline |
| MAD-TCN 2026 | Lightweight adaptive-dependency candidate; official code currently incomplete |
| LatentFlow 2026 | Dynamic-dependency candidate; full reproduction not completed because of computational cost |
| Park-Shim 2026 | Dependency plug-in candidate; excluded from the strict main table because of Test-driven weight selection and source conflicts |

## Decision

The public repository is authentic but partial. It permits a source-grounded architecture audit, not a faithful baseline reproduction. B5 training is blocked. No resource preflight is scheduled because exact constructor arguments, target/loss, and score are missing and the constructor is not independently runnable.

Final status: `MAD_TCN_2026_OFFICIAL_CODE_INCOMPLETE`.

Controls: `NO TRAINING`, `NO TEST DATA READ`, `NO HP SEARCH`, `NO SCORE INVENTION`, `NO TEST TUNING`, `NO PA`, `NO POT`, `NO COMMIT`, `NO PUSH`.

## Sources

- Official repository: <https://github.com/qianmo2001/MAD-TCN>
- Published paper: <https://www.sciencedirect.com/science/article/pii/S0925231226003516>
- DOI: <https://doi.org/10.1016/j.neucom.2026.132954>
