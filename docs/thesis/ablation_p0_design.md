# Thesis Ablation-ABL-P0 design freeze

Date: 2026-09-23  
Status: `ABLATION_PROTOCOL_READY`  
Evidence class for every new variant: `RETROSPECTIVE_ABLATION`

## 1. Freeze gate

`outputs/thesis_registry/registry_manifest.json` was read before this design was created and reports `THESIS_PRE_ABLATION_STATE_FROZEN`. The following main-method elements remain frozen:

- Ours, REC, Unified Reliability Gate, Self/Cross lags, Ridge alpha, Dual score, and tail calibration;
- primary datasets: SWaT 2015 Official Physical and HAI 21.03;
- WADI.A2 role: `DOMAIN_SHIFT_STRESS_TEST` only;
- primary evaluation: Normal Validation q99, no Point Adjustment, no Test-best threshold;
- ablation results may isolate or remove components, but may not redefine the frozen main method.

SWaT, HAI, WADI, Historical CoMuT, and the controlled SWaT corruption results have already been observed. New rows therefore cannot be described as untouched-Test or pre-registered evidence. Variant definitions below come only from source structure, frozen normal-data logic, and nested component removal. They must not be revised after new Test results are produced.

## 2. Minimal sufficient matrix

The machine-readable plan is `outputs/thesis_ablation/ablation_execution_matrix.csv`. It contains 35 rows: 31 retained and 4 explicitly dropped. The four permitted action classes are `EXISTING_RESULT`, `EVAL_ONLY`, `RIDGE_REFIT_ONLY`, and `REC_RETRAIN_REQUIRED`.

### Innovation-1 — lightweight REC

The formal REC is `BackboneAE`, not `BackboneAEPlus`. Two internal variants are justified because they each replace one real structural choice without changing widths, optimizer, loss, splits, window, scaler, threshold, or seed.

| ID | Variant | Isolated question | Decision |
|---|---|---|---|
| I1-R0 | Frozen REC | Reference accuracy and efficiency | KEEP; existing result |
| I1-R1 | Replace the three depthwise temporal `k=3` convolutions with `k=1` | Does the effective temporal receptive field add value beyond instantaneous channel mixing? | KEEP; retrain SWaT and HAI |
| I1-R2 | Replace each depthwise-separable block with a same-width dense `k=3` Conv1d block, retaining BN/GELU/dropout | Does depthwise-separable factorization retain performance while providing the claimed lightweight efficiency? | KEEP; retrain SWaT and HAI |

No hidden-width sweep, latent-width sweep, residual version, or `BackboneAEPlus` comparison is admitted. Those would change more than one design choice or test capacity rather than the frozen REC structure.

### Innovation-2 — Dual-View

Final reference:

- Self lags: `[1,5,10,20,30]`;
- Cross lags: `[0,1,5,10,20,30]`;
- Ridge alpha: `1`;
- per-target raw score: `abs(pred_self - pred_cross)`;
- per-target calibration: Normal Validation median and `max(1.4826*MAD, 1e-6)`;
- global score: maximum calibrated target disagreement;
- target selection: frozen Unified Reliability Gate.

The nested short-lag definition is frozen as Self `[1]` and Cross `[0,1]`. This is the smallest variant that preserves an autoregressive Self view, instantaneous Cross context, and one-step Cross dynamics while removing all long lags. It is not the outcome of a lag search.

| ID | Definition | Decision | Execution logic |
|---|---|---|---|
| D0 | Final Dual | KEEP | Existing SWaT, HAI, Historical CoMuT, and SWaT-controlled results |
| D1 | Self `[1]`, Cross `[0,1]`; alpha, score, candidate rule, and gate logic unchanged | KEEP | Refit Ridge only; reapply the exact frozen gate logic to short-lag models |
| D2 | All base candidates; no Reliability Gate; full frozen lags and alpha | KEEP | Refit Ridge only; no Attack/Test target selection |
| D3-A | Candidate only | KEEP | Reuse D2 candidate-level score matrix |
| D3-B | Candidate + positive Discovery Self/Cross skill | KEEP | Evaluation-only subset |
| D3-C | D3-B + positive full Validation and both Validation halves | KEEP | Evaluation-only subset |
| D3-D | D3-C + median/q95 disagreement stability = Final | KEEP | Identical endpoint to D0 |

D3 is retained for SWaT and HAI because the candidate-level Ridge state needed by D2 already contains the required normal-only diagnostics. Its incremental cost is small, and it directly attributes the gate. It is not repeated on CoMuT because CoMuT has no compatible industrial stage/process gate semantics. It is not expanded into four controlled-corruption tables because D0/D1/D2 already answer that mechanism question.

### Mechanism datasets

- Historical CoMuT: reuse D0; run D1 across the same 13 official historical cases. D2/D3 are dropped because D0 already uses all channels and no industrial gate exists.
- SWaT controlled relation-breaking: reuse D0; evaluate D1 using the SWaT D1 Ridge state and frozen donors/samples; evaluate D2 using the SWaT all-candidate state. No new corruption or donor is allowed.
- Metrics are AUROC and AUPR first. These rows establish relation-change sensitivity, not industrial leaderboard performance.

### Real datasets

- HAI21: D0, D1, D2, and D3 are mandatory primary ablations. Report AUROC, AUPR, q99 F1/FPR, target count, and HIGH19 median hit.
- SWaT: D0, D1, D2, and D3 are mandatory boundary evidence. Report the same strict point metrics and 36-event hit count.
- WADI: no new fitting or evaluation. Existing F0/F1/F3/F4 results may appear only as zero-cost domain-shift stress evidence.

## 3. Fusion trace and decisions

| ID | Verified historical definition | Decision |
|---|---|---|
| F0 | REC endpoint mean absolute reconstruction error; Normal Validation q99 | KEEP; existing |
| F1 | DUAL maximum Normal-Validation-calibrated Self/Cross disagreement; Normal Validation q99 | KEEP; existing |
| F2 | No formal historical uncalibrated/simple fusion artifact was found | DROP; do not invent a definition |
| F3 | `max(max(0,(REC-med_REC)/scale_REC), max(0,(DUAL-med_DUAL)/scale_DUAL))`, with median/MAD fitted on Normal Validation; q99 | KEEP; historical `OLD_FUSION` |
| F4 | Per-branch empirical Normal-tail probability from Calibration-A, evidence `-log10(p_tail)`, fusion `max(E_REC,E_DUAL)`, Threshold-B q99 | KEEP; final frozen fusion |

The existing evidence answers the five required mechanism questions:

1. Raw heterogeneous scores are not directly comparable because REC and DUAL have different units, centers, spreads, and tails; no raw-max F2 is admitted.
2. Robust-z does not equalize upper tails: on Audit-B, F3 top-1 normal-tail dominance is 100% DUAL for SWaT, HAI21, and WADI.A2.
3. F4 maps both branches to empirical Normal-tail evidence. The top-1 normal-tail shares become REC/DUAL 65.19%/34.81% on SWaT, 59.15%/40.85% on HAI21, and 51.78%/48.22% on WADI.A2.
4. HAI complementarity appears after comparable evidence calibration: F4 AUPR is 0.617897 versus 0.536294 for REC and 0.570193 for DUAL.
5. SWaT remains a boundary: F4 AUPR is 0.206747 and q99 FPR is 0.692531; tail comparability does not repair the frozen threshold/domain instability.

## 4. Protocol and selection firewall

- Primary ablation tables use only strict legal endpoints and Normal Validation q99 thresholds.
- No PA, POT, Test-best threshold, Test-normal adaptation, or Oracle result can select a variant.
- Literature-compatible Oracle/PA results remain supplementary only.
- Test has already been seen; every newly executed row must carry `RETROSPECTIVE_ABLATION` in its manifest.
- After execution, variants are reported whether favorable or unfavorable. No second lag, gate, fusion, or architecture variant may be added in response to Test performance.

## 5. Cost and recommended execution order

Planning estimate, not a measured runtime:

- 4 REC training jobs: I1-R1 and I1-R2 on SWaT and HAI;
- 5 Ridge fitting batches: SWaT D1, HAI D1, 13-case CoMuT D1, SWaT D2/D3, HAI D2/D3;
- 4 incremental evaluation-only jobs: controlled D1, controlled D2, SWaT D3 subsets, HAI D3 subsets;
- sequential total: approximately 43–78 minutes, including artifact QA and table fill.

Recommended first real execution batch is CPU-only Innovation-2: SWaT and HAI D1 short-lag fits, SWaT and HAI D2/D3 all-candidate fits, reuse the two SWaT states for controlled D1/D2, then run CoMuT D1. Only after those artifacts are frozen should the four GPU REC retraining jobs begin.

## 6. Final keep/drop recommendation

Keep:

- REC R0/R1/R2 on SWaT and HAI;
- Dual D0/D1/D2 on SWaT and HAI;
- D3 decomposition on SWaT and HAI;
- D0/D1 on Historical CoMuT;
- D0/D1/D2 on controlled SWaT relation breaking;
- Fusion F0/F1/F3/F4 on SWaT and HAI, plus zero-cost WADI stress rows.

Drop:

- CoMuT D2 and D3;
- controlled SWaT D3 expansion;
- F2 uncalibrated/simple fusion;
- all hidden-width, latent-width, lag, alpha, threshold, fusion-weight, and WADI refit sweeps.

No training, inference, score computation, file deletion, Git add, commit, or push was performed in ABL-P0.
