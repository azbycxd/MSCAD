# CoMuT-D1R Historical Relation Diagnostic Re-evaluation

Status: **RELATION_REPRESENTATION_FAILED_ON_VALID_COMUT**.

## Controlled-rerun contract

The only D1 protocol change was the dataset path: `dataset/CoMuT_compat` was replaced by `dataset/CoMuT_historical_v1`. Provenance was verified as GutenTAG 1.0.0, commit `5638dadf9b1e83699ca317ce9eb4569a6c350064`, official configuration SHA256 `4D90D3871BFBB6C088F0AE4A3EAFE279D365FDBE4CACB4F85397908826570717`.

All D1 implementation functions were reused directly. All 13 cases used normal-only training, raw 80/20 train/validation splits, train-fit-only scaling and relation references, window 64, stride 1, causal endpoints `t≥63`, all channel pairs, seed 1, and the same AE/RawCorr/V2 formulas. Test labels were exposed only after all three score arrays were frozen.

## Overall results

| Method | Median AUROC | Mean AUROC | Median AUPR | Mean AUPR |
|---|---:|---:|---:|---:|
| Lightweight AE | 1.000000 | 0.999926 | 1.000000 | 0.972769 |
| RawCorr | 0.676580 | 0.724669 | 0.012751 | 0.136202 |
| V2 independent relation | 0.680844 | 0.726228 | 0.012751 | 0.131626 |

V2 exceeded AE AUROC by at least 0.10 in 0/13 cases; RawCorr did so in 0/13. The strongest V2 case was `rmj-large-mode-correlation-on-2` (AUROC 0.945171), and the weakest was `rmj-medium-mode-correlation-on-4` (0.624393).

## Category median AUROC

| Category | AE | RawCorr | V2 |
|---|---:|---:|---:|
| Short | 0.999899 | 0.634988 | 0.676892 |
| Medium | 1.000000 | 0.654471 | 0.638569 |
| Large | 1.000000 | 0.945171 | 0.934164 |
| Multiple-short | 0.999949 | 0.690595 | 0.688329 |

The fixed 64-point relation window performs strongly on large anomalies but substantially worse on short and medium categories. This is reported descriptively only; no window or aggregation search was performed.

## Score orientation

For both RawCorr and V2, anomaly median score exceeded normal median score in all 13/13 cases. Score orientation is therefore **PASS**. No score was flipped.

## Comparison with invalid compatibility D1

Historical minus compatibility median AUROC is **+0.282687** for RawCorr and **+0.269891** for V2. Recovering the historical generation semantics materially restores relation-signal discriminability, but not enough to satisfy the preregistered overall support gate.

The compatibility D1 remains preserved but is **INVALID_FOR_METHOD_CONCLUSION**, reason **INVALID_COMUT_GENERATION_SEMANTICS**. It may be used only as generation-failure evidence and as the fixed comparison source in this rerun.

## Scientific adjudication

RawCorr and V2 overall median AUROC are both below 0.70, so the unchanged preregistered Case C applies: **RELATION_REPRESENTATION_FAILED_ON_VALID_COMUT**. The current 64-point synchronous Pearson/max representation is not sufficiently reliable across the valid 13-case suite.

At the same time, the lightweight AE is nearly perfect, showing that historical CoMuT mode-correlation anomalies are also extremely visible to reconstruction. Because relation methods trail AE by roughly 0.32 median AUROC, they are not close enough for the Case D status. This result does not authorize tuning or a claim that all relation representations are ineffective.

All 15 protocol assertions passed. Historical data hashes matched before and after execution. Published results were not used for parameter selection or adjudication.

**HISTORICAL COMUT SEMANTICS; 13/13 CASES; NORMAL-ONLY TRAINING; NO TEST TUNING; NO PA; NO V3; NO FUSION; NO COMMIT; NO PUSH.**
