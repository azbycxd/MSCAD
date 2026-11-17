# CoMuT-D1 Correlation-Anomaly Discriminability Diagnostic

Historical status: **INVALID_FOR_METHOD_CONCLUSION**.

Reason: **INVALID_COMUT_GENERATION_SEMANTICS**. Subsequent N1 auditing showed that this compatibility generation did not reproduce CoMuT's normal correlation semantics (`median |Pearson| ≈ 0.041`, versus ≈0.9975 under the historical official configuration). The results below are retained as generation-failure evidence and must not support a conclusion about AE, RawCorr, or V2 effectiveness.

Original run status before the generation audit: `CURRENT_RELATION_REPRESENTATION_NOT_SUPPORTED_EVEN_ON_COMUT`.

## Frozen protocol

All 13 compatibility CoMuT cases were evaluated. Each case used its complete 10,000-point `train_no_anomaly.csv`; raw rows `[0,8000)` formed `train_fit` and `[8000,10000)` formed normal validation, with no window crossing that boundary. The scaler and both relation references used only `train_fit`. Test labels were exposed only after AE, RawCorr, and V2 score arrays were complete.

All methods used causal trailing windows of 64 points, stride 1, and the same 9,937 endpoints (`t=63..9999`). Endpoint-and-label SHA256 identities match across all three methods for every case. Metrics are point-wise AUROC and AUPR with `anomaly=1`; there is no threshold search, F1, point adjustment, pair selection, fusion, or V3.

## Overall results

| Method | Median AUROC | Mean AUROC | Median AUPR | Mean AUPR |
|---|---:|---:|---:|---:|
| Lightweight AE | 0.406999 | 0.394016 | 0.007684 | 0.008040 |
| RawCorr | 0.393893 | 0.387709 | 0.006990 | 0.008429 |
| V2 independent relation | 0.410952 | 0.428949 | 0.011316 | 0.010172 |

V2 exceeded AE AUROC by at least 0.10 in 5/13 cases; RawCorr did so in 3/13. The strongest V2 case was `rmj-short-mode-correlation-on-3` (AUROC 0.874227), and the weakest was `rmj-medium-mode-correlation-on-4` (0.129201). These identities are descriptive, not a basis for selecting cases or parameters.

## Category results

| Category | Cases | AE median AUROC | RawCorr median AUROC | V2 median AUROC |
|---|---:|---:|---:|---:|
| Short | 3 | 0.336013 | 0.173229 | 0.196625 |
| Medium | 3 | 0.422806 | 0.393893 | 0.390906 |
| Large | 3 | 0.417232 | 0.346657 | 0.410952 |
| Multiple-short | 4 | 0.371062 | 0.512769 | 0.511957 |

The preregistered short-anomaly dilution pattern was not observed: short was poor, but medium and large were also poor. The result is therefore `NO_PREREGISTERED_WINDOW_DILUTION_PATTERN`; window 64 was not changed.

## Answers to the research questions

1. The fixed lightweight AE does not reliably discriminate these compatibility CoMuT anomalies: median AUROC is 0.4070 and median AUPR is 0.00768.
2. Explicit synchronous relation scoring is not clearly stronger. RawCorr is slightly below AE in median AUROC, while V2 is only 0.00395 higher and fails the preregistered support gate.
3. Under this exact compatibility generation and frozen protocol, the present V2 synchronous Fisher/MAD/max representation is not supported even on CoMuT. The SWaT failure therefore cannot be attributed solely to benchmark mismatch. This does not establish that all correlation-anomaly representations are ineffective.

All 15 protocol assertions passed, including train-only fit scope, label isolation, endpoint alignment, all-pair use, seed 1, finite scores, and fixed label direction. Published Series2Graph++ values remain **REFERENCE_ONLY / NOT_DIRECTLY_COMPARABLE** and were not used for adjudication.

**COMUT NORMAL-ONLY TRAINING; 13/13 CASES; NO TEST TUNING; NO PA; NO V3; NO FUSION; PUBLISHED RESULTS REFERENCE ONLY; NO COMMIT; NO PUSH.**
