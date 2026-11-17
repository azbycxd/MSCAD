# Chapter 6.3 result audit

## 1. Scope and preconditions

Audit target: `docs/thesis/chapters/chapter6_experiments_part2.md` and `docs/thesis/chapter6_3_result_trace.csv`.

| Gate | Result |
|---|---|
| `THESIS_CHAPTER6_PART1_READY` | PASS |
| `THESIS_EXPERIMENTAL_PHASE_CLOSED` | PASS |
| `THESIS_CONTRIBUTION_STRUCTURE_FROZEN` | PASS |

Only Section 6.3 was created. Sections 6.1/6.2, method definitions, evaluation protocols and historical outputs were not modified.

## 2. Result-source verification

- Primary source: `outputs/thesis_tables/table_1_strict_baselines.csv`.
- Independent protocol cross-check: `outputs/thesis_evaluation/literature_compatible/final/table_a_historical_strict.csv`.
- All 12 dataset/method rows were present.
- AUROC, AUPR, F1 and FPR matched the independent Historical Strict table exactly.
- Every independent row was marked `HISTORICAL_STRICT`, `Test_tuning=false`, `PA=false`.
- Precision and Recall were read directly from the frozen strict-baseline CSV.
- Table 6-3 was generated from source rows and formatted to six decimals; it was not manually transcribed.
- The trace contains 72 metric rows: 2 datasets × 6 methods × 6 metrics.

Derived arithmetic is limited to subtraction of frozen values: HAI DUAL minus GDN AUROC/AUPR/F1 equals 0.038691/0.072419/0.048503, and HAI TAIL minus Ours-DUAL AUPR equals 0.047703. These values are labeled `derived arithmetic` in the chapter and are not new model metrics.

## 3. Required self-audit

| Check | Result | Finding |
|---|---|---|
| A. Oracle/PA used as Primary | PASS | No Oracle, PA or Test-best value appears in Section 6.3 or Table 6-3. |
| B. Common-mask result overwrote Primary | PASS | Only Historical Strict source rows are used. |
| C. SWaT DUAL/TAIL negative result hidden | PASS | Both low AUPR values and their boundary interpretation are reported explicitly. |
| D. TAIL described as universal gain | PASS | HAI gain is dataset-scoped; SWaT non-gain is retained. |
| E. HAI selected-metric advantage inflated to all metrics | PASS | DUAL AUROC, TAIL AUPR/F1 and GDN lower FPR are separately stated. |
| F. CoMuT/controlled/HIGH19 introduced early | PASS | No mechanism result or event-subgroup result is used; Section 6.5 is referenced only as future analysis. |
| G. Multi-seed mean replaced formal seed | PASS | Table 6-3 contains the frozen formal-seed Historical Strict values only. |
| H. Every number traced to frozen artifact | PASS | All 72 table cells are represented in `chapter6_3_result_trace.csv`; deterministic deltas are separately documented. |

## 4. Writing-boundary checks

- The section follows fact → interpretation → boundary for both datasets.
- It does not claim SOTA, universal superiority or statistical significance.
- It does not use “显著优于”, “全面最佳” or “所有指标最优”.
- No duplicate bar chart was created; the primary result table is sufficient for this section.
- The chapter contains Section 6.3 only and does not copy Sections 6.1/6.2.

## 5. Negative-action assertions

NO TRAINING; NO INFERENCE; NO NEW EXPERIMENT; NO RESULT RECALCULATION; NO NEW METRIC; NO METHOD CHANGE; NO CLAIM INFLATION; NO NEW LITERATURE SEARCH; NO COMMIT; NO PUSH.

## 6. Final status

`THESIS_CHAPTER6_PART2_READY`
