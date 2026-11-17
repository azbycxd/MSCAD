# HAI 21.03 Final Unified Dual-View Evaluation

Unified reliability used exactly positive Self/Cross skill on Discovery, full Validation, and both Validation halves, plus median ratio <=2 and q95 ratio <=3. The 54 targets and SHA `a78e820567eaa15dd5a0cd2a9f6491d04102c0a50732a4de7aa918f1b20be5bb` were written before Test access. The frozen prior REC checkpoint was reused after all Normal source, schema, split, scaler-row, and architecture checks passed.

| Method | AUROC | AUPR | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|
| REC | 0.852899 | 0.536294 | 0.459109 | 0.601095 | 0.520594 | 0.016133 |
| DUAL | 0.916334 | 0.570193 | 0.554350 | 0.679446 | 0.610556 | 0.012443 |
| FUSION | 0.907834 | 0.576716 | 0.559176 | 0.694423 | 0.619503 | 0.012471 |

HIGH-19 median hit: REC 0.650190, DUAL 0.841085, FUSION 0.841085. DUAL beat REC by >=0.10 in 9/19.

| HIGH event | REC hit | DUAL hit | FUSION hit | DUAL top target/process |
|---|---:|---:|---:|---|
| T2-E02 | 0.838863 | 0.824645 | 0.834123 | P1_PIT01 / P1 |
| T2-E04 | 0.536680 | 0.563707 | 0.563707 | P1_B3005 / P1 |
| T2-E06 | 0.085938 | 0.511719 | 0.511719 | P1_LIT01 / P1 |
| T2-E08 | 0.632184 | 0.869732 | 0.869732 | P1_LIT01 / P1 |
| T2-E10 | 0.199525 | 0.066508 | 0.066508 | P1_PIT01 / P1 |
| T2-E12 | 0.552632 | 0.506579 | 0.513158 | P2_CO_rpm / P2 |
| T2-E13 | 0.669291 | 0.854331 | 0.854331 | P1_LIT01 / P1 |
| T2-E15 | 0.410596 | 0.390728 | 0.390728 | P2_CO_rpm / P2 |
| T3-E02 | 0.940618 | 0.990499 | 0.990499 | P1_PIT01 / P1 |
| T3-E08 | 0.721569 | 0.976471 | 0.976471 | P1_LIT01 / P1 |
| T4-E01 | 0.625984 | 0.909449 | 0.909449 | P1_LIT01 / P1 |
| T4-E02 | 0.729008 | 0.744275 | 0.744275 | P1_B3005 / P1 |
| T4-E03 | 0.650190 | 0.893536 | 0.893536 | P1_LIT01 / P1 |
| T4-E04 | 0.713178 | 0.841085 | 0.841085 | P1_LIT01 / P1 |
| T5-E02 | 0.387597 | 0.519380 | 0.527132 | P1_LIT01 / P1 |
| T5-E03 | 0.722656 | 0.929688 | 0.933594 | P1_LIT01 / P1 |
| T5-E06 | 0.411765 | 0.385621 | 0.392157 | P2_CO_rpm / P2 |
| T5-E10 | 0.942748 | 0.984733 | 0.984733 | P1_LIT01 / P1 |
| T5-E12 | 0.946565 | 0.965649 | 0.969466 | P1_LIT01 / P1 |

50-event hits >=0.10: {'REC': 45, 'DUAL': 44, 'FUSION': 46}. Innovation-2 gates `{'H1': True, 'H2': True, 'H3': True, 'H4': True}` produce **DUAL_VIEW_REAL_HAI_UNIFIED_SUPPORTED**. Fusion gates `{'F1': True, 'F2': True, 'F3': True, 'F4': True}` produce **NORMAL_CALIBRATED_MAX_FUSION_PROMISING**; Fusion is secondary and does not determine the Innovation-2 decision.

Old-gate HAI and unified-gate HAI are compared only after the unified target list was frozen. No Test-driven selection, Top-K, source/lag/alpha/threshold search, point adjustment, or target removal was performed.
