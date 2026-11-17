# HAI-B1: Frozen Rec / Dual / Fusion Real-Attack Evaluation

## Scope and protocol

This is a single frozen evaluation on the official HAI 21.03 real-attack data. Three normal files were split independently into 80% Normal Train and 20% Normal Validation; windows and lag samples never cross file boundaries. The 79-variable `float64` `StandardScaler`, target discovery, Ridge models, reconstruction model, robust branch references, and q99 thresholds use Normal data only. The five Test files and their labels were opened only after the HIGH-event and Dual-target lists had been written and SHA256-frozen.

The reconstruction branch is the existing lightweight A0 architecture adapted only from 51 to 79 channels (`window=64`, hidden 64, latent 32, dropout 0.1, SmoothL1, Adam 1e-3, weight decay 1e-4, batch 256, max 15 epochs, patience 4). It has 15,482 parameters; early stopping selected epoch 3 with Normal Validation loss 0.0314837890. The Dual branch is exactly the frozen predictive disagreement `abs(pred_self-pred_cross)`, with self lags `[1,5,10,20,30]`, cross lags `[0,1,5,10,20,30]`, Ridge alpha 1.0, per-target Normal Validation median/MAD calibration, and a max over targets. Fusion is the fixed `max(Z_rec,Z_dual)`; no weights were searched.

All point comparisons use the common valid endpoint mask beginning at index 63 of each file. There is no point adjustment. Thresholds are Normal Validation q99: Rec 0.3079403102, Dual 31.9453687253, and Fusion 13.2764139158.

## Frozen Dual targets

There are 24 stable targets: P1 13, P2 5, P3 0, and P4 6. In frozen order they are:

`P1_B400B`, `P1_FCV01D`, `P1_FCV01Z`, `P1_FCV02Z`, `P1_FCV03D`, `P1_FT01`, `P1_FT01Z`, `P1_FT02`, `P1_FT02Z`, `P1_FT03`, `P1_FT03Z`, `P1_LCV01D`, `P1_LIT01`, `P2_VT01`, `P2_VXT02`, `P2_VXT03`, `P2_VYT02`, `P2_VYT03`, `P4_HT_LD`, `P4_HT_PO`, `P4_LD`, `P4_ST_GOV`, `P4_ST_LD`, `P4_ST_PO`.

The target-list SHA256 is `cc2c8845231985ae61e07cc646cc5fc4aef7096b652c7acb0a00db55136587c9`. Full discovery and validation gains are preserved in `dual_targets_frozen.csv`; zero self-error candidates were conservatively assigned no positive relative gain and could not pass a stability gate.

## Global point results

| Method | AUROC | AUPR | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|
| REC | 0.852899 | 0.536294 | 0.459109 | 0.601095 | 0.520594 | 0.016133 |
| DUAL | 0.871911 | 0.583895 | 0.546992 | 0.614731 | 0.578886 | 0.011598 |
| FUSION | 0.889494 | 0.625597 | 0.566727 | 0.667822 | 0.613135 | 0.011631 |

At event hit-rate 0.10, Rec hits 45/50 events, Dual 36/50, and Fusion 42/50.

## Frozen HIGH collaboration occurrences

The 19 occurrences below were frozen from the HAI-A0 official interval-to-AP mapping before Test inference. The freeze SHA256 is `cf18da4e5cb64b134fd8e2b54fcc8be34db2e5c195f3296b1f96cc116f401d48`.

| Event / AP | Process | Rec hit | Dual hit | Fusion hit | Dual top target / process |
|---|---|---:|---:|---:|---|
| T2-E02 / AP02 | P1 | 0.8389 | 0.7915 | 0.8365 | P1_LIT01 / P1 |
| T2-E04 / AP07 | P1 | 0.5367 | 0.6100 | 0.6100 | P1_FT03Z / P1 |
| T2-E06 / AP09 | P1 | 0.0859 | 0.5195 | 0.5195 | P1_LIT01 / P1 |
| T2-E08 / AP12 | P1 | 0.6322 | 0.8736 | 0.8736 | P1_LIT01 / P1 |
| T2-E10 / AP04 | P1 | 0.1995 | 0.0356 | 0.0356 | P1_FT01Z / P1 |
| T2-E12 / AP17 | P2 | 0.5526 | 0.0066 | 0.4408 | P2_VXT03 / P2 |
| T2-E13 / AP14 | P1 | 0.6693 | 0.8583 | 0.8583 | P1_LIT01 / P1 |
| T2-E15 / AP19 | P2 | 0.4106 | 0.0000 | 0.0066 | P4_HT_PO / P4 |
| T3-E02 / AP02+AP06 | P1 | 0.9406 | 0.9881 | 0.9881 | P1_LIT01 / P1 |
| T3-E08 / AP14+AP20 | P1+P2 | 0.7216 | 0.9804 | 0.9804 | P1_LIT01 / P1 |
| T4-E01 / AP03+AP12 | P1 | 0.6260 | 0.9094 | 0.9094 | P1_LIT01 / P1 |
| T4-E02 / AP07+AP25 | P1+P3 | 0.7290 | 0.8130 | 0.8130 | P1_FT03Z / P1 |
| T4-E03 / AP12+AP25 | P1+P3 | 0.6502 | 0.8935 | 0.8935 | P1_LIT01 / P1 |
| T4-E04 / AP19+AP14 | P1+P2 | 0.7132 | 0.8527 | 0.8566 | P1_LIT01 / P1 |
| T5-E02 / AP09+AP18 | P1+P2 | 0.3876 | 0.5194 | 0.5310 | P1_LIT01 / P1 |
| T5-E03 / AP12+AP18 | P1+P2 | 0.7227 | 0.9297 | 0.9336 | P1_LIT01 / P1 |
| T5-E06 / AP19+AP25 | P2+P3 | 0.4118 | 0.0000 | 0.0131 | P2_VXT03 / P2 |
| T5-E10 / AP11+AP14 | P1 | 0.9427 | 0.9847 | 0.9847 | P1_LIT01 / P1 |
| T5-E12 / AP06+AP09 | P1 | 0.9466 | 0.9542 | 0.9618 | P1_LIT01 / P1 |

All nine HIGH events where Dual exceeds Rec by at least 0.10 are: T2-E06/AP09, T2-E08/AP12, T2-E13/AP14, T3-E08/AP14+AP20, T4-E01/AP03+AP12, T4-E03/AP12+AP25, T4-E04/AP19+AP14, T5-E02/AP09+AP18, and T5-E03/AP12+AP18.

## Complementarity and category analysis

Among attack points, the Rec/Dual hit partition is: both hit 4,498; Rec-only 880; Dual-only 1,002; both miss 2,567. Among normal points, 3,347 are Rec-correct but Dual false positives.

There are five Rec near-miss events (`Rec hit < 0.10`): T2-E05/AP05, T2-E06/AP09, T2-E14/AP03, T2-E20/AP25, and T5-E09/AP01+AP03. Exactly one is Dual-recovered (`Dual hit >= 0.10`): T2-E06/AP09. Therefore all-events near-miss/recovery is 5/1; within HIGH it is 1/1.

| Category | Events | Median Rec hit | Median Dual hit | Median Fusion hit |
|---|---:|---:|---:|---:|
| HIGH | 19 | 0.650190 | 0.852713 | 0.856589 |
| DIRECT | 31 | 0.558824 | 0.622093 | 0.853933 |

## Preregistered decisions

Innovation-2 gates H1/H2/H3/H4 are all **PASS**. The required status is `DUAL_VIEW_REAL_HAI_COLLABORATION_SUPPORTED`. This supports the frozen Dual-view score as complementary real-attack evidence in this one HAI 21.03 evaluation; it is not attack-driven target selection and does not establish multi-seed uncertainty.

Fusion gates F1, F2, and F4 pass, but F3 fails because Fusion hits 42 events versus Rec's 45 at hit-rate 0.10. The required status is `NORMAL_CALIBRATED_MAX_FUSION_NOT_YET_SUPPORTED`. Its stronger global point metrics do not override the failed preregistered event gate, and the formula was not changed after seeing Test.

## Reproducibility artifacts

The machine-readable evidence is in `outputs/thesis_hai21_b1_real/`: frozen lists, global and event metrics, point/event complementarity, attribution, category summary, thresholds, manifest, and summary. The raw HAI files and HAI-A0 audit inputs were hash-checked before and after execution and were unchanged. No point adjustment, POT, Test-best threshold, lag/alpha/AE hyperparameter search, attack-driven feature/target selection, fusion weight search, commit, or push was performed.
