# WADI-A0 Dataset, Attack, and Process Audit

## Version and source gate

The local bundle contains `WADI_14days_new.csv`, `WADI_attackdataLABLE.csv`, and the one-page `table_WADI.pdf`. The PDF body is the updated/corrected WADI.A2 attack table: it enumerates Attack IDs 1-15, with IDs 3-4 intentionally sharing one table row, and covers 9-11 October 2017. The source files were read-only and SHA256-recorded in `dataset_manifest.json`.

## CSV audit

| Split | Physical rows | Columns | Process variables | Audited time extent | Nominal interval | Gaps | Largest gap |
|---|---:|---:|---:|---|---:|---:|---:|
| Normal | 784,571 | 130 | 127 | 2017-09-25 18:00:00 to 2017-10-07 21:16:10 | 1 s | 1 | 264,001 s |
| Attack | 172,801 | 131 | 127 | 2017-10-09 18:00:00 to 2017-10-11 18:00:00 | 1 s | 0 | 1 s |

The Attack CSV begins with a numeric header and stores its real header in the next physical row; two trailing non-data rows contain only label `1`. The audit excludes those two non-observation rows without changing the source file.

The raw `Time` field stores minute:second within an hour and cycles every 3,600 rows, so raw `Date+Time` is not unique (Normal 744,971 duplicates; Attack 162,001). The audit preserves the raw fields and explicitly reconstructs hour-of-day from within-date row order and date boundaries. This is an audit representation, not a rewrite. Reconstructed timestamps have no duplicates.

Normal has 3,138,318 missing cells. Four columns are wholly missing: `2_LS_001_AL`, `2_LS_002_AL`, `2_P_001_STATUS`, and `2_P_002_STATUS`; another 34 cells are missing across `1_AIT_002_PV`, `1_AIT_004_PV`, `2B_AIT_004_PV`, and `3_AIT_004_PV`. Attack has 691,204 missing cells, exactly the same four wholly missing columns. Neither split contains infinity. `Row` has no duplicates; Normal has one recorded Row discontinuity matching the session boundary.

## Label and official-event binding

The actual label column is `Attack LABLE (1:No Attack, -1:Attack)`. Its frozen mapping is exact: `1 -> 0` and `-1 -> 1`. Among physical Attack rows, 162,824 are normal and 9,977 are attack, for a 5.7737% attack ratio.

The CSV yields 14 automatic contiguous attack-label intervals. Fourteen of the 15 official event IDs overlap a label interval; IDs 3 and 4 correctly share one interval because the PDF groups them. Most boundaries are shifted by +3 seconds. Attack 1 ends 13 seconds earlier than the PDF, and Attack 11 starts 42 seconds earlier. Official Attack 13 (12:07:30-12:10:52) has no overlapping CSV label interval. An otherwise duration-matching unused interval occurs at 00:07:33-00:10:55, approximately 12 hours early. This is retained as a source conflict; labels were not edited or silently rebound.

Status: `WADI_A2_PROTOCOL_AUDIT_COMPLETE_WITH_LABEL_BINDING_CONFLICT` (14/15 official IDs bound).

## Process schema

Process groups come only from raw prefixes, not correlation, Attack labels, or model behavior.

| Group | Channels | Continuous candidates | Discrete sources | Constant | Uncertain |
|---|---:|---:|---:|---:|---:|
| 1 | 19 | 7 | 5 | 7 | 0 |
| 2 | 82 | 43 | 15 | 18 | 6 |
| 2A | 4 | 4 | 0 | 0 | 0 |
| 2B | 4 | 4 | 0 | 0 | 0 |
| 3 | 15 | 7 | 0 | 8 | 0 |
| PLANT | 3 | 2 | 0 | 1 | 0 |

There are 67 continuous target candidates, 20 discrete sources, 34 constants, and 6 uncertain variables. Constants are excluded from future candidate modeling; uncertain variables remain unresolved.

## Confirmed physical relationships

Only relationships explicitly supported by names and the PDF descriptions are frozen: `1_MV_001 -> 1_FIT_001 / 1_LT_001` (Attack 1); `1_FIT_001 -> chemical dosing response and primary-tank level context` (Attack 2); `2_MCV_007 -> 2_FIT_002 / 2_PIT_002` (Attacks 8, 11, 12); `2_PIC_003_SP -> 2_FIT_003 / 2_PIT_003` (Attack 13); and the coordinated `1_AIT_002 + 2_MV_003` contamination action (Attack 7). Unnamed dosing pumps and raw-water pumps remain `UNKNOWN`; no channel identity was invented.

## Normal-only protocol candidate

Normal is not one continuous session. The 264,001-second discontinuity creates two hard segments:

- Segment 1 `[0,335999)`: Train `[0,268799)`, Validation `[268799,335999)`.
- Segment 2 `[335999,784571)`: Train `[335999,694856)`, Validation `[694856,784571)`.

The saved gap position is index 335,998 -> 335,999, from `2017-09-29 15:19:58` to `2017-10-02 16:39:59` (264,001 seconds).

Future lag/window construction must remain inside each segment. Total candidate Train is 627,656 rows and Validation is 156,915 rows. The complete Attack CSV is Official Test; labels are evaluation-only and may not select targets, sources, lags, thresholds, or fusion.

## Cross-dataset protocol context

| Dataset | Variables | Normal data | Test data | Sampling | Official events | Physical groups | Labels | HIGH | MIXED |
|---|---:|---|---|---:|---:|---|---|---:|---:|
| SWaT 2015 | 51 | official Normal Physical | official Attack Physical | 1 s | 36 intervals | six physical stages | Normal/Attack | not frozen by this audit | not frozen by this audit |
| HAI 21.03 | 79 | three Normal files | five Test files | 1 s | 50 | P1-P4 | attack 0/1 | 19 occurrences | protocol uses HIGH/DIRECT, so n.a. |
| WADI.A2 | 127 | two audited Normal sessions | one two-day Attack file | 1 s | 15 | 1/2/2A/2B/3/PLANT | 1 normal, -1 attack | 5 events | 6 events |

Only WADI.A2 files were used. WADI.A1 and WaDi.A3 were neither searched as substitutes nor read.

No model training, model inference, anomaly metric, attack-driven selection, or timestamp repair was performed.
