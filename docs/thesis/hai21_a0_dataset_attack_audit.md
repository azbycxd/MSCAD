# HAI 21.03 Dataset, Process, and Attack Audit

## Audit scope and sources

This is a read-only benchmark audit of `dataset/hai/hai-21.03`. The source priority was the local official `hai_dataset_technical_details.pdf` (especially pp. 7-11, 12-15, 24-28, and 36-38), followed by the local official README. All eight `.csv.gz` files were read directly; no decompressed copies, model calls, inference, or detection metrics were produced.

Final status: `HAI21_BENCHMARK_PROTOCOL_CANDIDATE_READY`.

## Version and files

The local release is HAI 21.03 and contains the expected three normal Train files and five Attack Test files. All files share the same ordered 84-column schema and parse with `time` first and `attack`, `attack_P1`, `attack_P2`, `attack_P3` last.

| File | Rows | Hours | Start | End | Attack points |
|---|---:|---:|---|---|---:|
| train1.csv.gz | 216,001 | 60.0003 | 2020-07-11 00:00:00 | 2020-07-13 12:00:00 | 0 |
| train2.csv.gz | 226,801 | 63.0003 | 2020-07-31 22:00:00 | 2020-08-03 13:00:00 | 0 |
| train3.csv.gz | 478,801 | 133.0003 | 2020-08-04 22:00:00 | 2020-08-10 11:00:00 | 0 |
| test1.csv.gz | 43,201 | 12.0003 | 2020-07-07 15:00:00 | 2020-07-08 03:00:00 | see manifest |
| test2.csv.gz | 118,801 | 33.0003 | 2020-07-09 15:00:00 | 2020-07-11 00:00:00 | see manifest |
| test3.csv.gz | 108,001 | 30.0003 | 2020-07-13 00:00:00 | 2020-07-14 06:00:00 | see manifest |
| test4.csv.gz | 39,601 | 11.0003 | 2020-07-28 12:00:00 | 2020-07-28 23:00:00 | see manifest |
| test5.csv.gz | 92,401 | 25.6669 | 2020-07-30 10:50:00 | 2020-07-31 12:30:00 | see manifest |

Train totals are 921,603 rows and 256.0008 hours. Test totals are 402,005 rows and 111.6681 hours. Every Train value in all four label columns is zero, so the Train source is pure normal under the released labels.

Every file has a nominal one-second interval, zero missing timestamps, zero timestamp gaps, zero duplicate timestamps, and maximum gap of one second. Every future lag/window must remain inside its source file; files must never be joined into a continuous sequence.

## Schema and process grouping

The official technical manual states that HAI 21.03 has 78 SCADA points, but the local files contain 79 process-variable columns: `84 total = time + 79 process + 4 labels`. This is a source/file discrepancy and was not silently repaired. The actual benchmark schema is frozen from the files.

| Prefix | Process | Actual channels |
|---|---|---:|
| P1 | Boiler | 38 |
| P2 | Turbine | 22 |
| P3 | Water treatment | 7 |
| P4 | HIL simulation | 12 |

Grouping uses only exact `P1_` through `P4_` prefixes. Label columns are excluded from process features. Dataset-wide observed-value audit classified 19 channels as constant, 56 as continuous, and 4 as discrete. Classification uses the manual's Boolean descriptions plus observed unique states; it does not use attack performance or correlation clustering.

## Confirmed control relationships

| Controller | SP | PV | CV | Notes |
|---|---|---|---|---|
| P1-PC | P1_B2016 | P1_PIT01 | P1_PCV01D, P1_PCV02D | confirmed, PDF p.7 |
| P1-LC | P1_B3004 | P1_LIT01 | P1_LCV01D | FCV03D feed-forward, PDF p.7 |
| P1-FC | P1_B3005 | P1_FT03 | P1_FCV03D | confirmed, PDF p.7 |
| P1-TC | P1_B4022 / P1_B4002 | P1_TIT01 | P1_FCV01D, P1_FCV02D | official-name conflict between loop text and HAI21 section |
| P1-CC | P1_PP04SP | P1_TIT03 | P1_PP04 | documented but unavailable in local 21.03 schema |
| P2-SC | P2_AutoSD (local P2_ASD) | P2_SIT01 | P2_SCO (local P2_CO_rpm) | confirmed with local aliases, PDF p.10 |
| P3-LC | UNKNOWN | P3_LIT01 (manual LT01) | P3_LCV01D, P3_LCP01D | no explicit SP mapping, PDF p.11 |

Unknown or conflicting mappings remain explicit. No correlation-based mapping was introduced.

## Attack intervals and official mapping

The released global `attack == 1` labels produce exactly 50 continuous intervals: test1 5, test2 20, test3 8, test4 5, and test5 12. This matches the official HAI 21.03 count without merging or splitting intervals.

The technical manual provides an official interval-to-scenario Attack Operation table on pp. 36-38. Forty-nine local intervals match its date and start minute exactly. The last event, `T5-E12`, starts locally at `2020-07-31 11:20:59`, one second earlier than the table's `11:21`; it is retained and explicitly marked as a source conflict while preserving the official AP06+AP09 metadata. No local label was changed.

Process-label attribution is: P1 only 23 events, P2 only 11, P3 only 2, and multi-process 14. Multi-process comprises P1+P2 8, P1+P3 3, and P2+P3 3. P4 has no dedicated label, so absence of a P4 label must not be interpreted as proof that P4 is unaffected.

## Candidate benchmark protocol

For each of train1, train2, and train3 independently, use its first 80% as Normal Train candidate and final 20% as Normal Validation candidate. Fit operations may combine the three Train prefixes, and normal calibration may combine the three Validation suffixes, but lag/window construction must never cross a file boundary.

All five Test files form Official Test. Test labels are evaluation-only and may not select processes, targets, lags, features, or thresholds. No model was run in this phase.

HAI 21.03 is suitable for the thesis because the official design couples boiler, turbine, water treatment, and HIL simulation processes through feedback control relationships, SP/PV/CV behavior, and both single and combined attacks.

