# WADI-A0.5 Missingness + Attack13 Binding Protocol Freeze

## Scope

This protocol uses only WADI.A2: `WADI_14days_new.csv`, `WADI_attackdataLABLE.csv`, and `table_WADI.pdf`. It performs no model training or inference and does not modify source data.

## Frozen missingness policy

- Frozen Normal Train ranges: `[0,268799)` and `[335999,694856)` (627,656 points).
- Exclude a feature iff its missing ratio is exactly 1 across the combined Normal Train. Attack/Test is never used for exclusion.
- Excluded (4): `2_LS_001_AL`, `2_LS_002_AL`, `2_P_001_STATUS`, `2_P_002_STATUS`.
- Frozen model schema: 123 process variables.
- No imputation. Any remaining NaN makes that point invalid and creates a hard segment boundary; windows/lags cannot cross it.
- Train valid ratio: 0.999955390; Validation: 0.999961763; Test: 1.000000000.
- Status: **MISSINGNESS_PROTOCOL_READY**.

## Attack13 source adjudication

The official WADI table was visually and textually rechecked. ID13 is `2017-10-11 12:07:30` to `12:10:52`, reducing the booster set-point pressure with effects expected in `2FIT003` and `2PIT003`. The A0 parser is not a 12-hour parser: raw `Time` contains minute/second only, while hours and day rollover are reconstructed globally from monotonic `Row` and raw Date blocks. There is no AM/PM field, no `12 -> 00` conversion, and no event-specific patch.

The Attack CSV instead labels the canonical 00h interval (2017-10-11T00:07:33 to 2017-10-11T00:10:55). The four Attack labels at 12:05:00-12:05:03 in the requested 12:05-12:13 audit range are the tail of bound ID12; the exact official ID13 interval contains 0 Attack-labeled points. This is preserved as a source conflict; neither source is rewritten.

- Point-wise ground truth: actual Attack CSV labels.
- Official event primary evaluation: 14 successfully bound events.
- ID13: excluded from the primary event aggregate and retained as `SOURCE_CONFLICT_UNRESOLVED`.
- HIGH relevance IDs 2, 3, 4, 7, 15: all `BOUND`.
- Binding status: **ATTACK13_SOURCE_CONFLICT**.

## Frozen safeguards

No imputation, Attack-driven feature removal, event-specific 12-hour patch, model training, model inference, thresholding, or metric optimization was performed.
