# Thesis experiment status

## Frozen baseline commit

131f87fff8df1965eb3b8bd3923caf3d21bf9c66

## Current phase

Midterm Sprint S3 completed: unchanged-score A0++ static POT threshold diagnostic

## Formal benchmark

SWaT 2015 official Physical

source audit status: VALID_DIAGNOSTIC

protocol adjudication status: VALID_DIAGNOSTIC

formal benchmark status: PRELIMINARY single-seed diagnostic; final multi-seed benchmark not run

historical T2 source_readiness: E. label_alignment_conflict

historical T2 protocol_readiness: D. needs_official_semantics

adjudicated point labels: Physical exact mapping (`Normal=0`, `Attack=1`, `A ttack=1`)

adjudicated events: 36 attack-list intervals; 41 raw records retained

adjudicated gap policy: hard_session_boundary

## Secondary benchmark

HAI 22.04

## Diagnostic dataset

SWaT A12 2026

## Final method

NOT FROZEN

## Collaboration branch

V1 legacy

V2 diagnostic

V3 proposed; V3-A Normal-only gate=`PROMISING_INCREMENTAL_CROSS_CHANNEL_SIGNAL` (not an anomaly detector)

## Midterm evidence

- `S15-A0`: `VALID_DIAGNOSTIC / PRELIMINARY_OFFICIAL_SWAT2015`, seed 1, no point adjustment.
- `S15-A0+`: `VALID_DIAGNOSTIC / PRELIMINARY`; FPR/F1 operating point improved but AUPR/AUROC degraded, status=`ROBUST_TOPK_SCORING_NOT_SUPPORTED`.
- `S15-A0++`: `VALID_DIAGNOSTIC / PRELIMINARY`; ranking invariant passed but static POT was slightly worse than empirical q99, status=`STATIC_POT_NOT_SUFFICIENT`.
- `V3-A`: `VALID_DIAGNOSTIC`, Normal Train/Validation only; no Attack evaluation.
- Frozen S1 implementation: 396000/99000 ordered raw rows, 51 channels, window 64, stride 1, Train-only scaler, two hard-boundary Test segments, Validation q99 scores.

## Next gate

Do not promote A0+ top-6 or A0++ static POT as a strong baseline. Any further threshold/scoring design, multi-seed work, V3 Attack scoring, fusion, or final-method freezing requires a later explicitly authorized phase.
