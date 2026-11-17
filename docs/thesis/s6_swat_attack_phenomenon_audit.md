# S6 SWaT 2015 Attack Phenomenon / Collaboration-Relevance Audit

## Scope

This is a benchmark-phenomenon audit, not a detector experiment. It uses the frozen T2.5 protocol: Official SWaT 2015 Normal v1, Attack v0, and the 36 canonical physical-impact events. IDs 5, 9, 12, 15, and 18 remain excluded as official no-physical-impact markers; IDs 37–41 use the adjudicated 2016 dates. No label, event boundary, model, fusion, threshold, or point prediction was modified.

Normal Train alone supplies robust level/difference references, self-model residual references, state frequencies, transition matrices, and dwell distributions. Normal Validation alone supplies the continuous q99.5 evidence thresholds and the frozen A0 q99 threshold. Attack labels do not estimate any threshold.

## Frozen evidence rules

Continuous attacked variables are classified from the event fractions exceeding Validation q99.5 for raw level, first difference, and frozen V3-I1 self-history residual. Any fraction at least 0.20 is `STRONG_DIRECT`; otherwise any fraction at least 0.05 is `MEDIUM_DIRECT`; otherwise the variable is `WEAK_DIRECT`.

Discrete and Normal-constant actuators are audited as states. An unseen Train transition, Train state frequency below 0.005, or event dwell above the state's Train q99.5 is `STRONG_DIRECT`. Otherwise transition probability below 0.01 or state frequency below 0.02 is `MEDIUM_DIRECT`; otherwise it is `WEAK_DIRECT`. Constant remains the recorded schema type—it is not reclassified based on Attack outcomes.

The event class is the strongest mapped variable evidence. An event whose only attack point cannot be mapped remains `UNRESOLVED_MAPPING`; it is not silently treated as weak. Contextual relevance follows the preregistered mechanism and direct-evidence rules exactly.

## Main result

Among 36 official physical-impact events:

| Direct evidence | Count / 36 |
|---|---:|
| STRONG_DIRECT | 19 / 36 |
| MEDIUM_DIRECT | 13 / 36 |
| WEAK_DIRECT | 3 / 36 |
| UNRESOLVED_MAPPING | 1 / 36 |

| Collaboration relevance | Count / 36 |
|---|---:|
| HIGH_CONTEXTUAL_CANDIDATE | 3 / 36 |
| MIXED_CONTEXTUAL | 13 / 36 |
| DIRECT_DOMINATED | 19 / 36 |
| UNCLEAR | 1 / 36 |

Status: **SWAT_CONTEXTUAL_ATTACKS_ARE_NARROW**.

The auto-classification is an audit aid, not final ground truth. All High, Mixed, and Unclear cases are included in `manual_review_candidates.csv` for human adjudication.

## High contextual candidates

| ID | Attack points | Stage | Mechanism | Direct class | A0 median/max percentile | A0 q99 hit ratio | Official expected impact |
|---:|---|---|---|---|---:|---:|---|
| 13 | MV304 | P3 | ACTUATOR_MANIPULATION | WEAK_DIRECT | 96.682 / 97.835 | 0.000000 | Halt of stage 3 because change in the backwash process |
| 14 | MV303 | P3 | ACTUATOR_MANIPULATION | WEAK_DIRECT | 97.358 / 99.670 | 0.023202 | Halt of stage 3 because change in the backwash process |
| 29 | P201; P203; P205 | P2 | MULTI_POINT_COORDINATED | WEAK_DIRECT | 99.999 / 99.999 | 1.000000 | Wastage of chemicals |

For IDs 13 and 14 the event-modal actuator state is common in Normal Train (frequencies 0.960798 and 0.970654), no unseen transition appears, and event dwell is below the corresponding Normal q99.5. Their direct actuator evidence is therefore weak even though the official metadata describes forced valve behavior. A0 also gives no q99 hits for ID 13 and only 2.32% for ID 14.

ID 29 coordinates three P2 actuators. Every observed event-modal state is common in Normal Train and its event dwell is below the Normal q99.5. A0 responds strongly throughout the event, showing that weak attacked-variable evidence does not imply invisibility to a multichannel reconstruction score.

## Stage distribution

Events can contribute to more than one stage when multiple attack points span stages.

| Stage | Events | Direct dominated | High contextual | Mixed | Unclear |
|---|---:|---:|---:|---:|---:|
| P1 | 10 | 2 | 0 | 8 | 0 |
| P2 | 4 | 1 | 1 | 2 | 0 |
| P3 | 12 | 7 | 2 | 3 | 0 |
| P4 | 9 | 8 | 0 | 1 | 0 |
| P5 | 7 | 5 | 0 | 1 | 1 |
| P6 | 1 | 0 | 0 | 1 | 0 |

The three High candidates are concentrated in P2–P3 and in actuator/coordinated manipulation, not distributed broadly across SWaT stages.

## Mapping and protocol exceptions

Exact attack-point normalization removes hyphens/whitespace and uppercases identifiers while preserving both raw and canonical strings. `MV-504` (event 4) and `DIT-301` within event 23 do not map to the audited 51-variable schema. They remain explicit mapping errors; neither is replaced by a guessed variable. Event 23 still has mapped evidence for P602 and MV302, while event 4 remains `UNRESOLVED_MAPPING` and `UNCLEAR`.

The Physical-only label interval 2015-12-29 06:30:00–06:42:00 is independently registered as `UNMAPPED_PHYSICAL_LABEL_INTERVAL`, has no fabricated official event ID, and is not included among the 36 event identities.

## Research interpretation

SWaT contains real contextual/collaboration candidates, but only three under the frozen definition, so it does not provide a broad enough subset to serve as the only benchmark for a general collaboration-anomaly claim.

The prior V2/V3 failures are best explained by both factors. V3-R's high raw-level predictability but failed controlled corruption and V3-I1's zero stable innovation targets indicate a representation/mechanism problem; S6's three High candidates indicate that the target phenomenon is also narrow in this benchmark.

The most defensible next method direction is stage-local physical consistency on the preregistered P2–P3 candidates rather than another global correlation score. A second dataset is still needed if the thesis retains a general contextual/collaboration claim; otherwise the innovation claim should be narrowed to local physical-consistency detection.

## Artifacts

Ignored outputs are under `outputs/thesis_s6_swat_phenomenon_audit/`: event and attacked-variable audits, the manual-review table, stage summary, summary JSON, the separately registered Physical-only interval, frozen A0 Validation scores, and four representative figures (events 1, 2, 13, and 14).

Explicit scope: **PHENOMENON AUDIT ONLY; NO NEW DETECTOR; NO ATTACK-DRIVEN TRAINING; NO PARAMETER SEARCH; NO FUSION; NO POINT ADJUSTMENT.**
