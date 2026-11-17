# Chapter 6 Part 4 Audit

## Gate verification

- THESIS_CHAPTER6_PART1_READY: PASS
- THESIS_CHAPTER6_PART2_READY: PASS
- THESIS_CHAPTER6_PART3_READY: PASS
- THESIS_EXPERIMENTAL_PHASE_CLOSED: PASS
- THESIS_CONTRIBUTION_STRUCTURE_FROZEN: PASS
- Scope limited to Chapter 6.5: PASS
- Training / inference / new experiment / result recalculation: NONE

## Artifact verification

- Historical CoMuT: 13 cases and four frozen categories verified from D0 artifacts.
- Controlled SWaT: 6 frozen targets, Attack data exclusion, and target-level metrics verified.
- HAI21: DUAL strict metrics, 50 events, HIGH19 grouping, recovered event IDs, and normal-shift summary verified.
- SWaT Official: strict metrics, 36 events, contextual IDs 13/14/29, zero contextual DUAL hit rates, and normal-shift summary verified.
- A1 references: D2 No-Gate and HAI D1 Short-lag values verified from frozen ablation tables.
- GDN comparison: HAI strict metrics verified from the frozen baseline result.
- Result trace rows: 81
- Spreadsheet validation: artifact-tool import, recalculate, A1:I82 shape validation, workbook inspection, and error-token scan PASS.
- Optional evidence-chain plot was not generated because metric semantics differ across evidence levels and a single performance chart could be misleading.

## Semantic self-audit

| Check | Requirement | Result |
|---|---|---|
| A | CoMuT is not described as real industrial data | PASS |
| B | Controlled SWaT is not described as a real attack | PASS |
| C | SWaT Official negative result is reported explicitly | PASS |
| D | SWaT weakness is not attributed solely to domain shift | PASS |
| E | HAI single-dataset evidence is not generalized as universality | PASS |
| F | Section 6.3 baseline ranking is not repeated | PASS |
| G | Section 6.6 ablation is referenced only for mechanism context | PASS |
| H | Event counts, recovered IDs, and contextual candidates come from frozen artifacts | PASS |
| I | HIGH19 retains the pre-frozen grouping semantics | PASS |

## Frozen chapter preservation

- Chapter 6.1 SHA256: `4908e2ae236984029e03df8a823b1a8bf95dc5d2f80e0df6a2876262139563f0`
- Chapter 6.2/6.3 SHA256: `da2943d7532073db1c445aab0ea972d5998e66890d383c8e18c6991b3bfb55aa`
- Chapter 6.4 SHA256: `4502f9ded85a6c8c044168c197fbde005fc34f2f7f47fcd0cdafb2bbd26a1b0a`

## Final status

THESIS_CHAPTER6_PART4_READY
