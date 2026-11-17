# WADI.A2 Attack Taxonomy Frozen Before Modeling

This taxonomy uses only `table_WADI.pdf` page 1 and prefix-based process identification. It is frozen before any dependency selection or model execution.

| ID | Official time | Target(s) | Process | Description / intention | Taxonomy | Confidence |
|---:|---|---|---|---|---|---|
| 1 | 10-09 19:25:00-19:50:16 | 1_MV_001; 1_LT_001; 1_FIT_001 | 1 | Open valve; overflow should appear in level and flow | MIXED_COLLABORATION | MEDIUM |
| 2 | 10-10 10:24:10-10:34:00 | 1_FIT_001; dosing pump | 1 | False flow reading activates dosing while level stays constant | HIGH_COLLABORATION_RELEVANCE | HIGH |
| 3 | 10-10 10:55:00-11:24:00 | 2_LT_002 | 2 | Stealthily manipulate reservoir draining/filling dynamics | HIGH_COLLABORATION_RELEVANCE | HIGH |
| 4 | 10-10 10:55:00-11:24:00 | 1_AIT_001 | 1 | Coordinate water-quality spoofing with tank manipulation | HIGH_COLLABORATION_RELEVANCE | HIGH |
| 5 | 10-10 11:30:40-11:44:50 | six 2_MCV consumer valves | 2 | Close consumer valves and stop supply | DIRECT_DOMINATED | HIGH |
| 6 | 10-10 13:39:30-13:50:40 | 2_MCV_101; 2_MCV_201 | 2 | Maliciously open two valves | DIRECT_DOMINATED | HIGH |
| 7 | 10-10 14:48:17-14:59:55 | 1_AIT_002; 2_MV_003 | 1+2 | False quality value plus valve opening to supply contaminated water | HIGH_COLLABORATION_RELEVANCE | HIGH |
| 8 | 10-10 17:40:00-17:49:40 | 2_MCV_007; 2_PIT_002; 2_FIT_002 | 2 | Leakage valve action expected in pressure/flow; unsuccessful | MIXED_COLLABORATION | HIGH |
| 9 | 10-11 10:55:00-10:56:27 | 1_P_006 | 1 | Start pump to cause pipe burst | DIRECT_DOMINATED | HIGH |
| 10 | 10-11 11:17:54-11:31:20 | 1_MV_001; raw-water pump | 1 | Damage valve and pump to drain reservoir | MIXED_COLLABORATION | MEDIUM |
| 11 | 10-11 11:36:31-11:47:00 | same as Attack 8 | 2 | Leakage scenario similar to Attack 8 | MIXED_COLLABORATION | MEDIUM |
| 12 | 10-11 11:59:00-12:05:00 | same as Attack 8 | 2 | Leakage scenario similar to Attack 8 | MIXED_COLLABORATION | MEDIUM |
| 13 | 10-11 12:07:30-12:10:52 | 2_PIC_003_SP; 2_FIT_003; 2_PIT_003 | 2 | Reduce booster setpoint; intermittent supply reflected in flow/pressure | MIXED_COLLABORATION | HIGH |
| 14 | 10-11 12:16:00-12:25:36 | chemical dosing | 1 | Stop raw-water chemical dosing | DIRECT_DOMINATED | MEDIUM |
| 15 | 10-11 15:26:30-15:37:00 | 2_LT_002; 1_AIT_001 | 1+2 | Stealthy inverse of Attack 3 | HIGH_COLLABORATION_RELEVANCE | MEDIUM |

Counts: HIGH 5, MIXED 6, DIRECT 4, UNCLEAR 0. HIGH IDs are 2, 3, 4, 7, 15; MIXED IDs are 1, 8, 10, 11, 12, 13.

Because HIGH >= 3, the preregistered contextual status is `WADI_REAL_CONTEXTUAL_BENCHMARK_PROMISING`. This describes attack coverage only; it is not model-performance evidence.
