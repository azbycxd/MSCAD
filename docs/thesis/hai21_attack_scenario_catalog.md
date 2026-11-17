# HAI 21.03 Official Attack Scenario Catalog

## Source boundary

This catalog is limited to the 25 attack primitives explicitly used by the HAI 21.03 Attack Operation table in the local official technical-details PDF, pp. 36-38. The same table maps these primitives and 25 official combinations to the 50 released attacks. Later-version AP/AE numbering is not substituted for the HAI 21.03 table.

## Collaboration-relevance taxonomy

The taxonomy uses official scenario semantics only. No detector result, AUROC/AUPR, or observed test response was used.

- `HIGH_COLLABORATION_RELEVANCE`: AP02, AP04, AP07, AP09, AP12, AP14, AP17, AP19. These jointly manipulate an SP/control output and conceal or hold its PV, directly violating loop consistency.
- `MIXED_COLLABORATION`: none under the strict stated rule.
- `DIRECT_DOMINATED`: AP01, AP03, AP05, AP06, AP08, AP10, AP11, AP13, AP15, AP16, AP18, AP20, AP21, AP22, AP23, AP24, AP25. The official primitive is a single direct SP/control-output or short-term manipulation.
- `UNCLEAR`: none in the official 25-primitive catalog.

Counts are HIGH 8, MIXED 0, DIRECT 17, and UNCLEAR 0.

The catalog contains setpoint manipulation, control-output/actuator manipulation, short-term attacks, PV concealment/hold, and multi-point coordinated primitives. The 50-event Attack Operation table additionally contains 25 combinations of two primitives.

Machine-readable details, target points, methods, official support flags, taxonomy reasons, and source locators are preserved in `outputs/thesis_hai21_a0_audit/attack_scenario_catalog.csv` and `scenario_collaboration_taxonomy.csv`.
