# WADI-B1 Frozen Rec / Dual / Fusion External Validation

## Pre-Test target gate result

WADI.A2 A0.5 passed the source, 123-variable schema, missingness, and segment gates. StandardScaler and target discovery used only the frozen valid Normal Train points. The exact self lags `[1,5,10,20,30]`, joint lags `[1,5,10,20,30]`, Ridge alpha 1.0, physical process groups, and four stability thresholds were unchanged.

All 67 eligible continuous candidates were evaluated. Pass counts were: Discovery 1, full Normal Validation 0, first-half Validation 0, second-half Validation 0, and all four 0. The maximum full-validation gain was -0.002541166. The frozen target list is therefore empty (SHA256 `e93bc24e4096fa7b3e8aeb8f989d8d29db2a7580ad036240ea9b5331b49e33db`).

Status: **WADI_B1_STOPPED_NO_STABLE_DUAL_TARGETS**.

Because `D(t)=max_j S_j(t)` is undefined for an empty frozen target set, the run stopped before REC training and before opening WADI Test. No Test labels, Attack-driven selection, Top-K fallback, threshold, Fusion result, or Attack13 patch was used. The intended WADI external comparison remains unexecuted rather than being silently redefined.
