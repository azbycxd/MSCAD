# V3-B Normal-only controlled mechanism validation

> This is a Normal-only controlled mechanism experiment. It is not SWaT real-Attack performance, and no Attack cache or labels were read.

## Frozen protocol

The four targets are copied directly from the V3-A `relative_gain >= 0.10` output: FIT201 (0.272803), DPIT301 (0.173826), FIT301 (0.315733), and FIT601 (0.406098). V3-A lags `[1,5,10,20,30]`, Ridge alpha 1.0, and float64 coefficients are unchanged. Normal Validation valid times are divided chronologically into disjoint calibration and evaluation halves. Each cell uses 2,000 deterministic clean/corrupted pairs. Temporal shift is fixed at 60 seconds; matched donors come from a fixed 2,048-window evaluation pool and remain at least 300 seconds away.

For target `j`, the frozen score is `max(0, (median_calibration_gain_j - gain_j(t)) / max(1.4826*MAD_calibration_gain_j, 1e-6))`, where `gain=e_self-e_full`. Its q99 threshold uses calibration Normal only. Target values, complete 64-point target histories, and Self inputs are identical before and after corruption. Other-channel contexts are complete real Normal windows. A0 uses its unchanged epoch-13, 11,730-parameter checkpoint and raw endpoint mean-MAE score.

## Target-level results

| Target | V3-A gain | Injection | V3 AUROC/AUPR | A0 AUROC/AUPR | Self AUROC | Full AUROC | q99 recall | Normal FPR | Normal gain | Corrupted gain | Gain drop |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FIT201 | .272803 | Shift60 | .529541/.551942 | .507910/.512820 | .500000 | .515039 | .0495 | .0110 | .00005739 | -.00000944 | .00006683 |
| FIT201 | .272803 | MatchedSwap | .494676/.495712 | .452442/.474082 | .500000 | .524845 | .0105 | .0110 | .00005739 | .00008025 | -.00002286 |
| DPIT301 | .173826 | Shift60 | .538290/.572795 | .515890/.534474 | .500000 | .530054 | .0850 | .0080 | .00040226 | .00026748 | .00013478 |
| DPIT301 | .173826 | MatchedSwap | .503844/.508561 | .495762/.496202 | .500000 | .501236 | .0110 | .0080 | .00040226 | .00039334 | .00000892 |
| FIT301 | .315733 | Shift60 | .543685/.579710 | .513516/.534516 | .500000 | .520663 | .0790 | .0105 | .00002069 | -.00012257 | .00014326 |
| FIT301 | .315733 | MatchedSwap | .503172/.502926 | .536057/.587493 | .500000 | .517490 | .0095 | .0105 | .00002069 | .00001260 | .00000809 |
| FIT601 | .406098 | Shift60 | .503239/.519911 | .503313/.512373 | .500000 | .491812 | .0305 | .0095 | .00040251 | .00039132 | .00001119 |
| FIT601 | .406098 | MatchedSwap | .485497/.511449 | .185049/.409795 | .500000 | .522075 | .0020 | .0095 | .00040251 | .00029207 | .00011045 |

## Preregistered gate

| Gate | Requirement | Result | Status |
|---|---|---:|---|
| A | 8-cell median V3 AUROC >= .80 | .503542 | FAIL |
| B | V3 AUROC beats both A0 and Self in >=6/8 cells | 4/8 | FAIL |
| C | Median q99 corruption recall >= .50 | .020750 | FAIL |
| D | Median clean-Normal FPR <= .03 | .010000 | PASS |

Temporal Shift medians: V3 AUROC .533915, V3 AUPR .562369, A0 AUROC .510713, Self AUROC .500000. Matched Swap medians: V3 AUROC .498924, V3 AUPR .505743, A0 AUROC .474102, Self AUROC .500000. Across all eight cells, median V3 AUPR is .515680; median Normal gain, corrupted gain, and gain drop are .00022982, .00017387, and .00003901.

## Scientific conclusion

Status: `V3B_MECHANISM_NOT_SUPPORTED`.

The frozen collaboration-degradation score is near chance across the controlled cells and detects very few corruptions at its Normal-only q99 threshold. A positive median gain drop exists, but it is too small and inconsistent to satisfy the preregistered mechanism gates. Under this protocol, V3-A's aggregate Normal prediction gains do not translate into robust detection of broken target-context correspondence. No target, shift, donor rule, lag, alpha, aggregation, or threshold was changed after observing the result; progression to V3-C requires a new human decision.
