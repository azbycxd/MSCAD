# Efficiency-E1: Final MSCAD / Dual-View Efficiency Audit

## Status and scope

**Status: `OURS_EFFICIENCY_AUDIT_COMPLETED`**

This audit measures the frozen Ours-REC, Ours-DUAL, and Ours-FULL implementations on Official SWaT 2015 and HAI 21.03. It performs inference and deterministic reconstruction of the already-frozen Ridge coefficient artifacts only. It does not train a model, change targets, prune, quantize, tune scores, read attack/test data, or rerun baselines.

The target manifests were verified before measurement:

| Dataset | Frozen targets | SHA-256 |
|---|---:|---|
| SWaT | 15 | `af9f9aac9cb8b369015a94d4b88a4729d817c219d305eb3276050c5d33ad31bf` |
| HAI21 | 54 | `a78e820567eaa15dd5a0cd2a9f6491d04102c0a50732a4de7aa918f1b20be5bb` |

## Parameter accounting

REC parameters are obtained from `sum(p.numel() for p in model.parameters() if p.requires_grad)`. Each Dual target counts all Self Ridge and Cross Ridge coefficients plus one intercept per Ridge. Tail ECDF arrays and the fusion threshold are calibration state and therefore contribute zero trainable parameters.

| Dataset | REC | Self Ridge | Cross Ridge | DUAL | FULL |
|---|---:|---:|---:|---:|---:|
| SWaT | 11,730 | 90 | 681 | 771 | 12,501 |
| HAI21 | 15,482 | 324 | 6,714 | 7,038 | 22,520 |

Per-target Self parameter counts are 6/6/6 (min/median/max) on both datasets. Cross counts are 7/49/67 on SWaT and 25/181/181 on HAI21. Detailed target and stage/process totals are recorded in `ours_parameter_breakdown.csv`.

## Serialized size

The Dual and Tail artifacts are uncompressed NPZ containers. FULL bytes are the sum of the REC checkpoint, Dual coefficients, and Tail calibration state; this is a reproducibility footprint rather than a claim that all bytes must be resident simultaneously.

| Dataset | Component | Bytes | Trainable params | Calibration elements |
|---|---|---:|---:|---:|
| SWaT | REC | 63,981 | 11,730 | 0 |
| SWaT | DUAL | 68,436 | 771 | 0 |
| SWaT | TAIL | 792,802 | 0 | 98,937 |
| SWaT | FULL | 925,219 | 12,501 | 98,937 |
| HAI21 | REC | 77,981 | 15,482 | 0 |
| HAI21 | DUAL | 594,324 | 7,038 | 0 |
| HAI21 | TAIL | 1,474,858 | 0 | 184,135 |
| HAI21 | FULL | 2,147,163 | 22,520 | 184,135 |

## Latency protocol and equivalence

All timings use batch size 1, 100 warm-up endpoints, and 1,000 legal Normal endpoints on the same RTX 4060 host. REC is GPU inference, DUAL is the frozen CPU Ridge path, and FULL is the hybrid end-to-end path (REC + all frozen Dual targets + empirical-tail calibration + max fusion). GPU timings synchronize CUDA around each endpoint.

The rebuilt Dual coefficient scorer was checked against the original frozen scorer on 100 Normal endpoints. Maximum absolute differences were 0 for SWaT and `4.2788e-12` for HAI21, both below the preregistered `1e-10` limit.

| Dataset | Method | Device | Mean ms | Median ms | P95 ms | Peak VRAM MiB |
|---|---|---|---:|---:|---:|---:|
| SWaT | Ours-REC | GPU | 0.722302 | 0.641150 | 1.054905 | 12.8730 |
| SWaT | Ours-DUAL | CPU | 0.231988 | 0.211400 | 0.356035 | N/A |
| SWaT | Ours-FULL | HYBRID | 1.156281 | 1.047550 | 1.705535 | 12.8730 |
| HAI21 | Ours-REC | GPU | 0.786008 | 0.678550 | 1.271355 | 19.7090 |
| HAI21 | Ours-DUAL | CPU | 1.403966 | 1.342750 | 1.772435 | N/A |
| HAI21 | Ours-FULL | HYBRID | 2.728708 | 2.644000 | 3.430980 | 19.7090 |

FULL mean-time shares were REC/DUAL/calibration+fusion = 62.47%/20.06%/17.47% for SWaT and 28.81%/51.45%/19.74% for HAI21. These shares are descriptive differences of separately benchmarked components and the complete path; they are not a new optimization result.

## Baseline comparison

Baseline parameters, sizes, and latency are read from their existing frozen B1/B2/B3 artifacts; no baseline was rerun.

| Dataset | Comparator | Comparator params / Ours-FULL | Comparator latency / Ours-FULL |
|---|---|---:|---:|
| SWaT | LSTM-AE | 5.3233× | 0.8497× |
| SWaT | USAD | 2,852.8239× | 1.2262× |
| SWaT | GDN | 0.6400× | 0.6459× |
| HAI21 | LSTM-AE | 3.3541× | 0.3473× |
| HAI21 | USAD | 3,207.2632× | 4.5383× |
| HAI21 | GDN | 0.4349× | 0.2706× |

Thus FULL has fewer parameters than LSTM-AE and USAD on both datasets, but more parameters than GDN. FULL is faster than USAD on both datasets, slower than LSTM-AE and GDN on HAI21, and slightly slower than LSTM-AE and GDN on SWaT. This mixed latency result is retained rather than selecting only favorable comparators.

## Performance–efficiency interpretation

The performance table joins existing final results to the measured efficiency values. On HAI21, Ours-FULL has AUPR 0.617897, AUROC 0.901850, F1 0.617350, 22,520 parameters, and 2.728708 ms mean latency. On SWaT, Ours-FULL has AUPR 0.206747, AUROC 0.708375, F1 0.260789, 12,501 parameters, and 1.156281 ms mean latency.

The known SWaT applicability boundary must remain visible: Ours-FULL is substantially worse than Ours-REC and GDN in AUPR/AUROC on SWaT. Therefore the lightweight parameter count does not establish universal superiority, and the final method's strongest empirical support remains dataset-dependent.

## Reproducibility artifacts

- `ours_parameter_breakdown.csv`: target and stage/process parameter accounting.
- `ours_model_sizes.csv`: actual serialized sizes and Tail calibration elements.
- `ours_latency.csv`: latency distributions, device, VRAM, and equivalence errors.
- `efficiency_comparison.csv`: baseline plus Ours efficiency table.
- `performance_efficiency.csv`: performance–efficiency joint table.
- `efficiency_audit.xlsx`: formatted multi-sheet audit workbook.
- `manifest.json` and `efficiency_summary.json`: protocol gates, hashes, ratios, and status.

No training, target change, Top-K target reduction, pruning, quantization, score change, test tuning, baseline rerun, commit, or push was performed.
