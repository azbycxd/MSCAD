# Formal REC architecture trace

Date: 2026-09-23  
Scope: frozen Ours-REC used by the formal SWaT and HAI result line

## 1. Binding

The formal model is `models.backbone_ae.BackboneAE` (`models/backbone_ae.py:9-26`). The SWaT config binds that class explicitly in `configs/thesis/swat2015_midterm_seed1.yaml:43-48`. `BackboneAEPlus` exists in the same source file but is not the model used by the frozen REC checkpoints and is therefore outside the ablation reference.

Formal dimensions:

- SWaT: input channels 51, hidden channels 64, latent channels 32, dropout 0.1, 11,730 trainable parameters;
- HAI21: input channels 79, hidden channels 64, latent channels 32, dropout 0.1, 15,482 trainable parameters;
- input and output shape: `[B,T,C]`;
- internal Conv1d layout: transpose to `[B,C,T]`, process, transpose back to `[B,T,C]`.

## 2. Exact encoder/decoder

| Order | Formal layer | Channel map | Temporal kernel | Other operations |
|---:|---|---|---:|---|
| 1 | `DepthwiseSeparableConv1d` | `C → 64` | 3 | depthwise Conv, BN, GELU, pointwise 1x1 Conv, BN, GELU, dropout 0.1 |
| 2 | `DepthwiseSeparableConv1d` | `64 → 32` | 3 | same block |
| 3 | `DepthwiseSeparableConv1d` | `32 → 64` | 3 | same block |
| 4 | output `Conv1d` | `64 → C` | 1 | bias enabled; no output activation |

The block implementation is `models/layers.py:7-22`:

1. depthwise temporal `Conv1d(in_ch,in_ch,k=3,padding=1,groups=in_ch,bias=False)`;
2. `BatchNorm1d(in_ch)` and `GELU`;
3. pointwise `Conv1d(in_ch,out_ch,k=1,bias=False)`;
4. `BatchNorm1d(out_ch)`, `GELU`, and `Dropout(0.1)`.

There is no pooling, stride change, dilation, recurrent state, attention, residual connection, or skip connection in the formal `BackboneAE`. Time length remains 64 throughout. The latent bottleneck is a 32-channel representation at every time step, not a single vector.

## 3. Temporal and channel semantics

- Three sequential temporal `k=3`, dilation-1 convolutions yield an effective temporal receptive field of 7 samples.
- Same-padding makes every output time position available; the frozen primary score uses only the endpoint reconstruction.
- Temporal kernels and 1x1 channel-mixing weights are shared across all time positions by Conv1d.
- Depthwise convolution performs per-channel temporal filtering; the following pointwise convolution performs learned cross-channel mixing.
- The final 1x1 projection reconstructs all `C` physical variables with no sigmoid or clipping.

## 4. Frozen training and scoring

- window 64, stride 1;
- Train-only StandardScaler;
- full-window SmoothL1 reconstruction loss;
- Adam, learning rate `1e-3`, weight decay `1e-4`, batch 256, at most 15 epochs, patience 4;
- primary anomaly score: endpoint mean absolute reconstruction error across channels;
- threshold: Normal Validation q99;
- no PA, POT, or Test-best threshold.

The model reconstructs the full `[B,T,C]` tensor even though the formal anomaly score uses its endpoint. This distinction must remain unchanged in every internal REC ablation.

## 5. Justified internal variants

### R1 — temporal-free depthwise substitute

Replace only the three depthwise temporal `k=3` convolutions with `k=1`. Retain pointwise channel mixing, BN, GELU, dropout, widths, output projection, loss, optimization, and evaluation. This isolates the value of the real temporal receptive field without a width sweep.

### R2 — dense Conv1d substitute

Replace each complete depthwise-separable block by the existing `ConvBNAct1d`-style dense `k=3` Conv1d at the same input/output widths, followed by the same BN/GELU/dropout. Retain layer count, widths, loss, and protocol. This isolates depthwise-separable factorization and enables the accuracy/Params/latency/VRAM trade-off to be reported.

## 6. Rejected variants

- hidden or latent width changes: capacity sweep, not component ablation;
- `BackboneAEPlus`: simultaneously introduces a stem, residual blocks, more layers, skip projection, and different default widths;
- removing BN, GELU, or dropout as extra rows: would expand the table without directly testing the two claimed contributions;
- endpoint/full-window score swap: scoring ablation, not REC architecture;
- any Test-selected architecture or retraining rule.

Conclusion: two independent real components exist, so `REC_INTERNAL_ABLATION_NOT_JUSTIFIED` does not apply.
