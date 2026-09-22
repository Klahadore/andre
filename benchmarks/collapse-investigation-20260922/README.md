# Investigating input-independent predictions

The original width-512, 30-layer post-LN run reached validation loss 9.3087 at
step 11,000, but a separate CPU checkpoint probe on 64 validation cells found
essentially identical predictions. Hiding all non-MASK context or zeroing the
MASK count left loss unchanged at 9.3092155. First attention-projection gradient
EMA RMS was 6.49e-17, versus 1.24e-5 in the output weight matrix. The loss curve
alone concealed the failure.

## Controlled learning probes

All probes use the same 64 real training cells, fixed gene samples and MASKs,
512 positions, 30 layers, 36,601 output classes, raw count embeddings capped at
100,000, BF16 autocast, AdamW, and gradient clipping. Each target is absent from
the input, with exactly one MASK per cell. All 64 targets in this sample differ.
The fixed-set unconditional target entropy is ln(64), approximately 4.159.

| Width / configuration | Updates | Fixed-set loss | Accuracy | Loss with context hidden |
|---|---:|---:|---:|---:|
| 512 post-LN, LR 3e-4, warmup 100 | 300 | 0.44094 | 84.375% | 9.3712 |
| 512 pre-LN + final LN, same schedule | 300 | 0.000831 | 100% | 8.2029 |
| 512 post-LN, LR 1e-4, warmup 300 | 300 | 0.006395 | 100% | 10.3025 |
| 768 pre-LN + final LN, LR 3e-4, warmup 100 | 200 | 0.000332 | 100% | 8.8289 |

Pre-LN reached 100% accuracy by update 100 at both widths, and its predictions
varied across cells. Context ablation and shuffled-target controls show that
low loss was not explained by a shared gene-frequency distribution.

This supports an optimization-stability diagnosis, not a broken label mapping.
It does not establish that post-LN can never learn: it did learn the fixed set,
and a conservative post-LN schedule performed well. Tiny-set memorization is
not evidence of long-run generalization; new runs must be monitored on held-out
cells with prediction diversity and early-layer gradients as well as loss.

## Full-run choice

Use pre-LN with final LN, retain 30 layers and LR 3e-4, and extend warmup to
1,000 optimizer updates for the full-data restarts. Start from fresh weights;
never resume the collapsed post-LN optimizer/model into the new architecture.
Keep effective batch size 128, context 512, 152,588 optimizer updates and the
same fixed 6,400-cell validation subset at both widths.

The H100 test at width 768 found batch 128 fits: 67.22 GiB peak allocated,
67.63 GiB reserved. Synthetic training throughput was 203.94 cells/s, compared
to 194.57 at batch 64. This is a brief hardware test, not a long-run throughput
or convergence guarantee. Raw measurements and per-probe curves are alongside
this report.

## References

- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- [Original failing run](https://wandb.ai/preston281s-berkeley-city-college/andre/runs/e23qdx5v)

The selected normalization change is motivated by these measurements and known
residual-path stability considerations; the precise long-run causal contribution
of normalization versus schedule has not been isolated by these short probes.
