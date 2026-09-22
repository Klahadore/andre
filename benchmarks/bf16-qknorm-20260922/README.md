# QK-normalized BF16 implementation — September 22, 2026

The earlier investigation reproduced incorrect BF16 attention gradients and
found strongly saturated attention. This change addresses both the kernel
selection and unbounded query/key scale. It is a new architecture,
`pre_ln_qknorm_v2`, and requires fresh training.

## Implementation

- Retain pre-LN blocks and final LayerNorm; normalize Q and K independently
  within each head, using non-affine RMS normalization in FP32.
- Return normalized Q/K to the autocast dtype for SDPA. Attention permits
  cuDNN, Flash, and math kernels, excluding memory-efficient attention.
- Keep weights, gradients, and AdamW state in FP32. Disable reduced-precision
  BF16 GEMM reductions and reduced-precision math-SDPA intermediates. Use full
  FP32 precision for FP32 matrix multiplications and cross-entropy.
- Retain attention dropout 0, residual/FFN dropout 0.1, gradient clipping 1,
  pre-update maximum-gradient guard 1000, and global cell shuffling.

For head dimension d, unit-RMS Q and K have norm at most sqrt(d), so their scaled
inner product is bounded approximately by +/-sqrt(d). This bounds scale growth;
it does not guarantee high attention entropy or useful predictions.
The explicit block forward also prevents PyTorch's fused evaluation path from
silently bypassing QK normalization.

## Checks

Thirteen training-loop integration tests pass, including padding and permutation
invariance, agreement between gradient-enabled and no-grad evaluation, bounded
scores under large projections, legacy-checkpoint rejection, compiled checkpoint
round trips, and context-dependent learning on the synthetic test.

`scripts/validate_bf16.py` compares FP32, BF16, and compiled BF16 at identical
weights on eight actual validation cells. For this stress test it transfers the
old width-512 step-111000 and width-768 step-41000 weights into the new
architecture. This does not resume training and does not imply the bad weights
have recovered. The probe also runs two training-mode residual/FFN dropout seeds.

Fresh real-data pilots use the full LMDB training split with global shuffling,
2000 updates, batch 128, accumulation 1, context 512, BF16, compiled blocks,
learning rate 3e-4 after 1000 warmup updates, and the same fixed 6400 validation
cells every 500 updates. They process 256000 cell visits each. W&B is disabled
for these bounded diagnostics; raw metrics and configuration are archived here.
The 768 pilot's first launch failed before any update because SSH session cleanup
removed spawned workers' semaphores. A retry with `PAMName=login` completed
worker startup; that is a launcher issue, separate from model numerics.

These tests establish implementation behavior and short-run observations.
They do not establish recovery over a full 10-billion-position training run.

## Kernel and dtype audit

An actual H100 eager forward/backward with padding dispatches to cuDNN attention
(`aten::_scaled_dot_product_cudnn_attention` and its backward), not the excluded
memory-efficient kernel. Hooks observe FP32 output from the block's first
LayerNorm and final LayerNorm, and BF16 output from the feed-forward projection.
Both reduced-precision switches are false. See `bf16-backend-profile.json`.
The numerical comparison separately exercises compiled execution.

The saved pilot checkpoints are inspected directly: all model tensors and AdamW
state tensors are FP32. They remain on each node's NVMe; the small configuration,
metric, precision-summary, and comparison JSON files are archived in this folder.

## Largest-width memory check

The width-2176 model (2,082,375,033 parameters) completes compiled BF16 full
forward/backward/clipping/AdamW steps at both batch 16 and batch 32 on an H100
80 GB. Batch 32 peaks at 72.70 GiB allocated and 76.96 GiB reserved, with two
measured steps averaging 0.499 seconds. Batch 16 peaks at 49.76 GiB allocated.
This is a short synthetic dense-input benchmark, not a full loader/evaluation
run or a fresh search for the maximum batch. See `qknorm-2176-memory.json`.
Batch 32 with four accumulated batches still gives effective batch 128.
