"""Reproduce attention-gradient roundoff without data or training checkpoints.

Every query attends entirely to key 0: its logit is about 1,021 while all other
logits are zero. Softmax is numerically one-hot, so query/key gradients should
be zero. Compare backend/precision/dropout combinations on the same H100 used
for training. This diagnostic reports errors; it does not update any weights.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('This diagnostic requires CUDA')
    results = []
    for dtype in (torch.bfloat16, torch.float32):
        for backend in (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH):
            for dropout in (0., .1):
                torch.manual_seed(42)
                query = torch.zeros(2, 8, 512, 96, device='cuda', dtype=dtype)
                query[..., 0] = 100
                query.requires_grad_()
                key = torch.zeros_like(query)
                key[:, :, 0, 0] = 100
                key.requires_grad_()
                value = torch.randn_like(query, requires_grad=True)
                output_gradient = torch.randn_like(query)
                with sdpa_kernel(backend):
                    output = torch.nn.functional.scaled_dot_product_attention(
                        query, key, value, dropout_p=dropout)
                (output * output_gradient).sum().backward()
                result = dict(dtype=str(dtype), backend=str(backend), dropout=dropout,
                              q_grad_norm=query.grad.float().norm().item(),
                              k_grad_norm=key.grad.float().norm().item(),
                              v_grad_norm=value.grad.float().norm().item())
                print(json.dumps(result), flush=True)
                results.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(torch_version=torch.__version__,
                                         gpu=torch.cuda.get_device_name(), results=results), indent=2) + '\n')


if __name__ == '__main__':
    main()
