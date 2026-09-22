"""Inspect actual eager CUDA attention dispatch and activation dtypes.

Run from the repo on an H100; optionally pass a JSON output path.
"""
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import Andre
from train import configure_cuda_precision

configure_cuda_precision()
torch.set_num_threads(2)
model = Andre(width=512).cuda().eval()
ids = torch.randint(2, 30000, (1, 512), device="cuda")
ids[:, 0] = 1
ids[:, 384:] = 0
counts = torch.ones_like(ids)
mask = ids != 0
position = torch.zeros(1, device="cuda", dtype=torch.long)
dtypes = {}
handles = []
for name, module in [("norm1", model.transformer_layers[0].norm1),
                     ("ffn_linear1", model.transformer_layers[0].linear1),
                     ("final_norm", model.final_norm)]:
    handles.append(module.register_forward_hook(
        lambda module, args, output, name=name: dtypes.update({name: str(output.dtype)})))
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(ids, counts, mask, position)
    logits.float().square().mean().backward()
operators = [event.key for event in profiler.key_averages() if "attention" in event.key]
assert not any("efficient_attention" in name for name in operators), operators
report = {
    "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(),
    "operators": operators, "activation_dtypes": dtypes,
    "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    "math_sdp_reduced_precision": torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
}
print(json.dumps(report), flush=True)
Path(sys.argv[1] if len(sys.argv) > 1 else "bf16-backend-profile.json").write_text(
    json.dumps(report, indent=2) + "\n")
