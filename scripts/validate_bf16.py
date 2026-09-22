"""Compare BF16 and FP32 gradients on real cells, without updating weights.

An optional checkpoint transfers weights only for a diagnostic, not a training
resume. The model always uses the current QK-normalized architecture.
"""
import argparse
import json
from pathlib import Path
import random
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import ScBaseCountDataset, collate_fn
from model import Andre
from train import configure_cuda_precision, predict


def gradient_norm(model):
    return torch.stack([
        parameter.grad.float().norm()
        for parameter in model.parameters() if parameter.grad is not None
    ]).norm().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This numerical comparison requires a BF16-capable CUDA GPU")
    torch.set_num_threads(4)
    configure_cuda_precision()
    torch.manual_seed(43)
    dataset = ScBaseCountDataset(args.data_root, "val")
    indices = random.Random(43).sample(range(len(dataset)), 8)
    batch = {name: tensor.cuda() for name, tensor in
             collate_fn(dataset.__getitems__(indices), 512).items()}
    torch.manual_seed(42)
    model = Andre(width=args.width).cuda()
    source_step = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
        model.load_state_dict(checkpoint["model"])
        source_step = checkpoint["step"]
        del checkpoint

    # Dropout off: all three executions see identical inputs and weights.
    model.eval()
    results = []
    reference = None
    for mode in ["fp32", "bf16", "compiled_bf16"]:
        if mode == "compiled_bf16":
            for layer in model.transformer_layers:
                layer.compile(dynamic=False)
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mode != "fp32"):
            logits = predict(model, batch)
        loss = F.cross_entropy(logits.float(), batch["targets"] - 2)
        loss.backward()
        # Compare both ends of the stack. Double precision keeps this large
        # CPU cosine reduction from rounding above 1.
        gradient = torch.cat([
            model.transformer_layers[index].self_attn.in_proj_weight.grad.flatten()
            for index in [0, len(model.transformer_layers) - 1]
        ]).detach().cpu().double()
        norm = gradient_norm(model)
        if reference is None:
            reference = (logits.detach().float().cpu(), gradient)
        result = {
            "mode": mode, "width": args.width, "source_step": source_step,
            "loss": loss.item(), "grad_norm": norm,
            "logits_dtype": str(logits.dtype),
            "parameter_dtype": str(next(model.parameters()).dtype),
            "gradient_dtype": str(next(model.parameters()).grad.dtype),
            "max_logit_error": (logits.detach().float().cpu() - reference[0]).abs().max().item(),
            "attention_gradient_relative_error": ((gradient - reference[1]).norm() / reference[1].norm()).item(),
            "attention_gradient_cosine": F.cosine_similarity(gradient, reference[1], dim=0).item(),
        }
        print(json.dumps(result), flush=True)
        results.append(result)
        # These are diagnostic tolerances, not a guarantee for every future run.
        assert torch.isfinite(loss) and norm < 1000, result
        assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
        assert all(parameter.grad.dtype == torch.float32 for parameter in model.parameters()
                   if parameter.grad is not None)
        assert result["max_logit_error"] < 0.5, result
        assert result["attention_gradient_relative_error"] < 0.1, result
        assert result["attention_gradient_cosine"] > 0.98, result

    # Exercise compiled backward with the actual residual/FFN dropout enabled.
    model.train()
    for seed in [42, 123]:
        torch.manual_seed(seed)
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(predict(model, batch).float(), batch["targets"] - 2)
        loss.backward()
        norm = gradient_norm(model)
        result = {"mode": "compiled_bf16_train_dropout", "seed": seed,
                  "width": args.width, "source_step": source_step,
                  "loss": loss.item(), "grad_norm": norm}
        results.append(result)
        print(json.dumps(result), flush=True)
        assert torch.isfinite(loss) and norm < 1000, result
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
