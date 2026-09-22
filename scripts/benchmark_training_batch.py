"""Measure synthetic training-step throughput and memory without saving weights.

Uses the actual Andre forward, BF16 autocast over FP32 parameters, AdamW and
 gradient clipping. No activation checkpointing or loader timing.
The largest fitting batch is a hardware result, not a statistical optimum.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model as model_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-dim", type=int, default=2176)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16, 24, 32, 40, 48, 64])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-layers", action="store_true")
    args = parser.parse_args()
    if min(args.hidden_dim, args.length, args.warmup, args.steps, *args.batches) < 1:
        parser.error("dimensions, batch sizes and step counts must be positive")
    if args.hidden_dim % 8:
        parser.error("hidden dimension must be divisible by the model's eight heads")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requires a CUDA device supporting BF16")
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    with torch.device("cuda"):
        model = model_module.Andre(width=args.hidden_dim)
    model.train()
    if args.compile_layers:
        for layer in model.transformer_layers:
            layer.compile(dynamic=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    parameters = sum(p.numel() for p in model.parameters())
    report = {"gpu": torch.cuda.get_device_name(), "torch_version": torch.__version__,
              "total_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
              "hidden_dim": args.hidden_dim, "layers": len(model.transformer_layers),
              "heads": 8, "sequence_length": args.length, "parameters": parameters,
              "persistent_parameter_gradient_adam_bytes": parameters * 16,
              "precision": "FP32 parameters and optimizer states; BF16 autocast",
              "compile_layers": args.compile_layers,
              "scope": f"synthetic dense {args.length}-position cells; full forward/backward/clip/AdamW; no loader, evaluation or checkpoint saving",
              "warmup_steps": args.warmup, "timed_steps": args.steps, "trials": []}
    print(json.dumps({k: v for k, v in report.items() if k != "trials"}), flush=True)

    def step(batch):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch["ids"], batch["counts"], batch["mask"], batch["positions"])
            loss = nn.functional.cross_entropy(logits, batch["targets"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        return loss.detach()

    for size in args.batches:
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        batch = None
        result = {"batch_size": size}
        try:
            batch = {"ids": torch.randint(2, model.gene_embedding.num_embeddings,
                                          (size, args.length), device="cuda"),
                     "counts": torch.randint(1, 1000, (size, args.length), device="cuda").float(),
                     "mask": torch.ones(size, args.length, dtype=torch.bool, device="cuda"),
                     "positions": torch.zeros(size, dtype=torch.long, device="cuda"),
                     "targets": torch.randint(model.out_layer.out_features, (size,), device="cuda")}
            batch["ids"][:, 0] = 1
            torch.cuda.reset_peak_memory_stats()
            for _ in range(args.warmup):
                step(batch)
            torch.cuda.synchronize()
            warmup_peak = torch.cuda.max_memory_allocated() / 2**30
            torch.cuda.reset_peak_memory_stats()
            elapsed = []
            for _ in range(args.steps):
                t = time.perf_counter()
                loss = step(batch)
                torch.cuda.synchronize()
                elapsed.append(time.perf_counter() - t)
            result.update(status="ok", seconds_per_step=sum(elapsed) / len(elapsed),
                          cells_per_second=size * len(elapsed) / sum(elapsed),
                          gene_positions_per_second=size * args.length * len(elapsed) / sum(elapsed),
                          peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                          warmup_peak_allocated_gib=warmup_peak, step_seconds=elapsed,
                          last_loss=float(loss))
        except torch.cuda.OutOfMemoryError:
            result.update(status="oom", peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        report["trials"].append(result)
        print(json.dumps(result), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        del batch
        # Batch sizes are normally increasing; stop after the first failure.
        if result["status"] == "oom":
            break


if __name__ == "__main__":
    main()
