"""A small, single-device training loop for predicting one masked gene per cell.

Read main() from top to bottom: data -> model -> forward -> loss -> backward
-> update -> validation -> checkpoint. See TRAINING.md for a walkthrough.
"""

import argparse
from functools import partial
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import wandb

from dataset import BlockShuffleSampler, ScBaseCountDataset, collate_fn
from model import Andre


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Built LMDB directory containing catalog.json")
    parser.add_argument("--out", default="runs/andre", help="Checkpoints and logs go here")
    parser.add_argument("--steps", type=int, default=1000, help="Total optimizer steps, including resumed steps")
    parser.add_argument("--batch-size", type=int, default=16, help="Cells per optimizer step")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100, help="Also saves checkpoints")
    parser.add_argument("--val-batches", type=int, default=50, help="Bounded, fixed validation subset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--precision", choices=["auto", "float32", "bfloat16"], default="auto")
    parser.add_argument("--resume", help="Path to last.pt or best.pt")
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--wandb-project", default="andre")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B team/user")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)
    for name in ("steps", "batch_size", "context_length", "log_every", "eval_every", "val_batches"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.workers < 0 or args.warmup_steps < 0 or args.weight_decay < 0:
        parser.error("workers, warmup-steps, and weight-decay must be nonnegative")
    if args.lr <= 0 or args.grad_clip <= 0:
        parser.error("lr and grad-clip must be positive")
    return args


def move_batch(batch, device):
    return {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}


def predict(model, batch):
    # The model selects the MASK hidden vector and returns [batch, num_genes].
    return model(batch["gene_ids"], batch["counts"],
                 batch["attention_mask"], batch["mask_positions"])


@torch.no_grad()
def evaluate(model, loader, device, use_bf16, seed):
    """Same validation cells, sampled genes, and MASK positions at each check."""
    model.eval()  # Disable dropout. no_grad() also avoids storing gradients.
    loss_sum = correct = cells = 0
    # Validation collation runs on the main CPU. Restore its RNG afterward so
    # evaluation does not change subsequent training samples/dropout on CPU.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        for batch in loader:
            batch = move_batch(batch, device)
            targets = batch["targets"] - 2  # PAD=0, MASK=1; gene classes start at 0.
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = predict(model, batch)
                loss = nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Validation loss is not finite")
            n = targets.numel()
            loss_sum += loss.item() * n
            correct += (logits.argmax(dim=-1) == targets).sum().item()
            cells += n
    model.train()  # Re-enable dropout before returning to training.
    return {"val/loss": loss_sum / cells, "val/accuracy": correct / cells, "val/cells": cells}


def save_checkpoint(path, model, optimizer, step, epoch, best_loss, args):
    # Write a temporary file first so interruption does not destroy last.pt.
    temporary = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "epoch": epoch, "best_val_loss": best_loss,
                "config": vars(args)}, temporary)
    temporary.replace(path)


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    bf16_available = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if args.precision == "bfloat16" and not bf16_available:
        raise ValueError("This script uses bfloat16 only on a supported CUDA GPU")
    use_bf16 = bf16_available and args.precision != "float32"
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # 1. Load the completed dataset. Do not start from a builder's partial shards.
    if not (Path(args.data_root) / "catalog.json").is_file():
        raise FileNotFoundError("No catalog.json: wait for the dataset build to finish, then set --data-root")
    train_data = ScBaseCountDataset(args.data_root, split="train")
    val_data = ScBaseCountDataset(args.data_root, split="val")
    if not len(train_data) or not len(val_data):
        raise ValueError("Training requires nonempty train and val splits")
    sampler = BlockShuffleSampler(train_data, seed=args.seed)
    collate = partial(collate_fn, length=args.context_length)
    worker_options = {}
    if args.workers:
        worker_options = dict(persistent_workers=True, prefetch_factor=2,
                              multiprocessing_context="spawn")
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, sampler=sampler, collate_fn=collate,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed), **worker_options,
    )
    # Sample validation cells across the whole split, not just its first shard.
    n_val = min(len(val_data), args.val_batches * args.batch_size)
    val_indices = random.Random(args.seed + 1).sample(range(len(val_data)), n_val)
    val_loader = DataLoader(
        Subset(val_data, val_indices), batch_size=args.batch_size, collate_fn=collate,
        num_workers=0, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed + 1),
    )

    # 2. Create the model and optimizer. Parameters stay float32; autocast below
    # uses bfloat16 for suitable GPU operations. BF16 does not need a GradScaler.
    model = Andre().to(device)
    if (model.gene_embedding.num_embeddings != train_data.vocab_size
            or model.out_layer.out_features != train_data.vocab_size - 2):
        raise ValueError("Model vocabulary sizes must match catalog.json (including PAD/MASK at input only)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_step, epoch, best_loss = 0, 0, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"]
        best_loss = checkpoint["best_val_loss"]
        # Resume weights/optimizer, but start a fresh shuffle rather than claim
        # to restore worker prefetch queues and the exact previous data position.
        epoch = checkpoint["epoch"] + 1
        del checkpoint
    if start_step >= args.steps:
        raise ValueError("--steps must exceed the checkpoint's completed step count")
    sampler.set_epoch(epoch)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "last.pt").exists() and not args.resume:
        raise FileExistsError("This output directory already has last.pt; use --resume or a new --out")
    parameters = sum(p.numel() for p in model.parameters())
    config = {**vars(args), "parameters": parameters, "actual_device": str(device),
              "actual_precision": "bfloat16" if use_bf16 else "float32",
              "train_cells": len(train_data), "validation_cells": n_val}
    print(f"Device: {device}; precision: {config['actual_precision']}; parameters: {parameters:,}")
    print(f"Train cells: {len(train_data):,}; fixed validation cells: {n_val:,}", flush=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    # Each invocation gets its own W&B run, even when resuming a checkpoint.
    # Disabled/offline modes never require a W&B login.
    with wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                    name=args.run_name, mode=args.wandb_mode, dir=str(out_dir),
                    config=config) as run, (out_dir / "metrics.jsonl").open("a") as metrics_file:
        run.define_metric("step")
        run.define_metric("*", step_metric="step")
        model.train()
        batches = iter(train_loader)
        loss_sum = correct = cells = 0
        window_start = time.perf_counter()

        for step in range(start_step + 1, args.steps + 1):
            try:
                batch = next(batches)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                batches = iter(train_loader)
                batch = next(batches)
            batch = move_batch(batch, device)
            targets = batch["targets"] - 2

            # 3. Warm up the learning rate, then hold it constant.
            lr = args.lr * min(1.0, step / max(1, args.warmup_steps))
            for group in optimizer.param_groups:
                group["lr"] = lr

            # 4. The core training step: clear -> predict -> loss -> gradients -> update.
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = predict(model, batch)
                loss = nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at step {step}")
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip,
                                                error_if_nonfinite=True)
            optimizer.step()

            n = targets.numel()
            loss_sum += loss.item() * n
            correct += (logits.detach().argmax(dim=-1) == targets).sum().item()
            cells += n
            should_eval = step % args.eval_every == 0 or step == args.steps
            should_log = step == start_step + 1 or step % args.log_every == 0 or should_eval
            if should_log:
                metrics = {"step": step, "epoch": epoch, "train/loss": loss_sum / cells,
                           "train/accuracy": correct / cells, "train/lr": lr,
                           "train/grad_norm": grad_norm.item(),
                           "train/cells_per_second": cells / (time.perf_counter() - window_start)}
                print(f"Step {step:>6} | loss {metrics['train/loss']:.4f} | "
                      f"accuracy {metrics['train/accuracy']:.2%} | lr {lr:.2g}", flush=True)
                loss_sum = correct = cells = 0

                # 5. Evaluate without learning from validation; save resumable state.
                if should_eval:
                    validation = evaluate(model, val_loader, device, use_bf16, args.seed + 1)
                    metrics.update(validation)
                    improved = validation["val/loss"] < best_loss
                    best_loss = min(best_loss, validation["val/loss"])
                    save_checkpoint(out_dir / "last.pt", model, optimizer, step, epoch, best_loss, args)
                    if improved:
                        save_checkpoint(out_dir / "best.pt", model, optimizer, step, epoch, best_loss, args)
                    print(f"  Validation | loss {validation['val/loss']:.4f} | "
                          f"accuracy {validation['val/accuracy']:.2%} | checkpoints saved", flush=True)
                run.log(metrics)
                metrics_file.write(json.dumps(metrics) + "\n")
                metrics_file.flush()
                window_start = time.perf_counter()
        run.summary["best_val_loss"] = best_loss


if __name__ == "__main__":
    main()
