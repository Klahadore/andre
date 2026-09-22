"""Measure end-to-end CPU batch loading (including sampling/masking).

Reports warm-cache throughput when the pilot fits RAM; this is not a cold-NVMe
benchmark and does not include GPU transfer or model compute.
"""
import argparse
from functools import partial
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import BlockShuffleSampler, ScBaseCountDataset, collate_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--length", type=int, default=512)
    args = parser.parse_args()
    if args.batches < 1 or args.workers < 0:
        parser.error("batches must be positive; workers must be nonnegative")
    torch.set_num_threads(1)
    ds = ScBaseCountDataset(args.root, args.split)
    kwargs = ({"persistent_workers": True, "prefetch_factor": 2,
               "multiprocessing_context": "spawn"} if args.workers else {})
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
                        sampler=BlockShuffleSampler(ds),
                        collate_fn=partial(collate_fn, length=args.length), **kwargs)
    if not len(ds):
        raise ValueError("Selected split is empty")
    started = time.perf_counter()
    iterator = iter(loader)
    batch = next(iterator)
    print(f"First batch (including worker startup): {time.perf_counter()-started:.3f}s")
    del batch
    cells = batches = 0
    started = time.perf_counter()
    for batch in iterator:
        cells += len(batch["targets"])
        batches += 1
        if batches >= args.batches:
            break
    elapsed = time.perf_counter() - started
    print(f"{cells:,} cells / {batches} subsequent batches / {elapsed:.3f}s "
          f"= {cells/max(elapsed, 1e-9):,.0f} cells/s; workers={args.workers}")


if __name__ == "__main__":
    main()
