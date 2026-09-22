"""Benchmark packed LMDB records and the real CPU training input pipeline.

First-touch reads use the current OS cache, not a deliberately cold SSD. Raw
reads copy every value byte. Pipeline trials explicitly prewarm the same input
cells, include IPC/sampling/masking, and exclude GPU transfer/model execution.
"""
import argparse
from datetime import datetime, timezone
from functools import partial
from itertools import islice
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"n": len(values), "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99))}


def read_bytes():
    path = Path("/proc/self/io")
    if not path.exists():
        return None
    return int(dict(line.split(": ") for line in path.read_text().splitlines())["read_bytes"])


def validate_batch(batch, vocab_size):
    ids, counts, mask = (batch[k] for k in ("gene_ids", "counts", "attention_mask"))
    assert ids.shape == counts.shape == mask.shape
    assert ids.dtype == torch.int64 and counts.dtype == torch.float32
    assert mask.dtype == torch.bool and torch.isfinite(counts).all()
    assert ((ids >= 0) & (ids < vocab_size)).all()
    assert (counts[mask] > 0).all() and (counts[~mask] == 0).all()
    assert (ids[~mask] == 0).all() and ((ids == dataset.MASK_ID).sum(1) == 1).all()
    rows = torch.arange(len(ids))
    assert (ids[rows, batch["mask_positions"]] == dataset.MASK_ID).all()
    assert mask[rows, batch["mask_positions"]].all()
    assert ((batch["targets"] >= 2) & (batch["targets"] < vocab_size)).all()


def benchmark_records(ds, args):
    rng = np.random.default_rng(args.seed)
    sizes, genes, tensor_bytes, open_us = [], [], [], []
    first_us, hot_us, decode_us, cell_us, loader_us = [], [], [], [], []
    checks = []
    physical_before = read_bytes()
    # Choose shards proportional to their cell count, then uniform full records.
    # Partial tail records are excluded from the 32-cell latency/size statistics.
    for global_index in rng.integers(len(ds), size=args.shard_samples):
        shard, _ = ds._locate(int(global_index))
        start = ds.ends[shard - 1] if shard else 0
        count = ds.ends[shard] - start
        full_records = count // dataset.CELLS_PER_RECORD
        if not full_records:
            continue
        t = time.perf_counter_ns()
        env = dataset._environment(ds.root / ds.shards[shard])
        open_us.append((time.perf_counter_ns() - t) / 1000)
        for record in rng.integers(full_records, size=args.records_per_shard):
            key = dataset.cell_key(ds.split, int(record))
            t = time.perf_counter_ns()
            with env.begin(buffers=False) as txn:
                value = txn.get(key)
            first_us.append((time.perf_counter_ns() - t) / 1000)
            assert value is not None
            t = time.perf_counter_ns()
            with env.begin(buffers=False) as txn:
                hot = txn.get(key)
            hot_us.append((time.perf_counter_ns() - t) / 1000)
            assert value == hot
            sizes.append(len(value))
            t = time.perf_counter_ns()
            cell = dataset.unpack_cell(value, 0)
            cell_us.append((time.perf_counter_ns() - t) / 1000)
            t = time.perf_counter_ns()
            decoded = [dataset.unpack_cell(value, i) for i in range(dataset.CELLS_PER_RECORD)]
            decode_us.append((time.perf_counter_ns() - t) / 1000)
            n_genes = sum(len(c["gene_ids"]) for c in decoded)
            genes.append(n_genes)
            tensor_bytes.append(n_genes * 12)  # int64 gene ID + float32 count.
            indices = range(start + int(record) * dataset.CELLS_PER_RECORD,
                            start + (int(record) + 1) * dataset.CELLS_PER_RECORD)
            t = time.perf_counter_ns()
            loaded = ds.__getitems__(indices)
            loader_us.append((time.perf_counter_ns() - t) / 1000)
            for expected, actual in zip(decoded, loaded):
                assert all(torch.equal(expected[k], actual[k]) for k in expected)
            checks.append((indices.start, cell))
    physical_after = read_bytes()
    if not sizes:
        raise ValueError("No full 32-cell records available")
    # Verify request order and duplicates across shards and LRU eviction.
    requested = checks[::max(1, len(checks) // 64)][::-1]
    requested += requested[:3]
    loaded = ds.__getitems__([i for i, _ in requested])
    for (_, expected), actual in zip(requested, loaded):
        assert all(torch.equal(expected[k], actual[k]) for k in expected)
    dataset.close_environments()
    return {
        "sampling": "cell-weighted shard sampling with replacement; uniform full records within shard",
        "cells_per_record": dataset.CELLS_PER_RECORD,
        "packed_bytes": stats(sizes), "expressed_gene_pairs_per_record": stats(genes),
        "decoded_tensor_bytes": stats(tensor_bytes),
        "latency_us": {"shard_handle_lookup_or_open": stats(open_us),
                       "current_cache_get_and_copy": stats(first_us),
                       "hot_get_and_copy": stats(hot_us),
                       "decode_one_cell": stats(cell_us),
                       "decode_all_32_cells": stats(decode_us),
                       "hot_dataset_getitems_32_cells": stats(loader_us)},
        "process_storage_read_bytes_during_record_test": (
            physical_after - physical_before if physical_before is not None else None),
        "correctness": "every sampled full record equals batched loader output; mixed-shard ordering and duplicates pass",
    }


def benchmark_pipeline(ds, args):
    n_batches = args.warmup_batches + args.batches + 1
    indices = list(islice(iter(dataset.BlockShuffleSampler(ds, seed=args.seed)),
                          n_batches * args.batch_size))
    if len(indices) != n_batches * args.batch_size:
        raise ValueError("Split too small for the requested benchmark")
    print(f"Prewarming {len(indices):,} cells for identical pipeline trials...", flush=True)
    for offset in range(0, len(indices), 4096):
        ds.__getitems__(indices[offset:offset + 4096])
    dataset.close_environments()
    rng = np.random.default_rng(args.seed)
    results = []
    for repeat in range(args.repeats):
        for workers in rng.permutation(args.worker_counts):
            workers = int(workers)
            torch.manual_seed(args.seed)
            kwargs = ({"multiprocessing_context": "spawn", "prefetch_factor": 2}
                      if workers else {})
            loader = DataLoader(ds, batch_size=args.batch_size, num_workers=workers,
                                sampler=indices, collate_fn=partial(dataset.collate_fn, length=args.length),
                                **kwargs)
            t = time.perf_counter()
            iterator = iter(loader)
            batch = next(iterator)
            startup = time.perf_counter() - t
            validate_batch(batch, ds.vocab_size)
            for _ in range(args.warmup_batches):
                next(iterator)
            waits = []
            t = time.perf_counter()
            for _ in range(args.batches):
                before = time.perf_counter_ns()
                batch = next(iterator)
                waits.append((time.perf_counter_ns() - before) / 1e6)
            elapsed = time.perf_counter() - t
            validate_batch(batch, ds.vocab_size)
            result = {"repeat": repeat, "workers": workers, "batch_size": args.batch_size,
                      "sequence_length": args.length, "measured_batches": args.batches,
                      "warmup_batches": args.warmup_batches, "pin_memory": False,
                      "startup_seconds": startup, "elapsed_seconds": elapsed,
                      "cells_per_second": args.batches * args.batch_size / elapsed,
                      "batch_wait_ms": stats(waits),
                      "batch_tensor_bytes": sum(v.numel() * v.element_size() for v in batch.values())}
            results.append(result)
            print(json.dumps(result), flush=True)
            # Finite sampler: normal exhaustion joins all spawned workers.
            assert next(iterator, None) is None
            del iterator, loader
            dataset.close_environments()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--split", default="train", choices=dataset.SPLITS)
    parser.add_argument("--worker-counts", nargs="+", type=int, default=[0, 2, 4, 8])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--batches", type=int, default=512)
    parser.add_argument("--warmup-batches", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--shard-samples", type=int, default=128)
    parser.add_argument("--records-per-shard", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(v < 1 for v in (args.batch_size, args.length, args.batches, args.repeats,
                          args.shard_samples, args.records_per_shard)):
        parser.error("sample sizes and repetitions must be positive")
    if args.warmup_batches < 0 or any(w < 0 for w in args.worker_counts):
        parser.error("worker counts and warmup must be nonnegative")
    torch.set_num_threads(1)
    ds = dataset.ScBaseCountDataset(args.root, args.split)
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "platform": platform.platform(), "logical_cpus": os.cpu_count(),
              "torch_version": torch.__version__, "split": args.split, "split_cells": len(ds),
              "seed": args.seed, "shuffle_block_cells": 4096,
              "cache_policy": "no cache eviction; record first-touch uses current OS cache; pipeline input explicitly prewarmed",
              "scope": "CPU only; no GPU transfer/model; pipeline includes decompression, random gene selection, masking and worker IPC"}
    report["records"] = benchmark_records(ds, args)
    print(json.dumps({"records": report["records"]}), flush=True)
    report["pipeline"] = benchmark_pipeline(ds, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"All live-data checks passed. Report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
