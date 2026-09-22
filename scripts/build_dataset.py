"""Convert local scBaseCount H5ADs into immutable, resumable LMDB shards.

Run with --limit first to measure output size before converting the corpus.
Source H5ADs are never modified or deleted. Only X is read, not alternate layers.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
# Each worker uses one CPU; avoid multiplying BLAS/DuckDB thread pools.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import fcntl
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack
import zlib

import anndata as ad
import duckdb
# DuckDB's module-level default connection otherwise starts a thread pool sized
# to the whole host in every process, even when explicit connections use one.
duckdb.execute("SET threads=1")
import h5py
import lmdb
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_codec import CELLS_PER_RECORD, FORMAT_VERSION, SPLITS, cell_key, encode_cell, pack_cells
from scripts.fast_h5 import read_vector
from scripts.fast_encode import prepare_cells


MIN_GENES, MIN_UMIS = 300, 500


def atomic_json(path, content):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(content, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def load_vocab(path):
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"Vocabulary not found: {Path(path).resolve()}. Copy the notebook's "
            "data/gene_to_id.csv from your local checkout to this path, or pass "
            "--vocab /absolute/path/gene_to_id.csv. Do not regenerate IDs from "
            "an arbitrary sample: existing model token IDs must remain stable."
        )
    frame = pd.read_csv(path)
    if (frame["gene"].duplicated().any() or frame["token_id"].duplicated().any()
            or frame["gene"].isna().any()):
        raise ValueError("Vocabulary genes and token IDs must be unique and non-null")
    frame = frame.sort_values("token_id")
    if not np.array_equal(frame.token_id, np.arange(len(frame))):
        raise ValueError("Vocabulary token IDs must be contiguous starting at zero")
    if frame.gene.iloc[:2].tolist() != ["<pad>", "<mask>"] or len(frame) > 65536:
        raise ValueError("Vocabulary must begin with PAD/MASK and fit uint16")
    mapping = dict(zip(frame.gene, frame.token_id))
    digest = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
    return mapping, digest


def notebook_splits(accession, barcodes):
    """Same DuckDB hash and 80/10/10 cell-level assignment as the notebook."""
    frame = pd.DataFrame({"cell_barcode": barcodes})
    with duckdb.connect(config={"threads": 1}) as db:
        db.register("cells", frame)
        buckets = db.execute(
            "SELECT hash(? || cell_barcode) % 10 FROM cells", [accession]
        ).fetchnumpy()
    buckets = next(iter(buckets.values()))
    return np.where(buckets < 8, 0, np.where(buckets == 8, 1, 2))


def _write_batch(env, records):
    while True:
        try:
            with env.begin(write=True) as txn:
                for key, value in records:
                    if not txn.put(key, value, overwrite=False):
                        raise ValueError(f"Duplicate cell key: {key!r}")
            return
        except lmdb.MapFullError:
            env.set_mapsize(env.info()["map_size"] * 2)


def build_shard(source, destination, accession, vocab, vocab_hash, *,
                batch_size=2048, matrix_memory_gib=8, reserve_gib=10):
    source, destination = Path(source), Path(destination)
    stat = source.stat()
    signature = {
        "format_version": FORMAT_VERSION, "accession": accession,
        "source_bytes": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
        "vocab_sha256": vocab_hash, "min_genes": MIN_GENES, "min_umis": MIN_UMIS,
        "split_policy": "duckdb_hash_accession_concat_barcode_mod10_80_10_10",
        "duckdb_version": duckdb.__version__, "codec": "zlib1_delta_u16_shuffle_f32",
        "cells_per_record": CELLS_PER_RECORD,
    }
    if destination.exists():
        meta = json.loads((destination / "metadata.json").read_text())
        if meta["signature"] != signature or not (destination / "data.mdb").is_file():
            raise ValueError(f"Existing shard differs from input/config: {destination}")
        return meta

    with h5py.File(source, "r") as h5:
        x = h5["X"]
        encoding = x.attrs.get("encoding-type")
        if encoding not in ("csc_matrix", "csr_matrix"):
            raise ValueError(f"Expected sparse CSC/CSR X, got {encoding}")
        n_cells, n_genes = map(int, x.attrs["shape"])
        # Conservative source+CSR+conversion workspace bound. Never load layers.
        estimate = len(x["data"]) * 32 + (n_cells + n_genes + 2) * 64
        if estimate > matrix_memory_gib * 2**30:
            raise MemoryError(
                f"{accession}: estimated conversion workspace {estimate / 2**30:.1f} GiB "
                f"exceeds --matrix-memory-gib={matrix_memory_gib}; raise the budget "
                "on a host with sufficient available RAM"
            )
        obs = h5["obs"]
        barcodes = np.asarray(ad.io.read_elem(obs[obs.attrs["_index"]]), dtype=str)
        gene_names = np.asarray(ad.io.read_elem(h5["var"][h5["var"].attrs["_index"]]), dtype=str)
        if len(set(gene_names)) != n_genes or len(set(barcodes)) != n_cells:
            raise ValueError(f"Duplicate gene IDs or cell barcodes in {accession}")
        if "SRX_accession" in obs:
            source_accessions = ad.io.read_elem(obs["SRX_accession"])
            if not np.all(np.asarray(source_accessions) == accession):
                raise ValueError(f"Accession mismatch in {source}")
        unknown = set(gene_names) - vocab.keys()
        if unknown:
            raise ValueError(f"{accession}: {len(unknown)} genes outside vocabulary, e.g. {sorted(unknown)[:3]}")
        column_tokens = np.array([vocab[g] for g in gene_names], dtype=np.uint16)
        keep = np.flatnonzero((obs["gene_count_Unique"][:] >= MIN_GENES)
                              & (obs["umi_count_Unique"][:] >= MIN_UMIS))
        split_ids = notebook_splits(accession, barcodes[keep])
        constructor = csc_matrix if encoding == "csc_matrix" else csr_matrix
        matrix = constructor((read_vector(x["data"]), read_vector(x["indices"]),
                              read_vector(x["indptr"])), shape=(n_cells, n_genes)).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if np.any(column_tokens[1:] <= column_tokens[:-1]):
        order = np.argsort(column_tokens)
        matrix = matrix[:, order].tocsr()
        column_tokens = column_tokens[order]
    matrix.sort_indices()
    prepared = prepare_cells(matrix, column_tokens, keep)
    if source.stat().st_mtime_ns != stat.st_mtime_ns or source.stat().st_size != stat.st_size:
        raise RuntimeError(f"Source changed while reading: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{accession}-", dir=destination.parent))
    env = None
    counts = {split: 0 for split in SPLITS}
    payload_bytes = nnz = 0
    try:
        # Temporary shards can be rebuilt. Flush once before publication instead
        # of fsync on every small transaction; final published shards are durable.
        env = lmdb.open(str(temporary), map_size=64 * 2**20, max_spare_txns=0,
                        sync=False, metasync=False)
        records = []
        pending = {split: [] for split in SPLITS}
        for cell_index, (row, split_id) in enumerate(zip(keep, split_ids)):
            start, end = matrix.indptr[row:row+2]
            # Fail on invalid data rather than silently changing cohort membership.
            if prepared is None:
                value = encode_cell(column_tokens[matrix.indices[start:end]], matrix.data[start:end])
            else:
                raw, offsets = prepared
                value = zlib.compress(memoryview(raw)[offsets[cell_index]:offsets[cell_index+1]], level=1)
            split = SPLITS[int(split_id)]
            pending[split].append(value)
            counts[split] += 1
            nnz += int(end - start)
            payload_bytes += len(value)
            if len(pending[split]) == CELLS_PER_RECORD:
                block_index = (counts[split] - 1) // CELLS_PER_RECORD
                records.append((cell_key(split, block_index), pack_cells(pending[split])))
                pending[split] = []
            if len(records) >= max(1, batch_size // CELLS_PER_RECORD):
                _check_space(temporary, records, reserve_gib)
                _write_batch(env, records)
                records = []
        for split, cells in pending.items():
            if cells:
                records.append((cell_key(split, counts[split] // CELLS_PER_RECORD), pack_cells(cells)))
        if records:
            _check_space(temporary, records, reserve_gib)
            _write_batch(env, records)
        env.sync(True)
        env.close()
        env = None
        meta = {"signature": signature, "counts": counts, "source_cells": n_cells,
                "nnz": nnz, "payload_bytes": payload_bytes,
                "database_bytes": (temporary / "data.mdb").stat().st_size}
        atomic_json(temporary / "metadata.json", meta)
        temporary.rename(destination)
        return meta
    finally:
        if env is not None:
            env.close()
        if temporary.exists():
            shutil.rmtree(temporary)


def _check_space(path, records, reserve_gib):
    # Include page rounding and transaction overhead, not just payload bytes.
    needed = sum(((len(v) // 4096) + 2) * 4096 for _, v in records)
    if shutil.disk_usage(path).free < reserve_gib * 2**30 + 2 * needed:
        raise OSError("Insufficient disk space; completed shards are reusable on the next run")


def selected_accessions(data_root):
    """Use the local sample metadata for the notebook's sample-level filter."""
    with duckdb.connect(config={"threads": 1}) as db:
        rows = db.execute("""
            SELECT DISTINCT srx_accession FROM read_parquet(?)
            WHERE organism = 'Homo sapiens' AND tech_10x = '3_prime_gex'
              AND cell_prep = 'single_cell' ORDER BY srx_accession
        """, [str(data_root / "metadata/sample_metadata.parquet")]).fetchall()
    eligible = {row[0] for row in rows}
    # The downloader's list additionally restricts to samples with passing cells.
    requested = (data_root / "accessions.txt").read_text().splitlines()
    invalid = set(requested) - eligible
    if invalid:
        raise ValueError(f"Downloaded accessions outside the notebook cohort: {sorted(invalid)[:5]}")
    return sorted(set(requested))


_WORKER = None


def _init_worker(config):
    global _WORKER
    # Passing the 36k-entry mapping through every spawn pipe serializes worker
    # startup. Load its small local CSV once inside each child instead.
    if "vocab" not in config:
        config["vocab"], digest = load_vocab(config["vocab_path"])
        if digest != config["vocab_hash"]:
            raise ValueError("Vocabulary changed while starting workers")
    _WORKER = config


def _convert_accession(accession):
    config = _WORKER
    source = config["data_root"] / "h5ad" / f"{accession}.h5ad"
    destination = config["out"] / "shards" / accession
    published = config["publish_to"] / "shards" / accession if config["publish_to"] else destination
    # A published shard is immutable; validate it rather than copying/rebuilding.
    target = published if published.exists() else destination
    meta = build_shard(source, target, accession, config["vocab"], config["vocab_hash"],
                       matrix_memory_gib=config["matrix_memory_gib"],
                       reserve_gib=config["reserve_gib"])
    if published != destination and not published.exists():
        published.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(published.parent).free < (config["reserve_gib"] * 2**30
                                                        + meta["database_bytes"]):
            raise OSError("Insufficient free space at publication destination")
        temporary = Path(tempfile.mkdtemp(prefix=f".{accession}-copy-", dir=published.parent))
        try:
            for name in ("data.mdb", "metadata.json"):
                shutil.copy2(destination / name, temporary / name)
                with (temporary / name).open("rb") as f:
                    os.fsync(f.fileno())
            temporary.rename(published)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        shutil.rmtree(destination)
    return accession, meta


def convert_all(accessions, config, workers, memory_budget_gib):
    """Bound concurrency and estimated total memory; schedule large files first."""
    sizes = {acc: (config["data_root"] / "h5ad" / f"{acc}.h5ad").stat().st_size
             for acc in accessions}
    ordered = sorted(accessions, key=lambda a: sizes[a], reverse=True)
    budget = memory_budget_gib * 2**30
    def weight(acc):
        return min(config["matrix_memory_gib"] * 2**30 * 1.25,
                   sizes[acc] * 6 + 512 * 2**20)
    running = {}
    allocated = 0
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init_worker, initargs=(config,)) as pool:
        while ordered or running:
            while ordered and len(running) < workers:
                # Backfill with smaller files when a large file cannot fit in
                # remaining RAM, rather than leaving otherwise idle CPUs.
                choice = next((i for i, acc in enumerate(ordered)
                               if allocated + weight(acc) <= budget), None)
                if choice is None:
                    if not running:
                        raise MemoryError("A source exceeds --memory-budget-gib")
                    break
                acc = ordered.pop(choice)
                amount = weight(acc)
                running[pool.submit(_convert_accession, acc)] = amount
                allocated += amount
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                allocated -= running.pop(future)
                yield future.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, default=Path("data/gene_to_id.csv"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="build an explicit pilot subset")
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 16))
    parser.add_argument("--memory-budget-gib", type=float, default=128,
                        help="total estimated workspace budget across workers")
    parser.add_argument("--publish-to", type=Path,
                        help="copy completed local shards here, then remove local staging shards")
    parser.add_argument("--matrix-memory-gib", type=float, default=8)
    parser.add_argument("--reserve-gib", type=float, default=10)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.workers < 1 or args.memory_budget_gib <= 0:
        parser.error("workers and total memory budget must be positive")
    if args.matrix_memory_gib <= 0 or args.reserve_gib < 0:
        parser.error("memory budget must be positive; reserve must be nonnegative")
    vocab, vocab_hash = load_vocab(args.vocab)
    accessions = selected_accessions(args.data_root)
    if args.limit:
        accessions = accessions[:args.limit]
    if not accessions:
        raise ValueError("No selected accessions")
    for accession in accessions:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", accession):
            raise ValueError(f"Invalid accession: {accession!r}")
        path = args.data_root / "h5ad" / f"{accession}.h5ad"
        if not path.is_file():
            raise FileNotFoundError(f"Missing source {path}; finish downloading first")
    args.out.mkdir(parents=True, exist_ok=True)
    output_root = args.publish_to or args.out
    output_root.mkdir(parents=True, exist_ok=True)
    if args.publish_to and args.out.resolve() == args.publish_to.resolve():
        parser.error("--publish-to must differ from the local --out staging directory")
    config = dict(data_root=args.data_root, out=args.out, publish_to=args.publish_to,
                  vocab_path=args.vocab, vocab_hash=vocab_hash, matrix_memory_gib=args.matrix_memory_gib,
                  reserve_gib=args.reserve_gib)
    # Hold both locks when staging locally and publishing to another filesystem.
    with ExitStack() as stack:
        for root in sorted({args.out.resolve(), output_root.resolve()}):
            lock = stack.enter_context((root / ".build.lock").open("w"))
            # POSIX locks interoperate between a local writer and an NFS client;
            # mixing local BSD flock with NFS-emulated flock does not.
            fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        shards = {}
        totals = Counter()
        total_bytes = 0
        started = time.monotonic()
        for i, (accession, meta) in enumerate(convert_all(
                accessions, config, args.workers, args.memory_budget_gib)):
            totals.update(meta["counts"])
            total_bytes += meta["database_bytes"]
            shards[accession] = {"path": f"shards/{accession}", "counts": meta["counts"],
                                "database_bytes": meta["database_bytes"]}
            elapsed = time.monotonic() - started
            progress = {"files_done": i+1, "files_total": len(accessions),
                        "cells_done": sum(totals.values()), "elapsed_seconds": round(elapsed, 1),
                        "output_bytes": total_bytes}
            if (i+1) % 10 == 0 or i == 0 or i+1 == len(accessions):
                atomic_json(output_root / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        catalog = {"format_version": FORMAT_VERSION, "vocab_size": len(vocab),
                   "vocab_sha256": vocab_hash, "duckdb_version": duckdb.__version__,
                   "pilot": args.limit is not None, "counts": dict(totals),
                   "shards": [shards[acc] for acc in accessions]}
        atomic_json(output_root / "vocabulary.json", vocab)
        atomic_json(output_root / "catalog.json", catalog)
        size = sum(s["database_bytes"] for s in shards.values())
        cells = sum(totals.values())
        print(f"Ready: {cells:,} cells, {size/2**30:.2f} GiB, "
              f"{size/max(cells, 1):.0f} bytes/cell, {time.monotonic()-started:.1f}s")


if __name__ == "__main__":
    main()
