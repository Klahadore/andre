"""Convert local scBaseCount H5ADs into immutable, resumable LMDB shards.

Run with --limit first to measure output size before converting the corpus.
Source H5ADs are never modified or deleted. Only X is read, not alternate layers.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import fcntl

import anndata as ad
import duckdb
import h5py
import lmdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import CELLS_PER_RECORD, FORMAT_VERSION, SPLITS, cell_key, encode_cell, pack_cells


MIN_GENES, MIN_UMIS = 300, 500


def atomic_json(path, content):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(content, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def load_vocab(path):
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
    with duckdb.connect() as db:
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
        matrix = ad.io.read_elem(x).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if source.stat().st_mtime_ns != stat.st_mtime_ns or source.stat().st_size != stat.st_size:
        raise RuntimeError(f"Source changed while reading: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{accession}-", dir=destination.parent))
    env = None
    counts = {split: 0 for split in SPLITS}
    payload_bytes = nnz = 0
    try:
        env = lmdb.open(str(temporary), map_size=64 * 2**20, max_spare_txns=0)
        records = []
        pending = {split: [] for split in SPLITS}
        for row, split_id in zip(keep, split_ids):
            start, end = matrix.indptr[row:row+2]
            gene_ids = column_tokens[matrix.indices[start:end]]
            values = matrix.data[start:end]
            # Fail on invalid data rather than silently changing cohort membership.
            value = encode_cell(gene_ids, values)
            split = SPLITS[int(split_id)]
            pending[split].append(value)
            counts[split] += 1
            nnz += len(values)
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
        env.sync()
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
    with duckdb.connect() as db:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, default=Path("data/gene_to_id.csv"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="build an explicit pilot subset")
    parser.add_argument("--matrix-memory-gib", type=float, default=8)
    parser.add_argument("--reserve-gib", type=float, default=10)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
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
    # Prevent competing builders from publishing the same output root.
    with (args.out / ".build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        shards = []
        totals = Counter()
        started = time.monotonic()
        for i, accession in enumerate(accessions):
            relative = f"shards/{accession}"
            meta = build_shard(
                args.data_root / "h5ad" / f"{accession}.h5ad", args.out / relative,
                accession, vocab, vocab_hash, matrix_memory_gib=args.matrix_memory_gib,
                reserve_gib=args.reserve_gib,
            )
            totals.update(meta["counts"])
            shards.append({"path": relative, "counts": meta["counts"],
                           "database_bytes": meta["database_bytes"]})
            print(f"{i+1}/{len(accessions)} {accession}: {meta['counts']}, "
                  f"{meta['database_bytes']/2**20:.1f} MiB", flush=True)
        catalog = {"format_version": FORMAT_VERSION, "vocab_size": len(vocab),
                   "vocab_sha256": vocab_hash, "duckdb_version": duckdb.__version__,
                   "pilot": args.limit is not None, "counts": dict(totals), "shards": shards}
        atomic_json(args.out / "vocabulary.json", vocab)
        atomic_json(args.out / "catalog.json", catalog)
        size = sum(s["database_bytes"] for s in shards)
        cells = sum(totals.values())
        print(f"Ready: {cells:,} cells, {size/2**30:.2f} GiB, "
              f"{size/max(cells, 1):.0f} bytes/cell, {time.monotonic()-started:.1f}s")


if __name__ == "__main__":
    main()
