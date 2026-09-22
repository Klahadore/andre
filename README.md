# ANDRE dataset loader

`dataset.py` reads an **offline LMDB dataset on local SSD**. LMDB memory-maps its
files; the OS automatically keeps recently used pages in RAM and reclaims clean
pages under memory pressure. Workers share this page cache. There is no second
Python cell cache, database server, or fixed RAM/SSD split to manage. `map_size`
is an address-space limit, not a RAM reservation. This does not impose a hard
cache-memory limit. [LMDB memory documentation](https://lmdb.readthedocs.io/en/latest/index.html#memory-usage)

The source is the scBaseCount 2026-01-12 `GeneFull_Ex50pAS/Homo_sapiens` data in
`scripts/data_explore.ipynb`: human 10x 3′ GEX whole cells, at least 300 detected
genes and 500 UMIs. The notebook reports 153,486,103 retained cells and 36,603
vocabulary entries including PAD=0 and MASK=1. The actual converted totals are
recorded in `catalog.json`.

## Build once

Use the downloader's `data/metadata`, `data/accessions.txt`, and `data/h5ad`
layout, plus the notebook's `data/gene_to_id.csv`. The converter checks the
sample-level cohort against local sample metadata, then applies QC from each
H5AD's `obs`. It reads **only X**, preserving all expressed genes and raw counts.
It maps each source gene identifier to the vocabulary; it does not assume every
file has the same column order. Unknown genes fail explicitly.

On the observed server the data root is `/opt/dlami/nvme/andre/data` (the `/opt`
directory itself is on the smaller root volume). After making this code and
`gene_to_id.csv` available there, start with a separate pilot output:

```sh
uv run scripts/build_dataset.py \
  --data-root /opt/dlami/nvme/andre/data \
  --vocab data/gene_to_id.csv \
  --out /opt/dlami/nvme/andre/lmdb-pilot --limit 20
```

For the full build, omit `--limit` and choose the intended output directory.
Re-running the same command reuses completed shards, validating source size,
mtime, vocabulary, codec, QC, and DuckDB version. An incomplete shard is never
published. The catalog is published after all requested sources succeed.
Source files are never deleted. A hard kill can leave a hidden temporary shard
directory; it is not used on restart and can be removed while no builder runs.

Conversion runs in separate processes (`--workers`, default up to 16), loading
one sparse X matrix per worker and converting CSC to CSR once. A memory-aware
scheduler starts large samples early and fills unused capacity with smaller
ones. `--memory-budget-gib` bounds estimated combined conversion workspace.
Gzip-only HDF5 arrays use system libdeflate when available; Numba prepares the
record bytes in compiled loops. Other layouts and installations fall back to
the original decoders. Both paths produce the same lossless record format.
Temporary shards flush once before publication rather than on each transaction. The default estimated per-matrix workspace budget is 8 GiB;
`--matrix-memory-gib` increases it for larger samples on a host with room.
`--reserve-gib` defaults to 10 and stops conversion before consuming the disk
reserve. DuckDB's hash version is recorded because the notebook's cell-level
80/10/10 split depends on it. This preserves the existing split policy, which
does not hold out whole donors/studies.

Each accession becomes an immutable LMDB shard. Within each split, one key
holds 32 independently compressed cells plus offsets, reducing page waste while
allowing individual cells to be decoded. Records contain sorted delta-encoded
uint16 gene tokens and byte-shuffled float32 counts with zlib compression. Counts
are not capped or packed into uint16. Shards must stay immutable while readers
are running; the reader disables LMDB locking on that basis.

## Load batches

This replaces the old `ScBaseCountDataset(csv, gene_to_id)` API with
`ScBaseCountDataset(lmdb_root, split)`; training does not load the giant CSVs,
open HDF5, access the network, or load a whole matrix.

```python
import torch
from torch.utils.data import DataLoader
from dataset import ScBaseCountDataset, BlockShuffleSampler, collate_fn

# Put DataLoader construction inside main() with an __main__ guard when
# using spawn in a Python script.
dataset = ScBaseCountDataset("/opt/dlami/nvme/andre/lmdb", split="train")
sampler = BlockShuffleSampler(dataset, block_size=4096, seed=42)
loader = DataLoader(
    dataset, batch_size=256, sampler=sampler, collate_fn=collate_fn,
    num_workers=4, persistent_workers=True, prefetch_factor=2,
    multiprocessing_context="spawn", pin_memory=True,
    generator=torch.Generator().manual_seed(42),
)
for epoch in range(10):
    sampler.set_epoch(epoch)
    for batch in loader:
        # Transfer tensors to the GPU with non_blocking=True as needed.
        # Your training step goes here.
        pass
```

The sampler shuffles blocks and cells within each block using bounded memory.
It visits every cell once per epoch, but is a locality-aware shuffle rather than
a uniform global permutation. It is a single-process training sampler; it does
not partition work between distributed training ranks.

Each batch has `gene_ids`, `counts`, and `attention_mask` shaped `[B, 512]`, plus
`targets` and `mask_positions` shaped `[B]`. Every visit samples up to 512 genes
without replacement and masks one identity; the corresponding count remains.
`attention_mask=True` means a real entry, including MASK. Pass
`~batch["attention_mask"]` as PyTorch Transformer's `src_key_padding_mask`.
Use `dataset.vocab_size` for the model's embedding and prediction-head sizes
(36,603 for this vocabulary). The model consumes the same batch interface.

Reader handles open lazily in each worker, with at most 32 open shards per
process. Batch reads group requests by shard and packed record. Multiple
Dataset instances share handles inside a process; returned tensors own their
memory. Use `close_environments()` to release reader handles explicitly.

## Validation and measurements

```sh
uv run python -m unittest discover -s tests -v
uv run scripts/benchmark_dataset.py /path/to/lmdb-pilot --workers 4
```

The local pilot for ERX8792169 retained 7,410 cells and produced a 32.0 MiB
LMDB file (about 4,530 bytes/cell). Every retained cell's genes and counts matched
the source exactly. A short local warm-cache run with four workers delivered
about 23,500 cells/s after startup, including sampling and collation. This is
not a cold-NVMe or H100 training benchmark.

The remote snapshot had about 1,022 GiB free. Extrapolating this single sample
to the notebook total gives roughly 648 GiB, but sizes vary by sample. Measure a
broader pilot and retain headroom before the full conversion.

The optimized converter processed a 64-file, 524,767-cell pilot in 15.7 seconds
on a temporary c7i.48xlarge using 32 workers, including worker startup. The
source was read over private NFS from the H100 host. A 30,885-cell comparison
against the original converter matched every gene ID and count. This pilot
timing is not a full-corpus runtime guarantee.

For conversion on another machine, use `--out /local/staging` and
`--publish-to /mounted/destination`. LMDB writes happen on the local filesystem;
closed, flushed shards are copied and atomically published at the destination.
Local copies are removed after publication, and the final catalog appears only
after all requested shards finish. Do not write active LMDB environments over
NFS. Run only one builder for a destination; file locks enforce this.

The exact notebook vocabulary is required. If it is missing remotely, copy
`data/gene_to_id.csv` from the local checkout; do not regenerate a differently
ordered vocabulary. The converter now reports the absolute missing path and
this recovery step.

## Completed full build (2026-09-22 UTC)

The complete dataset is available on the H100 host at
`/opt/dlami/nvme/andre/lmdb`; no rebuild is needed to start training.

- 18,758 shards; 153,486,103 cells; 492.10 GiB of LMDB data.
- Train: 122,793,289; validation: 15,345,138; test: 15,347,676.
- Conversion: approximately 10 minutes 20 seconds including a scheduler restart;
  the resumed run took 515.5 seconds. It used 96 workers on a temporary
  c7i.48xlarge in the same availability zone, publishing to the H100 NVMe.
- Verification: all shard sizes and packed-record counts checked; cell and
  nonzero totals match the independent source audit exactly (347,909,504,795
  nonzero entries). 124 cells sampled across source files, including each
  sampled file's maximum-count cell where retained, match independent H5AD
  reads. Verification took 65 seconds.
- H100-host CPU loader smoke benchmark: 25,600 cells in 1.540 seconds after
  startup, or about 16,600 cells/s with four workers and batches of 256. This
  includes sampling/collation and excludes model execution and GPU transfer;
  it is not a controlled cold-cache benchmark.

The vocabulary CSV is tracked in Git. The output directory includes
`build.log`, `build-run.json`, and `verification.json` for provenance.

```sh
uv run scripts/verify_dataset.py /opt/dlami/nvme/andre/lmdb \
  --data-root /opt/dlami/nvme/andre/data \
  --expected-cells 153486103 --expected-nnz 347909504795
```
