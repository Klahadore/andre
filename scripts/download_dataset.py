"""
This file was written by Claude
"""


"""Download the scBaseCount metadata and the h5ad matrices we train on.

Every step is idempotent: re-running the same command skips files that are
already complete and retries anything that failed.

    uv run scripts/download_dataset.py --accessions SRX16217046   # smoke test
    uv run scripts/download_dataset.py --dry-run                  # total size only
    uv run scripts/download_dataset.py --workers 16               # the real thing
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
import gcsfs
from tqdm import tqdm


BUCKET = "arc-institute-virtual-cell-atlas/scbasecount/2026-01-12"
METADATA_PREFIX = f"{BUCKET}/metadata/GeneFull_Ex50pAS/Homo_sapiens"
H5AD_PREFIX = f"{BUCKET}/h5ad/GeneFull_Ex50pAS/Homo_sapiens"
METADATA_FILES = ["sample_metadata.parquet", "obs_metadata.parquet"]

fs = gcsfs.GCSFileSystem(token="anon")


def download_one(remote: str, local: Path, size: int, retries: int = 5) -> str:
    """Download one object to `local`, verifying its byte size.

    Writes to a `.part` file and renames on success so an interrupted run never
    leaves a truncated file that looks complete.
    """
    if local.exists() and local.stat().st_size == size:
        return "skipped"

    tmp = local.with_suffix(local.suffix + ".part")
    last: Exception | None = None
    for attempt in range(retries):
        try:
            fs.get_file(remote, str(tmp))
            actual = tmp.stat().st_size
            if actual != size:
                raise IOError(f"size mismatch: got {actual}, expected {size}")
            tmp.rename(local)
            return "downloaded"
        except Exception as err:
            tmp.unlink(missing_ok=True)
            last = err
            time.sleep(2**attempt)
    raise RuntimeError(f"{remote}: {last}")


def download_metadata(meta_dir: Path) -> None:
    for name in METADATA_FILES:
        remote = f"{METADATA_PREFIX}/{name}"
        size = fs.size(remote)
        status = download_one(remote, meta_dir / name, size)
        print(f"{name}: {status} ({size / 1e9:.2f} GB)")


def select_accessions(meta_dir: Path, memory_limit: str) -> list[str]:
    """Same filter as scripts/data_explore.ipynb, run against the local parquet files.

    Human, 10x 3' gene expression, single cell, and only samples that have at
    least one cell passing the gene/UMI thresholds.
    """
    db = duckdb.connect()
    db.execute(f"SET memory_limit = '{memory_limit}'")
    db.execute(f"SET temp_directory = '{meta_dir / 'duckdb_tmp'}'")
    rows = db.execute(
        """
        SELECT DISTINCT c.SRX_accession
        FROM read_parquet(?) AS c
        JOIN read_parquet(?) AS s
          ON c.SRX_accession = s.srx_accession
        WHERE s.organism = 'Homo sapiens'
          AND s.tech_10x = '3_prime_gex'
          AND s.cell_prep = 'single_cell'
          AND c.gene_count_Unique >= 300
          AND c.umi_count_Unique >= 500
        ORDER BY 1
        """,
        [str(meta_dir / "obs_metadata.parquet"), str(meta_dir / "sample_metadata.parquet")],
    ).fetchall()
    return [row[0] for row in rows]


def list_remote_sizes() -> dict[str, int]:
    """One paginated listing gives every h5ad and its size, instead of 23k HEADs."""
    entries = fs.ls(H5AD_PREFIX, detail=True)
    return {Path(e["name"]).stem: e["size"] for e in entries if e["name"].endswith(".h5ad")}


def download_matrices(
    accessions: list[str], sizes: dict[str, int], h5ad_dir: Path, workers: int
) -> list[tuple[str, str]]:
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                download_one, f"{H5AD_PREFIX}/{acc}.h5ad", h5ad_dir / f"{acc}.h5ad", sizes[acc]
            ): acc
            for acc in accessions
        }
        for future in tqdm(as_completed(futures), total=len(futures), unit="file"):
            acc = futures[future]
            try:
                future.result()
            except Exception as err:
                failures.append((acc, str(err)))
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("data"), help="output root")
    parser.add_argument("--workers", type=int, default=16, help="parallel downloads")
    parser.add_argument("--limit", type=int, default=None, help="only the first N matrices")
    parser.add_argument(
        "--accessions",
        nargs="+",
        default=None,
        help="download only these accessions and skip the metadata step",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print file count and total size, then stop"
    )
    parser.add_argument(
        "--memory-limit", default="24GB", help="DuckDB memory limit for the cell metadata join"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta_dir = args.out / "metadata"
    h5ad_dir = args.out / "h5ad"
    meta_dir.mkdir(parents=True, exist_ok=True)
    h5ad_dir.mkdir(parents=True, exist_ok=True)

    if args.accessions:
        accessions = args.accessions
    else:
        download_metadata(meta_dir)
        accessions = select_accessions(meta_dir, args.memory_limit)
        (args.out / "accessions.txt").write_text("\n".join(accessions) + "\n")
        print(f"{len(accessions)} accessions pass the filter")

    if args.limit:
        accessions = accessions[: args.limit]

    sizes = list_remote_sizes()
    missing = [acc for acc in accessions if acc not in sizes]
    accessions = [acc for acc in accessions if acc in sizes]
    total_bytes = sum(sizes[acc] for acc in accessions)
    print(f"{len(accessions)} matrices, {total_bytes / 1e12:.3f} TB, {len(missing)} not in bucket")
    if missing:
        (args.out / "missing_in_bucket.txt").write_text("\n".join(missing) + "\n")

    if args.dry_run:
        return

    failures = download_matrices(accessions, sizes, h5ad_dir, args.workers)
    failures_path = args.out / "download_failures.txt"
    failures_path.write_text("\n".join(f"{acc}\t{err}" for acc, err in failures))
    print(f"done: {len(accessions) - len(failures)} ok, {len(failures)} failed")
    if failures:
        print(f"failed accessions written to {failures_path}; re-run to retry them")


if __name__ == "__main__":
    main()
