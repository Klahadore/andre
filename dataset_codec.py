"""Portable LMDB record format shared by the builder and PyTorch reader."""
import struct
import zlib

import numpy as np

PAD_ID, MASK_ID = 0, 1
FORMAT_VERSION = 1
CELLS_PER_RECORD = 32
SPLITS = ("train", "val", "test")


def cell_key(split, index):
    return bytes([SPLITS.index(split)]) + struct.pack(">Q", index)


def encode_cell(gene_ids, counts):
    """Sorted delta IDs and byte-shuffled float32 counts, compressed losslessly."""
    ids = np.asarray(gene_ids)
    values = np.asarray(counts)
    if ids.ndim != 1 or values.shape != ids.shape or not len(ids):
        raise ValueError("Expected equally sized, nonempty gene/count vectors")
    if np.any(ids < 2) or np.any(ids > 65535) or np.any(ids != ids.astype(np.uint16)):
        raise ValueError("Gene token IDs must be integers in [2, 65535]")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Counts must be finite and positive")
    if not np.array_equal(values, values.astype("<f4")):
        raise ValueError("Counts cannot be represented losslessly as float32")
    order = np.argsort(ids)
    ids = ids[order].astype("<u2")
    if np.any(ids[1:] == ids[:-1]):
        raise ValueError("Duplicate gene IDs")
    deltas = np.diff(ids, prepend=np.uint16(0)).astype("<u2")
    values = values[order].astype("<f4")
    raw = (struct.pack("<I", len(ids))
           + deltas.view("u1").reshape(-1, 2).T.tobytes()
           + values.view("u1").reshape(-1, 4).T.tobytes())
    return zlib.compress(raw, level=1)


def decode_cell(value):
    raw = zlib.decompress(value)
    n, = struct.unpack_from("<I", raw)
    if n == 0 or len(raw) != 4 + 6 * n:
        raise ValueError("Invalid cell record")
    # Owned, writable arrays: no tensor outlives an LMDB transaction buffer.
    deltas = np.frombuffer(raw, "u1", 2*n, 4).reshape(2, n).T.copy().view("<u2").ravel()
    counts = np.frombuffer(raw, "u1", 4*n, 4+2*n).reshape(4, n).T.copy().view("<f4").ravel()
    return {"gene_ids": np.cumsum(deltas, dtype=np.int64), "counts": counts}


def pack_cells(cells):
    """Pack independently compressed cells, avoiding per-cell LMDB page waste."""
    if not 0 < len(cells) <= CELLS_PER_RECORD:
        raise ValueError("Invalid cell block size")
    offsets = np.zeros(CELLS_PER_RECORD + 1, dtype="<u4")
    offsets[1:len(cells)+1] = np.cumsum([len(c) for c in cells])
    offsets[len(cells)+1:] = offsets[len(cells)]
    return offsets.tobytes() + b"".join(cells)


def unpack_cell(block, index):
    start, end = struct.unpack_from("<II", block, 4*index)
    header = 4 * (CELLS_PER_RECORD + 1)
    if start >= end or header + end > len(block):
        raise ValueError("Invalid cell block offsets")
    return decode_cell(block[header+start:header+end])

