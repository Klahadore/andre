"""Read immutable scBaseCount LMDB shards; see scripts/build_dataset.py.

Only paths and a small shard catalog are held in the Dataset. LMDB's mmap uses
shared, reclaimable OS cache rather than a separate Python cache of cell data.
"""

from bisect import bisect_right
from collections import OrderedDict, defaultdict
import json
import operator
import os
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from dataset_codec import (
    PAD_ID, MASK_ID, FORMAT_VERSION, CELLS_PER_RECORD, SPLITS,
    cell_key, encode_cell, pack_cells,
    decode_cell as _decode_cell, unpack_cell as _unpack_cell,
)

# Per process, shared across Dataset instances: LMDB must not be opened twice
# at the same path in a process. Bound descriptors even with thousands of shards.
_ENVIRONMENTS = OrderedDict()
_ENV_PID = os.getpid()
_MAX_OPEN_SHARDS = 32


def _environment(path):
    global _ENV_PID
    if _ENV_PID != os.getpid():
        # Discard inherited handles; never perform reads using a parent's env.
        _ENVIRONMENTS.clear()
        _ENV_PID = os.getpid()
    key = str(Path(path).resolve())
    if key not in _ENVIRONMENTS:
        while len(_ENVIRONMENTS) >= _MAX_OPEN_SHARDS:
            _, old = _ENVIRONMENTS.popitem(last=False)
            old.close()
        _ENVIRONMENTS[key] = lmdb.open(
            key, readonly=True, create=False, lock=False, readahead=False,
            max_spare_txns=0,
        )
    _ENVIRONMENTS.move_to_end(key)
    return _ENVIRONMENTS[key]


def close_environments():
    """Release this process's reader handles (e.g. before forking)."""
    for env in _ENVIRONMENTS.values():
        env.close()
    _ENVIRONMENTS.clear()


def decode_cell(value):
    return {name: torch.from_numpy(array) for name, array in _decode_cell(value).items()}


def unpack_cell(block, index):
    return {name: torch.from_numpy(array) for name, array in _unpack_cell(block, index).items()}


class ScBaseCountDataset(Dataset):
    def __init__(self, root, split="train"):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.root = Path(root).resolve()
        self.split = split
        catalog = json.loads((self.root / "catalog.json").read_text())
        if catalog["format_version"] != FORMAT_VERSION:
            raise ValueError("Unsupported dataset format; rebuild the shards")
        self.vocab_size = catalog["vocab_size"]
        self.shards = []
        self.ends = []
        total = 0
        for shard in catalog["shards"]:
            count = shard["counts"][split]
            if count:
                self.shards.append(shard["path"])
                total += count
                self.ends.append(total)
        self.size = total

    def __len__(self):
        return self.size

    def _locate(self, index):
        index = operator.index(index)
        if index < 0:
            index += self.size
        if not 0 <= index < self.size:
            raise IndexError(index)
        shard = bisect_right(self.ends, index)
        start = self.ends[shard - 1] if shard else 0
        return shard, index - start

    def __getitem__(self, index):
        return self.__getitems__([index])[0]

    def __getitems__(self, indices):
        """One transaction per shard in a batch, preserving requested order."""
        groups = defaultdict(list)
        for output_index, index in enumerate(indices):
            shard, local_index = self._locate(index)
            groups[shard].append((output_index, local_index))
        result = [None] * len(indices)
        for shard, requests in groups.items():
            env = _environment(self.root / self.shards[shard])
            with env.begin(buffers=True) as txn:
                blocks = {}
                for output_index, local_index in requests:
                    block_index, offset = divmod(local_index, CELLS_PER_RECORD)
                    if block_index not in blocks:
                        blocks[block_index] = txn.get(cell_key(self.split, block_index))
                    value = blocks[block_index]
                    if value is None:
                        raise KeyError(f"Missing {self.split}/{local_index} in {self.shards[shard]}")
                    result[output_index] = unpack_cell(value, offset)
        return result


class BlockShuffleSampler(Sampler):
    """Shuffle blocks and cells within blocks with bounded permutation memory.

    This is a locality-aware shuffle, not a uniform permutation of all cells.
    Call set_epoch(epoch) before each epoch for a reproducible new order.
    """
    def __init__(self, dataset, block_size=4096, seed=0):
        if block_size < 1:
            raise ValueError("block_size must be positive")
        self.size = len(dataset)
        self.block_size = block_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.size

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        blocks = rng.permutation((self.size + self.block_size - 1) // self.block_size)
        for block in blocks:
            start = int(block) * self.block_size
            for offset in rng.permutation(min(self.block_size, self.size - start)):
                yield start + int(offset)


def collate_fn(samples, length=512):
    """Sample expressed genes and mask one gene identity per cell.

    attention_mask is True for real entries. Pass ~attention_mask as the
    nn.TransformerEncoder src_key_padding_mask (True means ignore there).
    Counts remain raw, including the count at the masked position.
    """
    if not samples or length < 1:
        raise ValueError("A nonempty batch and positive length are required")
    batch_size = len(samples)
    batch_ids = torch.zeros(batch_size, length, dtype=torch.long)
    batch_counts = torch.zeros(batch_size, length, dtype=torch.float32)
    attention_mask = torch.zeros(batch_size, length, dtype=torch.bool)
    targets = torch.empty(batch_size, dtype=torch.long)
    mask_positions = torch.empty(batch_size, dtype=torch.long)
    for i, sample in enumerate(samples):
        gene_ids, counts = sample["gene_ids"], sample["counts"]
        if gene_ids.ndim != 1 or counts.shape != gene_ids.shape or not len(gene_ids):
            raise ValueError("Expected equally sized, nonempty gene/count vectors")
        n = min(len(gene_ids), length)
        selected = torch.randperm(len(gene_ids))[:n]
        batch_ids[i, :n] = gene_ids[selected]
        batch_counts[i, :n] = counts[selected]
        attention_mask[i, :n] = True
        mask_index = torch.randint(n, ()).item()
        targets[i] = batch_ids[i, mask_index]
        mask_positions[i] = mask_index
        batch_ids[i, mask_index] = MASK_ID
    return {
        "gene_ids": batch_ids, "counts": batch_counts,
        "attention_mask": attention_mask, "targets": targets,
        "mask_positions": mask_positions,
    }
