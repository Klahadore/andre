from typing import TypedDict

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import fsspec as fs
import anndata as ad
import math
import random



base_url = (
    "https://storage.googleapis.com/"
    "arc-institute-virtual-cell-atlas/"
    "scbasecount/2026-01-12/h5ad/"
    "GeneFull_Ex50pAS/Homo_sapiens/"
)

class ScBaseCountDataset(Dataset):
    def __init__(self, dataset_csv: str, gene_to_id: dict):
        self.df = pd.read_csv(dataset_csv)
        self.gene_to_id = gene_to_id


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        srx_accession = row["srx_accession"]
        cell_barcode = row["cell_barcode"]
        file_path = base_url + srx_accession + ".h5ad"

        local_path = fs.open_local(
            f"simplecache::{file_path}",
            simplecache={"cache_storage": "data/h5ad_cache"},
        )

        adata = ad.read_h5ad(local_path, backed="r")

        try:
            # X uses integer row positions, not barcode strings.
            cell_idx = adata.obs_names.get_loc(cell_barcode)
            row = adata.X[cell_idx].tocoo()
            gene_ids = torch.tensor(row.col, dtype=torch.long) + 2 # +2 for PAD and MASK
            counts = torch.tensor(row.data, dtype=torch.float32)

            return {"gene_ids": gene_ids, "counts": counts}
        finally:
            adata.file.close()



def collate_fn(samples):
    batch_size = len(samples)
    length = 512

    # Allocate space for these tensors. 
    batch_ids = torch.zeros(batch_size, length, dtype=torch.long)
    batch_counts = torch.zeros(batch_size, length, dtype=torch.float32)
    attention_mask = torch.zeros(batch_size, length, dtype=torch.bool)

    targets = torch.empty(batch_size, dtype=torch.long)
    mask_positions = torch.empty(batch_size, dtype=torch.long)

    for i, sample in enumerate(samples):
        gene_ids = sample["gene_ids"]
        counts = sample["counts"]

        if len(gene_ids) == 0:
            raise ValueError("Cell has no expressed genes")
        if len(gene_ids) != len(counts):
            raise ValueError("Gene IDs and counts have different lengths")

        n = min(len(gene_ids), length)
        selected = torch.randperm(len(gene_ids))[:n]
        selected_gene_ids = gene_ids[selected]

        batch_ids[i, :n] = selected_gene_ids
        batch_counts[i, :n] = counts[selected]
        attention_mask[i, :n] = False

        if n < length:
            batch_ids[i, n:] = 0
            batch_counts[i, n:] = 0
            attention_mask[i, n:] = False
        
        mask_index = torch.randint(n, ()).item()

        targets[i] = batch_ids[i, mask_index]
        mask_positions[i] = mask_index
        batch_ids[i, mask_index] = 1
        

        return {
        "gene_ids": batch_ids,
        "counts": batch_counts,
        "attention_mask": attention_mask,
        "targets": targets,
        "mask_positions": mask_positions,
    }


        
if __name__ == "__main__":
    gene_to_id = pd.read_csv("data/gene_to_id.csv", index_col="token_id").to_dict()
    dataset = ScBaseCountDataset("data/train.csv", gene_to_id)
 
    loader = DataLoader(
        dataset,
        batch_size=12,
        collate_fn=collate_fn,
        num_workers=0,
        shuffle=False,
    )
    batch = next(iter(loader))

    for name, tensor in batch.items():
        print(name, tensor.shape, tensor.dtype)
