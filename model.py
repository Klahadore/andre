import torch
from torch import nn
from einops import einsum
import dataset

context_length = 512
hidden_dim = 512
num_transformer_layers = 30
output_dim = 36601

dataset_path = ""

layers = 30
class Andre(nn.Module):
    def __init__(self):
        super().__init__()
        self.count_embedding = nn.Embedding(num_embeddings=100_001, padding_idx=0, embedding_dim=hidden_dim)
        self.gene_embedding = nn.Embedding(num_embeddings=36601+2, padding_idx=0, embedding_dim=hidden_dim)

        self.transformer_layers = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*4, batch_first=True) for i in range(num_transformer_layers)])

        self.out_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, gene_ids, counts, attention_mask, mask_position):
        count_ids = counts.clamp(max=100_000).long()

        x = self.gene_embedding(gene_ids)
        x = x + self.count_embedding(count_ids)

        for _, layer in enumerate(self.transformer_layers):
            x = layer(x, src_key_padding_mask=~attention_mask)

        is_mask = torch.nn.functional.one_hot(
              mask_position,
              num_classes=x.shape[1],
          ).to(device=x.device, dtype=x.dtype)

        # Get only the embedding for the embedding at the hidden_dim
        masked_hidden = einsum(x, is_mask, "b p h, b p -> b h")
        out = self.out_layer(masked_hidden)

        return out


if __name__ == "__main__":
    model = Andre()

    train_dataset = dataset.ScBaseCountDataset()
    test_dataset = dataset.ScBaseCountDataset(split="test")

    sc_dataset_shuffler = dataset.BlockShuffleSampler(train_dataset, )

    loss = nn.functional.cross_entropy(

    )
