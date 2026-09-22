import torch
from torch import nn
from einops import einsum

context_length = 512
hidden_dim = 512
num_transformer_layers = 30
output_dim = 36601

class Andre(nn.Module):
    def __init__(self, width=None):
        super().__init__()
        width = hidden_dim if width is None else width
        if width < 1 or width % 8:
            raise ValueError("Model width must be positive and divisible by 8 attention heads")
        self.count_embedding = nn.Embedding(num_embeddings=100_001, padding_idx=0, embedding_dim=width)
        self.gene_embedding = nn.Embedding(num_embeddings=36601+2, padding_idx=0, embedding_dim=width)

        self.transformer_layers = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=width, nhead=8, dim_feedforward=width*4, batch_first=True) for i in range(num_transformer_layers)])

        self.out_layer = nn.Linear(width, output_dim)

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

        # Select the MASK position's hidden vector for each cell.
        masked_hidden = einsum(x, is_mask, "b p h, b p -> b h")
        out = self.out_layer(masked_hidden)

        return out


if __name__ == "__main__":
    model = Andre()
    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
