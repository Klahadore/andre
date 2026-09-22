import torch
from torch import nn
from einops import einsum

context_length = 512
hidden_dim = 512
num_transformer_layers = 30
output_dim = 36601

class Andre(nn.Module):
    def __init__(self, width=None, norm_first=True):
        super().__init__()
        width = hidden_dim if width is None else width
        if width < 1 or width % 8:
            raise ValueError("Model width must be positive and divisible by 8 attention heads")
        self.count_embedding = nn.Embedding(num_embeddings=100_001, padding_idx=0, embedding_dim=width)
        self.gene_embedding = nn.Embedding(num_embeddings=36601+2, padding_idx=0, embedding_dim=width)

        self.transformer_layers = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=width, nhead=8, dim_feedforward=width*4, batch_first=True,
            norm_first=norm_first) for i in range(num_transformer_layers)])

        # Pre-LN keeps the residual path open through the deep stack. Normalize
        # its final hidden vector before predicting the missing gene.
        self.final_norm = nn.LayerNorm(width) if norm_first else nn.Identity()
        self.architecture = "pre_ln_v1" if norm_first else "post_ln_v0"
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
        masked_hidden = self.final_norm(masked_hidden)
        out = self.out_layer(masked_hidden)

        return out


if __name__ == "__main__":
    model = Andre()
    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
