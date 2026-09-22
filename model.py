import torch
from torch import nn

context_length = 512
hidden_dim = 512
num_transformer_layers = 30
output_dim = 36601

layers = 30
class Andre(nn.Module):
    def __init__(self):
        super().__init__()
        self.count_embedding = nn.Embedding(num_embeddings=100_001, padding_idx=0, embedding_dim=hidden_dim)
        self.gene_embedding = nn.Embedding(num_embeddings=36601+2, padding_idx=0, embedding_dim=hidden_dim)

        self.transformer_layers = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*4, batch_first=True) for i in range(num_transformer_layers)])

        self.out_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, gene_ids, counts, attention_mask):
        count_ids = counts.clamp(max=100_000).long()

        x = self.gene_embedding(gene_ids)
        x = x + self.count_embedding(count_ids)

        for _, layer in enumerate(self.transformer_layers):
            x = layer(x, src_key_padding_mask=~attention_mask)

        x = self.out_layer(x)
        return x


if __name__ == "__main__":
    model = Andre()
    print(model)
