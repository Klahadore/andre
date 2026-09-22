import torch
from torch import nn

hidden_dim = 512
layers = 30
class Andre(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.transformer_blocks = nn.ModuleList(    )
        for _ in range(layers):
            self.transformer_blocks.append(nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*4, batch_first=True))

        self.embedding = nn.Embedding(num_embeddings=36601, embedding_dim=hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 36601)

    def forward(self, gene_ids, counts, attention_mask):
        x = self.embedding(gene_ids)
        counts_embeddings 

if __name__ == "__main__":
    model = Andre()
    print(model)