from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from einops import einsum, rearrange

context_length = 512
hidden_dim = 512
num_transformer_layers = 30
output_dim = 36601


def normalize_qk(x):
    """Unit RMS per head, with the reduction in FP32 and no learnable gain.

    With the usual 1/sqrt(head_dim) attention scale, this bounds each
    query-key score to approximately +/-sqrt(head_dim), even if weights grow.
    Cast back afterward so the attention matrix operations can still use BF16.
    """
    with torch.autocast(device_type=x.device.type, enabled=False):
        normalized = F.rms_norm(x.float(), (x.shape[-1],), eps=1e-6)
    return normalized.to(x.dtype)


class QKNormEncoderLayer(nn.TransformerEncoderLayer):
    """A pre-LN block with explicit, normalized self-attention.

    Reuse PyTorch's parameters and feed-forward layers, but write the forward
    pass explicitly: its fused evaluation path would otherwise skip QK norm.
    """
    def forward(self, src, src_key_padding_mask=None):
        x = src
        normalized = self.norm1(x)
        qkv = F.linear(normalized, self.self_attn.in_proj_weight,
                       self.self_attn.in_proj_bias)
        q, k, v = rearrange(qkv, "b s (three heads d) -> three b heads s d",
                            three=3, heads=self.self_attn.num_heads).unbind(0)
        q, k = normalize_qk(q), normalize_qk(k)
        # SDPA True means attend; Transformer padding-mask True means ignore.
        allowed = None if src_key_padding_mask is None else ~src_key_padding_mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed,
            dropout_p=self.self_attn.dropout if self.training else 0.0)
        attended = rearrange(attended, "b heads s d -> b s (heads d)")
        attended = self.self_attn.out_proj(attended)
        x = x + self.dropout1(attended)
        hidden = self.norm2(x)
        hidden = self.dropout(self.activation(self.linear1(hidden)))
        hidden = self.linear2(hidden)
        return x + self.dropout2(hidden)


class Andre(nn.Module):
    def __init__(self, width=None, norm_first=True, attention_dropout=0.0, qk_norm=True):
        super().__init__()
        width = hidden_dim if width is None else width
        if width < 1 or width % 8:
            raise ValueError("Model width must be positive and divisible by 8 attention heads")
        if not 0 <= attention_dropout < 1:
            raise ValueError("attention_dropout must be between 0 (inclusive) and 1")
        self.count_embedding = nn.Embedding(num_embeddings=100_001, padding_idx=0, embedding_dim=width)
        self.gene_embedding = nn.Embedding(num_embeddings=36601+2, padding_idx=0, embedding_dim=width)

        self.qk_norm = bool(qk_norm and norm_first)
        layer_type = QKNormEncoderLayer if self.qk_norm else nn.TransformerEncoderLayer
        self.transformer_layers = nn.ModuleList([layer_type(
            d_model=width, nhead=8, dim_feedforward=width*4, batch_first=True,
            norm_first=norm_first) for i in range(num_transformer_layers)])
        # BF16 memory-efficient attention with dropout produced spurious large
        # gradients once attention became saturated. Keep the other dropout
        # layers, but default attention dropout to zero. Evidence is recorded in
        # benchmarks/attention-instability-20260922/.
        for layer in self.transformer_layers:
            layer.self_attn.dropout = attention_dropout

        # Pre-LN keeps the residual path open through the deep stack. Normalize
        # its final hidden vector before predicting the missing gene.
        self.final_norm = nn.LayerNorm(width) if norm_first else nn.Identity()
        self.architecture = "pre_ln_qknorm_v2" if self.qk_norm else ("pre_ln_v1" if norm_first else "post_ln_v0")
        self.out_layer = nn.Linear(width, output_dim)

    def forward(self, gene_ids, counts, attention_mask, mask_position):
        count_ids = counts.clamp(max=100_000).long()

        x = self.gene_embedding(gene_ids)
        x = x + self.count_embedding(count_ids)

        # Never select the memory-efficient backend that failed our BF16
        # gradient probe. cuDNN/Flash can use BF16; math is the safe fallback.
        # Keep this context outside compiled blocks so it also covers tracing.
        kernels = sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
                               SDPBackend.MATH]) if self.qk_norm else nullcontext()
        with kernels:
            for layer in self.transformer_layers:
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
