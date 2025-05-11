import torch
import torch.nn as nn
import torch.nn.functional as F
from flashattention_kernel import flash_attention

class FlashDecoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        assert embed_dim % n_heads == 0

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, lens=None):
        B, T, D = x.shape
        H = self.n_heads

        x_norm = self.norm1(x)
        qkv = self.qkv_proj(x_norm)
        qkv = qkv.view(B, T, H, 3 * self.head_dim).transpose(1, 2)  # [B, H, T, 3*D_head]
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Flash attention with causal mask
        out = flash_attention(q, k, v, lens=lens, causal=True)
        out = out.transpose(1, 2).reshape(B, T, D)
        x = x + self.dropout(self.out_proj(out))

        # Feedforward block
        x = x + self.ffn(self.norm2(x))
        return x

class FlashTransformerDecoderOnly(nn.Module):
    def __init__(self, vocab_size, embed_dim, depth, n_heads, max_seq_len, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            FlashDecoderBlock(embed_dim, n_heads, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x, lens=None):
        B, T = x.shape
        x = self.token_emb(x) + self.pos_emb[:, :T, :]
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, lens=lens)
        x = self.norm(x)
        return self.output_proj(x)  # [B, T, vocab_size]

# Example usage
if __name__ == "__main__":
    model = FlashTransformerDecoderOnly(
        vocab_size=50257,
        embed_dim=512,
        depth=6,
        n_heads=8,
        max_seq_len=256,
    ).cuda()

    tokens = torch.randint(0, 50257, (4, 128)).cuda()
    lens = torch.tensor([128, 120, 90, 70], dtype=torch.int32).cuda()
    logits = model(tokens, lens=lens)  # [4, 128, 50257]
    print(logits.shape)
