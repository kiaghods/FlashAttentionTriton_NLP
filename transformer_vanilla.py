import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal=False, dropout_p=0.0):
        B, H, T, D = q.shape
        scale = 1.0 / math.sqrt(D)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if causal:
            mask = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        ctx.save_for_backward(q, k, v, weights)
        ctx.scale = scale
        return torch.matmul(weights, v)

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, weights = ctx.saved_tensors
        scale = ctx.scale

        dV = torch.matmul(weights.transpose(-2, -1), grad_output)
        dW = torch.matmul(grad_output, v.transpose(-2, -1))

        dS = dW * weights
        dS = dS - weights * dS.sum(dim=-1, keepdim=True)
        dS *= scale

        dQ = torch.matmul(dS, k)
        dK = torch.matmul(dS.transpose(-2, -1), q)

        return dQ, dK, dV, None, None


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, causal=False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lens=None):
        B, T, D = x.shape
        H = self.num_heads
        qkv = self.qkv_proj(x).view(B, T, H, 3 * self.head_dim).transpose(1, 2)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        attn_output = ScaledDotProductAttention.apply(q, k, v, self.causal, self.dropout.p)
        attn_output = self.dropout(attn_output)
        out = attn_output.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1, causal=False):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout, causal)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-5)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, lens=None):
        x = x + self.attn(self.norm1(x), lens=lens)
        x = x + self.mlp(self.norm2(x))
        return x


class VanillaTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_len=512, dropout=0.1, causal=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout, causal=causal)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-5)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, lens=None):
        B, T = x.shape
        x = self.embed(x) + self.pos_embed[:, :T, :]
        for block in self.blocks:
            x = block(x, lens=lens)
        x = self.norm(x)
        return self.output_proj(x)


if __name__ == "__main__":
    model = VanillaTransformer(
        vocab_size=10000,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        max_len=1024,
        dropout=0.1,
        causal=True,
    )

    dummy_input = torch.randint(0, 10000, (2, 128)).cuda()
    model = model.cuda()

    out = model(dummy_input)
    print(out.shape)  # (2, 128, 10000)

    loss = out.mean()
    loss.backward()  # Uses custom attention backward

