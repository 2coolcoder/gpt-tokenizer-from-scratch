"""
Lecture 7: a decoder-only Transformer (GPT), following "Attention Is All You Need" and GPT-2.

Two interchangeable self-attention implementations:
  attn="heads"    the lecture's version: one Head module per head (key/query/value Linear),
                  explicit causal mask + softmax, concatenated by MultiHeadAttention
  attn="batched"  the same computation with all heads in one QKV projection and
                  F.scaled_dot_product_attention (flash / memory-efficient kernels)
Given the same weights both produce identical outputs (tests/test_gpt.py).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 256  # maximum context length
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2
    attn: str = "batched"  # "heads" | "batched"
    # attention logits are scaled by 1/sqrt(d): "head_size" (the paper, default) or "n_embd" (the
    # lecture video's final code, which uses C**-0.5 with C = n_embd; softer attention)
    attn_scale: str = "head_size"
    # "gpt2": N(0, 0.02) weights, zero biases (Karpathy's final gpt.py); "pytorch": PyTorch's default
    # init (the original video, which did not cover weight initialisation)
    init: str = "gpt2"


class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, config, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)
        self.scale = (config.n_embd if config.attn_scale == "n_embd" else head_size) ** -0.5

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        # affinities between every pair of positions, scaled so softmax stays diffuse at init
        wei = q @ k.transpose(-2, -1) * self.scale  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # no peeking at the future
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)  # (B, T, hs)


class MultiHeadAttention(nn.Module):
    """Several heads in parallel, concatenated and projected back into the residual stream."""

    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class CausalSelfAttention(nn.Module):
    """MultiHeadAttention with the heads batched: one (C -> 3C) projection, heads as a dimension."""

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.attn_dropout = config.dropout
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)  # [queries | keys | values]
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.scale = config.n_embd ** -0.5 if config.attn_scale == "n_embd" else None  # None: 1/sqrt(head_size)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale,
                                           dropout_p=self.attn_dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(y))


class FeedForward(nn.Module):
    """Position-wise MLP: the per-token 'computation' after attention's 'communication'."""

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: pre-LayerNorm, attention and MLP, each with a residual connection."""

    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadAttention(config) if config.attn == "heads" else CausalSelfAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        if config.init == "gpt2":
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        x = self.blocks(tok_emb + pos_emb)
        logits = self.lm_head(self.ln_f(x))  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]  # the position table only covers block_size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx


def heads_to_batched_state_dict(sd, n_head):
    """Convert attn='heads' weights into attn='batched' weights (same function, fused layout)."""
    out = {}
    for k, v in sd.items():
        if ".sa.heads." in k:
            continue
        out[k] = v
    layers = sorted({int(k.split(".")[1]) for k in sd if k.startswith("blocks.")})
    for l in layers:
        pre = f"blocks.{l}.sa"
        qkv = [torch.cat([sd[f"{pre}.heads.{h}.{name}.weight"] for h in range(n_head)], 0)
               for name in ("query", "key", "value")]
        out[f"{pre}.qkv.weight"] = torch.cat(qkv, 0)
    return out
