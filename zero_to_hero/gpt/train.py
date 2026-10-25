"""
Lecture 7: train on Tiny Shakespeare, character level.

  python -m zero_to_hero.gpt.train bigram   bigram baseline (the lecture's starting point)
  python -m zero_to_hero.gpt.train gpt      the full GPT with the lecture's hyperparameters
                                            (6 layers, 6 heads, n_embd 384, context 256,
                                            dropout 0.2, AdamW 3e-4, 5000 iterations, batch 64)
  python -m zero_to_hero.gpt.train gpt_lecture_scale
                                            same, with the video's attention scaling 1/sqrt(n_embd)
                                            instead of the paper's 1/sqrt(head_size)
  python -m zero_to_hero.gpt.train gpt_video
                                            the original video's model: 1/sqrt(n_embd) scaling and
                                            PyTorch default init instead of N(0, 0.02)
"""

import os
import sys
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from zero_to_hero.common import RESULTS_DIR, device, save_results
from zero_to_hero.gpt.data import CharDataset
from zero_to_hero.gpt.model import GPT, GPTConfig


class BigramLanguageModel(nn.Module):
    """Each token reads the logits for the next token straight from a lookup table."""

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx


@torch.no_grad()
def estimate_loss(model, ds, batch_size, block_size, eval_iters, dev):
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = ds.get_batch(split, batch_size, block_size, dev)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(which="gpt"):
    torch.manual_seed(1337)
    dev = device()
    ds = CharDataset()
    if which == "bigram":
        hp = dict(batch_size=32, block_size=8, max_iters=3000, eval_interval=300, lr=1e-2, eval_iters=200)
        model = BigramLanguageModel(ds.vocab_size)
    else:
        hp = dict(batch_size=64, block_size=256, max_iters=5000, eval_interval=500, lr=3e-4, eval_iters=200)
        # attn="heads" runs the lecture's per-head implementation, in fp32 as in the lecture
        scale = "n_embd" if which in ("gpt_lecture_scale", "gpt_video") else "head_size"
        init = "pytorch" if which == "gpt_video" else "gpt2"
        model = GPT(GPTConfig(vocab_size=ds.vocab_size, block_size=hp["block_size"], attn="heads",
                              attn_scale=scale, init=init))
    model = model.to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{which}: {n_params / 1e6:.2f}M parameters on {dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"])
    history = []
    t0 = time.time()
    for it in range(hp["max_iters"]):
        if it % hp["eval_interval"] == 0 or it == hp["max_iters"] - 1:
            losses = estimate_loss(model, ds, hp["batch_size"], hp["block_size"], hp["eval_iters"], dev)
            history.append({"step": it, **losses})
            print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}", flush=True)
        xb, yb = ds.get_batch("train", hp["batch_size"], hp["block_size"], dev)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    train_time = time.time() - t0

    context = torch.zeros((1, 1), dtype=torch.long, device=dev)
    sample = ds.decode(model.generate(context, max_new_tokens=1000)[0].tolist())
    print(sample[:500])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if which.startswith("gpt"):
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"{which}_shakespeare_char.pt"))
    with open(os.path.join(RESULTS_DIR, f"gpt_{which}_sample.txt"), "w") as f:
        f.write(sample)
    return save_results(f"gpt_{which}", {"model": which, "n_params": n_params, "hyperparameters": hp,
                                         "final": history[-1], "history": history,
                                         "train_time_s": train_time, "device": dev})


if __name__ == "__main__":
    train(sys.argv[1] if len(sys.argv) > 1 else "gpt")
