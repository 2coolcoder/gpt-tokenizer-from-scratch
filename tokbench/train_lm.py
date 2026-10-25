"""
Train the course GPT (gpt.py: 6 layers, 6 heads, n_embd 384, context 256 tokens, dropout 0.2,
AdamW lr 3e-4 constant, batch 64) on a cached token stream under a fixed training-FLOPs budget,
and evaluate held-out bits-per-byte.

The model is this project's lecture-7 GPT (zero_to_hero/gpt/model.py) with attn="batched": the
per-head attention modules fused into one QKV projection + F.scaled_dot_product_attention
(identical outputs, tests/test_gpt.py), trained in bf16 autocast. Only the vocab size changes.

Compute accounting (Kaplan et al. 2020): training FLOPs/token = 6 * N + 6 * n_layer * T * C,
where N counts every matmul parameter, including lm_head (V * C) but not the embedding lookups.
A bigger vocab therefore costs more per token, so it gets fewer steps for the same budget.

bpb = (sum of per-token NLL in nats) / ln(2) / (UTF-8 bytes of the scored tokens), computed on
the full split with a sliding window (context 256, stride 128), so every token is scored exactly
once with >= 128 tokens of context (except the first window).
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.nn import functional as F

from tokbench.tok_suite import TOKENS_DIR, get_tokenizer
from zero_to_hero.gpt.model import GPT, GPTConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(ROOT, "tokbench_out/runs")

# course hyperparameters (007_GPT/_src/gpt.py)
BATCH_SIZE = 64
BLOCK_SIZE = 256
LR = 3e-4
N_EMBD = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2


def model_config(vocab_size):
    return GPTConfig(vocab_size=vocab_size, block_size=BLOCK_SIZE, n_embd=N_EMBD, n_head=N_HEAD,
                     n_layer=N_LAYER, dropout=DROPOUT, attn="batched")


def flops_per_token(vocab_size):
    n_matmul = N_LAYER * 12 * N_EMBD ** 2 + vocab_size * N_EMBD
    return 6 * n_matmul + 6 * N_LAYER * BLOCK_SIZE * N_EMBD


def load_tokens(corpus, tok, split, device):
    arr = np.load(os.path.join(TOKENS_DIR, corpus, tok, f"{split}.npy")).astype(np.int64)
    return torch.from_numpy(arr).to(device)


@torch.no_grad()
def eval_bpb(model, data, token_nbytes, stride=BLOCK_SIZE // 2, batch=256):
    """Sliding-window bits-per-byte over the whole token stream `data`."""
    model.eval()
    T = BLOCK_SIZE
    n = data.numel()
    starts = list(range(0, max(n - T - 1, 0) + 1, stride))
    if starts[-1] + T + 1 < n:
        starts.append(n - T - 1)
    # window starting at s scores targets with absolute index in (prev window's last target, s + T]
    prev_end = [0] + [st + T for st in starts[:-1]]
    nll = torch.zeros((), dtype=torch.float64, device=data.device)
    nbytes = torch.zeros((), dtype=torch.int64, device=data.device)
    ar = torch.arange(T, device=data.device)
    for i in range(0, len(starts), batch):
        s = torch.tensor(starts[i:i + batch], device=data.device)
        pe = torch.tensor(prev_end[i:i + batch], device=data.device)
        win = data[s[:, None] + torch.arange(T + 1, device=data.device)]
        x, y = win[:, :-1], win[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
                               reduction="none").view(y.shape)
        keep = (s[:, None] + 1 + ar) > pe[:, None]
        nll += (loss.double() * keep).sum()
        nbytes += (token_nbytes[y] * keep).sum()
    nll, nbytes = nll.item(), nbytes.item()
    assert nbytes == token_nbytes[data[1:]].sum().item(), "sliding window must score every token once"
    model.train()
    return nll / math.log(2) / nbytes, nll, nbytes


def train(corpus, tok, budget_pflops, seed, n_evals=25, device="cuda", out_dir=RUNS_DIR):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokenizer = get_tokenizer(corpus, tok)
    V = tokenizer.vocab_size
    token_nbytes = torch.tensor([len(tokenizer.token_bytes(i)) for i in range(V)], device=device)
    train_data = load_tokens(corpus, tok, "train", device)
    val_data = load_tokens(corpus, tok, "val", device)
    test_data = load_tokens(corpus, tok, "test", device)
    # a train slice with the same number of bytes as val, to measure the generalization gap
    cum = torch.cumsum(token_nbytes[train_data], 0)
    val_bytes = token_nbytes[val_data].sum()
    train_slice = train_data[: int(torch.searchsorted(cum, val_bytes)) + 1]
    train_bytes_total = int(cum[-1])

    model = GPT(model_config(V)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    fpt = flops_per_token(V)
    tokens_per_step = BATCH_SIZE * BLOCK_SIZE
    max_steps = int(budget_pflops * 1e15 / (fpt * tokens_per_step))
    eval_steps = sorted({int(round(max_steps * k / n_evals)) for k in range(1, n_evals + 1)})
    bytes_per_tok_train = train_bytes_total / train_data.numel()

    def get_batch():
        ix = torch.randint(train_data.numel() - BLOCK_SIZE - 1, (BATCH_SIZE,), device=device)
        win = train_data[ix[:, None] + torch.arange(BLOCK_SIZE + 1, device=device)]
        return win[:, :-1], win[:, 1:]

    curve, best = [], {"val_bpb": float("inf")}
    train_time, step = 0.0, 0
    torch.cuda.reset_peak_memory_stats(device)
    for target in eval_steps:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while step < target:
            x, y = get_batch()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
            loss = F.cross_entropy(logits.float().view(-1, V), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
        torch.cuda.synchronize()
        train_time += time.perf_counter() - t0
        val_bpb, _, _ = eval_bpb(model, val_data, token_nbytes)
        train_bpb, _, _ = eval_bpb(model, train_slice, token_nbytes)
        tokens_seen = step * tokens_per_step
        point = {"step": step, "pflops": tokens_seen * fpt / 1e15, "tokens_seen": tokens_seen,
                 "epochs": tokens_seen / train_data.numel(), "val_bpb": val_bpb, "train_bpb": train_bpb,
                 "train_loss_nats": loss.item()}
        curve.append(point)
        if val_bpb < best["val_bpb"]:
            best = dict(point, state={k: v.detach().clone() for k, v in model.state_dict().items()})
        print(f"[{corpus}/{tok}/s{seed}] step {step}/{max_steps} val_bpb {val_bpb:.4f} "
              f"train_bpb {train_bpb:.4f} epochs {point['epochs']:.1f}", flush=True)

    final_test_bpb, _, _ = eval_bpb(model, test_data, token_nbytes)
    final_val_bpb = curve[-1]["val_bpb"]
    model.load_state_dict(best.pop("state"))
    best_test_bpb, test_nll, test_bytes = eval_bpb(model, test_data, token_nbytes)

    # a short sample from the best checkpoint, for qualitative inspection
    model.eval()
    idx = val_data[:16][None]
    with torch.no_grad():
        for _ in range(200):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(idx[:, -BLOCK_SIZE:])[0][:, -1].float()
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
    sample = tokenizer.decode(idx[0].tolist())

    result = {
        "corpus": corpus, "tokenizer": tok, "seed": seed, "vocab_size": V, "n_params": n_params,
        "flops_per_token": fpt, "budget_pflops": budget_pflops, "max_steps": max_steps,
        "tokens_seen": max_steps * tokens_per_step, "train_tokens": train_data.numel(),
        "bytes_per_token_train": bytes_per_tok_train,
        "bytes_seen": max_steps * tokens_per_step * bytes_per_tok_train,
        "epochs": max_steps * tokens_per_step / train_data.numel(),
        "context_bytes": BLOCK_SIZE * bytes_per_tok_train,
        "train_time_s": train_time, "train_tokens_per_s": max_steps * tokens_per_step / train_time,
        "train_bytes_per_s": max_steps * tokens_per_step * bytes_per_tok_train / train_time,
        "peak_mem_GB": torch.cuda.max_memory_allocated(device) / 1e9,
        "best_step": best["step"], "best_val_bpb": best["val_bpb"], "best_test_bpb": best_test_bpb,
        "final_val_bpb": final_val_bpb, "final_test_bpb": final_test_bpb,
        "test_nll_nats": test_nll, "test_bytes": test_bytes,
        "curve": curve, "sample": sample, "gpu": torch.cuda.get_device_name(device),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{corpus}_{tok}_s{seed}.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(f"[{corpus}/{tok}/s{seed}] DONE best test bpb {best_test_bpb:.4f} (step {best['step']}), "
          f"final test bpb {final_test_bpb:.4f}, {result['train_tokens_per_s']:.0f} tok/s", flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--budget_pflops", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_evals", type=int, default=25)
    ap.add_argument("--out_dir", default=RUNS_DIR)
    a = ap.parse_args()
    train(a.corpus, a.tok, a.budget_pflops, a.seed, a.n_evals, out_dir=a.out_dir)
