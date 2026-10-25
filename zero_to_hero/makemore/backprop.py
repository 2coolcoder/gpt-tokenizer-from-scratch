"""
Lecture 5: becoming a backprop ninja.

A 2-layer MLP with BatchNorm, with the forward pass broken into atomic tensor operations.
Every gradient is derived and computed by hand, then checked against PyTorch autograd:
  exercise 1  backprop through every intermediate tensor
  exercise 2  fused cross-entropy backward: dlogits = (softmax - onehot) / n
  exercise 3  fused BatchNorm backward in one expression
  exercise 4  train the network with the manual gradients only (no loss.backward())
"""

import torch
import torch.nn.functional as F

from zero_to_hero.common import save_results
from zero_to_hero.makemore.data import VOCAB_SIZE, sample_names, splits


def init_params(n_embd=10, n_hidden=64, block_size=3, seed=2147483647, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    fan_in = n_embd * block_size
    p = {
        "C": torch.randn((VOCAB_SIZE, n_embd), generator=g),
        "W1": torch.randn((fan_in, n_hidden), generator=g) * (5 / 3) / fan_in ** 0.5,
        "b1": torch.randn(n_hidden, generator=g) * 0.1,  # useless with BN, kept for the exercise
        "W2": torch.randn((n_hidden, VOCAB_SIZE), generator=g) * 0.1,
        "b2": torch.randn(VOCAB_SIZE, generator=g) * 0.1,
        # non-trivial gain/bias so that mistakes in their gradients are not hidden
        "bngain": torch.randn((1, n_hidden), generator=g) * 0.1 + 1.0,
        "bnbias": torch.randn((1, n_hidden), generator=g) * 0.1,
    }
    return {k: v.to(dtype) for k, v in p.items()}


def forward(p, Xb, Yb):
    """Forward pass as a dict of every intermediate tensor."""
    n = Xb.shape[0]
    t = {}
    t["emb"] = p["C"][Xb]
    t["embcat"] = t["emb"].view(n, -1)
    t["hprebn"] = t["embcat"] @ p["W1"] + p["b1"]
    t["bnmeani"] = 1 / n * t["hprebn"].sum(0, keepdim=True)
    t["bndiff"] = t["hprebn"] - t["bnmeani"]
    t["bndiff2"] = t["bndiff"] ** 2
    t["bnvar"] = 1 / (n - 1) * t["bndiff2"].sum(0, keepdim=True)  # Bessel's correction
    t["bnvar_inv"] = (t["bnvar"] + 1e-5) ** -0.5
    t["bnraw"] = t["bndiff"] * t["bnvar_inv"]
    t["hpreact"] = p["bngain"] * t["bnraw"] + p["bnbias"]
    t["h"] = torch.tanh(t["hpreact"])
    t["logits"] = t["h"] @ p["W2"] + p["b2"]
    t["logit_maxes"] = t["logits"].max(1, keepdim=True).values
    t["norm_logits"] = t["logits"] - t["logit_maxes"]  # numerical stability
    t["counts"] = t["norm_logits"].exp()
    t["counts_sum"] = t["counts"].sum(1, keepdim=True)
    t["counts_sum_inv"] = t["counts_sum"] ** -1
    t["probs"] = t["counts"] * t["counts_sum_inv"]
    t["logprobs"] = t["probs"].log()
    t["loss"] = -t["logprobs"][range(n), Yb].mean()
    return t


def manual_backward(p, t, Xb, Yb):
    """Exercise 1: the chain rule through every atomic op, in reverse order."""
    n = Xb.shape[0]
    d = {}
    d["logprobs"] = torch.zeros_like(t["logprobs"])
    d["logprobs"][range(n), Yb] = -1.0 / n
    d["probs"] = (1.0 / t["probs"]) * d["logprobs"]
    d["counts_sum_inv"] = (t["counts"] * d["probs"]).sum(1, keepdim=True)
    d["counts"] = t["counts_sum_inv"] * d["probs"]
    d["counts_sum"] = -t["counts_sum"] ** -2 * d["counts_sum_inv"]
    d["counts"] = d["counts"] + torch.ones_like(t["counts"]) * d["counts_sum"]
    d["norm_logits"] = t["counts"] * d["counts"]
    d["logit_maxes"] = (-d["norm_logits"]).sum(1, keepdim=True)
    d["logits"] = d["norm_logits"].clone()
    d["logits"] += F.one_hot(t["logits"].max(1).indices, num_classes=VOCAB_SIZE).to(t["logits"].dtype) * d["logit_maxes"]
    d["h"] = d["logits"] @ p["W2"].T
    d["W2"] = t["h"].T @ d["logits"]
    d["b2"] = d["logits"].sum(0)
    d["hpreact"] = (1.0 - t["h"] ** 2) * d["h"]
    d["bngain"] = (t["bnraw"] * d["hpreact"]).sum(0, keepdim=True)
    d["bnraw"] = p["bngain"] * d["hpreact"]
    d["bnbias"] = d["hpreact"].sum(0, keepdim=True)
    d["bndiff"] = t["bnvar_inv"] * d["bnraw"]
    d["bnvar_inv"] = (t["bndiff"] * d["bnraw"]).sum(0, keepdim=True)
    d["bnvar"] = -0.5 * (t["bnvar"] + 1e-5) ** -1.5 * d["bnvar_inv"]
    d["bndiff2"] = 1.0 / (n - 1) * torch.ones_like(t["bndiff2"]) * d["bnvar"]
    d["bndiff"] = d["bndiff"] + 2 * t["bndiff"] * d["bndiff2"]
    d["bnmeani"] = (-d["bndiff"]).sum(0, keepdim=True)
    d["hprebn"] = d["bndiff"].clone() + 1.0 / n * torch.ones_like(t["hprebn"]) * d["bnmeani"]
    d["embcat"] = d["hprebn"] @ p["W1"].T
    d["W1"] = t["embcat"].T @ d["hprebn"]
    d["b1"] = d["hprebn"].sum(0)
    d["emb"] = d["embcat"].view(t["emb"].shape)
    d["C"] = torch.zeros_like(p["C"]).index_add_(0, Xb.view(-1), d["emb"].view(-1, d["emb"].shape[-1]))
    return d


def fused_backward(p, Xb, Yb):
    """Exercises 2 + 3: analytic cross-entropy and BatchNorm backward, fewer intermediates."""
    n = Xb.shape[0]
    emb = p["C"][Xb]
    embcat = emb.view(n, -1)
    hprebn = embcat @ p["W1"] + p["b1"]
    bnmean = hprebn.mean(0, keepdim=True)
    bnvar = hprebn.var(0, keepdim=True, unbiased=True)
    bnvar_inv = (bnvar + 1e-5) ** -0.5
    bnraw = (hprebn - bnmean) * bnvar_inv
    hpreact = p["bngain"] * bnraw + p["bnbias"]
    h = torch.tanh(hpreact)
    logits = h @ p["W2"] + p["b2"]
    loss = F.cross_entropy(logits, Yb)

    d = {}
    dlogits = F.softmax(logits, 1)
    dlogits[range(n), Yb] -= 1
    dlogits /= n
    d["logits"] = dlogits
    d["h"] = dlogits @ p["W2"].T
    d["W2"] = h.T @ dlogits
    d["b2"] = dlogits.sum(0)
    dhpreact = (1.0 - h ** 2) * d["h"]
    d["bngain"] = (bnraw * dhpreact).sum(0, keepdim=True)
    d["bnbias"] = dhpreact.sum(0, keepdim=True)
    d["hprebn"] = p["bngain"] * bnvar_inv / n * (
        n * dhpreact - dhpreact.sum(0) - n / (n - 1) * bnraw * (dhpreact * bnraw).sum(0))
    d["W1"] = embcat.T @ d["hprebn"]
    d["b1"] = d["hprebn"].sum(0)
    demb = (d["hprebn"] @ p["W1"].T).view(emb.shape)
    d["C"] = torch.zeros_like(p["C"]).index_add_(0, Xb.view(-1), demb.view(-1, demb.shape[-1]))
    return loss, d


def gradient_check(Xtr, Ytr, batch_size=32, dtype=torch.float64):
    """Compare manual and fused gradients with autograd; returns max |diff| per tensor."""
    p = init_params(dtype=dtype)
    for v in p.values():
        v.requires_grad = True
    g = torch.Generator().manual_seed(42)
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]
    t = forward(p, Xb, Yb)
    for v in t.values():
        v.retain_grad()
    t["loss"].backward()
    with torch.no_grad():
        d = manual_backward(p, t, Xb, Yb)
        _, df = fused_backward(p, Xb, Yb)
    report = {}
    for k, dv in d.items():
        ref = p[k].grad if k in p else t[k].grad
        report[k] = (dv - ref).abs().max().item()
    fused = {k: (df[k] - (p[k].grad if k in p else t[k].grad)).abs().max().item() for k in df}
    return report, fused


def train_manual(Xtr, Ytr, Xdev, Ydev, steps=200_000, batch_size=32, seed=2147483647):
    """Exercise 4: train with the fused manual gradients only (hidden 200, as in the lecture)."""
    p = init_params(n_hidden=200)
    params = ["C", "W1", "b1", "W2", "b2", "bngain", "bnbias"]
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for i in range(steps):
            ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
            loss, d = fused_backward(p, Xtr[ix], Ytr[ix])
            lr = 0.1 if i < 100_000 else 0.01
            for k in params:
                p[k] += -lr * d[k]
            if i % 20_000 == 0:
                print(f"{i:7d}/{steps}: {loss.item():.4f}")

        # calibrate BatchNorm statistics on the whole training set, then evaluate
        hpre = p["C"][Xtr].view(Xtr.shape[0], -1) @ p["W1"] + p["b1"]
        bnmean, bnvar = hpre.mean(0, keepdim=True), hpre.var(0, keepdim=True, unbiased=True)

        def logits_of(X):
            hp = p["C"][X].view(X.shape[0], -1) @ p["W1"] + p["b1"]
            h = torch.tanh(p["bngain"] * (hp - bnmean) * (bnvar + 1e-5) ** -0.5 + p["bnbias"])
            return h @ p["W2"] + p["b2"]

        return {"train": F.cross_entropy(logits_of(Xtr), Ytr).item(),
                "dev": F.cross_entropy(logits_of(Xdev), Ydev).item(),
                "samples": sample_names(lambda c: F.softmax(logits_of(c), 1), block_size=3)}


def main():
    torch.set_num_threads(1)
    (Xtr, Ytr), (Xdev, Ydev), _ = splits(block_size=3)
    manual, fused = gradient_check(Xtr, Ytr)
    print("max |manual - autograd| over all tensors:", max(manual.values()))
    print("max |fused  - autograd| over all tensors:", max(fused.values()))
    trained = train_manual(Xtr, Ytr, Xdev, Ydev)
    print(f"manual-gradient training: train {trained['train']:.4f} dev {trained['dev']:.4f}")
    return save_results("makemore_backprop", {
        "gradient_check_float64_max_abs_diff": {"per_tensor_manual": manual, "per_tensor_fused": fused,
                                                "max_manual": max(manual.values()),
                                                "max_fused": max(fused.values())},
        "manual_training": trained,
    })


if __name__ == "__main__":
    main()
