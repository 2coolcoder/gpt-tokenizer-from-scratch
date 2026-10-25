"""
Lecture 6: building a WaveNet.

Context grows from 3 to 8 characters, and instead of squashing all 8 into one hidden layer
the model fuses them pairwise in a tree (FlattenConsecutive(2) three times), like the dilated
causal convolutions of WaveNet (2016). Built from the hand-written layers in layers.py.

Ablation ladder (lecture 6 progression):
  mlp_ctx3      flat MLP, 3-char context, n_embd 10, hidden 200   (~12K params)
  mlp_ctx8      flat MLP, 8-char context, n_embd 10, hidden 200   (~22K params)
  wavenet_22k   hierarchical, n_embd 10, hidden 68                (~22K params)
  wavenet_76k   hierarchical, n_embd 24, hidden 128               (~76K params)
"""

import sys

import torch
import torch.nn.functional as F

from zero_to_hero.common import save_results
from zero_to_hero.makemore.data import VOCAB_SIZE, sample_names, splits
from zero_to_hero.makemore.layers import (BatchNorm1d, Embedding, Flatten, FlattenConsecutive, Linear,
                                          Sequential, Tanh)


def flat_mlp(block_size, n_embd, n_hidden, g):
    return Sequential([Embedding(VOCAB_SIZE, n_embd, generator=g), Flatten(),
                       Linear(n_embd * block_size, n_hidden, bias=False, generator=g), BatchNorm1d(n_hidden), Tanh(),
                       Linear(n_hidden, VOCAB_SIZE, generator=g)])


def wavenet(n_embd, n_hidden, g):
    return Sequential([
        Embedding(VOCAB_SIZE, n_embd, generator=g),
        FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False, generator=g), BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False, generator=g), BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False, generator=g), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden, VOCAB_SIZE, generator=g),
    ])


CONFIGS = {
    "mlp_ctx3": dict(block_size=3, build=lambda g: flat_mlp(3, 10, 200, g)),
    "mlp_ctx8": dict(block_size=8, build=lambda g: flat_mlp(8, 10, 200, g)),
    "wavenet_22k": dict(block_size=8, build=lambda g: wavenet(10, 68, g)),
    "wavenet_76k": dict(block_size=8, build=lambda g: wavenet(24, 128, g)),
}


def run(name, steps=200_000, batch_size=32, seed=42):
    cfg = CONFIGS[name]
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    (Xtr, Ytr), (Xdev, Ydev), (Xte, Yte) = splits(cfg["block_size"])
    model = cfg["build"](g)
    with torch.no_grad():
        model.layers[-1].weight *= 0.1  # less confident output layer at init
    params = model.parameters()
    for p in params:
        p.requires_grad = True
    for i in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        loss = F.cross_entropy(model(Xtr[ix]), Ytr[ix])
        for p in params:
            p.grad = None
        loss.backward()
        lr = 0.1 if i < 150_000 else 0.01
        for p in params:
            p.data += -lr * p.grad
        if i % 20_000 == 0:
            print(f"[{name}] {i:7d}/{steps}: {loss.item():.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        res = {"n_params": sum(p.nelement() for p in params), "block_size": cfg["block_size"],
               "train": F.cross_entropy(model(Xtr), Ytr).item(), "dev": F.cross_entropy(model(Xdev), Ydev).item(),
               "test": F.cross_entropy(model(Xte), Yte).item()}
    res["samples"] = sample_names(lambda c: F.softmax(model(c), 1), block_size=cfg["block_size"])
    print(f"[{name}] params {res['n_params']} train {res['train']:.4f} dev {res['dev']:.4f}", flush=True)
    return res


def main(names=tuple(CONFIGS)):
    torch.set_num_threads(1)
    results = {n: run(n) for n in names}
    suffix = "" if len(names) == len(CONFIGS) else "_" + "_".join(names)
    return save_results("makemore_wavenet" + suffix, results)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or tuple(CONFIGS))
