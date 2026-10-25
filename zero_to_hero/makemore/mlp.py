"""
Lecture 3: MLP character-level language model (Bengio et al. 2003) on names.txt.

The previous `block_size` characters are embedded with a shared lookup table C, concatenated,
passed through a tanh hidden layer and a softmax output layer. Written with raw tensors,
as in the lecture.
"""

import torch
import torch.nn.functional as F

from zero_to_hero.common import save_results
from zero_to_hero.makemore.data import VOCAB_SIZE, sample_names, splits


class MLP:
    def __init__(self, block_size=3, n_embd=10, n_hidden=200, seed=2147483647):
        g = torch.Generator().manual_seed(seed)
        self.block_size = block_size
        self.C = torch.randn((VOCAB_SIZE, n_embd), generator=g)
        self.W1 = torch.randn((n_embd * block_size, n_hidden), generator=g)
        self.b1 = torch.randn(n_hidden, generator=g)
        self.W2 = torch.randn((n_hidden, VOCAB_SIZE), generator=g)
        self.b2 = torch.randn(VOCAB_SIZE, generator=g)
        for p in self.parameters():
            p.requires_grad = True

    def parameters(self):
        return [self.C, self.W1, self.b1, self.W2, self.b2]

    def __call__(self, X):
        emb = self.C[X]  # (B, block_size, n_embd)
        h = torch.tanh(emb.view(emb.shape[0], -1) @ self.W1 + self.b1)
        return h @ self.W2 + self.b2


@torch.no_grad()
def evaluate(model, X, Y):
    return F.cross_entropy(model(X), Y).item()


def train(model, Xtr, Ytr, steps=200_000, batch_size=32, lr_fn=lambda i: 0.1 if i < 100_000 else 0.01,
          seed=2147483647, log_every=10_000):
    g = torch.Generator().manual_seed(seed)
    lossi = []
    for i in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        loss = F.cross_entropy(model(Xtr[ix]), Ytr[ix])
        for p in model.parameters():
            p.grad = None
        loss.backward()
        for p in model.parameters():
            p.data += -lr_fn(i) * p.grad
        lossi.append(loss.log10().item())
        if i % log_every == 0:
            print(f"{i:7d}/{steps}: {loss.item():.4f}")
    return lossi


def main():
    torch.set_num_threads(1)
    (Xtr, Ytr), (Xdev, Ydev), (Xte, Yte) = splits(block_size=3)
    model = MLP()
    lossi = train(model, Xtr, Ytr)
    res = {
        "config": {"block_size": 3, "n_embd": 10, "n_hidden": 200, "steps": 200_000, "batch_size": 32},
        "n_params": sum(p.nelement() for p in model.parameters()),
        "train": evaluate(model, Xtr, Ytr), "dev": evaluate(model, Xdev, Ydev), "test": evaluate(model, Xte, Yte),
        "loss_curve_log10": [sum(lossi[i:i + 1000]) / 1000 for i in range(0, len(lossi), 1000)],
        "samples": sample_names(lambda c: F.softmax(model(c), dim=1), block_size=3),
    }
    print(f"train {res['train']:.4f} dev {res['dev']:.4f} test {res['test']:.4f}")
    print("samples:", res["samples"])
    return save_results("makemore_mlp", res)


if __name__ == "__main__":
    main()
