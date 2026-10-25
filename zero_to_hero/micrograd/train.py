"""
Train a micrograd MLP as a binary classifier on the two-moons dataset (micrograd demo).

MLP 2 -> 16 -> 16 -> 1, max-margin (SVM) loss + L2 regularization, full-batch SGD with
a decaying learning rate. Every gradient comes from the scalar autograd engine.
"""

import math
import random

import numpy as np

from zero_to_hero.common import save_results
from zero_to_hero.micrograd.engine import Value
from zero_to_hero.micrograd.nn import MLP


def make_moons(n=100, noise=0.1, seed=1):
    """Two interleaving half circles, labels in {-1, +1} (same recipe as sklearn.make_moons)."""
    rng = np.random.default_rng(seed)
    n_out, n_in = n // 2, n - n // 2
    t_out, t_in = np.linspace(0, math.pi, n_out), np.linspace(0, math.pi, n_in)
    X = np.vstack([np.stack([np.cos(t_out), np.sin(t_out)], 1),
                   np.stack([1 - np.cos(t_in), 1 - np.sin(t_in) - 0.5], 1)])
    y = np.hstack([-np.ones(n_out), np.ones(n_in)])
    X += rng.normal(scale=noise, size=X.shape)
    return X, y


def loss_fn(model, X, y, alpha=1e-4):
    scores = [model([Value(float(a)) for a in x]) for x in X]
    losses = [(1 + -yi * si).relu() for yi, si in zip(y, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    reg_loss = alpha * sum(p * p for p in model.parameters())
    acc = sum((yi > 0) == (si.data > 0) for yi, si in zip(y, scores)) / len(y)
    return data_loss + reg_loss, acc


def main(steps=100):
    random.seed(1337)
    X, y = make_moons()
    model = MLP(2, [16, 16, 1])
    history = []
    for k in range(steps):
        loss, acc = loss_fn(model, X, y)
        model.zero_grad()
        loss.backward()
        lr = 1.0 - 0.9 * k / steps
        for p in model.parameters():
            p.data -= lr * p.grad
        history.append({"step": k, "loss": loss.data, "acc": acc})
        if k % 10 == 0 or k == steps - 1:
            print(f"step {k} loss {loss.data:.4f} accuracy {acc * 100:.0f}%")
    return save_results("micrograd", {
        "task": "two-moons binary classification (100 points)",
        "model": "MLP 2-16-16-1 (ReLU), micrograd scalar autograd",
        "n_params": len(model.parameters()),
        "final_loss": history[-1]["loss"], "final_accuracy": history[-1]["acc"],
        "history": history,
    })


if __name__ == "__main__":
    main()
