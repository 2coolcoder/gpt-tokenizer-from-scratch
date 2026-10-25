"""
Lecture 2: bigram character-level language models on names.txt.

1. Counting: P(next | prev) from bigram counts with +1 smoothing.
2. Neural: one linear layer (27x27) on one-hot inputs, softmax, trained by gradient descent
   on the NLL. It converges to the counting model (W ~ log counts); the L2 term on W plays
   the role of count smoothing.
"""

import torch
import torch.nn.functional as F

from zero_to_hero.common import save_results
from zero_to_hero.makemore.data import VOCAB_SIZE, WORDS, build_dataset, sample_names, split_words


def bigram_pairs(words):
    X, Y = build_dataset(words, block_size=1)
    return X[:, 0], Y


def nll(P, xs, ys):
    return -P[xs, ys].log().mean().item()


def counting_model(xs, ys, smoothing=1):
    N = torch.zeros((VOCAB_SIZE, VOCAB_SIZE), dtype=torch.int32)
    N.index_put_((xs, ys), torch.ones_like(xs, dtype=torch.int32), accumulate=True)
    P = (N + smoothing).float()
    return P / P.sum(1, keepdim=True)


def neural_model(xs, ys, steps=300, lr=50.0, reg=0.01, seed=2147483647):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn((VOCAB_SIZE, VOCAB_SIZE), generator=g, requires_grad=True)
    xenc = F.one_hot(xs, num_classes=VOCAB_SIZE).float()
    losses = []
    for _ in range(steps):
        logits = xenc @ W  # log-counts
        loss = F.cross_entropy(logits, ys) + reg * (W ** 2).mean()
        W.grad = None
        loss.backward()
        W.data += -lr * W.grad
        losses.append(loss.item())
    return F.softmax(W.detach(), dim=1), losses


def main():
    tr, dev, te = split_words()
    xs_tr, ys_tr = bigram_pairs(tr)
    xs_dev, ys_dev = bigram_pairs(dev)
    xs_all, ys_all = bigram_pairs(WORDS)

    P_all = counting_model(xs_all, ys_all)  # the lecture's number: fit + evaluate on all names
    P_count = counting_model(xs_tr, ys_tr)
    P_nn, losses = neural_model(xs_tr, ys_tr)
    res = {
        "counting_all_words_nll": nll(P_all, xs_all, ys_all),
        "counting": {"train": nll(P_count, xs_tr, ys_tr), "dev": nll(P_count, xs_dev, ys_dev)},
        "neural": {"train": nll(P_nn, xs_tr, ys_tr), "dev": nll(P_nn, xs_dev, ys_dev),
                   "loss_curve": losses[::10]},
        "max_abs_prob_diff_neural_vs_counting": (P_nn - P_count).abs().max().item(),
        "samples_counting": sample_names(lambda c: P_count[c[:, -1]], block_size=1),
        "samples_neural": sample_names(lambda c: P_nn[c[:, -1]], block_size=1),
    }
    for k in ("counting", "neural"):
        print(f"{k:9s} train {res[k]['train']:.4f}  dev {res[k]['dev']:.4f}")
    print("counting model on all words:", round(res["counting_all_words_nll"], 4))
    print("samples:", res["samples_counting"][:5])
    return save_results("makemore_bigram", res)


if __name__ == "__main__":
    main()
