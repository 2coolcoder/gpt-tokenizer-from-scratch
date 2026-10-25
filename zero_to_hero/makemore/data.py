"""names.txt dataset shared by every makemore lecture: vocab, (context -> next char) examples, splits."""

import random

import torch

from zero_to_hero.common import data_path


def load_words():
    with open(data_path("names.txt")) as f:
        return f.read().splitlines()


WORDS = load_words()
CHARS = sorted(set("".join(WORDS)))
STOI = {s: i + 1 for i, s in enumerate(CHARS)}
STOI["."] = 0  # '.' marks both the start and the end of a name
ITOS = {i: s for s, i in STOI.items()}
VOCAB_SIZE = len(STOI)


def build_dataset(words, block_size):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = STOI[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


def split_words(seed=42):
    """80/10/10 train/dev/test split of the names, as in the lectures."""
    words = WORDS[:]
    random.seed(seed)
    random.shuffle(words)
    n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
    return words[:n1], words[n1:n2], words[n2:]


def splits(block_size, seed=42):
    tr, dev, te = split_words(seed)
    return build_dataset(tr, block_size), build_dataset(dev, block_size), build_dataset(te, block_size)


@torch.no_grad()
def sample_names(next_probs, block_size, n=10, seed=2147483647):
    """Sample names; next_probs(context LongTensor (1, block_size)) -> probs (1, vocab)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        context, name = [0] * block_size, []
        while True:
            probs = next_probs(torch.tensor([context]))
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            if ix == 0:
                break
            name.append(ITOS[ix])
            context = context[1:] + [ix]
        out.append("".join(name))
    return out
