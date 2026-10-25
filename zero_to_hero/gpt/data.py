"""Character-level Tiny Shakespeare dataset for lecture 7."""

import torch

from zero_to_hero.common import data_path


class CharDataset:
    def __init__(self, path=None, train_frac=0.9):
        with open(path or data_path("input.txt"), encoding="utf-8") as f:
            self.text = f.read()
        self.chars = sorted(set(self.text))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        data = torch.tensor(self.encode(self.text), dtype=torch.long)
        n = int(train_frac * len(data))  # first 90% train, last 10% val
        self.train_data, self.val_data = data[:n], data[n:]

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def get_batch(self, split, batch_size, block_size, device="cpu"):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)
