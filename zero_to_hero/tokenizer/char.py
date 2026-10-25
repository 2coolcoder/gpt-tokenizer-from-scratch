"""Character-level tokenizer (the one used by makemore and the lecture-7 GPT)."""

import json


class CharTokenizer:
    """One token per character; the vocabulary is the set of characters seen in training."""

    kind = "char"

    def __init__(self, chars=None):
        self.chars = sorted(chars) if chars else []
        self._build()

    def _build(self):
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    def train(self, text):
        self.chars = sorted(set(text))
        self._build()
        return self

    def encode(self, text):
        stoi = self.stoi
        return [stoi[c] for c in text]

    def decode(self, ids):
        itos = self.itos
        return "".join(itos[i] for i in ids)

    def token_bytes(self, i):
        return self.itos[i].encode("utf-8")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"kind": "char", "chars": self.chars}, f)
