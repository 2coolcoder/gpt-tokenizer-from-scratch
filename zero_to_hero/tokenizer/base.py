"""
Lecture 8: shared BPE machinery (minbpe-style).

A BPE tokenizer is a list of merges: (token a, token b) -> new token id, applied in the
order they were learned, starting from the 256 raw bytes. Training finds the merges;
encode replays them; decode concatenates each token's bytes.
"""

import json


def get_stats(ids, counts=None):
    """Count consecutive pairs: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}."""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """Replace every non-overlapping occurrence of `pair` in `ids` with `idx`."""
    out = []
    i = 0
    n = len(ids)
    p0, p1 = pair
    while i < n:
        if i < n - 1 and ids[i] == p0 and ids[i + 1] == p1:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def encode_bytes(ids, merges):
    """Apply learned merges to a byte list: always merge the pair that was learned first."""
    ids = list(ids)
    while len(ids) >= 2:
        best, best_rank = None, None
        for p in zip(ids, ids[1:]):
            r = merges.get(p)
            if r is not None and (best_rank is None or r < best_rank):
                best, best_rank = p, r
        if best is None:
            break
        ids = merge(ids, best, best_rank)
    return ids


class Tokenizer:
    """Base class: merges, special tokens, vocab, save/load."""

    kind = "base"

    def __init__(self):
        self.merges = {}  # (int, int) -> int, in learning order
        self.pattern = ""
        self.special_tokens = {}  # str -> int
        self.vocab = self._build_vocab()

    @property
    def vocab_size(self):
        return 256 + len(self.merges) + len(self.special_tokens)

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def set_merges(self, merges):
        self.merges = dict(merges)
        self.vocab = self._build_vocab()

    def token_bytes(self, idx):
        return self.vocab[idx]

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"kind": self.kind, "pattern": self.pattern, "special_tokens": self.special_tokens,
                       "merges": [[p0, p1, idx] for (p0, p1), idx in self.merges.items()]}, f)
