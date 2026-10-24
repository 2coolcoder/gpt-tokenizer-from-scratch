"""
Tokenizers for the ablation suite.

- CharTokenizer: the character-level tokenizer from the course (gpt.py / makemore).
- BPETokenizer: byte-level BPE with the GPT-4 regex pre-split, i.e. minbpe's
  RegexTokenizer. Training uses the same greedy "merge the most frequent pair"
  algorithm as minbpe, but runs on de-duplicated regex chunks with incremental
  pair-count updates and a lazy max-heap, so an 8K vocab trains in seconds
  instead of hours. Ties are broken deterministically (smallest pair id wins).
  `train_reference` is the naive O(merges * corpus) version, kept for tests.
"""

import heapq
import json
from collections import Counter, defaultdict

import regex as re

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def merge(ids, pair, idx):
    """Replace every non-overlapping occurrence of `pair` in `ids` with `idx` (minbpe's merge)."""
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


class CharTokenizer:
    """One token per character; the vocabulary is the set of characters in the corpus."""

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


class BPETokenizer:
    """Byte-level BPE with regex pre-splitting (minbpe RegexTokenizer semantics)."""

    kind = "bpe"

    def __init__(self, pattern=GPT4_SPLIT_PATTERN):
        self.pattern = pattern
        self.compiled = re.compile(pattern)
        self.merges = {}  # (int, int) -> int, in insertion (= rank) order
        self.vocab = {i: bytes([i]) for i in range(256)}
        self._cache = {}

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    def _set_merges(self, merges):
        self.merges = dict(merges)
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        self._cache = {}

    def _chunk_counts(self, text):
        return Counter(self.compiled.findall(text))

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        counts = self._chunk_counts(text)
        words = [list(ch.encode("utf-8")) for ch in counts]
        freqs = list(counts.values())

        pair_counts = defaultdict(int)
        where = defaultdict(set)  # pair -> indices of words that (may) contain it
        for wi, w in enumerate(words):
            f = freqs[wi]
            for p in zip(w, w[1:]):
                pair_counts[p] += f
                where[p].add(wi)
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        merges = {}
        for i in range(num_merges):
            # pop the most frequent pair, skipping stale heap entries
            while heap:
                negc, pair = heapq.heappop(heap)
                if pair_counts.get(pair, 0) == -negc and -negc > 0:
                    break
            else:
                break  # nothing left to merge
            idx = 256 + i
            merges[pair] = idx
            touched = set()
            for wi in where.pop(pair):
                w = words[wi]
                if len(w) < 2:
                    continue
                f = freqs[wi]
                new_w = merge(w, pair, idx)
                if len(new_w) == len(w):
                    continue  # stale index entry
                for p in zip(w, w[1:]):
                    pair_counts[p] -= f
                    touched.add(p)
                for p in zip(new_w, new_w[1:]):
                    pair_counts[p] += f
                    where[p].add(wi)
                    touched.add(p)
                words[wi] = new_w
            for p in touched:
                c = pair_counts[p]
                if c > 0:
                    heapq.heappush(heap, (-c, p))
                else:
                    pair_counts.pop(p, None)
            if verbose and (i + 1) % 1000 == 0:
                print(f"  merge {i + 1}/{num_merges}")
        self._set_merges(merges)
        return self

    def train_reference(self, text, vocab_size):
        """Naive trainer: recount all pairs after every merge (same tie-break as `train`)."""
        counts = self._chunk_counts(text)
        words = [list(ch.encode("utf-8")) for ch in counts]
        freqs = list(counts.values())
        merges = {}
        for i in range(vocab_size - 256):
            stats = defaultdict(int)
            for w, f in zip(words, freqs):
                for p in zip(w, w[1:]):
                    stats[p] += f
            if not stats:
                break
            pair = min(stats, key=lambda p: (-stats[p], p))
            idx = 256 + i
            merges[pair] = idx
            words = [merge(w, pair, idx) for w in words]
        self._set_merges(merges)
        return self

    def _encode_chunk(self, chunk_bytes):
        ids = list(chunk_bytes)
        merges = self.merges
        while len(ids) >= 2:
            # the pair with the lowest merge rank gets merged first (minbpe encode)
            best = None
            best_rank = None
            for p in zip(ids, ids[1:]):
                r = merges.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = p, r
            if best is None:
                break
            ids = merge(ids, best, best_rank)
        return ids

    def encode(self, text, use_cache=True):
        out = []
        cache = self._cache
        for chunk in self.compiled.findall(text):
            ids = cache.get(chunk) if use_cache else None
            if ids is None:
                ids = self._encode_chunk(chunk.encode("utf-8"))
                if use_cache:
                    cache[chunk] = ids
            out.extend(ids)
        return out

    def decode(self, ids):
        vocab = self.vocab
        return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

    def token_bytes(self, i):
        return self.vocab[i]

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"kind": "bpe", "pattern": self.pattern,
                       "merges": [[p0, p1, idx] for (p0, p1), idx in self.merges.items()]}, f)


def load_tokenizer(path):
    with open(path) as f:
        d = json.load(f)
    if d["kind"] == "char":
        return CharTokenizer(d["chars"])
    tok = BPETokenizer(d["pattern"])
    tok._set_merges({(p0, p1): idx for p0, p1, idx in d["merges"]})
    return tok
