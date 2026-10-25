"""
Lecture 8: byte-level BPE with a regex pre-split and special tokens (minbpe RegexTokenizer).

The text is first split into chunks (words, numbers, punctuation, whitespace) with the GPT-2 /
GPT-4 regex, so merges never cross chunk boundaries. Three trainers share the same greedy
"merge the most frequent pair" algorithm:

  train_minbpe     the lecture's algorithm: recount every pair over every chunk after each merge
  train_reference  the same over de-duplicated chunks weighted by frequency
  train            de-duplicated chunks + incremental pair-count updates + a lazy max-heap;
                   identical merges to train_reference, about 400x faster than train_minbpe

train / train_reference break count ties by the smallest pair, so they are deterministic.
"""

import heapq
from collections import Counter, defaultdict

import regex as re

from zero_to_hero.tokenizer.base import Tokenizer, encode_bytes, get_stats, merge

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):
    kind = "bpe"

    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled = re.compile(self.pattern)
        self.inverse_special_tokens = {}
        self._cache = {}

    def register_special_tokens(self, special_tokens):
        """special_tokens: dict str -> int, e.g. {"<|endoftext|>": 100257}."""
        self.special_tokens = dict(special_tokens)
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
        self.vocab = self._build_vocab()

    def set_merges(self, merges):
        super().set_merges(merges)
        self._cache = {}

    # ---------------------------------------------------------------- training

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        counts = Counter(self.compiled.findall(text))
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
        self.set_merges(merges)
        return self

    def train_reference(self, text, vocab_size):
        counts = Counter(self.compiled.findall(text))
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
        self.set_merges(merges)
        return self

    def train_minbpe(self, text, vocab_size):
        ids = [list(ch.encode("utf-8")) for ch in self.compiled.findall(text)]
        merges = {}
        for i in range(vocab_size - 256):
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            merges[pair] = idx
        self.set_merges(merges)
        return self

    # ---------------------------------------------------------------- encode / decode

    def _encode_chunk(self, chunk_bytes):
        return encode_bytes(chunk_bytes, self.merges)

    def encode_ordinary(self, text, use_cache=True):
        """Encode ignoring special tokens."""
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

    def encode(self, text, allowed_special="none_raise", use_cache=True):
        """allowed_special: "all" | "none" | "none_raise" | set of special-token strings."""
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(tok not in text for tok in self.special_tokens), "text contains a special token"
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            return self.encode_ordinary(text, use_cache)
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        out = []
        for part in re.split(special_pattern, text):
            if part in special:
                out.append(special[part])
            elif part:
                out.extend(self.encode_ordinary(part, use_cache))
        return out

    def decode(self, ids):
        parts = []
        for i in ids:
            if i in self.inverse_special_tokens:
                parts.append(self.inverse_special_tokens[i].encode("utf-8"))
            else:
                parts.append(self.vocab[i])
        return b"".join(parts).decode("utf-8", errors="replace")
