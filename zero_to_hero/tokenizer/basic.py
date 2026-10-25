"""Lecture 8: the simplest byte-level BPE, trained on the raw byte stream (no regex pre-split)."""

from zero_to_hero.tokenizer.base import Tokenizer, encode_bytes, get_stats, merge


class BasicTokenizer(Tokenizer):
    kind = "basic"

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        ids = list(text.encode("utf-8"))
        merges = {}
        for i in range(vocab_size - 256):
            stats = get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)  # the most frequent pair
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            if verbose:
                print(f"merge {i + 1}/{vocab_size - 256}: {pair} -> {idx} ({stats[pair]} occurrences)")
        self.set_merges(merges)
        return self

    def encode(self, text):
        return encode_bytes(text.encode("utf-8"), self.merges)
