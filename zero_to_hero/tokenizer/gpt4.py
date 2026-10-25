"""
Lecture 8: reproduce the GPT-4 tokenizer (tiktoken cl100k_base) with our RegexTokenizer.

tiktoken ships only the final token -> rank table, not the merge list, so the merges are
recovered by re-running BPE on each token's bytes with the ranks below it. GPT-4 also
permutes the 256 single-byte tokens, so encode and decode apply that byte shuffle.
"""

from zero_to_hero.tokenizer.base import encode_bytes
from zero_to_hero.tokenizer.regex import GPT4_SPLIT_PATTERN, RegexTokenizer

GPT4_SPECIAL_TOKENS = {
    "<|endoftext|>": 100257,
    "<|fim_prefix|>": 100258,
    "<|fim_middle|>": 100259,
    "<|fim_suffix|>": 100260,
    "<|endofprompt|>": 100276,
}


def bpe(mergeable_ranks, token, max_rank):
    """Split `token` into the two parts its final merge joined (BPE restricted to lower ranks)."""
    parts = [bytes([b]) for b in token]
    while True:
        min_idx, min_rank = None, None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx, min_rank = i, rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts


def recover_merges(mergeable_ranks):
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue
        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2
        merges[(mergeable_ranks[pair[0]], mergeable_ranks[pair[1]])] = rank
    return merges


class GPT4Tokenizer(RegexTokenizer):
    def __init__(self):
        super().__init__(pattern=GPT4_SPLIT_PATTERN)
        import tiktoken
        mergeable_ranks = tiktoken.get_encoding("cl100k_base")._mergeable_ranks
        self.byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
        self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}
        # vocab ids are tiktoken ranks; vocab bytes are in the shuffled byte space
        self.set_merges(recover_merges(mergeable_ranks))
        self.register_special_tokens(GPT4_SPECIAL_TOKENS)

    def _encode_chunk(self, chunk_bytes):
        return encode_bytes(bytes(self.byte_shuffle[b] for b in chunk_bytes), self.merges)

    def decode(self, ids):
        parts = []
        for i in ids:
            if i in self.inverse_special_tokens:
                parts.append(self.inverse_special_tokens[i].encode("utf-8"))
            else:
                parts.append(bytes(self.inverse_byte_shuffle[b] for b in self.vocab[i]))
        return b"".join(parts).decode("utf-8", errors="replace")

    def token_bytes(self, idx):
        return bytes(self.inverse_byte_shuffle[b] for b in self.vocab[idx])
