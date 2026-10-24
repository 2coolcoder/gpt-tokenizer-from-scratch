"""Wall-clock comparison of BPE trainers on a corpus' train split.

minbpe   : the course's RegexTokenizer.train - recount all pairs over every chunk after each merge
naive    : same, but over de-duplicated chunks weighted by frequency (BPETokenizer.train_reference)
fast     : de-duplicated chunks + incremental pair counts + lazy heap (BPETokenizer.train)
"""
import json
import os
import sys
import time

from tokbench.data import load_split
from tokbench.tok_suite import RESULTS_DIR
from tokbench.tokenization import BPETokenizer, merge


def minbpe_train(text, vocab_size, pattern):
    import regex as re
    ids = [list(ch.encode("utf-8")) for ch in re.findall(re.compile(pattern), text)]
    merges = {}
    for i in range(vocab_size - 256):
        stats = {}
        for chunk_ids in ids:
            for pair in zip(chunk_ids, chunk_ids[1:]):
                stats[pair] = stats.get(pair, 0) + 1
        pair = max(stats, key=stats.get)
        idx = 256 + i
        ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
        merges[pair] = idx
    return merges


def main(corpus="shakespeare", vocabs=(512, 1024, 2048, 4096, 8192), slow_max=1024):
    text = load_split(corpus, "train")
    rows = []
    for v in vocabs:
        row = {"corpus": corpus, "vocab_size": v}
        t0 = time.perf_counter(); fast = BPETokenizer().train(text, v); row["fast_s"] = time.perf_counter() - t0
        if v <= slow_max:
            t0 = time.perf_counter(); ref = BPETokenizer().train_reference(text, v); row["naive_s"] = time.perf_counter() - t0
            assert ref.merges == fast.merges
            t0 = time.perf_counter(); minbpe_train(text, v, fast.pattern); row["minbpe_s"] = time.perf_counter() - t0
        rows.append(row)
        print(row, flush=True)
    with open(os.path.join(RESULTS_DIR, f"bpe_trainer_speed_{corpus}.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:2])
