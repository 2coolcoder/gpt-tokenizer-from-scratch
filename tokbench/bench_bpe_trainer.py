"""Wall-clock comparison of BPE trainers on a corpus' train split.

minbpe   : the lecture algorithm - recount all pairs over every chunk after each merge (RegexTokenizer.train_minbpe)
naive    : same, but over de-duplicated chunks weighted by frequency (RegexTokenizer.train_reference)
fast     : de-duplicated chunks + incremental pair counts + lazy heap (RegexTokenizer.train)
"""
import json
import os
import sys
import time

from tokbench.data import load_split
from tokbench.tok_suite import RESULTS_DIR
from zero_to_hero.tokenizer import RegexTokenizer


def main(corpus="shakespeare", vocabs=(512, 1024, 2048, 4096, 8192), slow_max=1024):
    text = load_split(corpus, "train")
    rows = []
    for v in vocabs:
        row = {"corpus": corpus, "vocab_size": v}
        t0 = time.perf_counter(); fast = RegexTokenizer().train(text, v); row["fast_s"] = time.perf_counter() - t0
        if v <= slow_max:
            t0 = time.perf_counter(); ref = RegexTokenizer().train_reference(text, v); row["naive_s"] = time.perf_counter() - t0
            assert ref.merges == fast.merges
            t0 = time.perf_counter(); RegexTokenizer().train_minbpe(text, v); row["minbpe_s"] = time.perf_counter() - t0
        rows.append(row)
        print(row, flush=True)
    with open(os.path.join(RESULTS_DIR, f"bpe_trainer_speed_{corpus}.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:2])
