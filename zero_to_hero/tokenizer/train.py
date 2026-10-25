"""
Lecture 8 demo (as in minbpe's train.py): train BasicTokenizer and RegexTokenizer with a
512-token vocab on the Taylor Swift Wikipedia article, compare compression and training time,
and compare against the reproduced GPT-4 tokenizer.
"""

import os
import time

from zero_to_hero.common import RESULTS_DIR, data_path, save_results
from zero_to_hero.tokenizer import BasicTokenizer, RegexTokenizer


def main(vocab_size=512):
    text = open(data_path("taylorswift.txt"), encoding="utf-8").read()
    n_bytes = len(text.encode("utf-8"))
    out_dir = os.path.join(RESULTS_DIR, "tokenizers")
    os.makedirs(out_dir, exist_ok=True)
    res = {"corpus": "taylorswift.txt", "bytes": n_bytes, "vocab_size": vocab_size, "tokenizers": {}}
    trainers = [("basic", lambda: BasicTokenizer(), "train"),
                ("regex_minbpe_trainer", lambda: RegexTokenizer(), "train_minbpe"),
                ("regex_fast_trainer", lambda: RegexTokenizer(), "train")]
    for name, make, method in trainers:
        tok = make()
        t0 = time.perf_counter()
        getattr(tok, method)(text, vocab_size)
        dt = time.perf_counter() - t0
        ids = tok.encode(text)
        assert tok.decode(ids) == text
        tok.save(os.path.join(out_dir, f"{name}_{vocab_size}.json"))
        top = [tok.decode([idx]) for idx in list(tok.merges.values())[:12]]
        res["tokenizers"][name] = {"train_time_s": dt, "tokens": len(ids), "bytes_per_token": n_bytes / len(ids),
                                   "first_merges": top}
        print(f"{name:22s} trained in {dt:6.2f}s  {n_bytes / len(ids):.2f} bytes/token  first merges {top[:6]}")
    try:
        from zero_to_hero.tokenizer.gpt4 import GPT4Tokenizer
        gpt4 = GPT4Tokenizer()
        ids = gpt4.encode(text)
        res["tokenizers"]["gpt4_cl100k_reproduced"] = {"tokens": len(ids), "bytes_per_token": n_bytes / len(ids)}
        print(f"{'gpt4 (cl100k)':22s} {n_bytes / len(ids):.2f} bytes/token")
    except Exception as e:
        print("skipping GPT-4 comparison:", e)
    return save_results("tokenizer", res)


if __name__ == "__main__":
    main()
