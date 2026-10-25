"""
Train every tokenizer, measure its intrinsic metrics, and cache token streams for LM training.

Per corpus:
  char            character-level (the course's tokenizer)
  bpe{V}          byte-level regex BPE trained on this corpus' train split, V in VOCABS
  xbpe2048        BPE-2048 trained on the *other* corpus (domain-mismatch ablation)
  tiktoken refs   gpt2 / cl100k_base, intrinsic metrics only (trained on web-scale data)

Outputs: tokbench_out/tokenizers/*.json, tokbench_out/tokens/<corpus>/<tok>/<split>.npy,
         tokbench_out/results/tokenizer_metrics.json
"""

import json
import os
import statistics
import time
from collections import Counter

import numpy as np

from tokbench.data import DATA_DIR, load_docs, load_split
from zero_to_hero.tokenizer import CharTokenizer, RegexTokenizer, load_tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tokbench_out")
TOK_DIR = os.path.join(OUT, "tokenizers")
TOKENS_DIR = os.path.join(OUT, "tokens")
RESULTS_DIR = os.path.join(OUT, "results")
CORPORA = ("shakespeare", "python")
VOCABS = (512, 1024, 2048, 4096, 8192)
OTHER = {"shakespeare": "python", "python": "shakespeare"}


def tokenizer_names():
    return ["char"] + [f"bpe{v}" for v in VOCABS] + ["xbpe2048"]


def tok_path(corpus, name):
    return os.path.join(TOK_DIR, f"{corpus}_{name}.json")


def train_tokenizers():
    """Train char + BPE tokenizers on each corpus' train split; returns training times (s)."""
    os.makedirs(TOK_DIR, exist_ok=True)
    times = {}
    for corpus in CORPORA:
        text = load_split(corpus, "train")
        t0 = time.perf_counter()
        CharTokenizer().train(text).save(tok_path(corpus, "char"))
        times[(corpus, "char")] = time.perf_counter() - t0
        for v in VOCABS:
            t0 = time.perf_counter()
            tok = RegexTokenizer().train(text, v)
            times[(corpus, f"bpe{v}")] = time.perf_counter() - t0
            tok.save(tok_path(corpus, f"bpe{v}"))
            print(f"trained {corpus} bpe{v} in {times[(corpus, f'bpe{v}')]:.1f}s")
    return times


def get_tokenizer(corpus, name):
    if corpus == "python_large":  # data-scale ablation reuses the 1 MB python tokenizers
        corpus = "python"
    if name == "xbpe2048":
        return load_tokenizer(tok_path(OTHER[corpus], "bpe2048"))
    return load_tokenizer(tok_path(corpus, name))


def _timed(fn, reps=3):
    ts = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return out, min(ts)


def intrinsic_metrics(tok, corpus, encode=None, decode=None, vocab_size=None):
    encode = encode or tok.encode
    decode = decode or tok.decode
    vocab_size = vocab_size or tok.vocab_size
    heldout_docs = load_docs(corpus, "val") + load_docs(corpus, "test")
    heldout = "\n\n".join(heldout_docs)
    train = load_split(corpus, "train")
    hb = len(heldout.encode("utf-8"))

    def cold_encode():
        if hasattr(tok, "_cache"):
            tok._cache = {}
        return encode(heldout)

    ids, t_enc = _timed(cold_encode)
    text, t_dec = _timed(lambda: decode(ids))
    assert text == heldout, "lossless round-trip failed"
    train_ids = encode(train)
    doc_lens = [len(encode(d)) for d in heldout_docs]
    train_counts = Counter(train_ids)
    return {
        "vocab_size": vocab_size,
        "bytes_per_token_heldout": hb / len(ids),
        "bytes_per_token_train": len(train.encode("utf-8")) / len(train_ids),
        "tokens_heldout": len(ids),
        "tokens_per_doc_mean": statistics.mean(doc_lens),
        "tokens_per_doc_median": statistics.median(doc_lens),
        "vocab_used_heldout_frac": len(set(ids)) / vocab_size,
        "vocab_rare_train_frac": sum(1 for i in range(vocab_size) if train_counts.get(i, 0) < 10) / vocab_size,
        "encode_MBps": hb / t_enc / 1e6,
        "decode_MBps": hb / t_dec / 1e6,
    }


def cache_tokens(corpus, name, tok):
    d = os.path.join(TOKENS_DIR, corpus, name)
    os.makedirs(d, exist_ok=True)
    for split in ("train", "val", "test"):
        ids = tok.encode(load_split(corpus, split))
        assert max(ids) < 65536
        np.save(os.path.join(d, f"{split}.npy"), np.array(ids, dtype=np.uint16))
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"vocab_size": tok.vocab_size,
                   "first_token_bytes": {s: len(tok.token_bytes(int(np.load(os.path.join(d, f"{s}.npy"))[0])))
                                         for s in ("train", "val", "test")}}, f)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    train_times = train_tokenizers()
    rows = []
    for corpus in CORPORA:
        for name in tokenizer_names():
            tok = get_tokenizer(corpus, name)
            m = intrinsic_metrics(tok, corpus)
            m.update(corpus=corpus, tokenizer=name,
                     train_time_s=train_times.get((corpus, name)) if name != "xbpe2048" else None)
            rows.append(m)
            cache_tokens(corpus, name, tok)
            print(corpus, name, {k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()})
        try:
            import tiktoken
            for enc_name in ("gpt2", "cl100k_base"):
                enc = tiktoken.get_encoding(enc_name)
                m = intrinsic_metrics(enc, corpus, encode=enc.encode_ordinary, decode=enc.decode,
                                      vocab_size=enc.n_vocab)
                m.update(corpus=corpus, tokenizer=f"tiktoken-{enc_name}", train_time_s=None)
                rows.append(m)
                print(corpus, enc_name, round(m["bytes_per_token_heldout"], 3))
        except Exception as e:  # tiktoken refs are optional (need a network download once)
            print("skipping tiktoken reference:", e)
    with open(os.path.join(RESULTS_DIR, "tokenizer_metrics.json"), "w") as f:
        json.dump(rows, f, indent=1)


def cache_python_large():
    for name in tokenizer_names()[:-1]:
        cache_tokens("python_large", name, get_tokenizer("python_large", name))


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["python_large"]:
        cache_python_large()
    else:
        main()
