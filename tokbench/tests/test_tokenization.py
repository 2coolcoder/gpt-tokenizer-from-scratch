"""Correctness checks: fast BPE trainer == naive reference trainer; lossless round-trips."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tokbench.tokenization import BPETokenizer, CharTokenizer, load_tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHAKESPEARE = os.path.join(ROOT, "repo/Karpathy-Neural-Networks-Zero-to-Hero-main/data/input.txt")


def sample_text(n=60_000):
    with open(SHAKESPEARE) as f:
        text = f.read()[:n]
    # add some code-like and non-ASCII text so multi-byte paths get exercised
    return text + "\ndef f(x):\n    return x ** 2  # héllo wörld ✓ 日本語\n" * 20


def test_fast_trainer_matches_reference():
    text = sample_text()
    fast = BPETokenizer().train(text, 256 + 300)
    ref = BPETokenizer().train_reference(text, 256 + 300)
    assert list(fast.merges.items()) == list(ref.merges.items())


def test_roundtrip_and_save_load():
    text = sample_text()
    for tok in [BPETokenizer().train(text, 1024), CharTokenizer().train(text)]:
        ids = tok.encode(text)
        assert tok.decode(ids) == text
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        tok.save(path)
        tok2 = load_tokenizer(path)
        os.remove(path)
        assert tok2.encode(text) == ids


def test_encode_cache_consistent():
    text = sample_text()
    tok = BPETokenizer().train(text, 800)
    assert tok.encode(text, use_cache=False) == tok.encode(text)


def test_merges_compress_monotonically():
    text = sample_text()
    lens = [len(BPETokenizer().train(text, v).encode(text)) for v in (256, 400, 800, 1600)]
    assert lens[0] == len(text.encode("utf-8"))
    assert all(a > b for a, b in zip(lens, lens[1:]))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
