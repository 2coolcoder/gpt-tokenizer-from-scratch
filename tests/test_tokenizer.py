"""Tokenizer checks: trainer equivalence, lossless round-trips, special tokens, GPT-4 parity."""

import os
import tempfile

import pytest

from zero_to_hero.common import data_path
from zero_to_hero.tokenizer import BasicTokenizer, CharTokenizer, RegexTokenizer, load_tokenizer

SAMPLES = [
    "",
    "?",
    "hello world!!!? (안녕하세요!) lol123 😉",
    "def f(x):\n    return x ** 2  # héllo wörld ✓ 日本語\n",
    "   leading and trailing spaces   \n\n\tand tabs",
]


def sample_text(n=60_000):
    with open(data_path("input.txt")) as f:
        text = f.read()[:n]
    # add code-like and non-ASCII text so multi-byte paths get exercised
    return text + "\ndef f(x):\n    return x ** 2  # héllo wörld ✓ 日本語\n" * 20


def test_basic_tokenizer_wikipedia_example():
    """The Wikipedia BPE example: 'aaabdaaabac' with 3 merges -> 'XdXac'."""
    tok = BasicTokenizer().train("aaabdaaabac", 256 + 3)
    ids = tok.encode("aaabdaaabac")
    assert ids == [258, 100, 258, 97, 99]
    assert tok.decode(ids) == "aaabdaaabac"


def test_fast_trainer_matches_reference():
    text = sample_text()
    fast = RegexTokenizer().train(text, 256 + 300)
    ref = RegexTokenizer().train_reference(text, 256 + 300)
    assert list(fast.merges.items()) == list(ref.merges.items())


def test_minbpe_trainer_compresses_like_fast_trainer():
    # minbpe breaks count ties by first occurrence, so merges may differ in order; compression must not
    text = sample_text(20_000)
    a = len(RegexTokenizer().train(text, 400).encode(text))
    b = len(RegexTokenizer().train_minbpe(text, 400).encode(text))
    assert abs(a - b) / a < 0.01


@pytest.mark.parametrize("factory", [lambda t: BasicTokenizer().train(t, 512),
                                     lambda t: RegexTokenizer().train(t, 1024),
                                     lambda t: CharTokenizer().train(t)])
def test_roundtrip_and_save_load(factory):
    text = sample_text()
    tok = factory(text)
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    if not isinstance(tok, CharTokenizer):
        for s in SAMPLES:
            assert tok.decode(tok.encode(s)) == s
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    tok.save(path)
    tok2 = load_tokenizer(path)
    os.remove(path)
    assert tok2.encode(text) == ids


def test_encode_cache_consistent():
    text = sample_text()
    tok = RegexTokenizer().train(text, 800)
    assert tok.encode(text, use_cache=False) == tok.encode(text)


def test_merges_compress_monotonically():
    text = sample_text()
    lens = [len(RegexTokenizer().train(text, v).encode(text)) for v in (256, 400, 800, 1600)]
    assert lens[0] == len(text.encode("utf-8"))
    assert all(a > b for a, b in zip(lens, lens[1:]))


def test_special_tokens():
    tok = RegexTokenizer().train(sample_text(), 512)
    tok.register_special_tokens({"<|endoftext|>": 512})
    text = "first doc<|endoftext|>second doc"
    ids = tok.encode(text, allowed_special="all")
    assert ids.count(512) == 1 and tok.decode(ids) == text
    assert 512 not in tok.encode(text, allowed_special="none")
    with pytest.raises(AssertionError):
        tok.encode(text)  # default: refuse to silently encode special tokens


def test_gpt4_tokenizer_matches_tiktoken():
    tiktoken = pytest.importorskip("tiktoken")
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        from zero_to_hero.tokenizer.gpt4 import GPT4Tokenizer
        tok = GPT4Tokenizer()
    except Exception as e:  # needs a one-time download of the cl100k ranks
        pytest.skip(f"cl100k_base unavailable: {e}")
    texts = SAMPLES + [sample_text(3000), open(data_path("taylorswift.txt")).read()[:5000]]
    for s in texts:
        assert tok.encode(s) == enc.encode_ordinary(s)
        assert tok.decode(tok.encode(s)) == s
    special = "<|endoftext|>hello world<|fim_prefix|>"
    assert tok.encode(special, allowed_special="all") == enc.encode(special, allowed_special="all")
