import json

from zero_to_hero.tokenizer.base import Tokenizer, get_stats, merge
from zero_to_hero.tokenizer.basic import BasicTokenizer
from zero_to_hero.tokenizer.char import CharTokenizer
from zero_to_hero.tokenizer.regex import GPT2_SPLIT_PATTERN, GPT4_SPLIT_PATTERN, RegexTokenizer


def load_tokenizer(path):
    with open(path) as f:
        d = json.load(f)
    if d["kind"] == "char":
        return CharTokenizer(d["chars"])
    merges = {(p0, p1): idx for p0, p1, idx in d["merges"]}
    if d["kind"] == "basic":
        tok = BasicTokenizer()
    else:
        tok = RegexTokenizer(d["pattern"])
    tok.set_merges(merges)
    if d.get("special_tokens"):
        tok.register_special_tokens(d["special_tokens"])
    return tok


__all__ = ["Tokenizer", "BasicTokenizer", "RegexTokenizer", "CharTokenizer", "load_tokenizer",
           "get_stats", "merge", "GPT2_SPLIT_PATTERN", "GPT4_SPLIT_PATTERN"]
