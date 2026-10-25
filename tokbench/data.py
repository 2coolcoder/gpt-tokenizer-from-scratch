"""
Corpus preparation. Both corpora are size-matched (~1.1 MB) and split 90/5/5 into
train/val/test on document boundaries, so no document straddles two splits.

- shakespeare: Tiny Shakespeare (the course's English corpus). A document is a
  blank-line-separated block (one speech).
- python: source files from the CPython standard library of the local interpreter
  (ASCII-only files, no tests / site-packages / generated data tables, <= 40 KB each),
  shuffled with a fixed seed. A document is one file.

Both corpora are pure ASCII, so characters == bytes and bits-per-byte (bpb) is
bits-per-character for the char-level model.
"""

import glob
import json
import os
import random
import sys
import sysconfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHAKESPEARE = os.path.join(ROOT, "data", "input.txt")
DATA_DIR = os.path.join(ROOT, "tokbench_out/data")
SPLITS = ("train", "val", "test")
FRACS = (0.90, 0.05, 0.05)
SEP = "\n\n"


def shakespeare_docs():
    with open(SHAKESPEARE) as f:
        text = f.read()
    return [d for d in text.split("\n\n") if d.strip()]


def python_docs(target_bytes, seed=0, max_file_bytes=40_000):
    stdlib = sysconfig.get_paths()["stdlib"]
    files = sorted(glob.glob(os.path.join(stdlib, "**", "*.py"), recursive=True))
    skip = ("site-packages", "/test", "/tests", "idlelib", "pydoc_data", "/encodings/", "unicodedata")
    docs = []
    for path in files:
        if any(s in path for s in skip):
            continue
        raw = open(path, "rb").read()
        if not raw.strip() or len(raw) > max_file_bytes:
            continue
        try:
            text = raw.decode("ascii")
        except UnicodeDecodeError:
            continue
        docs.append((os.path.relpath(path, stdlib), text.replace("\r\n", "\n")))
    random.Random(seed).shuffle(docs)
    out, total = [], 0
    for name, text in docs:
        if total >= target_bytes:
            break
        out.append((name, text))
        total += len(text) + len(SEP)
    return out, stdlib


def split_docs(docs):
    """Contiguous 90/5/5 split by bytes, cut on document boundaries."""
    total = sum(len(d) + len(SEP) for d in docs)
    bounds = [FRACS[0] * total, (FRACS[0] + FRACS[1]) * total]
    splits = {s: [] for s in SPLITS}
    acc = 0
    for d in docs:
        s = "train" if acc < bounds[0] else ("val" if acc < bounds[1] else "test")
        splits[s].append(d)
        acc += len(d) + len(SEP)
    return splits


def prepare(out_dir=DATA_DIR):
    os.makedirs(out_dir, exist_ok=True)
    sh_docs = shakespeare_docs()
    target = len(SEP.join(sh_docs))
    py, stdlib = python_docs(target)
    corpora = {
        "shakespeare": (sh_docs, {"source": SHAKESPEARE, "doc_unit": "speech (blank-line block)"}),
        "python": ([t for _, t in py], {"source": stdlib, "python_version": sys.version.split()[0],
                                        "doc_unit": "source file", "files": [n for n, _ in py]}),
    }
    meta = {}
    for name, (docs, info) in corpora.items():
        d = os.path.join(out_dir, name)
        os.makedirs(d, exist_ok=True)
        splits = split_docs(docs)
        info = dict(info)
        for s, sdocs in splits.items():
            text = SEP.join(sdocs)
            assert text.isascii()
            with open(os.path.join(d, f"{s}.txt"), "w") as f:
                f.write(text)
            with open(os.path.join(d, f"{s}_docs.json"), "w") as f:
                json.dump(sdocs, f)
            info[f"{s}_bytes"] = len(text.encode("utf-8"))
            info[f"{s}_docs"] = len(sdocs)
        meta[name] = info
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(info, f, indent=1)
    return meta


def load_split(corpus, split, data_dir=DATA_DIR):
    with open(os.path.join(data_dir, corpus, f"{split}.txt")) as f:
        return f.read()


def load_docs(corpus, split, data_dir=DATA_DIR):
    with open(os.path.join(data_dir, corpus, f"{split}_docs.json")) as f:
        return json.load(f)


def prepare_python_large(out_dir=DATA_DIR, max_file_bytes=100_000):
    """Data-scale ablation: ~6.6x more Python training code, identical val/test files.

    Train = every eligible stdlib file (<= 100 KB) that is not in the `python` val/test splits.
    The `python` tokenizers are reused, so only the amount of LM training data changes.
    """
    with open(os.path.join(out_dir, "python", "meta.json")) as f:
        small = json.load(f)
    heldout_files = set(small["files"][small["train_docs"]:])
    docs, stdlib = python_docs(float("inf"), max_file_bytes=max_file_bytes)
    train = [t for n, t in docs if n not in heldout_files]
    d = os.path.join(out_dir, "python_large")
    os.makedirs(d, exist_ok=True)
    text = SEP.join(train)
    with open(os.path.join(d, "train.txt"), "w") as f:
        f.write(text)
    with open(os.path.join(d, "train_docs.json"), "w") as f:
        json.dump(train, f)
    for s in ("val", "test"):
        for ext in (".txt", "_docs.json"):
            with open(os.path.join(out_dir, "python", s + ext)) as src, \
                    open(os.path.join(d, s + ext), "w") as dst:
                dst.write(src.read())
    info = {"source": stdlib, "train_bytes": len(text), "train_docs": len(train),
            "val_test": "identical to python", "tokenizers": "python's (trained on the 1 MB split)"}
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(info, f, indent=1)
    return info


if __name__ == "__main__":
    for name, info in prepare().items():
        print(name, {k: v for k, v in info.items() if k != "files"})
    print("python_large", prepare_python_large())
