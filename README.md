# Tokenizer ablation suite for a from-scratch GPT

Does the tokenizer that compresses text best also train the best language model under a fixed
compute budget? This project turns the GPT (lecture 7) and BPE tokenizer (lecture 8) from
Andrej Karpathy's [*Neural Networks: Zero to Hero*](https://github.com/karpathy/nn-zero-to-hero)
into a controlled experiment to answer that.

**Setup:** a 6-layer GPT (10.8–17M params, 256-token context) trained on **Tiny Shakespeare** and
**Python standard-library source code**. It compares character-level tokenization against byte-level
BPE at 512 / 1K / 2K / 4K / 8K vocab, over 60 runs (3 seeds) at a fixed 5.5 PFLOP budget, measuring
held-out **bits per byte**.

![best test bpb by tokenizer](tokbench_out/results/figures/bpb_by_tokenizer.png)

## Key results

- **Better compression gave a worse model on 1 MB corpora.** Test bpb rose with every step up in
  vocab size: 2.27 (char) vs 2.47 (BPE-8K) on English. BPE models re-read the small corpus
  164–268 times under the same budget and overfit.
- **More data moves the optimum.** With 6.7× more Python code, BPE-512 beats char (1.24 vs 1.29
  bpb), while BPE-8K is still last.
- **The winner depends on the budget.** BPE-8K is best at small budgets on every corpus; finer
  tokenizers take over as compute grows.
- **Fast BPE training:** chunk de-duplication, incremental pair counts and a lazy heap make it 390×
  faster than the course implementation, with merge-for-merge identical output (unit-tested).

Full method, tables, figures and limitations are in **[REPORT.md](REPORT.md)**.

## Layout

```
tokbench/
  tokenization.py       char tokenizer + byte-level regex BPE (fast trainer, reference trainer)
  data.py               corpus preparation and 90/5/5 document-level splits
  tok_suite.py          trains tokenizers, intrinsic metrics, caches token ids
  train_lm.py           course GPT, FLOPs-budgeted training, sliding-window bits-per-byte eval
  run_all.py            runs the corpus x tokenizer x seed sweep across GPUs
  analyze.py            tables (tables.md) and figures
  bench_bpe_trainer.py  BPE trainer wall-clock comparison
  tests/                trainer equivalence, round-trip and save/load tests
tokbench_out/
  results/              tables.md, figures/, summary JSON
  runs/                 one JSON per training run (config, learning curve, metrics, sample)
REPORT.md               full write-up
```

## Reproduce

Requirements: Python 3.12, a CUDA GPU, and `pip install torch numpy regex matplotlib pytest tiktoken`.
`tiktoken` is optional; it's only used for the GPT-2/cl100k compression reference rows.

```bash
# Tiny Shakespeare at the path data.py expects
mkdir -p repo/Karpathy-Neural-Networks-Zero-to-Hero-main/data
wget -O repo/Karpathy-Neural-Networks-Zero-to-Hero-main/data/input.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

python -m pytest tokbench/tests                   # sanity checks (~2 s)
python -m tokbench.data                           # corpora + splits
python -m tokbench.tok_suite                      # tokenizers, intrinsic metrics, token ids
python -m tokbench.tok_suite python_large         # token ids for the data-scale ablation
python -m tokbench.run_all --gpus 0,1             # 42 runs: English + Python, 3 seeds
python -m tokbench.run_all --corpora python_large # 18 runs: data-scale ablation
python -m tokbench.bench_bpe_trainer shakespeare  # trainer speed comparison
python -m tokbench.analyze                        # tables + figures
```

Each run takes about 2 minutes on an RTX 6000 Ada; the full sweep is about 2 GPU-hours. The Python
corpus comes from the local interpreter's standard library. The results here used CPython 3.12.7,
and other versions give slightly different files.

## Credits

Built on Andrej Karpathy's *Neural Networks: Zero to Hero* lectures (GPT and minBPE) and the
[chizkidd/Karpathy-Neural-Networks-Zero-to-Hero](https://github.com/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero)
notebooks (MIT). Tiny Shakespeare is from [karpathy/char-rnn](https://github.com/karpathy/char-rnn).
