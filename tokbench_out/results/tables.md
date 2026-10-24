### Intrinsic metrics: English (Tiny Shakespeare, 1 MB)

| tokenizer | vocab | bytes/token (held-out) | bytes/token (train) | median tokens/doc | vocab used on held-out | vocab rare in train (<10) | train time (s) | encode MB/s | decode MB/s |
|---|---|---|---|---|---|---|---|---|---|
| char | 65 | 1.00 | 1.00 | 59 | 94% | 3% | 0.01 | 41.6 | 27.9 |
| BPE-512 | 512 | 1.99 | 2.04 | 32 | 61% | 38% | 0.37 | 3.9 | 35.9 |
| BPE-1K | 1024 | 2.44 | 2.62 | 26 | 73% | 22% | 0.40 | 3.8 | 44.3 |
| BPE-2K | 2048 | 2.81 | 3.16 | 23 | 73% | 18% | 0.48 | 3.5 | 44.5 |
| BPE-4K | 4096 | 3.24 | 3.61 | 20 | 63% | 21% | 0.49 | 3.4 | 56.9 |
| BPE-8K | 8192 | 3.59 | 3.95 | 17 | 43% | 67% | 0.56 | 3.3 | 61.8 |
| BPE-2K (other domain) | 2048 | 2.12 | 2.15 | 30 | 42% | 58% | - | 3.7 | 36.8 |
| tiktoken-gpt2 | 50257 | 3.09 | 3.32 | 20 | 8% | 95% | - | 10.5 | 112.3 |
| tiktoken-cl100k_base | 100277 | 3.54 | 3.71 | 18 | 4% | 97% | - | 11.5 | 130.8 |

### Intrinsic metrics: Python source (CPython stdlib, 1 MB)

| tokenizer | vocab | bytes/token (held-out) | bytes/token (train) | median tokens/doc | vocab used on held-out | vocab rare in train (<10) | train time (s) | encode MB/s | decode MB/s |
|---|---|---|---|---|---|---|---|---|---|
| char | 96 | 1.00 | 1.00 | 2377 | 98% | 0% | 0.01 | 51.6 | 28.2 |
| BPE-512 | 512 | 1.98 | 2.11 | 1007 | 67% | 31% | 0.35 | 4.3 | 34.2 |
| BPE-1K | 1024 | 2.47 | 2.69 | 795 | 80% | 17% | 0.44 | 3.9 | 43.3 |
| BPE-2K | 2048 | 2.89 | 3.23 | 665 | 74% | 11% | 0.48 | 3.5 | 49.8 |
| BPE-4K | 4096 | 3.25 | 3.76 | 596 | 55% | 16% | 0.55 | 3.1 | 48.6 |
| BPE-8K | 8192 | 3.60 | 4.22 | 546 | 35% | 61% | 0.58 | 3.4 | 58.3 |
| BPE-2K (other domain) | 2048 | 1.61 | 1.61 | 1597 | 45% | 53% | - | 4.0 | 29.4 |
| tiktoken-gpt2 | 50257 | 2.30 | 2.25 | 1179 | 6% | 95% | - | 9.4 | 88.5 |
| tiktoken-cl100k_base | 100277 | 4.18 | 4.28 | 511 | 3% | 97% | - | 8.7 | 153.0 |

### LM results: English (Tiny Shakespeare, 1 MB)

| tokenizer | params (M) | steps | epochs in budget | context (bytes) | best test bpb | final test bpb | epochs at best | final val-train gap (bpb) | train tok/s | train MB/s | peak mem (GB) | seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| char | 10.79 | 4999 | 82 | 256 | 2.267 ± 0.008 | 2.411 | 48.9 | 0.88 | 887K | 0.89 | 1.52 | 3 |
| BPE-512 | 11.13 | 4924 | 164 | 523 | 2.286 ± 0.012 | 3.380 | 35.6 | 2.82 | 841K | 1.72 | 1.54 | 3 |
| BPE-1K | 11.53 | 4840 | 207 | 671 | 2.307 ± 0.017 | 3.943 | 37.9 | 3.42 | 849K | 2.22 | 1.65 | 3 |
| BPE-2K | 12.31 | 4682 | 241 | 809 | 2.361 ± 0.024 | 4.410 | 38.2 | 3.81 | 767K | 2.42 | 1.90 | 3 |
| BPE-4K | 13.89 | 4393 | 259 | 925 | 2.416 ± 0.006 | 4.653 | 32.4 | 3.97 | 726K | 2.62 | 2.39 | 3 |
| BPE-8K | 17.04 | 3912 | 252 | 1012 | 2.472 ± 0.007 | 4.755 | 25.2 | 4.07 | 584K | 2.31 | 3.39 | 3 |
| BPE-2K (other domain) | 12.31 | 4682 | 165 | 551 | 2.313 ± 0.005 | 3.597 | 34.3 | 3.07 | 796K | 1.71 | 1.90 | 3 |

### LM results: Python source (CPython stdlib, 1 MB)

| tokenizer | params (M) | steps | epochs in budget | context (bytes) | best test bpb | final test bpb | epochs at best | final val-train gap (bpb) | train tok/s | train MB/s | peak mem (GB) | seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| char | 10.81 | 4994 | 81 | 256 | 1.908 ± 0.031 | 2.224 | 31.8 | 1.52 | 837K | 0.84 | 1.53 | 3 |
| BPE-512 | 11.13 | 4924 | 169 | 540 | 2.205 ± 0.010 | 3.223 | 33.7 | 2.86 | 861K | 1.82 | 1.54 | 3 |
| BPE-1K | 11.53 | 4840 | 211 | 688 | 2.229 ± 0.011 | 3.556 | 29.9 | 3.26 | 802K | 2.16 | 1.64 | 3 |
| BPE-2K | 12.31 | 4682 | 245 | 826 | 2.229 ± 0.007 | 3.678 | 26.6 | 3.41 | 801K | 2.59 | 1.90 | 3 |
| BPE-4K | 13.89 | 4393 | 268 | 962 | 2.232 ± 0.014 | 3.764 | 26.8 | 3.49 | 690K | 2.59 | 2.39 | 3 |
| BPE-8K | 17.04 | 3912 | 268 | 1080 | 2.319 ± 0.020 | 3.889 | 26.8 | 3.66 | 607K | 2.56 | 3.38 | 3 |
| BPE-2K (other domain) | 12.31 | 4682 | 122 | 411 | 2.196 ± 0.009 | 3.017 | 33.6 | 2.69 | 761K | 1.22 | 1.90 | 3 |

### LM results: Python source, 6.7 MB train (same val/test)

| tokenizer | params (M) | steps | epochs in budget | context (bytes) | best test bpb | final test bpb | epochs at best | final val-train gap (bpb) | train tok/s | train MB/s | peak mem (GB) | seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| char | 10.81 | 4994 | 12 | 256 | 1.287 ± 0.010 | 1.287 | 12.1 | 0.25 | 880K | 0.88 | 1.62 | 3 |
| BPE-512 | 11.13 | 4924 | 25 | 536 | 1.236 ± 0.005 | 1.234 | 23.9 | 0.40 | 827K | 1.73 | 1.58 | 3 |
| BPE-1K | 11.53 | 4840 | 31 | 670 | 1.311 ± 0.015 | 1.308 | 28.9 | 0.57 | 846K | 2.21 | 1.69 | 3 |
| BPE-2K | 12.31 | 4682 | 35 | 785 | 1.403 ± 0.014 | 1.394 | 25.5 | 0.80 | 762K | 2.34 | 1.93 | 3 |
| BPE-4K | 13.89 | 4393 | 37 | 891 | 1.466 ± 0.013 | 1.495 | 22.2 | 0.98 | 718K | 2.50 | 2.42 | 3 |
| BPE-8K | 17.04 | 3912 | 36 | 978 | 1.515 ± 0.012 | 1.589 | 15.7 | 1.18 | 584K | 2.23 | 3.41 | 3 |

### Compression vs model quality

- English (Tiny Shakespeare, 1 MB): Spearman(bytes/token, best test bpb) = +1.00 (negative = better compression -> better model)
- Python source (CPython stdlib, 1 MB): Spearman(bytes/token, best test bpb) = +0.94 (negative = better compression -> better model)
- Python source, 6.7 MB train (same val/test): Spearman(bytes/token, best test bpb) = +0.94 (negative = better compression -> better model)

### Best tokenizer at each compute budget (val bpb with early stopping, mean over seeds)

| corpus | budget (PFLOPs) | best tokenizer | val bpb | runner-up |
|---|---|---|---|---|
| English (Tiny Shakespeare, 1 MB) | 0.25 | BPE-8K | 2.355 | BPE-4K (2.512) |
| English (Tiny Shakespeare, 1 MB) | 0.5 | BPE-8K | 2.099 | BPE-4K (2.148) |
| English (Tiny Shakespeare, 1 MB) | 1.0 | BPE-1K | 2.027 | BPE-512 (2.028) |
| English (Tiny Shakespeare, 1 MB) | 2.0 | BPE-512 | 2.006 | BPE-1K (2.026) |
| English (Tiny Shakespeare, 1 MB) | 5.5 | char | 1.985 | BPE-512 (2.006) |
| Python source (CPython stdlib, 1 MB) | 0.25 | BPE-8K | 2.328 | BPE-4K (2.362) |
| Python source (CPython stdlib, 1 MB) | 0.5 | BPE-2K | 2.065 | BPE-4K (2.076) |
| Python source (CPython stdlib, 1 MB) | 1.0 | BPE-512 | 1.971 | BPE-2K (2.021) |
| Python source (CPython stdlib, 1 MB) | 2.0 | char | 1.685 | BPE-512 (1.949) |
| Python source (CPython stdlib, 1 MB) | 5.5 | char | 1.665 | BPE-512 (1.949) |
| Python source, 6.7 MB train (same val/test) | 0.25 | BPE-8K | 2.265 | BPE-4K (2.329) |
| Python source, 6.7 MB train (same val/test) | 0.5 | BPE-4K | 1.915 | BPE-8K (1.927) |
| Python source, 6.7 MB train (same val/test) | 1.0 | BPE-512 | 1.596 | BPE-4K (1.622) |
| Python source, 6.7 MB train (same val/test) | 2.0 | BPE-512 | 1.304 | BPE-1K (1.387) |
| Python source, 6.7 MB train (same val/test) | 5.5 | BPE-512 | 1.155 | char (1.215) |

### BPE trainer wall-clock (s)

| corpus | vocab | fast (ours) | naive (dedup chunks) | minbpe (course) | speedup vs minbpe |
|---|---|---|---|---|---|
| shakespeare | 512 | 0.38 | 5.9 | 65.6 | 174x |
| shakespeare | 1024 | 0.44 | 16.4 | 175.1 | 394x |
| shakespeare | 2048 | 0.50 | nan | nan | - |
| shakespeare | 4096 | 0.54 | nan | nan | - |
| shakespeare | 8192 | 0.59 | nan | nan | - |
