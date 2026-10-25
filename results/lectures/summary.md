| lecture | model | params | train loss | dev/val loss | course train | course dev/val | notes |
|---|---|---|---|---|---|---|---|
| 1 micrograd | MLP 2-16-16-1 on two moons (scalar autograd) | 337 | - | - | - | - | accuracy 100% (demo: 100%) |
| 2 bigram | counting, +1 smoothing | 729 | 2.4548 | 2.4533 | - | - | all names: 2.4546 (lecture: 2.454) |
| 2 bigram | one-layer neural net | 729 | 2.4628 | 2.4610 | - | - | converges to the counting model |
| 3 MLP | Bengio et al. MLP, context 3 | 11897 | 2.1241 | 2.1692 | 2.1241 | 2.1655 |  |
| 4 BatchNorm | + scaled-down output layer | 11.9K | 2.0689 | 2.1296 | 2.0696 | 2.1311 | loss at init 3.32, tanh saturated 64% |
| 4 BatchNorm | + Kaiming init (gain 5/3) | 11.9K | 2.0370 | 2.1070 | 2.0377 | 2.1070 | loss at init 3.31, tanh saturated 12% |
| 4 BatchNorm | + BatchNorm | 11.9K | 2.0679 | 2.1095 | 2.0668 | 2.1048 | loss at init 3.32, tanh saturated 0% |
| 5 backprop ninja | trained with manual gradients only | 12.3K | 2.0705 | 2.1094 | 2.0706 | 2.1112 | max |manual - autograd| = 4e-17 over all tensors |
| 6 WaveNet | flat MLP, context 3 | 12097 | 2.0583 | 2.1065 | 2.0583 | 2.1065 |  |
| 6 WaveNet | flat MLP, context 8 | 22097 | 1.9163 | 2.0342 | 1.9163 | 2.0342 |  |
| 6 WaveNet | WaveNet tree, context 8 | 22397 | 1.9110 | 2.0202 | 1.9110 | 2.0202 |  |
| 6 WaveNet | WaveNet tree, scaled up | 76579 | 1.7690 | 1.9937 | 1.7690 | 1.9936 |  |
| 7 GPT | bigram baseline (Shakespeare, chars) | 4225 | 2.4611 | 2.4903 | - | - | lecture: ~2.49 val |
| 7 GPT | 6-layer GPT: N(0, 0.02) init, 1/sqrt(head_size) scale (final gpt.py) | 10.8M | 0.8622 | 1.5549 | - | - | best val 1.4844 at step 3000 |
| 7 GPT | 6-layer GPT: N(0, 0.02) init, 1/sqrt(n_embd) scale | 10.8M | 0.8941 | 1.5450 | - | - | best val 1.4711 at step 3000 |
| 7 GPT | 6-layer GPT: default init, 1/sqrt(n_embd) scale (video) | 10.8M | 1.0748 | 1.4815 | 1.0777 | 1.4856 | best val 1.4732 at step 4500 |
| 8 tokenizer | BPE-512 on taylorswift.txt (basic / regex) | 512 | - | - | - | - | 2.36 / 2.13 bytes/token; GPT-4 tokenizer reproduced, matches tiktoken exactly |
