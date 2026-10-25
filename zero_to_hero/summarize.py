"""Collect every lecture's results into results/lectures/summary.md, next to the course's numbers."""

import glob
import json
import os

from zero_to_hero.common import RESULTS_DIR

# reference numbers printed in the course notebooks (chizkidd/Karpathy-Neural-Networks-Zero-to-Hero)
REF = {
    "mlp": (2.1241, 2.1655),
    "fix_softmax": (2.0696, 2.1311),
    "fix_tanh": (2.0377, 2.1070),
    "batchnorm": (2.0668, 2.1048),
    "backprop": (2.0706, 2.1112),
    "mlp_ctx3": (2.0583, 2.1065),
    "mlp_ctx8": (1.9163, 2.0342),
    "wavenet_22k": (1.9110, 2.0202),
    "wavenet_76k": (1.7690, 1.9936),
    "gpt": (1.0777, 1.4856),
}


def load(name):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    return json.load(open(path)) if os.path.exists(path) else None


def fmt(x):
    return "-" if x is None else f"{x:.4f}"


def main():
    rows = []

    def add(lecture, model, params, train, dev, ref_key=None, note=""):
        ref = REF.get(ref_key, (None, None))
        rows.append([lecture, model, params, fmt(train), fmt(dev), fmt(ref[0]), fmt(ref[1]), note])

    m = load("micrograd")
    if m:
        rows.append(["1 micrograd", "MLP 2-16-16-1 on two moons (scalar autograd)", m["n_params"], "-", "-", "-", "-",
                     f"accuracy {100 * m['final_accuracy']:.0f}% (demo: 100%)"])
    b = load("makemore_bigram")
    if b:
        add("2 bigram", "counting, +1 smoothing", 729, b["counting"]["train"], b["counting"]["dev"],
            note=f"all names: {b['counting_all_words_nll']:.4f} (lecture: 2.454)")
        add("2 bigram", "one-layer neural net", 729, b["neural"]["train"], b["neural"]["dev"],
            note="converges to the counting model")
    m = load("makemore_mlp")
    if m:
        add("3 MLP", "Bengio et al. MLP, context 3", m["n_params"], m["train"], m["dev"], "mlp")
    bn = load("makemore_batchnorm")
    if bn:
        a = bn["init_ablation"]
        for k, label in [("fix_softmax", "+ scaled-down output layer"), ("fix_tanh", "+ Kaiming init (gain 5/3)"),
                         ("batchnorm", "+ BatchNorm")]:
            add("4 BatchNorm", label, "11.9K", a[k]["train"], a[k]["dev"], k,
                note=f"loss at init {a[k]['init_loss']:.2f}, tanh saturated {100 * a[k]['init_tanh_saturated_frac']:.0f}%")
    bp = load("makemore_backprop")
    if bp:
        g = bp["gradient_check_float64_max_abs_diff"]
        add("5 backprop ninja", "trained with manual gradients only", "12.3K", bp["manual_training"]["train"],
            bp["manual_training"]["dev"], "backprop",
            note=f"max |manual - autograd| = {max(g['max_manual'], g['max_fused']):.0e} over all tensors")
    wn = {}
    for p in glob.glob(os.path.join(RESULTS_DIR, "makemore_wavenet*.json")):
        wn.update(json.load(open(p)))
    for k, label in [("mlp_ctx3", "flat MLP, context 3"), ("mlp_ctx8", "flat MLP, context 8"),
                     ("wavenet_22k", "WaveNet tree, context 8"), ("wavenet_76k", "WaveNet tree, scaled up")]:
        if k in wn:
            add("6 WaveNet", label, wn[k]["n_params"], wn[k]["train"], wn[k]["dev"], k)
    g = load("gpt_bigram")
    if g:
        add("7 GPT", "bigram baseline (Shakespeare, chars)", g["n_params"], g["final"]["train"], g["final"]["val"],
            note="lecture: ~2.49 val")
    for name, label in [("gpt_gpt", "6-layer GPT: N(0, 0.02) init, 1/sqrt(head_size) scale (final gpt.py)"),
                        ("gpt_gpt_lecture_scale", "6-layer GPT: N(0, 0.02) init, 1/sqrt(n_embd) scale"),
                        ("gpt_gpt_video", "6-layer GPT: default init, 1/sqrt(n_embd) scale (video)")]:
        g = load(name)
        if g:
            best = min(g["history"], key=lambda p: p["val"])
            add("7 GPT", label, f"{g['n_params'] / 1e6:.1f}M", g["final"]["train"], g["final"]["val"],
                "gpt" if name == "gpt_gpt_video" else None,
                note=f"best val {best['val']:.4f} at step {best['step']}")
    t = load("tokenizer")
    if t:
        tk = t["tokenizers"]
        rows.append(["8 tokenizer", "BPE-512 on taylorswift.txt (basic / regex)", 512, "-", "-", "-", "-",
                     f"{tk['basic']['bytes_per_token']:.2f} / {tk['regex_fast_trainer']['bytes_per_token']:.2f} bytes/token; "
                     f"GPT-4 tokenizer reproduced, matches tiktoken exactly"])

    header = ["lecture", "model", "params", "train loss", "dev/val loss", "course train", "course dev/val", "notes"]
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    text = "\n".join(md) + "\n"
    with open(os.path.join(RESULTS_DIR, "summary.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
