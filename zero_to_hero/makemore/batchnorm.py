"""
Lecture 4: activations, gradients and Batch Normalization.

Part A - initialisation ablation on the lecture-3 MLP (block 3, emb 10, hidden 200), one fix
at a time:
  naive        every parameter ~ N(0, 1): huge logits, loss at init ~27 instead of -ln(1/27)=3.30
  fix_softmax  scale the output layer down so the initial distribution is ~uniform
  fix_tanh     Kaiming init (gain 5/3) for the hidden layer so tanh does not saturate
  batchnorm    + a BatchNorm layer before the tanh (hidden bias dropped, running stats for eval)

Part B - diagnostics on a deep (5 hidden layers) tanh MLP built from the hand-written layers:
forward activation, backward gradient and update:data ratio statistics per layer, comparing
too-small / correct (5/3) / too-large init gains and BatchNorm.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from zero_to_hero.common import RESULTS_DIR, save_results  # noqa: E402
from zero_to_hero.makemore.data import VOCAB_SIZE, sample_names, splits  # noqa: E402
from zero_to_hero.makemore.layers import BatchNorm1d, Linear, Sequential, Tanh, Embedding, Flatten  # noqa: E402

# ---------------------------------------------------------------- Part A


class MLP2:
    def __init__(self, variant, block_size=3, n_embd=10, n_hidden=200, seed=2147483647):
        g = torch.Generator().manual_seed(seed)
        fan_in = n_embd * block_size
        self.variant = variant
        self.C = torch.randn((VOCAB_SIZE, n_embd), generator=g)
        w1_scale = (5 / 3) / fan_in ** 0.5 if variant in ("fix_tanh", "batchnorm") else 1.0
        self.W1 = torch.randn((fan_in, n_hidden), generator=g) * w1_scale
        self.b1 = torch.randn(n_hidden, generator=g) * (0.01 if variant != "naive" and variant != "fix_softmax" else 1.0)
        out_scale = 1.0 if variant == "naive" else 0.01
        self.W2 = torch.randn((n_hidden, VOCAB_SIZE), generator=g) * out_scale
        self.b2 = torch.randn(VOCAB_SIZE, generator=g) * (1.0 if variant == "naive" else 0.0)
        self.params = [self.C, self.W1, self.W2, self.b2]
        if variant != "batchnorm":
            self.params.append(self.b1)  # with BN the bias is redundant (subtracted by the mean)
        else:
            self.bngain, self.bnbias = torch.ones((1, n_hidden)), torch.zeros((1, n_hidden))
            self.bnmean_running, self.bnstd_running = torch.zeros((1, n_hidden)), torch.ones((1, n_hidden))
            self.params += [self.bngain, self.bnbias]
        for p in self.params:
            p.requires_grad = True

    def __call__(self, X, training=True):
        emb = self.C[X]
        hpreact = emb.view(emb.shape[0], -1) @ self.W1
        if self.variant == "batchnorm":
            if training:
                mean, std = hpreact.mean(0, keepdim=True), hpreact.std(0, keepdim=True)
                with torch.no_grad():
                    self.bnmean_running = 0.999 * self.bnmean_running + 0.001 * mean
                    self.bnstd_running = 0.999 * self.bnstd_running + 0.001 * std
            else:
                mean, std = self.bnmean_running, self.bnstd_running
            hpreact = self.bngain * (hpreact - mean) / std + self.bnbias
        else:
            hpreact = hpreact + self.b1
        self.h = torch.tanh(hpreact)
        return self.h @ self.W2 + self.b2


def train_mlp2(variant, Xtr, Ytr, Xdev, Ydev, steps=200_000, batch_size=32, seed=2147483647):
    model = MLP2(variant)
    g = torch.Generator().manual_seed(seed)
    init = {}
    for i in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        loss = F.cross_entropy(model(Xtr[ix]), Ytr[ix])
        if i == 0:
            init = {"init_loss": loss.item(),
                    "init_tanh_saturated_frac": (model.h.abs() > 0.99).float().mean().item()}
        for p in model.params:
            p.grad = None
        loss.backward()
        lr = 0.1 if i < 100_000 else 0.01
        for p in model.params:
            p.data += -lr * p.grad
    with torch.no_grad():
        res = dict(init, train=F.cross_entropy(model(Xtr, False), Ytr).item(),
                   dev=F.cross_entropy(model(Xdev, False), Ydev).item())
    print(f"{variant:12s} init loss {res['init_loss']:6.3f}  saturated {100 * res['init_tanh_saturated_frac']:4.1f}%"
          f"  train {res['train']:.4f}  dev {res['dev']:.4f}")
    if variant == "batchnorm":
        res["samples"] = sample_names(lambda c: F.softmax(model(c, training=False), 1), block_size=3)
    return res


# ---------------------------------------------------------------- Part B


def deep_mlp(gain, batchnorm, block_size=3, n_embd=10, n_hidden=100, n_layers=5, seed=2147483647):
    g = torch.Generator().manual_seed(seed)
    layers = [Embedding(VOCAB_SIZE, n_embd, generator=g), Flatten()]
    fan_in = n_embd * block_size
    for _ in range(n_layers):
        layers += [Linear(fan_in, n_hidden, bias=not batchnorm, generator=g)]
        layers += [BatchNorm1d(n_hidden)] if batchnorm else []
        layers += [Tanh()]
        fan_in = n_hidden
    layers += [Linear(n_hidden, VOCAB_SIZE, generator=g)] + ([BatchNorm1d(VOCAB_SIZE)] if batchnorm else [])
    with torch.no_grad():
        for layer in layers:
            if isinstance(layer, Linear):
                layer.weight *= gain
        if batchnorm:
            layers[-1].gamma *= 0.1  # make the last layer less confident
        else:
            layers[-1].weight *= 0.1
    model = Sequential(layers)
    for p in model.parameters():
        p.requires_grad = True
    return model


def diagnose(name, model, Xtr, Ytr, steps=1000, batch_size=32, lr=0.1, seed=2147483647):
    """Train a few steps and record the statistics the lecture inspects."""
    g = torch.Generator().manual_seed(seed)
    ud = []
    tanhs = [l for l in model.layers if isinstance(l, Tanh)]
    for i in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        loss = F.cross_entropy(model(Xtr[ix]), Ytr[ix])
        for layer in model.layers:
            if hasattr(layer, "out"):
                layer.out.retain_grad()
        for p in model.parameters():
            p.grad = None
        loss.backward()
        for p in model.parameters():
            p.data += -lr * p.grad
        with torch.no_grad():
            ud.append([((lr * p.grad).std() / p.data.std()).log10().item()
                       for p in model.parameters() if p.ndim == 2])
    stats = {"name": name, "final_minibatch_loss": loss.item(), "layers": []}
    for li, t in enumerate(tanhs):
        stats["layers"].append({
            "layer": li + 1, "act_mean": t.out.mean().item(), "act_std": t.out.std().item(),
            "saturated_frac": (t.out.abs() > 0.97).float().mean().item(),
            "grad_std": t.out.grad.std().item(),
            "act_hist": torch.histogram(t.out.detach().float(), bins=50, range=(-1, 1)).hist.tolist(),
            "grad_values": t.out.grad.detach().flatten().tolist()[:3200],
        })
    # weight matrices: [embedding, linear1..5, output]; keep the hidden linears for the ratio plot
    stats["update_ratio_log10"] = [row[1:-1] for row in ud]
    return stats


def plot_diagnostics(all_stats, path):
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # layer depth (ordinal)
    fig, axes = plt.subplots(len(all_stats), 3, figsize=(12, 2.6 * len(all_stats)))
    centers = torch.linspace(-1, 1, 51)[:-1] + 1 / 50
    for r, st in enumerate(all_stats):
        a_act, a_grad, a_ud = axes[r]
        for li, L in enumerate(st["layers"]):
            lab = f"layer {L['layer']}"
            a_act.plot(centers, L["act_hist"], color=ramp[li], lw=1.6, label=lab)
            gv = torch.tensor(L["grad_values"])
            h = torch.histogram(gv, bins=50)
            a_grad.plot((h.bin_edges[:-1] + h.bin_edges[1:]) / 2, h.hist, color=ramp[li], lw=1.6)
        ud = torch.tensor(st["update_ratio_log10"])
        for li in range(ud.shape[1]):
            a_ud.plot(ud[:, li], color=ramp[li], lw=1.2)
        a_ud.axhline(-3, color="#52514e", ls="--", lw=1)
        sat = sum(L["saturated_frac"] for L in st["layers"]) / len(st["layers"])
        a_act.set_title(f"{st['name']}: tanh outputs (saturated {100 * sat:.0f}%)", loc="left", fontsize=9)
        a_grad.set_title("gradients w.r.t. tanh outputs", loc="left", fontsize=9)
        a_ud.set_title("log10 update:data ratio (target ~ -3)", loc="left", fontsize=9)
        for ax in (a_act, a_grad, a_ud):
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(labelsize=7)
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    (Xtr, Ytr), (Xdev, Ydev), _ = splits(block_size=3)
    part_a = {v: train_mlp2(v, Xtr, Ytr, Xdev, Ydev) for v in ("naive", "fix_softmax", "fix_tanh", "batchnorm")}

    configs = [("gain 0.5, no BN", 0.5, False), ("gain 5/3, no BN", 5 / 3, False),
               ("gain 3, no BN", 3.0, False), ("gain 5/3 + BatchNorm", 5 / 3, True)]
    part_b = [diagnose(n, deep_mlp(gn, bn), Xtr, Ytr) for n, gn, bn in configs]
    import os
    fig_path = os.path.join(RESULTS_DIR, "batchnorm_diagnostics.png")
    plot_diagnostics(part_b, fig_path)
    for st in part_b:
        for L in st["layers"]:
            del L["grad_values"]
        print(st["name"], [f"L{L['layer']}: std {L['act_std']:.2f} sat {100 * L['saturated_frac']:.0f}%"
                           for L in st["layers"]])
    return save_results("makemore_batchnorm", {"init_ablation": part_a, "deep_diagnostics": part_b,
                                               "figure": fig_path})


if __name__ == "__main__":
    main()
