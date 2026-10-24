"""Aggregate tokenizer metrics + LM runs into tables (REPORT tables as markdown) and figures."""

import glob
import json
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tokbench.tok_suite import RESULTS_DIR, tokenizer_names  # noqa: E402
from tokbench.train_lm import RUNS_DIR  # noqa: E402

FIG_DIR = os.path.join(RESULTS_DIR, "figures")
CORPUS_LABEL = {"shakespeare": "English (Tiny Shakespeare, 1 MB)",
                "python": "Python source (CPython stdlib, 1 MB)",
                "python_large": "Python source, 6.7 MB train (same val/test)"}
CORPUS_COLOR = {"shakespeare": "#2a78d6", "python": "#eb6834", "python_large": "#1baf7a"}
# ordinal one-hue ramp (blue 250 -> 700) for tokenizers ordered by granularity
TOK_ORDER = ["char", "bpe512", "bpe1024", "bpe2048", "bpe4096", "bpe8192"]
TOK_COLOR = dict(zip(TOK_ORDER, ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]))
TOK_LABEL = {"char": "char", "bpe512": "BPE-512", "bpe1024": "BPE-1K", "bpe2048": "BPE-2K",
             "bpe4096": "BPE-4K", "bpe8192": "BPE-8K", "xbpe2048": "BPE-2K (other domain)"}
# hand-placed label offsets (points) where neighbouring labels would collide
LABEL_OFFSET = {("python", "xbpe2048"): (-6, -14), ("python", "bpe512"): (6, -12),
                ("python", "bpe4096"): (6, -14), ("python", "bpe2048"): (4, 7), ("python", "bpe1024"): (-6, 7),
                ("shakespeare", "bpe512"): (-6, -12),
                ("shakespeare", "xbpe2048"): (-6, 6), ("shakespeare", "bpe1024"): (6, -12)}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load_runs():
    runs = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(RUNS_DIR, "*.json"))):
        r = json.load(open(p))
        runs[(r["corpus"], r["tokenizer"])].append(r)
    return runs


def ms(xs):
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def summarize(runs, tokm):
    rows = []
    for (corpus, tok), rs in runs.items():
        m = tokm.get((corpus if corpus != "python_large" else "python", tok), {})
        row = {"corpus": corpus, "tokenizer": tok, "seeds": len(rs), "vocab_size": rs[0]["vocab_size"],
               "bytes_per_token": m.get("bytes_per_token_heldout"),
               "tokens_per_doc_median": m.get("tokens_per_doc_median"),
               "n_params_M": rs[0]["n_params"] / 1e6, "steps": rs[0]["max_steps"],
               "epochs": rs[0]["epochs"], "context_bytes": rs[0]["context_bytes"]}
        for k in ("best_test_bpb", "final_test_bpb", "best_val_bpb", "train_tokens_per_s",
                  "train_bytes_per_s", "peak_mem_GB"):
            row[k], row[k + "_std"] = ms([r[k] for r in rs])
        row["best_epochs"] = statistics.mean(
            r["best_step"] * 64 * 256 / r["train_tokens"] for r in rs)
        # generalization gap at the end of the budget
        row["final_gap_bpb"] = statistics.mean(r["curve"][-1]["val_bpb"] - r["curve"][-1]["train_bpb"] for r in rs)
        rows.append(row)
    return rows


def envelope(curve):
    """Best val bpb reachable with early stopping at each budget point (running minimum)."""
    x = np.array([p["pflops"] for p in curve])
    y = np.minimum.accumulate(np.array([p["val_bpb"] for p in curve]))
    return x, y


def winners_by_budget(runs, corpus, budgets):
    out = {}
    for b in budgets:
        scores = {}
        for tok in TOK_ORDER:
            rs = runs.get((corpus, tok))
            if not rs:
                continue
            vals = []
            for r in rs:
                x, y = envelope(r["curve"])
                if x[0] > b * 1.0001:
                    vals = None
                    break
                vals.append(y[x <= b * 1.0001][-1])
            if vals:
                scores[tok] = statistics.mean(vals)
        if scores:
            best = min(scores, key=scores.get)
            out[b] = (best, scores[best], scores)
    return out


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def fig_bpb_vs_tokenizer(summary, corpora):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    style(ax)
    xs = np.arange(len(TOK_ORDER))
    for c in corpora:
        rows = {r["tokenizer"]: r for r in summary if r["corpus"] == c}
        toks = [t for t in TOK_ORDER if t in rows]
        x = [TOK_ORDER.index(t) for t in toks]
        y = [rows[t]["best_test_bpb"] for t in toks]
        e = [rows[t]["best_test_bpb_std"] for t in toks]
        ax.errorbar(x, y, yerr=e, color=CORPUS_COLOR[c], lw=2, marker="o", ms=7, capsize=3,
                    markeredgecolor="white", markeredgewidth=1.5, label=CORPUS_LABEL[c])
        ax.annotate({"shakespeare": "English, 1 MB", "python": "Python, 1 MB"}.get(c, "Python, 6.7 MB"),
                    (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK2)
    ax.set_xticks(xs, [TOK_LABEL[t] for t in TOK_ORDER])
    ax.set_xlim(-0.4, len(TOK_ORDER) - 0.2)
    ax.set_ylabel("held-out bits per byte (lower is better)", color=INK2, fontsize=10)
    ax.set_title("Best test bpb at a fixed 5.5 PFLOP budget (mean ± sd over seeds)",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="center right", bbox_to_anchor=(1.0, 0.42))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "bpb_by_tokenizer.png"), dpi=160)
    plt.close(fig)


def fig_compression_vs_bpb(summary, corpora):
    fig, axes = plt.subplots(1, len(corpora), figsize=(4.4 * len(corpora), 4.0), sharey=False)
    for ax, c in zip(np.atleast_1d(axes), corpora):
        style(ax)
        rows = [r for r in summary if r["corpus"] == c and r["tokenizer"] in TOK_ORDER + ["xbpe2048"]]
        for r in rows:
            t = r["tokenizer"]
            color = TOK_COLOR.get(t, "#e34948")
            mk = "o" if t in TOK_ORDER else "D"
            ax.errorbar(r["bytes_per_token"], r["best_test_bpb"], yerr=r["best_test_bpb_std"], fmt=mk,
                        color=color, ms=8, capsize=3, markeredgecolor="white", markeredgewidth=1.2)
            off = LABEL_OFFSET.get((c, t), (6, 5))
            ax.annotate(TOK_LABEL[t], (r["bytes_per_token"], r["best_test_bpb"]), xytext=off,
                        textcoords="offset points", fontsize=8, color=INK2,
                        ha="right" if off[0] < 0 else "left")
        ax.set_xlabel("compression: held-out bytes per token", color=INK2, fontsize=10)
        ax.set_title(CORPUS_LABEL[c], loc="left", fontsize=10, color=INK)
    np.atleast_1d(axes)[0].set_ylabel("best test bits per byte", color=INK2, fontsize=10)
    fig.suptitle("Compression vs held-out bpb at a fixed 5.5 PFLOP budget",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "compression_vs_bpb.png"), dpi=160)
    plt.close(fig)


def fig_curves(runs, corpora):
    fig, axes = plt.subplots(1, len(corpora), figsize=(4.6 * len(corpora), 4.2))
    for ax, c in zip(np.atleast_1d(axes), corpora):
        style(ax)
        lo = float("inf")
        for tok in TOK_ORDER:
            rs = runs.get((c, tok))
            if not rs:
                continue
            x = np.array([p["pflops"] for p in rs[0]["curve"]])
            y = np.mean([[p["val_bpb"] for p in r["curve"]] for r in rs], axis=0)
            ax.plot(x, y, color=TOK_COLOR[tok], lw=2, label=TOK_LABEL[tok])
            lo = min(lo, y.min())
        ax.set_xscale("log")
        ticks = [0.2, 0.5, 1, 2, 5]
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([str(t) for t in ticks]))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("training compute (PFLOPs, log)", color=INK2, fontsize=10)
        ax.set_title(CORPUS_LABEL[c], loc="left", fontsize=10, color=INK)
        ax.set_ylim(lo - 0.15, lo + 1.5)
    np.atleast_1d(axes)[0].set_ylabel("validation bits per byte", color=INK2, fontsize=10)
    np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=8, loc="upper right", title="tokenizer",
                                  title_fontsize=8)
    fig.suptitle("Validation bpb vs training compute (constant LR, mean over seeds)", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "val_bpb_vs_compute.png"), dpi=160)
    plt.close(fig)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    tok_rows = json.load(open(os.path.join(RESULTS_DIR, "tokenizer_metrics.json")))
    tokm = {(r["corpus"], r["tokenizer"]): r for r in tok_rows}
    runs = load_runs()
    summary = summarize(runs, tokm)
    json.dump(summary, open(os.path.join(RESULTS_DIR, "lm_summary.json"), "w"), indent=1)
    main_corpora = [c for c in ("shakespeare", "python") if any(k[0] == c for k in runs)]
    all_corpora = main_corpora + (["python_large"] if any(k[0] == "python_large" for k in runs) else [])

    fig_bpb_vs_tokenizer(summary, all_corpora)
    fig_compression_vs_bpb(summary, main_corpora)
    fig_curves(runs, all_corpora)

    md = []
    # intrinsic table
    order = tokenizer_names() + ["tiktoken-gpt2", "tiktoken-cl100k_base"]
    for c in ("shakespeare", "python"):
        rows = []
        for t in order:
            r = tokm.get((c, t))
            if not r:
                continue
            rows.append([TOK_LABEL.get(t, t), r["vocab_size"], f"{r['bytes_per_token_heldout']:.2f}",
                         f"{r['bytes_per_token_train']:.2f}", f"{r['tokens_per_doc_median']:.0f}",
                         f"{100 * r['vocab_used_heldout_frac']:.0f}%", f"{100 * r['vocab_rare_train_frac']:.0f}%",
                         f"{r['train_time_s']:.2f}" if r["train_time_s"] is not None else "-",
                         f"{r['encode_MBps']:.1f}", f"{r['decode_MBps']:.1f}"])
        md.append(f"### Intrinsic metrics: {CORPUS_LABEL[c]}\n\n" + md_table(
            ["tokenizer", "vocab", "bytes/token (held-out)", "bytes/token (train)", "median tokens/doc",
             "vocab used on held-out", "vocab rare in train (<10)", "train time (s)", "encode MB/s",
             "decode MB/s"], rows))
    # LM table
    for c in all_corpora:
        rows = []
        for t in TOK_ORDER + ["xbpe2048"]:
            r = next((x for x in summary if x["corpus"] == c and x["tokenizer"] == t), None)
            if not r:
                continue
            rows.append([TOK_LABEL[t], f"{r['n_params_M']:.2f}", r["steps"], f"{r['epochs']:.0f}",
                         f"{r['context_bytes']:.0f}",
                         f"{r['best_test_bpb']:.3f} ± {r['best_test_bpb_std']:.3f}",
                         f"{r['final_test_bpb']:.3f}", f"{r['best_epochs']:.1f}", f"{r['final_gap_bpb']:.2f}",
                         f"{r['train_tokens_per_s'] / 1e3:.0f}K", f"{r['train_bytes_per_s'] / 1e6:.2f}",
                         f"{r['peak_mem_GB']:.2f}", r["seeds"]])
        md.append(f"### LM results: {CORPUS_LABEL[c]}\n\n" + md_table(
            ["tokenizer", "params (M)", "steps", "epochs in budget", "context (bytes)",
             "best test bpb", "final test bpb", "epochs at best", "final val-train gap (bpb)",
             "train tok/s", "train MB/s", "peak mem (GB)", "seeds"], rows))
    # correlations + winners by budget
    lines = []
    for c in all_corpora:
        rows = [r for r in summary if r["corpus"] == c and r["tokenizer"] in TOK_ORDER]
        if len(rows) >= 3 and all(r["bytes_per_token"] for r in rows):
            rho = spearman([r["bytes_per_token"] for r in rows], [r["best_test_bpb"] for r in rows])
            lines.append(f"- {CORPUS_LABEL[c]}: Spearman(bytes/token, best test bpb) = {rho:+.2f} "
                         f"(negative = better compression -> better model)")
    md.append("### Compression vs model quality\n\n" + "\n".join(lines))
    budgets = [0.1, 0.25, 0.5, 1.0, 2.0, 5.5]
    rows = []
    for c in all_corpora:
        w = winners_by_budget(runs, c, budgets)
        for b in budgets:
            if b in w:
                best, score, scores = w[b]
                runner = sorted(scores.items(), key=lambda kv: kv[1])[1]
                rows.append([CORPUS_LABEL[c], b, TOK_LABEL[best], f"{score:.3f}",
                             f"{TOK_LABEL[runner[0]]} ({runner[1]:.3f})"])
    md.append("### Best tokenizer at each compute budget (val bpb with early stopping, mean over seeds)\n\n"
              + md_table(["corpus", "budget (PFLOPs)", "best tokenizer", "val bpb", "runner-up"], rows))
    speed = glob.glob(os.path.join(RESULTS_DIR, "bpe_trainer_speed_*.json"))
    for p in speed:
        rs = json.load(open(p))
        md.append("### BPE trainer wall-clock (s)\n\n" + md_table(
            ["corpus", "vocab", "fast (ours)", "naive (dedup chunks)", "minbpe (course)", "speedup vs minbpe"],
            [[r["corpus"], r["vocab_size"], f"{r['fast_s']:.2f}", f"{r.get('naive_s', float('nan')):.1f}",
              f"{r.get('minbpe_s', float('nan')):.1f}",
              f"{r['minbpe_s'] / r['fast_s']:.0f}x" if "minbpe_s" in r else "-"] for r in rs]))
    open(os.path.join(RESULTS_DIR, "tables.md"), "w").write("\n\n".join(md) + "\n")
    print("\n\n".join(md))


if __name__ == "__main__":
    main()
