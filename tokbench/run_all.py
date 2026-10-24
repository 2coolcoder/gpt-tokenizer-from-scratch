"""Run the full LM sweep (corpus x tokenizer x seed) at a fixed FLOPs budget across GPUs."""
import argparse
import os
import queue
import subprocess
import sys
import threading

from tokbench.tok_suite import CORPORA, tokenizer_names
from tokbench.train_lm import RUNS_DIR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGET_PFLOPS = 5.52  # = the course's char-level run: 5000 steps x 64 x 256 tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--corpora", default=",".join(CORPORA))
    a = ap.parse_args()
    jobs = queue.Queue()
    for seed in range(a.seeds):
        for corpus in a.corpora.split(","):
            for tok in tokenizer_names():
                if corpus == "python_large" and tok == "xbpe2048":
                    continue
                if not os.path.exists(os.path.join(RUNS_DIR, f"{corpus}_{tok}_s{seed}.json")):
                    jobs.put((corpus, tok, seed))

    def worker(gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        while True:
            try:
                corpus, tok, seed = jobs.get_nowait()
            except queue.Empty:
                return
            cmd = [sys.executable, "-m", "tokbench.train_lm", "--corpus", corpus, "--tok", tok,
                   "--seed", str(seed), "--budget_pflops", str(BUDGET_PFLOPS), "--n_evals", "40"]
            r = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, text=True)
            lines = r.stdout.strip().splitlines()
            print(f"gpu{gpu}:", lines[-1] if lines else "", r.stderr[-2000:] if r.returncode else "", flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in a.gpus.split(",")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
