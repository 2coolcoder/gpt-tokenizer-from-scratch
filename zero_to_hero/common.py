"""Shared paths and helpers for the lecture scripts."""

import json
import os

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results", "lectures")


def data_path(name):
    return os.path.join(DATA_DIR, name)


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_results(name, results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"saved {path}")
    return path
