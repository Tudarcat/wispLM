#!/usr/bin/env python3
"""Reproduce the alpha ablation (see experiments/alpha_ablation.md).

Runs the same tiny overfit task for several (alpha_target, alpha_init) pairs
and prints a table. On a CPU-only box this takes ~10 minutes for the default
four settings; use --jobs to run them in parallel.

    python scripts/alpha_ablation.py --steps 400 --jobs 4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

SETTINGS: List[Tuple[str, float]] = [
    ("identity", 1e-4),
    ("identity", 1e-2),
    ("identity", 0.5),
    ("identity", 1.0),
    ("branch", 1e-4),
    ("branch", 0.0),
]

DEFAULT_SETTINGS = [("identity", 1e-4), ("identity", 1.0), ("branch", 1e-4), ("branch", 0.0)]


def run(setting: Tuple[str, float], args) -> Tuple[Tuple[str, float], float, float]:
    target, init = setting
    cmd = [
        PY, os.path.join(HERE, "train.py"),
        "--preset", "smoke",
        "--data", os.path.join(HERE, "data", "sample_zh_en.txt"),
        "--seq-len", "128", "--micro-batch", "2", "--grad-accum", "1",
        "--steps", str(args.steps), "--lr", "3e-3", "--warmup", "20",
        "--alpha-init", str(init), "--alpha-target", target,
        "--out", os.path.join(args.out, f"{target}_{init}"),
        "--device", "cpu", "--log-every", str(args.steps),
    ]
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    out = subprocess.run(cmd, capture_output=True, text=True, env=env).stdout
    first = last = float("nan")
    for line in out.splitlines():
        if line.startswith("step") and "loss" in line:
            parts = line.split("|")
            loss = float(parts[1].split()[-1])
            if first != first:
                first = loss
            last = loss
    return setting, first, last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="/tmp/wisp_alpha_ablation")
    ap.add_argument("--all", action="store_true", help="run all six settings instead of the default four")
    args = ap.parse_args()

    settings = SETTINGS if args.all else DEFAULT_SETTINGS
    print(f"alpha ablation: {args.steps} steps, tiny config, data/sample_zh_en.txt")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda s: run(s, args), settings))

    print(f"\n{'alpha_target':>13} {'alpha_init':>11} {'loss@1':>8} {'loss@'+str(args.steps):>10}")
    for (target, init), first, last in results:
        flag = "" if last < 2.0 else "   <- stuck"
        print(f"{target:>13} {init:>11} {first:>8.3f} {last:>10.3f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
