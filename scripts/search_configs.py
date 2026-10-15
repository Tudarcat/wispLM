#!/usr/bin/env python3
"""Search Wispv1 configs that maximise depth at a given parameter budget.

    python scripts/search_configs.py --budget 260 --share 0.5 --top 15
    python scripts/search_configs.py --budget 250 --share 0.5 --gqa 4 --sort depth

Only counted with the analytic estimator (it matches the real model to <0.1%,
see tests/test_model.py::test_estimate_matches_actual).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wisp.config import WispConfig  # noqa: E402
from wisp.model import estimate_parameter_breakdown  # noqa: E402


def build(d: int, layers: int, heads: int, kv: int, ffn: int, share: float, vocab: int, head_dim: int) -> WispConfig:
    share_layers = int(layers * share)
    share_layers -= share_layers % 2
    return WispConfig(
        vocab_size=vocab,
        hidden_size=d,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv,
        head_dim=head_dim,
        intermediate_size=ffn,
        ffn_share_layers=share_layers,
        max_position_embeddings=4096,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=260.0, help="parameter budget in millions")
    ap.add_argument("--tol", type=float, default=0.03, help="relative tolerance around the budget")
    ap.add_argument("--vocab", type=int, default=24000)
    ap.add_argument("--share", type=float, default=0.5, help="fraction of layers with shared FFN")
    ap.add_argument("--gqa", type=int, default=4, help="query heads per KV head")
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--sort", default="depth", choices=["depth", "params", "width"])
    args = ap.parse_args()

    lo, hi = args.budget * (1 - args.tol), args.budget * (1 + args.tol)
    rows: List[Dict] = []
    for d in range(512, 1537, 64):
        for head_dim in (64, 128):
            heads = d // head_dim
            if heads == 0 or d % head_dim or heads % args.gqa:
                continue
            kv = heads // args.gqa
            for ffn_mult in (2.5, 8 / 3, 3.0, 3.5, 4.0):
                ffn = int(d * ffn_mult / 128) * 128
                if ffn <= 0:
                    continue
                for layers in range(12, 97, 4):
                    cfg = build(d, layers, heads, kv, ffn, args.share, args.vocab, head_dim)
                    est = estimate_parameter_breakdown(cfg)["total"]
                    if lo <= est <= hi:
                        rows.append({"d": d, "layers": layers, "heads": heads, "kv": kv, "ffn": ffn,
                                     "params": est, "shared": cfg.ffn_share_layers})

    # keep the deepest variant per (d, ffn) family so the table is readable
    best: Dict[tuple, Dict] = {}
    for r in rows:
        key = (r["d"], r["ffn"], r["kv"])
        if key not in best or r["layers"] > best[key]["layers"]:
            best[key] = r
    cand = list(best.values())
    if args.sort == "depth":
        cand.sort(key=lambda r: (-r["layers"], abs(r["params"] - args.budget)))
    elif args.sort == "width":
        cand.sort(key=lambda r: (r["d"], -r["layers"]))
    else:
        cand.sort(key=lambda r: abs(r["params"] - args.budget))

    print(f"budget {args.budget}M (+-{args.tol:.0%}) | vocab {args.vocab} | FFN-shared fraction {args.share:.0%} | GQA {args.gqa}:1")
    print(f"{'d_model':>7} {'layers':>6} {'heads/kv':>9} {'ffn':>6} {'shared':>6} {'params(M)':>10}")
    for r in cand[: args.top]:
        print(f"{r['d']:>7} {r['layers']:>6} {r['heads']:>4}/{r['kv']:<4} {r['ffn']:>6} {r['shared']:>6} {r['params']:>10.1f}")
    if not cand:
        print("no configuration in range -- widen --tol")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
