#!/usr/bin/env python3
"""Minimal pre-training loop for Wispv1.

This machine is a *coding* box (no training data, weak GPU), so the script is
deliberately small and self-contained: it is meant for smoke tests and for
being run later on a real cluster.

Examples
--------
Smoke test on random tokens (no data, no tokenizer needed):

    python train.py --preset smoke --steps 50 --device cpu

Overfit a tiny bilingual sample to check the whole pipeline:

    python train.py --config configs/wispv1_250m.json \\
        --data data/sample_zh_en.txt --steps 200 --micro-batch 2 --seq-len 256

Real run:

    python train.py --config configs/wispv1_250m.json \\
        --data /path/to/ultra-fineweb-l3.jsonl --steps 100000 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Iterable, List, Optional

import torch

from wisp.config import WispConfig
from wisp.model import WispForCausalLM, count_parameters


# ----------------------------------------------------------------------
# config presets
# ----------------------------------------------------------------------
def build_config(args) -> WispConfig:
    if args.config:
        cfg = WispConfig.load(args.config)
    elif args.preset == "smoke":
        cfg = WispConfig(
            vocab_size=512,
            hidden_size=128,
            num_hidden_layers=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            intermediate_size=384,
            ffn_share_layers=4,
            max_position_embeddings=512,
            alpha_init=1e-4,
        )
    else:  # "250m"
        cfg = WispConfig()
    # CLI overrides
    for name in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                 "intermediate_size", "ffn_share_layers", "max_position_embeddings", "vocab_size",
                 "alpha_mode"):
        val = getattr(args, name, None)
        if val is not None:
            setattr(cfg, name, val)
    if args.alpha_init is not None:
        cfg.alpha_init = args.alpha_init
    if args.alpha_target is not None:
        cfg.alpha_target = args.alpha_target
    cfg.__post_init__()
    return cfg


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
class TokenBlockDataset:
    """Random blocks from one long token stream (tiny memmap-free version)."""

    def __init__(self, token_ids: List[int], seq_len: int, seed: int = 0):
        self.ids = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len
        self.rng = torch.Generator().manual_seed(seed)
        if len(self.ids) < seq_len + 1:
            raise ValueError(f"need at least {seq_len + 1} tokens, got {len(self.ids)}")

    def __len__(self) -> int:
        return max(1, (len(self.ids) - 1) // self.seq_len)

    def sample(self, batch_size: int, pad_token_id: int = 0):
        max_start = len(self.ids) - self.seq_len - 1
        starts = torch.randint(0, max_start + 1, (batch_size,), generator=self.rng)
        x = torch.stack([self.ids[s: s + self.seq_len] for s in starts])
        y = torch.stack([self.ids[s + 1: s + self.seq_len + 1] for s in starts])
        y = y.masked_fill(x == pad_token_id, -100)
        return x, y


def load_text_corpus(path: str, limit_chars: Optional[int] = None) -> Iterable[str]:
    """Yield raw strings from .txt/.jsonl/.json files or a directory of them."""
    paths: List[str] = []
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in sorted(files):
                if f.endswith((".txt", ".jsonl", ".json")):
                    paths.append(os.path.join(root, f))
    else:
        paths = [path]

    total = 0
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as fh:
            if p.endswith(".jsonl"):
                for line in fh:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = obj.get("text") or obj.get("content") or obj.get("raw_content") or ""
                    if text:
                        yield text
                        total += len(text)
                    if limit_chars and total > limit_chars:
                        return
            else:
                while True:
                    chunk = fh.read(1 << 20)
                    if not chunk:
                        break
                    yield chunk
                    total += len(chunk)
                    if limit_chars and total > limit_chars:
                        return


# ----------------------------------------------------------------------
def build_param_groups(model: torch.nn.Module, weight_decay: float):
    decay, no_decay = [], []
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (no_decay if p.ndim <= 1 else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def lr_at(step: int, total: int, warmup: int, base_lr: float, min_lr_ratio: float = 0.1) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if step >= total:
        return base_lr * min_lr_ratio
    progress = (step - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * coeff)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--preset", default="250m", choices=["250m", "smoke"])
    ap.add_argument("--data", default=None, help="text/jsonl file or directory")
    ap.add_argument("--smoke", action="store_true", help="use random tokens (no tokenizer/data)")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--out", default="runs/smoke")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-chars", type=int, default=None)
    # config overrides
    for name in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                 "intermediate_size", "ffn_share_layers", "max_position_embeddings", "vocab_size", "alpha_mode"):
        ap.add_argument(f"--{name.replace('_', '-')}", dest=name, type=(str if name == "alpha_mode" else int), default=None)
    ap.add_argument("--alpha-init", type=float, default=None)
    ap.add_argument("--alpha-target", default=None, choices=["identity", "branch"])
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = build_config(args)

    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev)

    dtype = args.dtype
    if dtype == "auto":
        dtype = "bfloat16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "float32"
    pt_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]

    # ---- tokenizer / vocab ------------------------------------------
    tok = None
    if not args.smoke:
        if args.data is None:
            print("[error] --data is required unless --smoke is given")
            return 2
        from wisp.tokenizer import WispTokenizer

        tok = WispTokenizer()
        cfg.vocab_size = tok.vocab_size
        cfg.bos_token_id = tok.bos_token_id
        cfg.eos_token_id = tok.eos_token_id
        cfg.pad_token_id = tok.pad_token_id
    if args.smoke and args.vocab_size is None:
        cfg.vocab_size = max(cfg.vocab_size, 512)

    # ---- model -------------------------------------------------------
    model = WispForCausalLM(cfg).to(device)
    n_params = count_parameters(model)
    print(f"Wispv1: {n_params/1e6:.2f}M unique params | {cfg.num_hidden_layers} layers "
          f"(first {cfg.ffn_share_layers} shared as {cfg.num_shared_pairs} pairs) | "
          f"d={cfg.hidden_size} heads={cfg.num_attention_heads}/{cfg.num_key_value_heads} "
          f"ffn={cfg.intermediate_size} | device={device} dtype={dtype}")
    print("breakdown (M):", {k: round(v, 2) for k, v in model.parameter_breakdown().items()})

    # ---- data --------------------------------------------------------
    if args.smoke:
        n = args.seq_len * 4096 + 8
        ids = torch.randint(0, cfg.vocab_size, (n,)).tolist()
    else:
        print(f"tokenizing {args.data} ...")
        stream: List[int] = []
        for text in load_text_corpus(args.data, limit_chars=args.max_chars):
            stream.extend(tok.encode(text, add_special_tokens=False))
            stream.append(tok.eos_token_id)
        print(f"  {len(stream):,} tokens")
        ids = stream
    dataset = TokenBlockDataset(ids, args.seq_len, seed=args.seed)

    # ---- optim -------------------------------------------------------
    optim = torch.optim.AdamW(build_param_groups(model, args.weight_decay), lr=args.lr,
                              betas=(0.9, 0.95), eps=1e-8, fused=(device.type == "cuda"))
    model.train()
    t0 = time.time()
    tokens_seen = 0
    losses: List[float] = []

    for step in range(args.steps):
        lr = lr_at(step, args.steps, args.warmup, args.lr)
        for g in optim.param_groups:
            g["lr"] = lr
        optim.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = dataset.sample(args.micro_batch, pad_token_id=cfg.pad_token_id)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=pt_dtype, enabled=(pt_dtype != torch.float32)):
                out = model(x, labels=y)
                loss = out.loss / args.grad_accum
            loss.backward()
            total_loss += loss.item()
            tokens_seen += x.numel()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        losses.append(total_loss)

        if (step + 1) % args.log_every == 0 or step == 0:
            dt = time.time() - t0
            tps = tokens_seen / max(dt, 1e-6)
            print(f"step {step+1:5d}/{args.steps} | loss {total_loss:.4f} | lr {lr:.2e} | "
                  f"gnorm {float(grad_norm):.3f} | {tps:,.0f} tok/s | {dt:.1f}s")
        if args.save_every and (step + 1) % args.save_every == 0:
            model.save_pretrained(os.path.join(args.out, f"step{step+1}"))

    # ---- alpha diagnostics ------------------------------------------
    if cfg.alpha_mode != "none":
        alphas = [layer.residual_attn.alpha.detach().float().mean().item()
                  for layer in model.model.layers if layer.residual_attn.alpha is not None]
        alphas += [layer.residual_ffn.alpha.detach().float().mean().item()
                   for layer in model.model.layers if layer.residual_ffn.alpha is not None]
        if alphas:
            print(f"alpha: init={cfg.alpha_init} -> min={min(alphas):.5f} max={max(alphas):.5f} "
                  f"|L1 grew|={(max(alphas)/cfg.alpha_init):.1f}x")

    model.save_pretrained(args.out)
    print(f"saved to {args.out}")

    if len(losses) >= 2:
        print(f"loss: first={losses[0]:.4f} last={losses[-1]:.4f} (delta {losses[0]-losses[-1]:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
