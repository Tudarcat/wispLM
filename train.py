#!/usr/bin/env python3
#    Copyright 2026 Yunda Wu, Yuyang Shen and contributors

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
Typical use on the training box::

    # 1) one-off: parquet -> uint16 token streams
    python scripts/tokenize_data.py --src /data/Ultra-FineWeb-L3 --dst /data/wispv1_tok --workers 16

    # 2) train (single 4090/3090)
    python train.py --config configs/wispv1_250m.json --data /data/wispv1_tok \\
        --seq-len 4096 --global-batch-tokens 524288 --total-tokens 7.1e9 --out runs/wispv1

    # 3) or multi-GPU
    torchrun --standalone --nproc_per_node=8 train.py --data /data/wispv1_tok ...

    # local CPU smoke test (no data needed)
    python train.py --preset smoke --smoke --steps 20 --device cpu
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import signal
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from wisp.config import WispConfig
from wisp.data import BatchPrefetcher, build_sampler
from wisp.model import WispForCausalLM, count_parameters

# bf16 dense peak FLOPs (TFLOPS); only used for the MFU estimate
GPU_PEAK_TFLOPS = {
    "4090": 165.0, "4080": 97.0, "3090": 71.0, "3080": 60.0,
    "A100": 312.0, "A800": 312.0, "H100": 989.0, "H800": 989.0,
    "L40S": 362.0, "A6000": 155.0, "V100": 125.0,
}


# ======================================================================
# distributed helpers
# ======================================================================
class Dist:
    def __init__(self) -> None:
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.enabled = self.world_size > 1
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.enabled:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        self.is_main = self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.local_rank}")
        return torch.device("cpu")

    def all_reduce_mean(self, value: float) -> float:
        if not self.enabled:
            return value
        t = torch.tensor([value], device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        return float(t.item())


def log0(dist_info: Dist, *args) -> None:
    if dist_info.is_main:
        print(*args, flush=True)


# ======================================================================
# config / smoke sampler
# ======================================================================
def build_config(args) -> WispConfig:
    if args.config:
        cfg = WispConfig.load(args.config)
    elif args.preset == "smoke":
        # 运行在CPU
        cfg = WispConfig(
            vocab_size=512, hidden_size=128, num_hidden_layers=8,
            num_attention_heads=4, num_key_value_heads=2, head_dim=32,
            intermediate_size=384, ffn_share_layers=4, max_position_embeddings=512,
            alpha_init=1e-4, torch_dtype="float32",
        )
    else:
        cfg = WispConfig()
    for name in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                 "intermediate_size", "ffn_share_layers", "max_position_embeddings", "vocab_size",
                 "alpha_mode", "alpha_target"):
        val = getattr(args, name, None)
        if val is not None:
            setattr(cfg, name, val)
    if args.alpha_init is not None:
        cfg.alpha_init = args.alpha_init
    cfg.__post_init__()
    return cfg


class RandomBlockSampler:
    """Drop-in sampler producing random tokens (smoke tests / benchmarking)."""

    def __init__(self, vocab_size: int, block_size: int, seed: int = 1234):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.seed = seed
        self.pos = 0
        self.steps_per_epoch = 10 ** 12
        self.corpora: List = []

    def next_batch(self, batch_size: int, out: Optional[np.ndarray] = None) -> np.ndarray:
        rng = np.random.default_rng([self.seed, self.pos])
        self.pos += batch_size
        return rng.integers(0, self.vocab_size, size=(batch_size, self.block_size + 1), dtype=np.int32)

    def next_val_batch(self, batch_size: int, out: Optional[np.ndarray] = None) -> np.ndarray:
        return self.next_batch(batch_size)

    def reset_val(self) -> None:
        pass

    def state_dict(self) -> dict:
        return {"epoch": 0, "pos": int(self.pos), "seed": self.seed, "block_size": self.block_size}

    def load_state_dict(self, state: dict) -> None:
        self.pos = int(state["pos"])


# ======================================================================
# mixed precision / memory
# ======================================================================
def autocast_ctx(args, device: torch.device):
    if args.dtype == "float32" or device.type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def optimizer_state_bytes(model: nn.Module) -> int:
    """Bytes AdamW allocates lazily (fp32 exp_avg + exp_avg_sq per parameter)."""
    total, seen = 0, set()
    for p in model.parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel() * 4 * 2
    return total


def _probe_once(args, model, optimizer, device, seq_len, vocab_size, mb: int, budget: int) -> Tuple[bool, int]:
    """One forward+backward at ``mb``; returns (fits, peak_bytes)."""
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        ids = torch.randint(0, vocab_size, (mb, seq_len), device=device)
        with autocast_ctx(args, device):
            out = model(ids, labels=ids, loss_chunk_size=args.loss_chunk)
            out.loss.backward()
        peak = torch.cuda.max_memory_allocated(device)
        optimizer.zero_grad(set_to_none=True)
        del ids, out
        return peak <= budget, peak
    except torch.cuda.OutOfMemoryError:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return False, budget + 1
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return False, budget + 1


def search_max_micro_batch(try_fn, max_micro_batch: int) -> Tuple[int, int]:
    """Largest micro-batch accepted by ``try_fn(mb) -> (fits, peak)``.

    Geometric growth to bracket the edge, then a binary search.  Kept separate
    from the CUDA probing so the search itself is unit-testable without a GPU.
    """
    lo, peak_at_lo = 0, 0
    hi = 0
    mb = 1
    while mb <= max_micro_batch:
        ok, peak = try_fn(mb)
        if ok:
            lo, peak_at_lo = mb, peak
            mb *= 2
        else:
            hi = mb
            break
    if lo == 0:
        return 0, 0
    if hi == 0:
        hi = min(max_micro_batch, lo * 2) + 1
    while lo + 1 < hi and lo < max_micro_batch:
        mid = (lo + hi) // 2
        if mid <= lo or mid > max_micro_batch:
            break
        ok, peak = try_fn(mid)
        if ok:
            lo, peak_at_lo = mid, peak
        else:
            hi = mid
    return lo, peak_at_lo


def probe_micro_batch(args, model, optimizer, device, seq_len, cfg) -> Tuple[int, float]:
    """Largest micro-batch that fits ``--mem-fraction`` of the GPU.

    Runs real forward+backward passes: geometric growth to find an upper bound,
    then a binary search.  Must be called *before* wrapping in DDP so the probe
    does not trigger gradient all-reduces.
    """
    if device.type != "cuda" or args.micro_batch > 0:
        return args.micro_batch or 1, 0.0
    total = torch.cuda.get_device_properties(device).total_memory
    opt_bytes = optimizer_state_bytes(model)
    budget = int(total * args.mem_fraction - opt_bytes)
    if budget <= 0:
        raise RuntimeError("optimizer state alone exceeds the memory budget")

    was_ckpt = model.model.gradient_checkpointing
    model.train()

    def try_fn(mb: int) -> Tuple[bool, int]:
        return _probe_once(args, model, optimizer, device, seq_len, cfg.vocab_size, mb, budget)

    best, peak = search_max_micro_batch(try_fn, args.max_micro_batch)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    model.model.gradient_checkpointing = was_ckpt
    if best == 0:
        raise RuntimeError(
            f"even micro-batch=1 OOMs at seq_len={seq_len}. Try --activation-checkpointing on, "
            f"a smaller --seq-len, or a smaller --loss-chunk."
        )
    return best, (peak + opt_bytes) / total


def make_loss_labels(input_ids: torch.Tensor, lut: Optional[torch.Tensor]) -> torch.Tensor:
    """Set ``-100`` on every position whose token id is flagged in ``lut``.

    Used to ignore ``<|unk|>`` (and optionally ``<|pad|>``) targets: predicting a
    single "unknown" token carries no learning signal, and it is 0.25% of the
    Chinese corpus.  ``lut`` is a boolean table of size ``vocab_size`` or None.
    """
    if lut is None:
        return input_ids
    return input_ids.masked_fill(lut[input_ids], -100)


# ======================================================================
# lr schedule (in tokens)
# ======================================================================
def lr_at(tokens_seen: float, args) -> float:
    if tokens_seen < args.warmup_tokens:
        return args.lr * max(tokens_seen / max(args.warmup_tokens, 1), 1e-8)
    if tokens_seen >= args.total_tokens:
        return args.lr * args.min_lr_ratio
    progress = (tokens_seen - args.warmup_tokens) / max(args.total_tokens - args.warmup_tokens, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * coeff)


def build_param_groups(model: nn.Module, weight_decay: float) -> List[dict]:
    decay, no_decay = [], []
    seen = set()
    for _, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (no_decay if p.ndim <= 1 else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


# ======================================================================
# checkpointing
# ======================================================================
def save_checkpoint(path: str, model, optimizer, args, cfg, step: int, tokens_seen: int,
                    sampler_state: dict) -> None:
    tmp = path + ".tmp"
    core = model.module if hasattr(model, "module") else model
    payload = {
        "format": "wispv1-train",
        "version": 1,
        "step": step,
        "tokens_seen": tokens_seen,
        "model": core.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg.to_dict(),
        "train_args": {k: v for k, v in vars(args).items()
                       if isinstance(v, (int, float, str, bool, type(None)))},
        "sampler_state": sampler_state,
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, model, optimizer, device) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    core = model.module if hasattr(model, "module") else model
    missing, unexpected = core.load_state_dict(ckpt["model"], strict=False)
    real_missing = [k for k in missing if "rotary" not in k]
    if real_missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={real_missing[:5]} unexpected={list(unexpected)[:5]}")
    if "optimizer" in ckpt and optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def find_latest(out_dir: str) -> Optional[str]:
    ptr = os.path.join(out_dir, "latest.json")
    if os.path.exists(ptr):
        with open(ptr, "r", encoding="utf-8") as fh:
            path = os.path.join(out_dir, json.load(fh)["checkpoint"])
        if os.path.exists(path):
            return path
    return None


def write_latest_pointer(out_dir: str, name: str) -> None:
    tmp = os.path.join(out_dir, "latest.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"checkpoint": name}, fh)
    os.replace(tmp, os.path.join(out_dir, "latest.json"))


def prune_checkpoints(out_dir: str, keep_last: int) -> None:
    if keep_last <= 0:
        return
    import glob
    ckpts = sorted(glob.glob(os.path.join(out_dir, "step*.pt")),
                   key=lambda p: int(os.path.basename(p)[4:-3]))
    for path in ckpts[:-keep_last]:
        try:
            os.remove(path)
        except OSError:
            pass


# ======================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # model
    ap.add_argument("--config", default=None)
    ap.add_argument("--preset", default="250m", choices=["250m", "smoke"])
    for name in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                 "intermediate_size", "ffn_share_layers", "max_position_embeddings", "vocab_size"):
        ap.add_argument(f"--{name.replace('_', '-')}", dest=name, type=int, default=None)
    ap.add_argument("--alpha-mode", default=None, choices=["shared", "all", "none"])
    ap.add_argument("--alpha-target", default=None, choices=["identity", "branch"])
    ap.add_argument("--alpha-init", type=float, default=None)
    # data
    ap.add_argument("--data", default=None, help="root of tokenized corpora (scripts/tokenize_data.py)")
    ap.add_argument("--mixture", default=None, help="e.g. 'pretrain/en=0.55,pretrain/zh=0.45'")
    ap.add_argument("--no-ram-cache", action="store_true", help="mmap instead of loading tokens into RAM")
    ap.add_argument("--seq-len", type=int, default=4096, help="block size")
    ap.add_argument("--smoke", action="store_true", help="random tokens, no data needed")
    ap.add_argument("--val-fraction", type=float, default=0.002)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-batches", type=int, default=20)
    # batch / memory
    ap.add_argument("--micro-batch", type=int, default=0, help="0 = probe the largest that fits")
    ap.add_argument("--max-micro-batch", type=int, default=64)
    ap.add_argument("--grad-accum", type=int, default=0, help="0 = derive from --global-batch-tokens")
    ap.add_argument("--global-batch-tokens", type=int, default=524288,
                    help="tokens per optimizer step across all ranks")
    ap.add_argument("--mem-fraction", type=float, default=0.90)
    ap.add_argument("--loss-chunk", type=int, default=512, help="positions per LM-head chunk (0 = full)")
    # optimisation
    ap.add_argument("--steps", type=int, default=0, help="stop after N optimizer steps (0 = use --total-tokens)")
    ap.add_argument("--total-tokens", type=float, default=-1,
                    help="LR schedule horizon; -1 = steps * global_batch_tokens")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup-tokens", type=float, default=-1, help="-1 = min(10M, 2%% of total)")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--mask-unk", default="on", choices=["on", "off"],
                    help="ignore <|unk|> targets in the loss (rare chars -> unk, ~0.25%% of zh)")
    ap.add_argument("--loss-mask-ids", default="",
                    help="extra token ids to ignore in the loss, comma separated (e.g. '3' for pad)")
    # runtime
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--activation-checkpointing", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--sdpa-backend", default="auto", choices=["auto", "flash", "mem_efficient", "math"])
    ap.add_argument("--prefetch-slots", type=int, default=4)
    ap.add_argument("--no-prefetch", action="store_true")
    # logging / io
    ap.add_argument("--out", default="runs/wispv1")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--keep-last", type=int, default=3)
    ap.add_argument("--resume", default="auto", help="'auto', 'none', or a checkpoint path")
    ap.add_argument("--strict-data-resume", action="store_true",
                    help="fail instead of restarting the data order when the mixture changed")
    ap.add_argument("--benchmark", action="store_true",
                    help="measure throughput and exit without writing checkpoints")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--gpu-peak-tflops", type=float, default=0.0)
    args = ap.parse_args()

    dist_info = Dist()
    device = torch.device(args.device) if args.device != "auto" else dist_info.device
    cfg = build_config(args)
    if device.type != "cuda" and args.dtype != "float32":
        args.dtype = "float32"  # autocast is a no-op off CUDA; don't pretend otherwise
    if args.activation_checkpointing == "auto":
        args.activation_checkpointing = "on" if args.seq_len >= 2048 else "off"

    torch.manual_seed(args.seed + dist_info.rank)
    np.random.seed(args.seed + dist_info.rank)
    random.seed(args.seed + dist_info.rank)
    torch.set_float32_matmul_precision("high")

    # ---------------- data ----------------
    if args.smoke:
        cfg.vocab_size = max(cfg.vocab_size, 512)
        train_sampler = RandomBlockSampler(cfg.vocab_size, args.seq_len, seed=args.seed)
        val_sampler = None
        corpora_info = "random tokens (smoke)"
        if not args.steps:
            args.steps = 20
    else:
        if not args.data:
            log0(dist_info, "[error] --data is required unless --smoke is given")
            return 2
        mixture = None
        if args.mixture:
            mixture = {}
            for part in args.mixture.split(","):
                k, v = part.split("=")
                mixture[k.strip()] = float(v)
        in_memory = not args.no_ram_cache
        train_sampler, _ = build_sampler(
            args.data, args.seq_len, mixture=mixture, seed=args.seed,
            val_fraction=args.val_fraction, split="train", in_memory=in_memory,
        )
        val_sampler = None
        if args.val_fraction > 0 and args.val_batches > 0:
            val_sampler, _ = build_sampler(
                args.data, args.seq_len, mixture=mixture, seed=args.seed,
                val_fraction=args.val_fraction, split="val", in_memory=in_memory,
            )
        from wisp.tokenizer import WispTokenizer
        try:
            tok = WispTokenizer()
            cfg.vocab_size = tok.vocab_size
            cfg.bos_token_id, cfg.eos_token_id, cfg.pad_token_id = (
                tok.bos_token_id, tok.eos_token_id, tok.pad_token_id)
        except FileNotFoundError:
            log0(dist_info, "[warn] local tokenizer not found; keeping vocab_size from config")
        corpora_info = ", ".join(f"{c.name}={c.num_tokens/1e9:.3f}B" for c in train_sampler.corpora)

    # ---------------- model ----------------
    # Order matters: build -> optimizer -> resume -> probe -> compile -> DDP.
    # The probe must run *before* DDP so it does not fire gradient all-reduces,
    # and torch.compile should wrap the raw module rather than the DDP wrapper.
    model = WispForCausalLM(cfg).to(device)
    if args.activation_checkpointing == "on":
        model.model.enable_gradient_checkpointing(True)
    n_params = count_parameters(model)
    log0(dist_info, f"Wispv1 | {n_params/1e6:.2f}M params | {cfg.num_hidden_layers} layers "
                   f"({cfg.ffn_share_layers} shared as {cfg.num_shared_pairs} pairs) | "
                   f"d={cfg.hidden_size} heads={cfg.num_attention_heads}/{cfg.num_key_value_heads} "
                   f"ffn={cfg.intermediate_size} | alpha={cfg.alpha_target}/{cfg.alpha_init}")
    log0(dist_info, f"device={device} dtype={args.dtype} world={dist_info.world_size} "
                   f"ckpt={args.activation_checkpointing} compile={args.compile}")
    log0(dist_info, f"data: {corpora_info}")
    if not args.smoke:
        n_tok = sum(c.num_tokens for c in train_sampler.corpora)
        log0(dist_info, f"corpus: {n_tok/1e9:.3f}B tokens in {len(train_sampler.corpora)} pools | "
                        f"samples/epoch={train_sampler.steps_per_epoch:,} | "
                        f"epoch tokens={train_sampler.steps_per_epoch*args.seq_len/1e9:.3f}B")

    if args.sdpa_backend != "auto" and device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(args.sdpa_backend == "flash")
        torch.backends.cuda.enable_mem_efficient_sdp(args.sdpa_backend == "mem_efficient")
        torch.backends.cuda.enable_math_sdp(args.sdpa_backend == "math")

    # ---------------- optimizer ----------------
    optimizer = torch.optim.AdamW(
        build_param_groups(model, args.weight_decay), lr=args.lr,
        betas=(args.beta1, args.beta2), eps=1e-8, fused=(device.type == "cuda"),
    )

    # ---------------- resume ----------------
    start_step, tokens_seen = 0, 0
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = find_latest(args.out) if args.resume == "auto" else (
        None if args.resume in ("none", "") else args.resume)
    if ckpt_path:
        info = load_checkpoint(ckpt_path, model, optimizer, device)
        start_step = int(info["step"])
        tokens_seen = int(info["tokens_seen"])
        state = info.get("sampler_state")
        if state:
            try:
                train_sampler.load_state_dict(state)
            except ValueError as exc:
                if args.strict_data_resume:
                    raise
                # expected when entering the decay stage: the mixture changed, so
                # the old position cannot be reused.  Restart the data order, but
                # keep tokens_seen so the LR schedule continues where it was.
                log0(dist_info, f"[data] {exc}")
                log0(dist_info, "[data] starting the data order fresh for this mixture "
                                "(the LR schedule continues from the restored token count)")
        log0(dist_info, f"resumed from {ckpt_path} at step {start_step}, {tokens_seen/1e9:.4f}B tokens")

    # ---------------- batching ----------------
    micro_batch, mem_used = probe_micro_batch(args, model, optimizer, device, args.seq_len, cfg)
    if dist_info.enabled:
        # every rank must use the same micro-batch (they normally have identical
        # cards, but a rank with less free VRAM must not desynchronise the group)
        t = torch.tensor([micro_batch], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        if int(t.item()) != micro_batch:
            log0(dist_info, f"micro-batch lowered {micro_batch} -> {int(t.item())} to match the "
                            f"smallest rank")
            micro_batch = int(t.item())
    tokens_per_micro = micro_batch * args.seq_len * dist_info.world_size
    grad_accum = args.grad_accum if args.grad_accum > 0 else max(1, round(args.global_batch_tokens / tokens_per_micro))
    global_batch_tokens = tokens_per_micro * grad_accum
    if args.total_tokens <= 0:
        if not args.steps:
            log0(dist_info, "[error] give --total-tokens or --steps")
            return 2
        args.total_tokens = args.steps * global_batch_tokens
    if args.warmup_tokens < 0:
        args.warmup_tokens = min(1e7, 0.02 * args.total_tokens)
    total_steps = args.steps if args.steps else math.ceil(args.total_tokens / global_batch_tokens)
    log0(dist_info, f"micro_batch={micro_batch} grad_accum={grad_accum} "
                   f"global_batch={global_batch_tokens:,} tokens "
                   f"({micro_batch}x{args.seq_len}x{grad_accum}x{dist_info.world_size})"
                   + (f" | probe used {mem_used:.1%} of VRAM" if mem_used else ""))
    log0(dist_info, f"total_steps={total_steps:,} | "
                   f"total_tokens={total_steps*global_batch_tokens/1e9:.3f}B | "
                   f"lr {args.lr} warmup {args.warmup_tokens/1e6:.1f}M")

    if args.compile:
        log0(dist_info, "compiling model (the first steps will be slow) ...")
        model = torch.compile(model)
    if dist_info.enabled:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[dist_info.local_rank] if device.type == "cuda" else None,
            find_unused_parameters=False, broadcast_buffers=False,
        )
    core = model.module if hasattr(model, "module") else model

    prefetcher = BatchPrefetcher(train_sampler, micro_batch, device=device,
                                 slots=args.prefetch_slots, pin_memory=not args.no_prefetch)
    val_prefetcher = None
    if val_sampler is not None:
        val_prefetcher = BatchPrefetcher(val_sampler, micro_batch, device=device,
                                         slots=2, validation=True)

    # ---------------- logging setup ----------------
    jsonl = open(os.path.join(args.out, "train.jsonl"), "a", encoding="utf-8") if dist_info.is_main else None
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    peak_tflops = args.gpu_peak_tflops or next(
        (v for k, v in GPU_PEAK_TFLOPS.items() if k.lower() in gpu_name.lower()), 0.0)
    attn_flops_per_token = 4 * cfg.num_hidden_layers * args.seq_len * cfg.num_attention_heads * cfg.head_dim
    flops_per_token = 6 * n_params + attn_flops_per_token

    stop_requested = {"flag": False}

    def _sigterm(_signo, _frame):
        stop_requested["flag"] = True

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    # ---------------- loss masking ----------------
    # <|unk|> (id 0) appears for rare hanzi / emoji / kana that the 24k vocab
    # cannot encode (~0.25% of Chinese tokens).  Asking the model to predict a
    # single "unknown" token carries no signal, so we mask it like padding.
    mask_ids: List[int] = []
    if args.mask_unk == "on" and not args.smoke:
        try:
            from wisp.tokenizer import WispTokenizer
            mask_ids.append(WispTokenizer().unk_token_id)
        except FileNotFoundError:
            pass
    for part in filter(None, args.loss_mask_ids.split(",")):
        mask_ids.append(int(part))
    mask_ids = sorted(set(i for i in mask_ids if i is not None))
    loss_mask_lut = None
    if mask_ids:
        valid = [i for i in mask_ids if 0 <= i < cfg.vocab_size]
        loss_mask_lut = torch.zeros(cfg.vocab_size, dtype=torch.bool, device=device)
        loss_mask_lut[torch.tensor(valid, device=device)] = True
        label_of = {0: "unk", cfg.pad_token_id: "pad", cfg.eos_token_id: "eos", cfg.bos_token_id: "bos"}
        names = [f"{label_of.get(i, 'id')}={i}" for i in valid]
        log0(dist_info, f"loss masking: ignoring targets {', '.join(names)}")

    def make_labels(input_ids: torch.Tensor) -> torch.Tensor:
        return make_loss_labels(input_ids, loss_mask_lut)

    # ---------------- train ----------------
    model.train()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    t_log = t0
    last_state = train_sampler.state_dict()
    step = start_step

    while step < total_steps:
        lr = lr_at(tokens_seen, args)
        for g in optimizer.param_groups:
            g["lr"] = lr
        loss_sum = 0.0
        for micro in range(grad_accum):
            batch = prefetcher.next()
            last_state = batch.sampler_state
            ctx = model.no_sync() if (dist_info.enabled and micro < grad_accum - 1) else contextlib.nullcontext()
            with ctx:
                with autocast_ctx(args, device):
                    out = model(batch.input_ids, labels=make_labels(batch.input_ids),
                                loss_chunk_size=args.loss_chunk)
                    loss = out.loss / grad_accum
                loss.backward()
            loss_sum += float(loss.detach())
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        step += 1
        tokens_seen += global_batch_tokens

        if step % args.log_every == 0 or step == start_step + 1:
            now = time.time()
            dt = now - t_log
            t_log = now
            tok_per_s = (global_batch_tokens * args.log_every) / max(dt, 1e-9)
            rec = {
                "step": step,
                "tokens": tokens_seen,
                "loss": round(dist_info.all_reduce_mean(loss_sum), 5),
                "lr": lr,
                "grad_norm": float(grad_norm),
                "tok_per_s": round(tok_per_s),
                "elapsed_s": round(now - t0, 1),
            }
            if not args.smoke and train_sampler.steps_per_epoch:
                rec["epoch"] = round(tokens_seen / (train_sampler.steps_per_epoch * args.seq_len), 4)
            if peak_tflops:
                rec["mfu"] = round(flops_per_token * tok_per_s / (peak_tflops * 1e12), 4)
            if device.type == "cuda":
                rec["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
            alphas = [l.residual_attn.alpha.detach().float().mean() for l in core.model.layers
                      if l.residual_attn.alpha is not None]
            if alphas:
                a = torch.stack(alphas)
                rec["alpha_mean"] = round(float(a.mean()), 8)
                rec["alpha_max"] = round(float(a.max()), 8)
            if dist_info.is_main:
                extra = f" | mfu {rec['mfu']:.1%}" if "mfu" in rec else ""
                mem = f" | mem {rec['peak_mem_gb']:.1f}GB" if "peak_mem_gb" in rec else ""
                alpha = f" | alpha {rec['alpha_mean']:.2e}" if "alpha_mean" in rec else ""
                print(f"step {step:6d}/{total_steps} | loss {rec['loss']:.4f} | lr {lr:.2e} "
                      f"| gnorm {rec['grad_norm']:.3f} | {tok_per_s/1e3:.1f}k tok/s{extra}{mem}{alpha} "
                      f"| {rec['elapsed_s']/60:.1f}min", flush=True)
                if jsonl:
                    jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    jsonl.flush()

        if val_prefetcher is not None and args.val_every and step % args.val_every == 0:
            model.eval()
            val_loss, n = 0.0, 0
            with torch.no_grad():
                for _ in range(args.val_batches):
                    b = val_prefetcher.next()
                    with autocast_ctx(args, device):
                        out = model(b.input_ids, labels=make_labels(b.input_ids),
                                    loss_chunk_size=args.loss_chunk)
                    val_loss += float(out.loss)
                    n += 1
            if val_sampler is not None:
                val_sampler.reset_val()
            val_loss /= max(n, 1)
            model.train()
            if dist_info.is_main:
                print(f"           val loss {val_loss:.4f} ({n} batches)", flush=True)
                if jsonl:
                    jsonl.write(json.dumps({"step": step, "tokens": tokens_seen,
                                            "val_loss": round(val_loss, 5)}) + "\n")
                    jsonl.flush()

        if args.save_every and step % args.save_every == 0:
            name = f"step{step}.pt"
            save_checkpoint(os.path.join(args.out, name), model, optimizer, args, cfg,
                            step, tokens_seen, last_state)
            if dist_info.is_main:
                write_latest_pointer(args.out, name)
                prune_checkpoints(args.out, args.keep_last)
                print(f"           saved {name}", flush=True)

        if stop_requested["flag"]:
            log0(dist_info, "\n[signal] saving checkpoint and exiting ...")
            break

        if args.benchmark and step >= start_step + args.steps:
            break

    # ---------------- final save ----------------
    if not args.benchmark:
        name = f"step{step}.pt"
        save_checkpoint(os.path.join(args.out, name), model, optimizer, args, cfg, step, tokens_seen, last_state)
        if dist_info.is_main:
            write_latest_pointer(args.out, name)
            prune_checkpoints(args.out, args.keep_last)
            core.save_pretrained(os.path.join(args.out, "final"))
            with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as fh:
                json.dump(cfg.to_dict(), fh, indent=2, ensure_ascii=False)
    prefetcher.close()
    if val_prefetcher:
        val_prefetcher.close()
    if jsonl:
        jsonl.close()
    if dist_info.enabled:
        dist.barrier()
        dist.destroy_process_group()
    wall = time.time() - t0
    log0(dist_info, f"done: {step - start_step} steps this run, total {tokens_seen/1e9:.3f}B tokens, "
                    f"{wall/60:.1f} min -> {args.out}")
    if args.benchmark and wall > 0:
        log0(dist_info, f"throughput: {tokens_seen/wall:,.0f} tok/s "
                        f"(seq_len={args.seq_len}, micro_batch={micro_batch}, "
                        f"grad_accum={grad_accum}, ckpt={args.activation_checkpointing})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
