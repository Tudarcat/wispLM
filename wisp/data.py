"""Efficient pre-training data pipeline for Wispv1.

Two stages:

1. :mod:`scripts.tokenize_data` turns the parquet shards into flat **uint16**
   token streams (one ``tokens.bin`` + ``meta.json`` per corpus).  uint16 is
   enough because the QiYuan tokenizer has a 24 000 token vocabulary, which
   halves both disk traffic and RAM footprint versus int32/int64.
2. This module loads those streams (fully into RAM when it fits) and produces
   shuffled ``[batch, seq_len + 1]`` blocks from a weighted mixture of corpora.
   The gather runs in a background thread into pinned memory, so the GPU never
   waits for the CPU.

Design notes
------------
* **No overlap inside an epoch.**  Each corpus is cut into disjoint blocks; an
  epoch visits every block exactly once in a seeded-shuffled order, so the
  token stream is seen precisely once per epoch.
* **Exact resumption.**  All randomness comes from
  ``numpy.random.default_rng(f"{seed}:{epoch}:{corpus}")``, so the schedule is
  reproducible from ``(seed, epoch, position)`` alone -- a checkpoint only has
  to store those three numbers, not a shard index table.
* **Mixture.**  A batch can mix corpora; the per-epoch block budget is
  ``steps_per_epoch = min_i(n_i / w_i)`` so no corpus runs out mid-epoch.
* **Validation** is a fixed, non-overlapping tail slice of every corpus.
"""

from __future__ import annotations

import json
import os
import queue as _queue
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = [
    "TokenizedCorpus",
    "MixtureBlockSampler",
    "BatchPrefetcher",
    "PrefetchedBatch",
    "discover_corpora",
    "build_sampler",
]


# ----------------------------------------------------------------------
class TokenizedCorpus:
    """A flat token stream produced by ``scripts/tokenize_data.py``."""

    def __init__(self, root: str, name: Optional[str] = None, in_memory: bool = True):
        self.root = root
        meta_path = os.path.join(root, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"{meta_path} not found (run scripts/tokenize_data.py)")
        with open(meta_path, "r", encoding="utf-8") as fh:
            self.meta = json.load(fh)
        self.name = name or self.meta.get("name") or os.path.basename(root.rstrip("/"))
        self.num_tokens = int(self.meta["num_tokens"])
        self.num_docs = int(self.meta.get("num_docs", 0))
        self.dtype = np.dtype(self.meta.get("dtype", "uint16"))
        self.bin_path = os.path.join(root, self.meta.get("bin_file", "tokens.bin"))
        if self.dtype.itemsize > 2:
            raise ValueError(f"{self.name}: dtype {self.dtype} is wider than uint16")
        if in_memory:
            self.tokens: np.ndarray = np.fromfile(self.bin_path, dtype=self.dtype)
        else:
            self.tokens = np.memmap(self.bin_path, dtype=self.dtype, mode="r")
        if len(self.tokens) != self.num_tokens:
            raise ValueError(
                f"{self.name}: {self.bin_path} has {len(self.tokens)} tokens, meta says {self.num_tokens}"
            )
        self.in_memory = in_memory
        self.nbytes = self.num_tokens * self.dtype.itemsize

    def __len__(self) -> int:
        return self.num_tokens

    def __repr__(self) -> str:
        where = "RAM" if self.in_memory else "mmap"
        return (f"TokenizedCorpus({self.name!r}, {self.num_tokens/1e9:.3f}B tokens, "
                f"{self.num_docs:,} docs, {self.dtype}, {where})")


def discover_corpora(root: str, in_memory: bool = True) -> Dict[str, TokenizedCorpus]:
    """Find every ``<name>/meta.json`` under ``root`` (recursively).

    Corpus names must be unique: tokenizing two different source trees without a
    ``--name-prefix`` can both yield ``en``/``zh``, which would silently make one
    pool shadow the other.
    """
    found: Dict[str, TokenizedCorpus] = {}
    for dirpath, _, filenames in os.walk(root):
        if "meta.json" in filenames:
            corp = TokenizedCorpus(dirpath, in_memory=in_memory)
            if corp.name in found:
                raise ValueError(
                    f"duplicate corpus name {corp.name!r}: {found[corp.name].root} and {dirpath}. "
                    f"Re-run scripts/tokenize_data.py with --name-prefix to disambiguate."
                )
            found[corp.name] = corp
    if not found:
        raise FileNotFoundError(f"no tokenized corpora (meta.json) under {root}")
    return found


# ----------------------------------------------------------------------
class MixtureBlockSampler:
    """Deterministic, epoch-based, weighted mixture of block streams."""

    def __init__(
        self,
        corpora: Sequence[TokenizedCorpus],
        weights: Optional[Sequence[float]] = None,
        block_size: int = 4096,
        seed: int = 1234,
        val_fraction: float = 0.0,
        split: str = "train",
    ):
        if not corpora:
            raise ValueError("need at least one corpus")
        self.corpora = list(corpora)
        self.block_size = int(block_size)
        self.seed = int(seed)
        self.split = split
        w = np.asarray(weights if weights is not None else np.ones(len(corpora)), dtype=np.float64)
        if len(w) != len(corpora):
            raise ValueError("weights length must match corpora")
        self.weights = w / w.sum()

        dtype = self.corpora[0].dtype
        for c in self.corpora:
            if c.dtype != dtype:
                raise ValueError(f"mixed dtypes ({c.name}: {c.dtype} vs {dtype}) are not supported")

        # disjoint blocks: sample i covers tokens [i*L, i*L + L]
        self.n_samples = [(c.num_tokens - 1) // self.block_size for c in self.corpora]
        n_val = [int(n * val_fraction) for n in self.n_samples]
        if split == "train":
            self.limits = [n - v for n, v in zip(self.n_samples, n_val)]
            self.offsets = [0] * len(self.corpora)
        elif split == "val":
            self.limits = list(n_val)
            self.offsets = [n - v for n, v in zip(self.n_samples, n_val)]
        else:
            raise ValueError(split)
        if any(l <= 0 for l in self.limits):
            raise ValueError(
                f"empty {split} split: {dict(zip([c.name for c in self.corpora], self.limits))}. "
                f"Too few tokens for block_size={self.block_size}/val_fraction={val_fraction}?"
            )

        # samples per epoch: no corpus may run out
        active = [i for i in range(len(corpora)) if self.weights[i] > 0]
        self.steps_per_epoch = int(min(self.limits[i] / self.weights[i] for i in active))
        self.samples_per_epoch = self.steps_per_epoch  # one sample per draw

        self.epoch = 0
        self.pos = 0
        self._build_epoch(0)
        self._build_val_blocks()

    # ------------------------------------------------------------------
    def _build_val_blocks(self) -> None:
        """Round-robin list of validation blocks (deterministic order)."""
        per_corpus = [
            [ (ci, self.offsets[ci] + k) for k in range(self.limits[ci]) ]
            for ci in range(len(self.corpora))
            if self.limits[ci] > 0
        ]
        blocks: List[Tuple[int, int]] = []
        cursor = [0] * len(per_corpus)
        while True:
            added = False
            for i, lst in enumerate(per_corpus):
                if cursor[i] < len(lst):
                    blocks.append(lst[cursor[i]])
                    cursor[i] += 1
                    added = True
            if not added:
                break
        self._val_blocks = blocks
        self._val_pos = 0

    def _build_epoch(self, epoch: int) -> None:
        """Rebuild the per-epoch assignment + per-corpus shuffled block queues."""
        self.epoch = epoch
        counts = np.floor(self.steps_per_epoch * self.weights).astype(np.int64)
        # hand out the rounding remainder to the largest fractional parts
        short = self.steps_per_epoch - int(counts.sum())
        if short > 0:
            frac = self.steps_per_epoch * self.weights - counts
            for i in np.argsort(-frac)[:short]:
                counts[i] += 1
        assignment = np.repeat(np.arange(len(self.corpora)), counts).astype(np.int32)
        np.random.default_rng([self.seed, epoch, 0]).shuffle(assignment)
        self.assignment = assignment
        self.queues = []
        self.ptrs = []
        for i, corpus in enumerate(self.corpora):
            n = self.limits[i]
            idx = np.arange(n, dtype=np.int64)
            np.random.default_rng([self.seed, epoch, i + 1]).shuffle(idx)
            self.queues.append(self.offsets[i] + idx)
            self.ptrs.append(0)
        self.pos = 0

    # ------------------------------------------------------------------
    def next_batch(self, batch_size: int, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Return ``[batch_size, block_size + 1]`` of token ids."""
        L = self.block_size
        if out is None:
            out = np.empty((batch_size, L + 1), dtype=np.int32)
        for j in range(batch_size):
            if self.pos >= self.steps_per_epoch:
                self._build_epoch(self.epoch + 1)
            ci = int(self.assignment[self.pos])
            self.pos += 1
            block = int(self.queues[ci][self.ptrs[ci]])
            self.ptrs[ci] += 1
            start = block * L
            out[j] = self.corpora[ci].tokens[start:start + L + 1]
        return out

    def next_val_batch(self, batch_size: int, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Fixed-order validation blocks (deterministic, no shuffling)."""
        L = self.block_size
        if not self._val_blocks:
            raise ValueError("no validation blocks (val_fraction=0?)")
        if out is None:
            out = np.empty((batch_size, L + 1), dtype=np.int32)
        for j in range(batch_size):
            ci, block = self._val_blocks[self._val_pos % len(self._val_blocks)]
            self._val_pos += 1
            start = block * L
            out[j] = self.corpora[ci].tokens[start:start + L + 1]
        return out

    def reset_val(self) -> None:
        self._val_pos = 0

    @property
    def num_val_blocks(self) -> int:
        return len(self._val_blocks)

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"epoch": int(self.epoch), "pos": int(self.pos), "seed": self.seed,
                "block_size": self.block_size, "split": self.split,
                "corpora": [c.name for c in self.corpora]}

    def load_state_dict(self, state: dict) -> None:
        if state.get("seed") != self.seed:
            raise ValueError(f"sampler seed mismatch: checkpoint {state.get('seed')} vs {self.seed}")
        if state.get("block_size") != self.block_size:
            raise ValueError("sampler block_size mismatch")
        saved_corpora = state.get("corpora")
        mine = [c.name for c in self.corpora]
        if saved_corpora is not None and list(saved_corpora) != mine:
            # happens when the run switches mixture (e.g. entering the decay
            # stage): the stored position is meaningless for a different corpus set
            raise ValueError(f"corpus set changed: checkpoint {saved_corpora} vs current {mine}")
        self._build_epoch(int(state["epoch"]))
        pos = int(state["pos"])
        if pos > self.steps_per_epoch:
            raise ValueError(f"position {pos} beyond epoch length {self.steps_per_epoch}")
        # fast-forward the per-corpus pointers deterministically
        consumed = np.bincount(self.assignment[:pos], minlength=len(self.corpora))
        self.ptrs = [int(p) + int(c) for p, c in zip(self.ptrs, consumed)]
        self.pos = pos


# ----------------------------------------------------------------------
@dataclass
class PrefetchedBatch:
    input_ids: torch.Tensor          # [B, L] int64 on device
    sampler_state: dict              # state *after* this batch was drawn
    seq_len: int


class BatchPrefetcher:
    """Gathers blocks on a background thread into pinned memory.

    The producer only does numpy gathers (CPU bound); the consumer issues the
    H2D copies on the current stream.  With ``slots`` >= 2 the gather for step
    ``k+1`` overlaps the GPU work of step ``k``.

    For causal LM pre-training the labels are the inputs themselves (the model
    shifts internally), so only ``input_ids`` is materialised.
    """

    def __init__(
        self,
        sampler: MixtureBlockSampler,
        batch_size: int,
        device: torch.device | str = "cuda",
        slots: int = 3,
        pin_memory: bool = True,
        validation: bool = False,
    ):
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.validation = validation
        self.pin_memory = bool(pin_memory) and self.device.type == "cuda"
        self.seq_len = sampler.block_size
        self._q: "_queue.Queue" = _queue.Queue(maxsize=max(1, slots))
        self._stop = threading.Event()
        self._exc: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._worker, name="wisp-prefetch", daemon=True)
        self._thread.start()

    # -- producer ------------------------------------------------------
    def _worker(self) -> None:
        try:
            while not self._stop.is_set():
                arr = self.sampler.next_batch(self.batch_size)          # [B, L+1] int32
                state = self.sampler.state_dict()
                cpu = torch.from_numpy(arr[:, :-1]).to(torch.int64)     # copy, releases numpy mem
                if self.pin_memory:
                    cpu = cpu.pin_memory()
                else:
                    cpu = cpu.contiguous()
                while not self._stop.is_set():
                    try:
                        self._q.put((cpu, state), timeout=0.5)
                        break
                    except _queue.Full:
                        continue
        except BaseException as exc:  # noqa: BLE001 - surfaced in the consumer
            self._exc = exc
            self._stop.set()

    # -- consumer ------------------------------------------------------
    def next(self) -> PrefetchedBatch:
        if self.validation:
            arr = self.sampler.next_val_batch(self.batch_size)
            cpu = torch.from_numpy(arr[:, :-1]).to(torch.int64)
            return PrefetchedBatch(
                cpu.to(self.device, non_blocking=self.pin_memory),
                self.sampler.state_dict(), self.seq_len,
            )
        while True:
            if self._exc is not None:
                raise RuntimeError("prefetch worker failed") from self._exc
            try:
                cpu, state = self._q.get(timeout=0.5)
                break
            except _queue.Empty:
                if not self._thread.is_alive() and self._q.empty():
                    raise RuntimeError("prefetch worker died") from self._exc
        return PrefetchedBatch(
            cpu.to(self.device, non_blocking=self.pin_memory), state, self.seq_len)

    def __iter__(self) -> "BatchPrefetcher":
        return self

    def __next__(self) -> PrefetchedBatch:
        return self.next()

    def close(self) -> None:
        self._stop.set()
        try:
            while True:
                self._q.get_nowait()
        except _queue.Empty:
            pass
        self._thread.join(timeout=5.0)

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


# ----------------------------------------------------------------------
def build_sampler(
    root: str,
    block_size: int,
    mixture: Optional[Dict[str, float]] = None,
    seed: int = 1234,
    val_fraction: float = 0.0,
    split: str = "train",
    in_memory: bool = True,
) -> Tuple[MixtureBlockSampler, Dict[str, TokenizedCorpus]]:
    """Discover corpora under ``root`` and build a sampler honouring ``mixture``."""
    corpora = discover_corpora(root, in_memory=in_memory)
    names = list(corpora.keys())
    if mixture:
        missing = [n for n in mixture if n not in corpora]
        if missing:
            raise KeyError(f"mixture names not found under {root}: {missing} (found {names})")
        names = [n for n in names if n in mixture]
    weights = [float(mixture[n]) for n in names] if mixture else [1.0] * len(names)
    sampler = MixtureBlockSampler(
        [corpora[n] for n in names], weights, block_size=block_size,
        seed=seed, val_fraction=val_fraction, split=split,
    )
    return sampler, {n: corpora[n] for n in names}
