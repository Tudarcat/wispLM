"""Rotary position embedding (RoPE), Llama-style (rotate the two halves of head_dim).

Also supports the two interpolation schemes that are most useful for the
length-extrapolation problem listed in ``Wispv1.md``:

* ``linear``  -- divide positions by ``factor`` (PI, Chen et al. 2023)
* ``dynamic`` -- NTK-aware scaling, the base is stretched as
  ``theta * factor ** (dim / (dim - 2))``
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding:
    """Pre-computes cos/sin tables, optionally rescaled for longer contexts."""

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10000.0,
        scaling: Optional[dict] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling = scaling or None
        self._cached_seq_len = 0
        self._cos: Optional[torch.Tensor] = None
        self._sin: Optional[torch.Tensor] = None
        self.dtype = dtype
        self._device = device or torch.device("cpu")
        self._build(max_position_embeddings)

    # ------------------------------------------------------------------
    def _effective_base_and_scale(self) -> Tuple[float, float]:
        """Return (rope_base, position_scale)."""
        base = float(self.base)
        scale = 1.0
        cfg = self.scaling
        if not cfg:
            return base, scale
        kind = str(cfg.get("type", "linear")).lower()
        factor = float(cfg.get("factor", 1.0))
        if kind == "linear":
            scale = 1.0 / factor
        elif kind == "dynamic":
            # NTK-aware: stretch the frequencies instead of the positions.
            base = base * factor ** (self.head_dim / (self.head_dim - 2))
        elif kind in ("yarn", "ntk"):
            base = base * factor ** (self.head_dim / (self.head_dim - 2))
        elif kind == "none":
            pass
        else:
            raise ValueError(f"unknown rope scaling type: {kind}")
        return base, scale

    def _build(self, seq_len: int) -> None:
        base, scale = self._effective_base_and_scale()
        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        t = torch.arange(seq_len, dtype=torch.float32) * scale
        freqs = torch.outer(t, inv_freq)  # [seq, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq, head_dim]
        self._cos = emb.cos().to(self.dtype)
        self._sin = emb.sin().to(self.dtype)
        self._cached_seq_len = seq_len

    def _ensure(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if seq_len > self._cached_seq_len:
            new_len = max(seq_len, int(self._cached_seq_len * 1.5) + 1)
            self._build(new_len)
        if self._cos.device != device or self._cos.dtype != dtype:
            self._cos = self._cos.to(device=device, dtype=dtype)
            self._sin = self._sin.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``q``/``k``: [B, H, T, D]."""
        B, H, T, D = q.shape
        self._ensure(T, q.device, q.dtype)
        if position_ids is None:
            cos = self._cos[:T].view(1, 1, T, D)
            sin = self._sin[:T].view(1, 1, T, D)
        else:
            cos = self._cos[position_ids].view(B, 1, T, D)
            sin = self._sin[position_ids].view(B, 1, T, D)
        q_out = q * cos + _rotate_half(q) * sin
        k_out = k * cos + _rotate_half(k) * sin
        return q_out.to(q.dtype), k_out.to(k.dtype)

    def state_dict(self) -> dict:
        return {
            "head_dim": self.head_dim,
            "max_position_embeddings": self.max_position_embeddings,
            "base": self.base,
            "scaling": self.scaling,
        }
