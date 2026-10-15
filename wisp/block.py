"""Decoder layer + the alpha-scaled residual connection.

Residual layout (approved design, ReZero form):

* **Pre-LN**: ``g = x + alpha * f(RMSNorm(x))``
* ``alpha`` is a *per-channel* learnable vector, initialised to ``1e-4``.
  A small ``alpha`` means the 48-layer stack starts out as an almost exact
  identity map, so every sub-layer learns a small residual correction instead
  of having to preserve the signal from scratch.
* ``alpha`` is only created for layers inside the FFN-shared (LayerPair)
  region -- see ``WispConfig.alpha_mode``.  Layers outside that region use a
  plain residual (``g = x + f(RMSNorm(x))``).

``alpha_target="identity"`` implements the literal ``g = f(x) + alpha * x``
instead; it is kept for the ablation in ``experiments/alpha_ablation.md``,
which shows that form stalls (any ``alpha < 1`` decays the residual stream by
``alpha ** depth``).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .attention import GQAttention
from .config import WispConfig
from .ffn import SwiGLUFFN
from .norm import RMSNorm
from .rope import RotaryEmbedding


class ScaledResidual(nn.Module):
    """Learnable per-channel scaling of a residual connection.

    ``target="branch"`` (DEFAULT, ReZero / LayerScale)::

        g = x + alpha * f(x)        alpha init 1e-4

    ``target="identity"`` (literal first draft, measured to stall)::

        g = f(x) + alpha * x        alpha init 1e-4

    If ``enabled`` is False this degrades to a plain ``x + f(x)`` add and
    registers no parameters at all.
    """

    def __init__(
        self,
        hidden_size: int,
        enabled: bool,
        alpha_init: float = 1e-4,
        target: str = "identity",
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.target = target
        if target not in ("identity", "branch"):
            raise ValueError(f"unknown alpha target: {target}")
        if enabled:
            self.alpha = nn.Parameter(torch.full((hidden_size,), float(alpha_init)))
        else:
            self.register_parameter("alpha", None)

    def forward(self, x: torch.Tensor, f_x: torch.Tensor) -> torch.Tensor:
        if self.alpha is None:
            return x + f_x
        if self.target == "identity":
            return f_x + self.alpha * x
        return x + self.alpha * f_x

    def extra_repr(self) -> str:
        return f"enabled={self.enabled}, target={self.target}"


class DecoderLayer(nn.Module):
    """One transformer layer: GQA sub-layer + FFN sub-layer.

    ``self.ffn`` is *not* created here; :class:`~wisp.model.WispModel` injects
    either a shared FFN (LayerPair region) or a dedicated one.
    """

    def __init__(self, config: WispConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GQAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn: SwiGLUFFN = None  # type: ignore[assignment]

        use_alpha = config.layer_uses_alpha(layer_idx)
        self.residual_attn = ScaledResidual(
            config.hidden_size, use_alpha, config.alpha_init, config.alpha_target
        )
        self.residual_ffn = ScaledResidual(
            config.hidden_size, use_alpha, config.alpha_init, config.alpha_target
        )
        self.uses_shared_ffn = False

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary: RotaryEmbedding,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        attn_out, present = self.self_attn(
            h,
            rotary,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = self.residual_attn(residual, attn_out)

        residual = hidden_states
        h = self.post_attention_layernorm(hidden_states)
        ffn_out = self.ffn(h)
        hidden_states = self.residual_ffn(residual, ffn_out)
        return hidden_states, present
