"""Wispv1 model configuration.

Wispv1 is a small (~0.25B) mono-lingual/bilingual decoder-only LM built to
maximise *depth* under a fixed parameter budget:

* **GQA** attention (per-layer, never shared).
* **SwiGLU** FFN, pair-wise *weight shared* over the first ``ffn_share_layers``
  layers (two consecutive layers form a ``LayerPair`` and share one FFN module).
* **Pre-LN** residual layout with a learnable per-residual **alpha** vector:
  ``g = x + alpha * f(RMSNorm(x))`` (ReZero form), ``alpha`` init ``1e-4``.
  The literal ``g = f(x) + alpha * x`` reading is still available via
  ``alpha_target="identity"`` but was measured to stall training.
* Embedding / LM-head **weight tying**.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional


@dataclass
class WispConfig:
    # ---- vocabulary / positions -------------------------------------
    vocab_size: int = 24000
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    # e.g. {"type": "linear", "factor": 4.0} or {"type": "dynamic", "factor": 8.0}
    rope_scaling: Optional[Dict[str, Any]] = None

    # ---- body --------------------------------------------------------
    hidden_size: int = 768
    num_hidden_layers: int = 48
    num_attention_heads: int = 12
    num_key_value_heads: int = 3
    head_dim: int = 64
    intermediate_size: int = 2048
    hidden_act: str = "silu"  # SwiGLU gate activation

    # ---- norm / init -------------------------------------------------
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = True
    # the diagram has no extra norm box after the stack, but Pre-LN needs one
    final_norm: bool = True

    # ---- FFN weight sharing -----------------------------------------
    # First ``ffn_share_layers`` layers are grouped into pairs; the two layers
    # of a pair share *one whole FFN module* (gate + up + down).
    ffn_share_layers: int = 24

    # ---- residual alpha ---------------------------------------------
    # g = x + alpha * f(x)   (alpha is a per-channel vector, init 1e-4)
    alpha_init: float = 1e-4
    # "shared": only layers inside the FFN-shared region get alpha (spec)
    # "all"   : every decoder layer gets alpha
    # "none"  : plain residual everywhere (baseline / ablation)
    alpha_mode: str = "shared"
    # "branch"  : g = x + alpha * f(x)   <- DEFAULT (ReZero / LayerScale).
    #             Small alpha => the stack starts near the identity map, which
    #             is what makes a 48-layer shared-FFN stack trainable.
    # "identity": g = f(x) + alpha * x   <- the literal first draft. Measured
    #             to stall: any alpha < 1 decays the residual stream by
    #             alpha**depth. See experiments/alpha_ablation.md.
    alpha_target: str = "branch"

    # ---- misc --------------------------------------------------------
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 3
    torch_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if self.head_dim is None:
            if self.hidden_size % self.num_attention_heads != 0:
                raise ValueError("hidden_size must be divisible by num_attention_heads")
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be a multiple of "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )
        if self.ffn_share_layers < 0 or self.ffn_share_layers > self.num_hidden_layers:
            raise ValueError("ffn_share_layers must be within [0, num_hidden_layers]")
        if self.ffn_share_layers % 2 != 0:
            raise ValueError("ffn_share_layers must be even (layers are shared two-by-two)")
        if self.alpha_mode not in ("shared", "all", "none"):
            raise ValueError(f"unknown alpha_mode: {self.alpha_mode}")
        if self.alpha_target not in ("identity", "branch"):
            raise ValueError(f"unknown alpha_target: {self.alpha_target}")

    # ------------------------------------------------------------------
    @property
    def num_shared_pairs(self) -> int:
        """Number of distinct shared FFN modules in the LayerPair region."""
        return self.ffn_share_layers // 2

    @property
    def num_independent_layers(self) -> int:
        return self.num_hidden_layers - self.ffn_share_layers

    @property
    def num_ffn_modules(self) -> int:
        return self.num_shared_pairs + self.num_independent_layers

    @property
    def num_query_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def layer_uses_alpha(self, layer_idx: int) -> bool:
        if self.alpha_mode == "none":
            return False
        if self.alpha_mode == "all":
            return True
        return layer_idx < self.ffn_share_layers

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WispConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "WispConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
