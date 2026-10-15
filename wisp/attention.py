"""Grouped-Query Attention (Ainslie et al., 2023) with RoPE.

The diagram shows RoPE feeding both Q and K of the ``GQA Scale dot attention``
block; K/V projections are compressed to ``num_key_value_heads`` groups.
Attention parameters are **never** shared between the two layers of a
``LayerPair``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import WispConfig
from .rope import RotaryEmbedding


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, kv, T, D] -> [B, kv*n_rep, T, D]"""
    if n_rep == 1:
        return x
    B, kv, T, D = x.shape
    return x[:, :, None].expand(B, kv, n_rep, T, D).reshape(B, kv * n_rep, T, D)


class GQAttention(nn.Module):
    def __init__(self, config: WispConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

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
        B, T, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = rotary(q, k, position_ids=position_ids)

        past_len = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_len = past_k.shape[2]
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present = (k, v) if use_cache else None

        k = repeat_kv(k, self.num_groups)
        v = repeat_kv(v, self.num_groups)

        # ---- mask handling -------------------------------------------
        # `attention_mask` is an optional [B, T] *padding* mask (1 = keep).
        # With a KV cache the causal diagonal must be offset by `past_len`.
        attn_mask = None
        is_causal = False
        if attention_mask is None:
            if past_len == 0:
                is_causal = True
            else:
                keep = torch.ones(T, past_len + T, dtype=torch.bool, device=q.device).tril(
                    diagonal=past_len
                )
                attn_mask = torch.zeros(keep.shape, dtype=q.dtype, device=q.device).masked_fill_(
                    ~keep, float("-inf")
                )
        else:
            key_len = past_len + T
            causal = torch.ones(T, key_len, dtype=torch.bool, device=q.device).tril(
                diagonal=past_len
            )
            pad = attention_mask[:, None, None, :].to(torch.bool)  # [B,1,1,key_len]
            keep = causal[None, None] & pad
            attn_mask = torch.zeros_like(keep, dtype=q.dtype).masked_fill_(~keep, float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal, dropout_p=self.dropout if self.training else 0.0,
            scale=self.scaling,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out), present
