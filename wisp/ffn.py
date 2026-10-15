"""SwiGLU feed-forward network (Shazeer, 2020).

``FFN(x) = W_down( silu(W_gate x) * (W_up x) )``

Inside a ``LayerPair`` the *whole* module is shared between the two layers
(per the approved design), which is why it lives in its own class: the model
attaches the very same instance to both layers of a pair.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import WispConfig

_ACT = {"silu": F.silu, "swish": F.silu, "gelu": F.gelu, "relu": F.relu}


class SwiGLUFFN(nn.Module):
    def __init__(self, config: WispConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_name = config.hidden_act
        self.act = _ACT[config.hidden_act]
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
