
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

"""Wispv1 - the model itself.

    Input -> Emb -> [LayerPair x n_shared_pairs] -> [DecoderLayer x n_rest]
          -> RMSNorm -> Linear(tied) -> Softmax -> Output prob

* the first ``config.ffn_share_layers`` layers are grouped two-by-two into
  ``LayerPair``s; both layers of a pair share one :class:`~wisp.ffn.SwiGLUFFN`;
* those same layers carry a per-channel ``alpha`` on every residual identity
  path (``g = f(x) + alpha * x``, ``alpha`` init ``1e-4``);
* attention (GQA) is never shared;
* the input embedding and the output projection share one weight matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import DecoderLayer
from .config import WispConfig
from .ffn import SwiGLUFFN
from .norm import RMSNorm
from .rope import RotaryEmbedding


@dataclass
class CausalLMOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None


class WispModel(nn.Module):
    """The bare decoder stack (no LM head)."""

    def __init__(self, config: WispConfig) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=self.padding_idx
        )
        self.rotary = RotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling=config.rope_scaling,
        )

        # ---- FFN modules: one per shared pair + one per free layer ----
        # They are created here but registered *through* the decoder layers,
        # so a shared FFN is stored under the first layer of its pair
        # (`model.layers.0.ffn.*`) and appears exactly once.
        ffn_share_layers = config.ffn_share_layers
        shared_ffns = [SwiGLUFFN(config) for _ in range(ffn_share_layers // 2)]
        free_ffns = [
            SwiGLUFFN(config) for _ in range(config.num_hidden_layers - ffn_share_layers)
        ]

        # ---- decoder layers ------------------------------------------
        layers = []
        for idx in range(config.num_hidden_layers):
            layer = DecoderLayer(config, idx)
            if idx < ffn_share_layers:
                layer.ffn = shared_ffns[idx // 2]
                layer.uses_shared_ffn = True
            else:
                layer.ffn = free_ffns[idx - ffn_share_layers]
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if config.final_norm else None
        self.gradient_checkpointing = False
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    @property
    def shared_ffns(self) -> List[Any]:
        """The distinct FFN modules used by the LayerPair region."""
        return [
            self.layers[i].ffn for i in range(0, self.config.ffn_share_layers, 2)
        ]

    @property
    def free_ffns(self) -> List[Any]:
        """The FFN modules of the layers outside the shared region."""
        return [
            self.layers[i].ffn
            for i in range(self.config.ffn_share_layers, self.config.num_hidden_layers)
        ]

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def enable_gradient_checkpointing(self, on: bool = True) -> None:
        self.gradient_checkpointing = on

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        B, T = input_ids.shape
        if position_ids is None:
            past_len = 0 if past_key_values is None else past_key_values[0][0].shape[2]
            position_ids = torch.arange(past_len, past_len + T, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(B, T)

        hidden_states = self.embed_tokens(input_ids)
        presents: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[i]
            if self.gradient_checkpointing and self.training:
                def custom_forward(hidden_states, layer=layer, past=past):
                    return layer(
                        hidden_states,
                        self.rotary,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past,
                        use_cache=False,
                    )

                hidden_states, _ = torch.utils.checkpoint.checkpoint(
                    custom_forward, hidden_states, use_reentrant=False
                )
            else:
                hidden_states, present = layer(
                    hidden_states,
                    self.rotary,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past,
                    use_cache=use_cache,
                )
                if use_cache:
                    presents.append(present)

        if self.norm is not None:
            hidden_states = self.norm(hidden_states)

        return hidden_states, presents


class WispForCausalLM(nn.Module):
    def __init__(self, config: WispConfig) -> None:
        super().__init__()
        self.config = config
        self.model = WispModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.post_init()

    # ------------------------------------------------------------------
    def post_init(self) -> None:
        # Only touch the LM head when it owns its own weight: with tying it is
        # the *same tensor* as the embedding, and re-initialising it here would
        # (a) re-randomise the embedding a second time and (b) destroy the
        # zeroed padding row.
        if not self.config.tie_word_embeddings:
            self.lm_head.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def num_parameters(self, unique: bool = True) -> int:
        seen = set()
        total = 0
        for p in self.parameters():
            if unique:
                if id(p) in seen:
                    continue
                seen.add(id(p))
            total += p.numel()
        return total

    def parameter_breakdown(self) -> Dict[str, float]:
        """Parameter count in millions, grouped by role (each tensor counted once)."""
        counted: set = set()
        groups: Dict[str, int] = {}

        def add(name: str, p: Optional[torch.Tensor]) -> None:
            if p is None or id(p) in counted:
                return
            counted.add(id(p))
            groups[name] = groups.get(name, 0) + p.numel()

        add("embedding", self.model.embed_tokens.weight)
        shared_ids = {id(m) for m in self.model.shared_ffns}
        for layer in self.model.layers:
            for p in layer.self_attn.parameters():
                add("attention", p)
            add("norm", layer.input_layernorm.weight)
            add("norm", layer.post_attention_layernorm.weight)
            add("alpha", layer.residual_attn.alpha)
            add("alpha", layer.residual_ffn.alpha)
        for ffn in self.model.shared_ffns:
            for p in ffn.parameters():
                add("ffn_shared", p)
        for ffn in self.model.free_ffns:
            for p in ffn.parameters():
                add("ffn_free", p)
        if self.model.norm is not None:
            add("norm", self.model.norm.weight)
        if not self.config.tie_word_embeddings:
            add("lm_head", self.lm_head.weight)
        return {k: v / 1e6 for k, v in groups.items()}

    def sharing_summary(self) -> Dict[str, Any]:
        return {
            "hidden_size": self.config.hidden_size,
            "num_hidden_layers": self.config.num_hidden_layers,
            "ffn_share_layers": self.config.ffn_share_layers,
            "num_layer_pairs": self.config.num_shared_pairs,
            "num_free_layers": self.config.num_independent_layers,
            "num_ffn_modules": len(self.model.shared_ffns) + len(self.model.free_ffns),
            "num_attention_modules": len(self.model.layers),
            "alpha_layers": [i for i in range(self.config.num_hidden_layers) if self.config.layer_uses_alpha(i)],
            "tie_word_embeddings": self.config.tie_word_embeddings,
            "unique_parameters": self.num_parameters(unique=True),
        }

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        hidden_states, presents = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(loss=loss, logits=logits, past_key_values=presents)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        eos_token_id = self.config.eos_token_id if eos_token_id is None else eos_token_id
        pad_token_id = self.config.pad_token_id if pad_token_id is None else pad_token_id
        past = None
        cur = input_ids
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
        for _ in range(max_new_tokens):
            out = self.forward(cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :].float()
            if temperature <= 0:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p is not None:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                    probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    remove = probs - F.softmax(sorted_logits, dim=-1) >= top_p
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            next_tok = torch.where(finished[:, None], torch.full_like(next_tok, pad_token_id), next_tok)
            cur = next_tok
            input_ids = torch.cat([input_ids, next_tok], dim=-1)
            finished |= next_tok.squeeze(-1) == eos_token_id
            if bool(finished.all()):
                break
        return input_ids

    # ------------------------------------------------------------------
    def canonical_state_dict(self) -> Dict[str, torch.Tensor]:
        """State dict with every *aliased* tensor stored exactly once.

        The two layers of a ``LayerPair`` are the same module object and tied
        embeddings are the same parameter, so ``state_dict()`` would repeat
        them once per path (12 extra FFN copies here -- ~226 MB in fp32).
        ``named_parameters()`` de-duplicates by identity; its keys are the
        canonical names we keep.
        """
        canonical = set(dict(self.named_parameters()).keys())
        canonical |= set(dict(self.named_buffers()).keys())
        return {k: v for k, v in self.state_dict().items() if k in canonical}

    def save_pretrained(self, save_dir: str, filename: str = "model.pt") -> None:
        import os

        os.makedirs(save_dir, exist_ok=True)
        self.config.save(os.path.join(save_dir, "config.json"))
        torch.save(
            {"model": self.canonical_state_dict(), "format": "wispv1", "version": 1},
            os.path.join(save_dir, filename),
        )

    @classmethod
    def from_pretrained(cls, save_dir: str, filename: str = "model.pt", map_location="cpu") -> "WispForCausalLM":
        import os

        config = WispConfig.load(os.path.join(save_dir, "config.json"))
        model = cls(config)
        blob = torch.load(os.path.join(save_dir, filename), map_location=map_location, weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        canonical = set(dict(model.named_parameters()).keys())
        canonical |= set(dict(model.named_buffers()).keys())
        missing = canonical - set(state.keys())
        unexpected = set(state.keys()) - canonical
        # a tied lm_head may legitimately be present in foreign checkpoints
        if config.tie_word_embeddings:
            unexpected.discard("lm_head.weight")
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint mismatch: missing={sorted(missing)[:5]} unexpected={sorted(unexpected)[:5]}"
            )
        model.load_state_dict(state, strict=False)
        return model


def count_parameters(model: nn.Module, unique: bool = True) -> int:
    seen = set()
    total = 0
    for p in model.parameters():
        if unique:
            if id(p) in seen:
                continue
            seen.add(id(p))
        total += p.numel()
    return total


def estimate_parameter_breakdown(config: WispConfig) -> Dict[str, float]:
    """Analytic parameter count (millions), useful for config search."""
    d = config.hidden_size
    hd = config.head_dim
    L = config.num_hidden_layers
    V = config.vocab_size
    S = config.ffn_share_layers
    I = config.intermediate_size
    attn = L * (d * config.num_attention_heads * hd + 2 * d * config.num_key_value_heads * hd + config.num_attention_heads * hd * d)
    ffn_per = 3 * d * I
    ffn = (S // 2 + (L - S)) * ffn_per
    emb = V * d
    out = {
        "embedding": emb / 1e6,
        "attention": attn / 1e6,
        "ffn_shared": (S // 2) * ffn_per / 1e6,
        "ffn_free": (L - S) * ffn_per / 1e6,
        "total": (emb + attn + ffn + (0 if config.tie_word_embeddings else emb)) / 1e6,
    }
    if config.alpha_mode != "none":
        n_alpha = sum(1 for i in range(L) if config.layer_uses_alpha(i)) * 2 * d
        out["alpha"] = n_alpha / 1e6
        out["total"] += n_alpha / 1e6
    out["norms"] = (L * 2 * d + (d if config.final_norm else 0)) / 1e6
    out["total"] += out["norms"]
    return out
