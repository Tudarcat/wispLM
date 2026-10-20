"""Local tokenizer wrapper for ``Morton-Li/QiYuanTokenizer-Small``.

The tokenizer files are vendored into ``<repo>/tokenizer`` by
``scripts/download_tokenizer.py`` so that training/inference needs no network.

Wispv1 renames the tokenizer's 15 reserved slots (ids 0..14) -- see
``scripts/setup_special_tokens.py`` and ``docs/special_tokens.md``::

    id  0..3   <|unk|> <|bos|> <|eos|> <|pad|>          (unchanged)
    id  4      <|end_of_turn|>
    id  5..7   <|user|> <|assistant|> <|system|>
    id  8..11  <|summarize|> <|rewrite|> <|polish|> <|expand|>
    id 12..13  <|think|> <|/think|>
    id 14      <|placeholder_9|>                        (spare)

Ids and ``vocab_size`` are unchanged, so this is transparent to the model.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Union

DEFAULT_TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tokenizer")

# names we expect to find at ids 4..13 after setup_special_tokens.py
ROLE_TOKENS = {"end_of_turn": 4, "user": 5, "assistant": 6, "system": 7}
TASK_TOKENS = {"summarize": 8, "rewrite": 9, "polish": 10, "expand": 11}
REASONING_TOKENS = {"think": 12, "end_think": 13}


class WispTokenizer:
    """Thin wrapper around the fast (Rust) Unigram tokenizer."""

    def __init__(self, tokenizer_dir: str = DEFAULT_TOKENIZER_DIR, add_bos: bool = True,
                 add_eos: bool = False, strict_special_tokens: bool = False):
        from tokenizers import Tokenizer  # local import: keeps the core model dep-free

        self.dir = tokenizer_dir
        self.path = os.path.join(tokenizer_dir, "tokenizer.json")
        if not os.path.exists(self.path):
            raise FileNotFoundError(
                f"{self.path} not found. Run `python scripts/download_tokenizer.py` first."
            )
        self._tok = Tokenizer.from_file(self.path)
        self._tok.no_padding()
        self._tok.no_truncation()
        self.add_bos = add_bos
        self.add_eos = add_eos

        with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        self.config = cfg
        self.bos_token = cfg["bos_token"]
        self.eos_token = cfg["eos_token"]
        self.pad_token = cfg["pad_token"]
        self.unk_token = cfg["unk_token"]
        self.mask_token = cfg.get("mask_token")  # None after setup_special_tokens.py
        self.extra_special_tokens = cfg.get("extra_special_tokens", [])
        self.reserved_tokens: Dict[str, str] = cfg.get("wispv1_reserved_tokens", {})

        self.bos_token_id = self._tok.token_to_id(self.bos_token)
        self.eos_token_id = self._tok.token_to_id(self.eos_token)
        self.pad_token_id = self._tok.token_to_id(self.pad_token)
        self.unk_token_id = self._tok.token_to_id(self.unk_token)
        self.mask_token_id = self._tok.token_to_id(self.mask_token) if self.mask_token else None

        # every reserved slot, i.e. ids 0..14
        self.special_token_ids = [i for i in range(15) if self._tok.id_to_token(i) is not None]
        self.special_tokens = [self._tok.id_to_token(i) for i in self.special_token_ids]

        # named accessors (only populated if the tokenizer was patched)
        for group, table in (("role_token_ids", ROLE_TOKENS),
                             ("task_token_ids", TASK_TOKENS),
                             ("reasoning_token_ids", REASONING_TOKENS)):
            ids = {}
            for name, tid in table.items():
                tok_str = self._tok.id_to_token(tid)
                if tok_str is not None and tok_str.startswith("<|") and "placeholder" not in tok_str:
                    ids[name] = tid
            setattr(self, group, ids)
        self.end_of_turn_token_id = self.role_token_ids.get("end_of_turn")
        self.user_token_id = self.role_token_ids.get("user")
        self.assistant_token_id = self.role_token_ids.get("assistant")
        self.system_token_id = self.role_token_ids.get("system")
        self.cot_token_ids = self.reasoning_token_ids
        # generation should stop on both end-of-document and end-of-turn
        stops = {self.eos_token_id}
        if self.end_of_turn_token_id is not None:
            stops.add(self.end_of_turn_token_id)
        self.stop_token_ids = sorted(stops)
        if strict_special_tokens and self.end_of_turn_token_id is None:
            raise RuntimeError(
                "the vendored tokenizer still has <|placeholder_*> ids; run "
                "`python scripts/setup_special_tokens.py` first"
            )

    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        """Size of the embedding matrix the LM must allocate."""
        return self._tok.get_vocab_size(with_added_tokens=True)

    def __len__(self) -> int:
        return self.vocab_size

    @property
    def is_wispv1_patched(self) -> bool:
        return self.end_of_turn_token_id is not None

    # ------------------------------------------------------------------
    def encode(self, text: str, add_special_tokens: Optional[bool] = None) -> List[int]:
        if add_special_tokens is None:
            add_special_tokens = self.add_bos or self.add_eos
        ids = self._tok.encode(text, add_special_tokens=False).ids
        if add_special_tokens and self.add_bos:
            ids = [self.bos_token_id] + ids
        if add_special_tokens and self.add_eos:
            ids = ids + [self.eos_token_id]
        return ids

    def reserved_name(self, token_id: int) -> str:
        """The literal string of a reserved slot (e.g. ``"<|user|>"``)."""
        name = self._tok.id_to_token(int(token_id))
        if name is None:
            raise KeyError(f"{token_id} is not a token id")
        return name

    def decode(self, ids: Union[Sequence[int], "object"], skip_special_tokens: bool = True) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()  # torch.Tensor / numpy array
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def batch_encode(self, texts: Sequence[str], add_special_tokens: Optional[bool] = None) -> List[List[int]]:
        return [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]

    # ---- training helpers -------------------------------------------
    def encode_to_fixed_length(self, text: str, block_size: int, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        ids = self._tok.encode(text, add_special_tokens=False).ids
        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        if len(ids) > block_size:
            ids = ids[:block_size]
        return ids + [self.pad_token_id] * (block_size - len(ids))

    def special_tokens_map(self) -> dict:
        return {
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "end_of_turn_token_id": self.end_of_turn_token_id,
            "role_token_ids": self.role_token_ids,
            "task_token_ids": self.task_token_ids,
            "reasoning_token_ids": self.reasoning_token_ids,
            "stop_token_ids": self.stop_token_ids,
            "vocab_size": self.vocab_size,
            "patched": self.is_wispv1_patched,
        }

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "tokenizer_info.json"), "w", encoding="utf-8") as fh:
            json.dump(self.special_tokens_map(), fh, indent=2)
