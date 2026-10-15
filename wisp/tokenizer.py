"""Local tokenizer wrapper for ``Morton-Li/QiYuanTokenizer-Small``.

The tokenizer files are vendored into ``<repo>/tokenizer`` by
``scripts/download_tokenizer.py`` so that training/inference needs no network.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Union

DEFAULT_TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tokenizer")


class WispTokenizer:
    """Thin wrapper around the fast (Rust) Unigram tokenizer."""

    def __init__(self, tokenizer_dir: str = DEFAULT_TOKENIZER_DIR, add_bos: bool = True, add_eos: bool = False):
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
        self.bos_token = cfg["bos_token"]
        self.eos_token = cfg["eos_token"]
        self.pad_token = cfg["pad_token"]
        self.unk_token = cfg["unk_token"]
        self.mask_token = cfg["mask_token"]
        self.extra_special_tokens = cfg.get("extra_special_tokens", [])
        self.special_tokens = [
            self.unk_token,
            self.bos_token,
            self.eos_token,
            self.pad_token,
            self.mask_token,
        ] + list(self.extra_special_tokens)

        self.bos_token_id = self._tok.token_to_id(self.bos_token)
        self.eos_token_id = self._tok.token_to_id(self.eos_token)
        self.pad_token_id = self._tok.token_to_id(self.pad_token)
        self.unk_token_id = self._tok.token_to_id(self.unk_token)
        self.mask_token_id = self._tok.token_to_id(self.mask_token)
        self.special_token_ids = [self._tok.token_to_id(t) for t in self.special_tokens if self._tok.token_to_id(t) is not None]

    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        """Size of the embedding matrix the LM must allocate."""
        return self._tok.get_vocab_size(with_added_tokens=True)

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str, add_special_tokens: Optional[bool] = None) -> List[int]:
        if add_special_tokens is None:
            add_special_tokens = self.add_bos or self.add_eos
        ids = self._tok.encode(text, add_special_tokens=False).ids
        if add_special_tokens and self.add_bos:
            ids = [self.bos_token_id] + ids
        if add_special_tokens and self.add_eos:
            ids = ids + [self.eos_token_id]
        return ids

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
            "vocab_size": self.vocab_size,
        }

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "tokenizer_info.json"), "w", encoding="utf-8") as fh:
            json.dump(self.special_tokens_map(), fh, indent=2)
