"""Tokenizer tests (skipped when the vendored tokenizer is missing)."""

from __future__ import annotations

import os

import pytest

from wisp.tokenizer import DEFAULT_TOKENIZER_DIR, WispTokenizer

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DEFAULT_TOKENIZER_DIR, "tokenizer.json")),
    reason="tokenizer not downloaded (run scripts/download_tokenizer.py)",
)


@pytest.fixture(scope="module")
def tok() -> WispTokenizer:
    return WispTokenizer()


def test_vocab_size(tok):
    # 24,000 Unigram entries that already contain the 15 special tokens
    assert tok.vocab_size == 24000


def test_special_token_ids(tok):
    assert tok.bos_token_id is not None
    assert tok.eos_token_id is not None
    assert tok.pad_token_id is not None
    assert tok.unk_token_id is not None
    assert len(set([tok.bos_token_id, tok.eos_token_id, tok.pad_token_id, tok.unk_token_id])) == 4
    assert len(tok.special_token_ids) == 15  # 5 core + 10 placeholders


def test_roundtrip(tok):
    import unicodedata

    # the tokenizer applies an NFKC normalizer, so compare normalised text
    for text in ["你好，世界！", "Hello world, 这是 Wispv1。", "12345 + abc = ?"]:
        ids = tok.encode(text)
        assert all(0 <= i < tok.vocab_size for i in ids)
        norm = lambda s: unicodedata.normalize("NFKC", s).replace(" ", "")  # noqa: E731
        assert norm(text) in norm(tok.decode(ids))


def test_chinese_compression(tok):
    ids = tok.encode("人工智能正在改变世界", add_special_tokens=False)
    assert len(ids) <= 12  # a 24k bilingual unigram vocab should not explode hanzi


def test_bos_eos(tok):
    ids = tok.encode("hi")
    assert ids[0] == tok.bos_token_id
    ids = tok.encode("hi", add_special_tokens=False)
    assert tok.bos_token_id not in ids


def test_fixed_length_padding(tok):
    ids = tok.encode_to_fixed_length("短文本", block_size=32)
    assert len(ids) == 32
    assert ids[-1] == tok.pad_token_id


def test_config_ids_match_tokenizer(tok):
    from wisp.config import WispConfig

    cfg = WispConfig()
    assert cfg.vocab_size == tok.vocab_size
    assert (cfg.bos_token_id, cfg.eos_token_id, cfg.pad_token_id) == (
        tok.bos_token_id,
        tok.eos_token_id,
        tok.pad_token_id,
    )
