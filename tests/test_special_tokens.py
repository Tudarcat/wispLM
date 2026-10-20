"""Tests for the Wispv1 reserved special tokens and the QA chat template."""

from __future__ import annotations

import os

import pytest

from wisp.chat import (
    format_conversation,
    format_sft_sample,
    parse_qa_document,
    prompt_prefix,
)
from wisp.tokenizer import DEFAULT_TOKENIZER_DIR, WispTokenizer

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DEFAULT_TOKENIZER_DIR, "tokenizer.json")),
    reason="tokenizer not downloaded (run scripts/download_tokenizer.py)",
)

ZH_DOC = (
    "淋球菌在多粘菌素B血液琼脂上培养24～48小时后可形成圆形、凸起、湿润的菌落。"
    "问题： 淋球菌培养后菌落有什么特征？ 答案： 圆形、凸起、湿润、光滑、半透明或灰白色的菌落。"
    "问题： 继续培养会怎样？ 答案： 菌落体积增大，表面粗糙，边缘皱缩。"
)
EN_DOC = (
    "Biomarker research commonly employs tissue microarrays to study many samples at once. "
    "Question: What is a tissue microarray used for? Answer: Studying many samples in one block. "
    "Question: How are cores transferred? Answer: From a donor block into a recipient block."
)


@pytest.fixture(scope="module")
def tok() -> WispTokenizer:
    return WispTokenizer()


# ------------------------------------------------------------ token slots
def test_tokenizer_is_patched(tok):
    assert tok.is_wispv1_patched, "run scripts/setup_special_tokens.py"
    assert tok.vocab_size == 24000, "the rename must not change the vocabulary size"


def test_reserved_ids_are_where_we_expect(tok):
    assert tok.unk_token_id == 0 and tok.bos_token_id == 1
    assert tok.eos_token_id == 2 and tok.pad_token_id == 3
    assert tok.end_of_turn_token_id == 4
    assert tok.role_token_ids == {"end_of_turn": 4, "user": 5, "assistant": 6, "system": 7}
    assert tok.task_token_ids == {"summarize": 8, "rewrite": 9, "polish": 10, "expand": 11}
    assert tok.reasoning_token_ids == {"think": 12, "end_think": 13}
    assert tok.mask_token_id is None, "the MLM mask slot was repurposed"
    assert tok.stop_token_ids == [2, 4]


def test_every_special_token_is_exactly_one_token(tok):
    for name, tid in list(tok.role_token_ids.items()) + list(tok.task_token_ids.items()) + \
            list(tok.reasoning_token_ids.items()):
        s = tok.reserved_name(tid)
        assert tok.encode(s, add_special_tokens=False) == [tid], (name, s)


def test_special_tokens_are_skipped_when_decoding(tok):
    s = "<|bos|><|user|>hi<|end_of_turn|><|assistant|>ok<|end_of_turn|><|eos|>"
    ids = tok.encode(s, add_special_tokens=False)
    assert ids == [1, 5, tok.encode("hi", add_special_tokens=False)[0], 4, 6,
                   tok.encode("ok", add_special_tokens=False)[0], 4, 2]
    assert "|" not in tok.decode(ids, skip_special_tokens=True)
    assert tok.decode(ids, skip_special_tokens=True) == "hiok"


def test_ordinary_text_is_untouched(tok):
    """The rename must not move a single id for normal text.

    These ids were captured from the *pristine* QiYuanTokenizer-Small (verified by
    tokenizing 240 real documents before/after the rename: zero differences).
    """
    expect = {
        "你好，世界！": [239, 266, 15, 541, 207],
        "人工智能正在改变世界": [11598, 1217, 2822, 541],
        "骨折术后康复": [2969, 2494, 5265, 91, 12943],
    }
    for text, want in expect.items():
        assert tok.encode(text, add_special_tokens=False) == want


def test_spare_slot_still_exists(tok):
    # id 14 is deliberately left over so it can be renamed later for free
    assert tok._tok.id_to_token(14) is not None
    assert tok._tok.token_to_id("<|placeholder_9|>") == 14


# ------------------------------------------------------------- the parser
def test_parse_chinese_document():
    doc = parse_qa_document(ZH_DOC)
    assert doc.language == "zh"
    assert len(doc.pairs) == 2
    assert doc.pairs[0].question.startswith("淋球菌培养后菌落")
    assert doc.pairs[0].answer.startswith("圆形、凸起")
    assert doc.passage.startswith("淋球菌在多粘菌素B")


def test_parse_english_document():
    doc = parse_qa_document(EN_DOC)
    assert doc.language == "en"
    assert len(doc.pairs) == 2
    assert doc.pairs[0].question.startswith("What is a tissue microarray")
    assert doc.pairs[1].answer.startswith("From a donor block")
    assert doc.passage.startswith("Biomarker research")


def test_parse_rejects_non_qa_text():
    assert not parse_qa_document("Just a plain paragraph without any markers.").is_usable
    assert not parse_qa_document("").is_usable


def test_parse_drops_unanswered_pairs():
    text = "正文。问题： a？ 答案： 这是足够长的答案。 问题： b？ 答案： 另一个足够长的答案。"
    doc = parse_qa_document(text, min_answer_chars=4)
    assert all(len(p.answer) >= 4 for p in doc.pairs)


# ------------------------------------------------------------ the template
def test_template_structure_zh(tok):
    doc = parse_qa_document(ZH_DOC)
    conv = format_conversation(doc, tok)
    ids = tok.encode(conv, add_special_tokens=False)
    markers = [i for i in ids if i <= 14]
    # bos, user, (eot, assistant, eot, user) x1, eot, eos
    assert markers == [1, 5, 4, 6, 4, 5, 4, 6, 4, 2]
    assert ids[-1] == tok.eos_token_id


def test_template_carries_the_passage_once(tok):
    doc = parse_qa_document(ZH_DOC)
    conv = format_conversation(doc, tok)
    assert conv.count(doc.pairs[0].question) == 1
    assert conv.count(doc.pairs[1].question) == 1
    assert doc.passage[:20] in conv


def test_template_with_task_token(tok):
    doc = parse_qa_document(ZH_DOC)
    conv = format_conversation(doc, tok, task="summarize")
    ids = tok.encode(conv, add_special_tokens=False)
    assert ids[:3] == [tok.bos_token_id, tok.task_token_ids["summarize"], tok.user_token_id]


def test_template_is_cheaper_than_the_raw_markers(tok):
    doc = parse_qa_document(ZH_DOC)
    conv = format_conversation(doc, tok)
    raw = len(tok.encode(ZH_DOC, add_special_tokens=False))
    new = len(tok.encode(conv, add_special_tokens=False))
    # 2 pairs -> 2 x 2 markers saved, minus bos/eos/newline overhead
    assert new <= raw + 2, (raw, new)


def test_template_roundtrips_through_decode(tok):
    doc = parse_qa_document(EN_DOC)
    conv = format_conversation(doc, tok)
    ids = tok.encode(conv, add_special_tokens=False)
    text = tok.decode(ids, skip_special_tokens=True)
    assert "tissue microarray" in text
    assert "donor block" in text
    assert "<|" not in text


def test_format_single_turn_sample(tok):
    s = format_sft_sample("概括这段话", "这是摘要。", tok, system="你是摘要助手。")
    ids = tok.encode(s, add_special_tokens=False)
    assert ids[0] == tok.bos_token_id
    assert tok.system_token_id in ids
    assert ids[-1] == tok.eos_token_id
    assert tok.decode(ids, skip_special_tokens=True).endswith("这是摘要。")


def test_prompt_prefix_ends_at_user(tok):
    p = prompt_prefix(tok, system="sys", task="polish")
    ids = tok.encode(p, add_special_tokens=False)
    assert ids[0] == tok.bos_token_id
    assert ids[-1] == tok.user_token_id
    assert tok.task_token_ids["polish"] in ids


def test_conversation_needs_patched_tokenizer():
    class FakeTok:
        is_wispv1_patched = False
    with pytest.raises(RuntimeError):
        format_conversation(parse_qa_document(ZH_DOC), FakeTok())


# --------------------------------------------------- tokenization boundaries
def test_document_boundaries_are_not_duplicated(tok):
    """encode_batch must not let the post_processor add its own BOS/EOS.

    Regression for a bug where every document got `<|eos|><|eos|>` (the template
    from tokenizer.json already adds BOS/EOS) and templated conversations got a
    doubled `<|bos|>`.
    """
    import numpy as np
    from tokenizers import Tokenizer
    import scripts.tokenize_data as td

    td._ENC = Tokenizer.from_file(tok.path)
    td._TOK = tok

    docs = ["The quick brown fox jumps over the lazy dog.", "人工智能正在改变世界。"]
    for doc_template, want_bos in ((None, 0), (tok.bos_token_id, 1)):
        arrays = td._encode_texts(docs, eos_id=tok.eos_token_id, bos_id=doc_template)
        for a in arrays:
            assert (a == tok.eos_token_id).sum() == 1, f"doubled EOS: {a[:5]}"
            assert (a == tok.bos_token_id).sum() == want_bos

    # chat template: exactly one BOS and one EOS per conversation
    arrays = td._encode_texts([ZH_DOC], eos_id=tok.eos_token_id, bos_id=None, chat_format="plain")
    assert len(arrays) == 1
    a = arrays[0]
    assert (a == tok.bos_token_id).sum() == 1
    assert (a == tok.eos_token_id).sum() == 1
    assert int(a[0]) == tok.bos_token_id and int(a[-1]) == tok.eos_token_id
    assert (a == tok.user_token_id).sum() == (a == tok.assistant_token_id).sum() == 2
    assert (a == tok.end_of_turn_token_id).sum() == 4
