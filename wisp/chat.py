"""Turn Ultra-FineWeb-L3 Q&A documents into Wispv1 chat-template text.

The L3 ``qa`` pool is *flat* text: an original passage followed by many inline
``问题：… 答案：…`` (or ``Question: … Answer: …``) pairs -- on average **14** pairs
per document.  To use it during the decay phase of pre-training (and later for
SFT) we parse those pairs and re-emit them with the reserved special tokens::

    <|bos|><|user|>{passage}

    {Q1}<|end_of_turn|><|assistant|>{A1}<|end_of_turn|>
    <|user|>{Q2}<|end_of_turn|><|assistant|>{A2}<|end_of_turn|>
    ...<|eos|>

Why bother: the raw markers cost 2 tokens each and the pairs are ambiguous to
parse; the special tokens cost 1 token each and are structural.  Measured on the
real shards, ``问题：``/``答案：`` are 2+2 tokens and ``<|user|>``/``<|assistant|>``
are 1+1, i.e. one *document* saves roughly 28 tokens (14 pairs x 2).

This module knows nothing about training: it only produces strings, which
``scripts/tokenize_data.py --chat-format`` then encodes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

# 问题： / 答案： (full-width or half-width colon, optional spaces)
Q_PATTERN = re.compile(r"问题\s*[:：]")
A_PATTERN = re.compile(r"答案\s*[:：]")
# NOTE: the English markers are *inline* (the passage, "Question: ...", "Answer: ..."
# all run together), so they must not be anchored to a line start.
Q_PATTERN_EN = re.compile(r"Question\s*[:：]\s*", re.IGNORECASE)
A_PATTERN_EN = re.compile(r"Answer\s*[:：]\s*", re.IGNORECASE)


@dataclass
class QAPair:
    question: str
    answer: str


@dataclass
class QADocument:
    passage: str = ""
    pairs: List[QAPair] = field(default_factory=list)
    language: str = "unknown"

    @property
    def is_usable(self) -> bool:
        return len(self.pairs) > 0

    def stats(self) -> dict:
        return {"passage_chars": len(self.passage), "pairs": len(self.pairs),
                "answer_chars": sum(len(p.answer) for p in self.pairs)}


def _split_pairs(text: str, q_pat: re.Pattern, a_pat: re.Pattern) -> Tuple[str, List[QAPair]]:
    """Split one document into (leading passage, [(question, answer), ...])."""
    marks = list(q_pat.finditer(text))
    if not marks:
        return text.strip(), []
    passage = text[: marks[0].start()].strip()
    pairs: List[QAPair] = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        chunk = text[m.end():end]
        am = a_pat.search(chunk)
        if am is None:
            question, answer = chunk.strip(), ""
        else:
            question, answer = chunk[: am.start()].strip(), chunk[am.end():].strip()
        if question or answer:
            pairs.append(QAPair(question, answer))
    return passage, pairs


def parse_qa_document(text: str, min_answer_chars: int = 2) -> QADocument:
    """Parse an Ultra-FineWeb-L3 QA document.

    Handles both the Chinese (``问题：/答案：``) and the English
    (``Question:/Answer:``) layout; the Chinese markers are tried first because
    they are unambiguous, then English ones.  Answers shorter than
    ``min_answer_chars`` are dropped (they are usually parsing artefacts).
    """
    if not text or not text.strip():
        return QADocument()
    zh_passage, zh_pairs = _split_pairs(text, Q_PATTERN, A_PATTERN)
    en_passage, en_pairs = _split_pairs(text, Q_PATTERN_EN, A_PATTERN_EN)
    if len(en_pairs) > len(zh_pairs):
        passage, pairs, lang = en_passage, en_pairs, "en"
    else:
        passage, pairs, lang = zh_passage, zh_pairs, "zh" if zh_pairs else "unknown"
    # drop pairs whose answer failed to parse (< 1% of them), but never drop all
    good = [p for p in pairs if len(p.answer) >= min_answer_chars]
    pairs = good if good else pairs
    return QADocument(passage=passage, pairs=pairs, language=lang)


# ----------------------------------------------------------------------
def format_conversation(
    doc: QADocument,
    tok,
    max_passage_chars: int = 6000,
    max_answers: int = 12,
    include_passage: bool = True,
    task: Optional[str] = None,
) -> str:
    """Render a parsed QA document as Wispv1 chat-template text.

    ``task`` optionally prefixes the first user turn with a task token such as
    ``summarize`` (see ``WispTokenizer.task_token_ids``).
    """
    if not doc.is_usable:
        return ""
    if not tok.is_wispv1_patched:
        raise RuntimeError(
            "this tokenizer still has <|placeholder_*> ids; run "
            "`python scripts/setup_special_tokens.py` before using the chat template"
        )
    eot = tok.reserved_name(tok.end_of_turn_token_id)
    user = tok.reserved_name(tok.user_token_id)
    assistant = tok.reserved_name(tok.assistant_token_id)
    bos = tok.bos_token
    eos = tok.eos_token

    parts: List[str] = [bos]
    if task and task in tok.task_token_ids:
        parts.append(tok.reserved_name(tok.task_token_ids[task]))
    parts.append(user)

    first = ""
    if include_passage and doc.passage:
        first = doc.passage[:max_passage_chars].strip() + "\n\n"
    pairs = doc.pairs[:max_answers]
    if not pairs:
        return ""
    first += pairs[0].question
    parts.append(first)
    parts.append(eot)
    parts.append(assistant)
    parts.append(pairs[0].answer)
    parts.append(eot)
    for pair in pairs[1:]:
        parts.append(user)
        parts.append(pair.question)
        parts.append(eot)
        parts.append(assistant)
        parts.append(pair.answer)
        parts.append(eot)
    parts.append(eos)
    # the special tokens are joined directly to the text: they are marked
    # `normalized: false` upstream, so NFKC never touches them, and the inference
    # helpers below build prompts the same way
    return "".join(parts)


def format_sft_sample(prompt: str, response: str, tok, system: Optional[str] = None,
                      task: Optional[str] = None) -> str:
    """A single-turn instruction sample (used by the future SFT stage)."""
    if not tok.is_wispv1_patched:
        raise RuntimeError("run `python scripts/setup_special_tokens.py` first")
    out = [tok.bos_token]
    if system:
        out += [tok.reserved_name(tok.system_token_id), system,
                tok.reserved_name(tok.end_of_turn_token_id)]
    if task and task in tok.task_token_ids:
        out.append(tok.reserved_name(tok.task_token_ids[task]))
    out += [tok.reserved_name(tok.user_token_id), prompt,
            tok.reserved_name(tok.end_of_turn_token_id),
            tok.reserved_name(tok.assistant_token_id), response,
            tok.reserved_name(tok.end_of_turn_token_id), tok.eos_token]
    return "".join(out)


def prompt_prefix(tok, system: Optional[str] = None, task: Optional[str] = None) -> str:
    """Everything up to and including ``<|assistant|>`` (for inference)."""
    out = [tok.bos_token]
    if system:
        out += [tok.reserved_name(tok.system_token_id), system,
                tok.reserved_name(tok.end_of_turn_token_id)]
    if task and task in tok.task_token_ids:
        out.append(tok.reserved_name(tok.task_token_ids[task]))
    return "".join(out) + tok.reserved_name(tok.user_token_id)
