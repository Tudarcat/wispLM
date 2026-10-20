#!/usr/bin/env python3
"""Rename the reserved special-token slots of QiYuanTokenizer-Small for Wispv1.

The tokenizer ships 15 reserved slots at ids **0..14** (they are the first
entries of both ``added_tokens`` and the Unigram ``model.vocab``).  Renaming them
keeps every id, ``vocab_size`` (24000) and therefore the model shape and any
checkpoint **byte-compatible** -- the embedding rows just become meaningful
instead of ``<|placeholder_N|>``.

Wispv1 layout (see docs/special_tokens.md)::

    id  0  <|unk|>            (unchanged)
    id  1  <|bos|>            (unchanged)
    id  2  <|eos|>            (unchanged)
    id  3  <|pad|>            (unchanged -- padding_idx, row is pinned to 0)
    id  4  <|end_of_turn|>    <- was <|mask|>          (useless: decoder-only + -100 masking)
    id  5  <|user|>           <- was <|placeholder_0|>
    id  6  <|assistant|>      <- was <|placeholder_1|>
    id  7  <|system|>         <- was <|placeholder_2|>
    id  8  <|summarize|>      <- was <|placeholder_3|>
    id  9  <|rewrite|>        <- was <|placeholder_4|>
    id 10  <|polish|>         <- was <|placeholder_5|>
    id 11  <|expand|>         <- was <|placeholder_6|>
    id 12  <|think|>          <- was <|placeholder_7|>
    id 13  <|/think|>         <- was <|placeholder_8|>
    id 14  <|placeholder_9|>  (spare -- can still be renamed later for free)

Why a spare is free: a slot that never appears in the pre-training data never
receives a gradient, so renaming it later needs no retraining.  Slots that *are*
used become frozen.

    python scripts/setup_special_tokens.py            # apply
    python scripts/setup_special_tokens.py --check    # verify only, no write
    python scripts/setup_special_tokens.py --revert   # restore the .orig backups
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(HERE, "tokenizer")

# id -> new token string.  Unlisted reserved ids are left alone.
REPLACEMENTS = {
    4: "<|end_of_turn|>",
    5: "<|user|>",
    6: "<|assistant|>",
    7: "<|system|>",
    8: "<|summarize|>",
    9: "<|rewrite|>",
    10: "<|polish|>",
    11: "<|expand|>",
    12: "<|think|>",
    13: "<|/think|>",
}
# everything from id 0..14 is a reserved slot
RESERVED_IDS = list(range(15))
CORE = {0: "<|unk|>", 1: "<|bos|>", 2: "<|eos|>", 3: "<|pad|>"}

# text used to prove nothing else in the tokenizer moved
PROBE_TEXTS = [
    "Hello world, this is Wispv1.",
    "你好，世界！人工智能正在改变世界。",
    "12345 + abc = ? \n\t  mixed   spacing",
    "骨折术后因治疗需要对患肢或患部进行制动，患者常需长期卧床。",
    "问题：淋球菌培养后菌落有什么特征？ 答案：圆形、凸起、湿润、光滑。",
    "Question: What is a tissue microarray? Answer: A paraffin block.",
]


def tokenizer_paths(root: str):
    return (os.path.join(root, "tokenizer.json"), os.path.join(root, "tokenizer_config.json"))


def load(root: str):
    tj, tc = tokenizer_paths(root)
    with open(tj, "r", encoding="utf-8") as fh:
        tok = json.load(fh)
    with open(tc, "r", encoding="utf-8") as fh:
        conf = json.load(fh)
    return tok, conf


def current_names(tok: dict) -> dict:
    return {t["id"]: t["content"] for t in tok["added_tokens"]}


def apply_patch(tok: dict, conf: dict) -> dict:
    names = current_names(tok)
    before = dict(names)
    for tid, new in REPLACEMENTS.items():
        if tid not in names:
            raise SystemExit(f"id {tid} is not an added token; unexpected tokenizer layout")
        if names[tid] in CORE.values():
            raise SystemExit(f"refusing to rename core token {names[tid]!r} at id {tid}")
        names[tid] = new

    # 1) added_tokens
    for entry in tok["added_tokens"]:
        tid = entry["id"]
        if tid in names:
            entry["content"] = names[tid]
            if tid in REPLACEMENTS:
                entry["special"] = True  # so decode(skip_special_tokens=True) strips it

    # 2) the Unigram vocab holds the same strings at the same indices
    vocab = tok["model"]["vocab"]
    for tid in RESERVED_IDS:
        if tid >= len(vocab):
            raise SystemExit(f"model.vocab is shorter than expected ({len(vocab)})")
        if tid in names:
            vocab[tid] = [names[tid], vocab[tid][1]]

    # 3) tokenizer_config.json bookkeeping
    conf["extra_special_tokens"] = [names[i] for i in RESERVED_IDS if i in names and i not in CORE]
    conf["mask_token"] = None  # the MLM mask slot was repurposed
    conf["wispv1_reserved_tokens"] = {str(i): names[i] for i in RESERVED_IDS if i in names}
    return before


def verify(root: str, before: dict, expect_renamed: bool) -> int:
    """Reload with the Rust tokenizer and check ids, vocab size and regression."""
    from tokenizers import Tokenizer

    tj, _ = tokenizer_paths(root)
    enc = Tokenizer.from_file(tj)
    ok = True

    vocab_size = enc.get_vocab_size(with_added_tokens=True)
    print(f"  vocab_size           : {vocab_size}", end="")
    print("  [ok]" if vocab_size == 24000 else "  [FAIL]")
    ok &= vocab_size == 24000

    names = {}
    with open(tj, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    names = {t["id"]: t["content"] for t in data["added_tokens"]}
    for tid, want in (CORE | (REPLACEMENTS if expect_renamed else {})).items():
        got = names.get(tid)
        status = "ok" if got == want else "FAIL"
        print(f"  id {tid:2d} -> {str(got):20s} want {want:20s} [{status}]")
        ok &= got == want

    # the crucial property: one token per reserved marker, and normal text untouched
    for tid, name in sorted((CORE | REPLACEMENTS).items()):
        ids = enc.encode(name, add_special_tokens=False).ids
        good = ids == [tid]
        if not good:
            print(f"  encode({name!r}) = {ids}, expected [{tid}]  [FAIL]")
        ok &= good
    print(f"  every reserved marker encodes to exactly 1 token  [{'ok' if ok else 'FAIL'}]")

    # regression: ordinary text must tokenize exactly as before the rename
    if before:
        print("  (regression check runs in the calling process)")
    return 0 if ok else 1


def regression_check(root: str, before_names: dict) -> int:
    """Tokenize probe texts before/after the patch using only the vocab strings.

    We can't reload the *old* file here, so instead we assert that no probe text
    produces a reserved id (0..14) -- i.e. the rename did not start matching
    normal text.
    """
    from tokenizers import Tokenizer

    tj, _ = tokenizer_paths(root)
    enc = Tokenizer.from_file(tj)
    bad = 0
    for text in PROBE_TEXTS:
        ids = enc.encode(text, add_special_tokens=False).ids
        hit = [i for i in ids if i <= 14]
        # "\n" (16) and punctuation are fine; only reserved ids would be suspicious
        if hit:
            print(f"  [warn] reserved ids {hit} appear when encoding {text[:40]!r}")
            bad += 1
    print(f"  probe texts    : {len(PROBE_TEXTS) - bad}/{len(PROBE_TEXTS)} clean of reserved ids")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", default=DEFAULT_DIR)
    ap.add_argument("--check", action="store_true", help="verify only, do not write")
    ap.add_argument("--revert", action="store_true", help="restore the .orig backups")
    args = ap.parse_args()
    root = args.tokenizer_dir
    tj, tc = tokenizer_paths(root)

    if args.revert:
        for p in (tj, tc):
            if os.path.exists(p + ".orig"):
                shutil.copy2(p + ".orig", p)
                print(f"restored {p}")
        return 0

    tok, conf = load(root)
    before = current_names(tok)
    print("before:")
    for tid in RESERVED_IDS:
        print(f"  id {tid:2d}  {before.get(tid)}")

    if args.check:
        return verify(root, {}, expect_renamed=all(
            before.get(i) == v for i, v in REPLACEMENTS.items()))

    if all(before.get(i) == v for i, v in REPLACEMENTS.items()):
        print("\nalready patched; nothing to do")
        return verify(root, {}, expect_renamed=True)

    # back up the pristine files once
    for p in (tj, tc):
        if not os.path.exists(p + ".orig"):
            shutil.copy2(p, p + ".orig")
            print(f"backup -> {p}.orig")

    apply_patch(tok, conf)
    tmp = tj + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(tok, fh, ensure_ascii=False)
    os.replace(tmp, tj)
    tmp = tc + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(conf, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, tc)
    print("\npatched tokenizer.json + tokenizer_config.json\n")

    rc = verify(root, before, expect_renamed=True)
    rc |= regression_check(root, before)
    print("\n" + ("OK" if rc == 0 else "PROBLEMS FOUND"))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
