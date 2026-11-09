#!/usr/bin/env python3
"""Derive an SFT ``mask.bin`` from an existing chat-templated token stream.

``tokenize_data.py --chat-format`` already produced a correct multi-turn stream;
all that is missing for SFT is the per-token supervision mask.  Re-tokenising
1B tokens takes ~40 min, while scanning the stream and writing the mask takes
~2 min and is exactly equivalent (same tokens, no second parse).

The state machine runs over the whole stream in order, so a turn that straddles
a chunk boundary is still classified correctly -- that is the one thing a
per-block "look at the marker ids" shortcut cannot do.

    python scripts/derive_sft_mask.py --root /path/to/Wispv1-Tokenized \
        --corpus decay/en --corpus decay/zh

Afterwards the corpus has ``mask_file: mask.bin`` in its meta.json and is usable
with ``train.py --sft``.  Verify with ``scripts/audit_sft_data.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from wisp.data import discover_corpora  # noqa: E402
from wisp.tokenizer import WispTokenizer  # noqa: E402

CHUNK = 20_000_000


def derive_mask(src: str, dst: str, tok, names: dict, verbose: bool = True) -> dict:
    """1 where the token belongs to an assistant turn (incl. its closing eot)."""
    arr = np.memmap(src, dtype=np.uint16, mode="r")
    total = len(arr)
    specials = np.fromiter(names.keys(), dtype=np.int64, count=len(names))
    asst_id = tok.assistant_token_id
    eot_id = tok.end_of_turn_token_id
    bos_id = tok.bos_token_id
    eos_id = tok.eos_token_id
    sys_id = tok.system_token_id

    n_sup = n_turns = n_docs = 0
    state = "between"          # between | user | assistant | system
    t0 = time.time()
    with open(dst, "wb") as out:
        for start in range(0, total, CHUNK):
            chunk = np.asarray(arr[start:start + CHUNK])
            mask = np.zeros(len(chunk), dtype=np.uint8)
            hit = np.flatnonzero(np.isin(chunk, specials))
            prev_pos = 0
            for idx in hit.tolist():
                tid = int(chunk[idx])
                # content before this marker belongs to the current turn
                if state == "assistant" and idx > prev_pos:
                    mask[prev_pos:idx] = 1
                    n_sup += idx - prev_pos
                if tid == bos_id:
                    state = "between"
                    n_docs += 1
                elif tid == eos_id:
                    state = "between"
                elif tid == sys_id:
                    state = "system"
                elif tid == tok.user_token_id:
                    state = "user"
                elif tid == asst_id:
                    state = "assistant"
                    n_turns += 1
                elif tid == eot_id:
                    if state == "assistant":
                        mask[idx] = 1          # learn to stop
                        n_sup += 1
                    state = "between"
                prev_pos = idx + 1
            # trailing content after the last marker of the chunk
            if state == "assistant" and len(chunk) > prev_pos:
                mask[prev_pos:] = 1
                n_sup += len(chunk) - prev_pos
            out.write(mask.tobytes())
        if verbose:
            print(f"    scanned {total:,} tokens, {n_docs:,} docs, {n_turns:,} assistant turns, "
                  f"{n_sup:,} supervised ({n_sup/max(total,1):.1%}) in {time.time()-t0:.0f}s")
    return {"num_tokens": total, "num_docs": n_docs, "num_turns": n_turns,
            "num_supervised_tokens": int(n_sup)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--corpus", action="append", default=None,
                    help="corpus name (repeatable); default = every corpus")
    ap.add_argument("--force", action="store_true", help="overwrite an existing mask.bin")
    args = ap.parse_args()

    tok = WispTokenizer()
    names = {tok.bos_token_id: "bos", tok.eos_token_id: "eos"}
    for n, i in tok.role_token_ids.items():
        names[i] = n
    for n, i in tok.task_token_ids.items():
        names[i] = f"task:{n}"
    for n, i in tok.reasoning_token_ids.items():
        names[i] = n

    corpora = discover_corpora(args.root, in_memory=False)
    todo = args.corpus or sorted(corpora)
    for name in todo:
        if name not in corpora:
            print(f"[error] {name!r} not found; available: {sorted(corpora)}")
            return 2
        corp = corpora[name]
        mask_path = os.path.join(corp.root, "mask.bin")
        if corp.mask is not None and not args.force:
            print(f"{name}: already has a mask (use --force to rebuild)")
            continue
        print(f"{name}: deriving mask -> {mask_path}")
        info = derive_mask(corp.bin_path, mask_path, tok, names)
        if info["num_supervised_tokens"] == 0:
            print(f"  [warn] no assistant turns found; this is not a chat corpus. "
                  f"mask.bin was still written but is all zeros.")
        meta = dict(corp.meta)
        meta.update({"mask_file": "mask.bin", "mask_dtype": "uint8", **info})
        tmp = os.path.join(corp.root, "meta.json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, os.path.join(corp.root, "meta.json"))
        print(f"  -> {info['num_supervised_tokens']:,} supervised tokens "
              f"({info['num_supervised_tokens']/max(info['num_tokens'],1):.1%}); "
              f"meta.json updated")
    print("\ndone. verify with: python scripts/audit_sft_data.py --root " + args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
