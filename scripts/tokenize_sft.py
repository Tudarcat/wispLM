#!/usr/bin/env python3
"""Tokenize Ultra-FineWeb-L3 QA documents **with a supervision mask** (for SFT).

Output is the same uint16 token stream as ``tokenize_data.py`` plus a parallel
``mask.bin`` (uint8, 1 = compute loss on this token).  The mask covers exactly
the assistant turns, including the ``<|end_of_turn|>`` that closes each one, so
the model learns both *what* to answer and *when to stop*; user/system text and
the ``<|assistant|>`` marker itself are masked out.

Aligned per token -- the char spans produced by ``wisp.chat.build_sft_sample`` are
mapped through the tokenizer's own byte offsets, so nothing is guessed from
marker ids (which would break when a turn straddles a block boundary).

    python scripts/tokenize_sft.py --src /data/Ultra-FineWeb-L3/sft \
        --dst /data/wispv1_sft --workers 16

    # then
    python train.py --sft --data /data/wispv1_sft --mixture "en=0.5,zh=0.5" ...
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from wisp.tokenizer import DEFAULT_TOKENIZER_DIR, WispTokenizer  # noqa: E402

_TOK = None
_ENC = None
_TEXT_COLUMNS = ("content", "text", "raw_content", "answer")


def _init_worker(tokenizer_dir: str, vocab_size: int) -> None:
    global _TOK, _ENC
    from tokenizers import Tokenizer

    _TOK = WispTokenizer(tokenizer_dir, strict_special_tokens=True)
    _ENC = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
    if _ENC.get_vocab_size(with_added_tokens=True) != vocab_size:
        raise RuntimeError("vocab size mismatch between workers")


def _render(texts: List[str], max_passage_chars: int, max_answers: int,
            task: Optional[str]) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
    """(token arrays, mask arrays, n_dropped)."""
    from wisp.chat import build_sft_sample, parse_qa_document, spans_to_mask

    docs, spans_list, dropped = [], [], 0
    for text in texts:
        doc = parse_qa_document(text)
        if not doc.is_usable:
            dropped += 1
            continue
        rendered, spans = build_sft_sample(
            doc, _TOK, max_passage_chars=max_passage_chars,
            max_answers=max_answers, task=task,
        )
        if not rendered or not spans:
            dropped += 1
            continue
        docs.append(rendered)
        spans_list.append(spans)
    if not docs:
        return [], [], dropped
    encs = _ENC.encode_batch(docs, add_special_tokens=False)
    toks, masks = [], []
    for enc, spans in zip(encs, spans_list):
        toks.append(np.asarray(enc.ids, dtype=np.uint16))
        masks.append(np.asarray(spans_to_mask(enc.offsets, spans), dtype=np.uint8))
    return toks, masks, dropped


def _tokenize_shard(job) -> Tuple[str, int, int, int, int, float]:
    """Worker: one parquet shard -> <part>.tok + <part>.msk (+ .done)."""
    (src, tok_path, msk_path, text_column, batch_rows, max_docs,
     max_passage_chars, max_answers, task) = job
    import pyarrow.parquet as pq

    stamp = tok_path + ".done"
    if os.path.exists(stamp):
        with open(stamp, "r", encoding="utf-8") as fh:
            info = json.load(fh)
        return (src, int(info["tokens"]), int(info["supervised"]), int(info["docs"]),
                int(info.get("dropped", 0)), 0.0)

    t0 = time.time()
    pf = pq.ParquetFile(src)
    cols = pf.schema_arrow.names
    col = text_column if text_column in cols else next(
        (c for c in _TEXT_COLUMNS if c in cols), cols[0])
    n_tokens = n_sup = n_docs = n_dropped = 0
    with open(tok_path, "wb") as ftok, open(msk_path, "wb") as fmsk:
        for batch in pf.iter_batches(batch_size=batch_rows, columns=[col]):
            texts = [t for t in batch.column(col).to_pylist() if isinstance(t, str) and t]
            if max_docs and n_docs + len(texts) > max_docs:
                texts = texts[: max(0, max_docs - n_docs)]
            if not texts:
                break
            toks, masks, dropped = _render(texts, max_passage_chars, max_answers, task)
            n_dropped += dropped
            if toks:
                flat_t = np.concatenate(toks)
                flat_m = np.concatenate(masks)
                ftok.write(flat_t.tobytes())
                fmsk.write(flat_m.tobytes())
                n_tokens += int(flat_t.size)
                n_sup += int(flat_m.sum())
                n_docs += len(toks)
            if max_docs and n_docs >= max_docs:
                break
    with open(stamp, "w", encoding="utf-8") as fh:
        json.dump({"tokens": n_tokens, "supervised": n_sup, "docs": n_docs,
                   "dropped": n_dropped}, fh)
    return src, n_tokens, n_sup, n_docs, n_dropped, time.time() - t0


def collect_sources(root: str, limit_files: int = 0,
                    prefix: str = "") -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for dirpath, _, filenames in os.walk(root):
        files = sorted(f for f in filenames if f.endswith(".parquet"))
        if not files:
            continue
        rel = os.path.relpath(dirpath, root)
        name = "all" if rel == "." else rel
        if prefix:
            name = f"{prefix.strip('/')}/{name}"
        groups.setdefault(name, []).extend(os.path.join(dirpath, f) for f in files)
    if limit_files:
        groups = {k: v[:limit_files] for k, v in groups.items()}
    return groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="qa parquet tree")
    ap.add_argument("--dst", required=True, help="output root (one dir per corpus)")
    ap.add_argument("--tokenizer-dir", default=DEFAULT_TOKENIZER_DIR)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--batch-rows", type=int, default=500)
    ap.add_argument("--limit-files", type=int, default=0)
    ap.add_argument("--max-docs-per-shard", type=int, default=0)
    ap.add_argument("--max-passage-chars", type=int, default=6000)
    ap.add_argument("--max-answers", type=int, default=12)
    ap.add_argument("--task", default=None,
                    choices=[None, "summarize", "rewrite", "polish", "expand"],
                    help="prepend this task token to every sample")
    ap.add_argument("--name-prefix", default="", help="e.g. 'sft' -> sft/en, sft/zh")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    with open(os.path.join(args.tokenizer_dir, "tokenizer_config.json"), "r", encoding="utf-8") as fh:
        tconf = json.load(fh)
    from tokenizers import Tokenizer

    vocab_size = Tokenizer.from_file(
        os.path.join(args.tokenizer_dir, "tokenizer.json")).get_vocab_size(with_added_tokens=True)

    groups = collect_sources(args.src, args.limit_files, args.name_prefix)
    if not groups:
        print(f"no parquet files under {args.src}")
        return 2
    print(f"tokenizer: {args.tokenizer_dir} (vocab {vocab_size})")
    print(f"source:    {args.src}")
    print(f"dest:      {args.dst}")
    for name, files in sorted(groups.items()):
        print(f"  {name:16s} {len(files):4d} shards")

    os.makedirs(args.dst, exist_ok=True)
    total_t = total_s = total_d = 0
    t_start = time.time()
    for name, files in sorted(groups.items()):
        out_dir = os.path.join(args.dst, name)
        parts = os.path.join(out_dir, "parts")
        os.makedirs(parts, exist_ok=True)
        jobs = []
        for src in files:
            base = os.path.splitext(os.path.basename(src))[0]
            tp = os.path.join(parts, base + ".tok")
            mp_ = os.path.join(parts, base + ".msk")
            if args.force and os.path.exists(tp + ".done"):
                os.remove(tp + ".done")
            jobs.append((src, tp, mp_, "auto", args.batch_rows,
                         args.max_docs_per_shard, args.max_passage_chars,
                         args.max_answers, args.task))

        print(f"\n[{name}] {len(jobs)} shards, {args.workers} workers ...", flush=True)
        n_tok = n_sup = n_docs = n_drop = 0
        t0 = time.time()
        with mp.Pool(args.workers, initializer=_init_worker,
                     initargs=(args.tokenizer_dir, vocab_size)) as pool:
            for k, (src, tk, sp, dc, dr, dt) in enumerate(
                    pool.imap_unordered(_tokenize_shard, jobs), 1):
                n_tok += tk; n_sup += sp; n_docs += dc; n_drop += dr
                print(f"  [{k:3d}/{len(jobs)}] {os.path.basename(src)[:40]:42s} "
                      f"{tk/1e6:7.2f}M tok ({sp/max(tk,1):5.1%} supervised) "
                      f"{n_tok/(time.time()-t0+1e-9)/1e6:5.2f}M tok/s", flush=True)

        tok_path = os.path.join(out_dir, "tokens.bin")
        msk_path = os.path.join(out_dir, "mask.bin")
        print(f"[{name}] concatenating ...", flush=True)
        written = 0
        with open(tok_path, "wb") as fo, open(msk_path, "wb") as fm:
            for src in files:
                base = os.path.splitext(os.path.basename(src))[0]
                for part, out, item in ((base + ".tok", fo, 2), (base + ".msk", fm, 1)):
                    p = os.path.join(parts, part)
                    if not os.path.exists(p):
                        raise RuntimeError(f"missing part {p}")
                    with open(p, "rb") as fh:
                        while True:
                            chunk = fh.read(1 << 24)
                            if not chunk:
                                break
                            out.write(chunk)
                            if item == 2:
                                written += len(chunk)
        if written // 2 != n_tok:
            raise RuntimeError(f"{name}: {written//2} tokens concatenated vs {n_tok} reported")
        meta = {
            "name": name,
            "num_tokens": int(n_tok),
            "num_docs": int(n_docs),
            "num_supervised_tokens": int(n_sup),
            "num_dropped_docs": int(n_drop),
            "dtype": "uint16",
            "bin_file": "tokens.bin",
            "mask_file": "mask.bin",
            "mask_dtype": "uint8",
            "tokenizer": os.path.abspath(args.tokenizer_dir),
            "vocab_size": vocab_size,
            "task_token": args.task,
            "max_answers": args.max_answers,
            "sources": [os.path.relpath(s, args.src) for s in files],
        }
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        shutil.rmtree(parts, ignore_errors=True)
        print(f"[{name}] done: {n_tok/1e9:.3f}B tokens, {n_sup/1e9:.3f}B supervised "
              f"({n_sup/max(n_tok,1):.1%}), {n_docs:,} docs, dropped {n_drop:,}, "
              f"{(time.time()-t0)/60:.1f} min")
        total_t += n_tok; total_s += n_sup; total_d += n_docs

    print(f"\nALL DONE: {total_t/1e9:.3f}B tokens, {total_s/1e9:.3f}B supervised "
          f"({total_s/max(total_t,1):.1%}), {total_d:,} docs, "
          f"{(time.time()-t_start)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
