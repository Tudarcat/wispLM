#!/usr/bin/env python3
"""Tokenize parquet shards into flat uint16 token streams (once, offline).

Output layout (one directory per corpus, discovered automatically by
``wisp.data.discover_corpora``)::

    <dst>/<corpus-name>/
        tokens.bin     flat uint16 little-endian token ids
        meta.json      {"name", "num_tokens", "num_docs", "dtype", "bin_file",
                        "eos_token_id", "doc_template", "tokenizer", "sources"}

Every document is encoded as ``text + <|eos|>`` (``--doc-template``), so the
stream is directly trainable by next-token prediction across packed blocks.

Tokenization is CPU bound and embarrassingly parallel: one worker process per
parquet shard, results concatenated afterwards.  On a 12-core laptop the 18 GB
Ultra-FineWeb-L3 slice (7B tokens) takes roughly {see README} -- run it on the
training machine instead if it has more cores.

    # local test on a couple of shards
    python scripts/tokenize_data.py --src "/media/.../Wispv1-Data" --dst /tmp/tok --limit-files 1

    # full run
    python scripts/tokenize_data.py --src "/media/.../Wispv1-Data" --dst /data/wispv1_tok --workers 16
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

# the Rust tokenizer parallelises internally with rayon; we parallelise with
# processes instead, so switch its internal threading off to avoid oversubscription
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from wisp.tokenizer import DEFAULT_TOKENIZER_DIR  # noqa: E402

_ENC = None
_TOK = None
_TEXT_COLUMNS = ("content", "text", "raw_content", "answer")


def _encode_texts(texts: List[str], eos_id: int, bos_id: Optional[int],
                  chat_format: str = "", max_passage_chars: int = 6000,
                  max_answers: int = 12) -> List[np.ndarray]:
    """Encode a batch of documents (one array per document).

    With ``chat_format`` the Ultra-FineWeb-L3 "passage + 问题：/答案：" layout is
    re-emitted as a multi-turn conversation using the reserved special tokens
    (``<|user|>``/``<|assistant|>``/``<|end_of_turn|>``); the template already
    carries ``<|bos|>``/``<|eos|>`` so no extra markers are added.
    """
    if chat_format:
        from wisp.chat import format_conversation, parse_qa_document

        rendered = []
        for text in texts:
            doc = parse_qa_document(text)
            conv = format_conversation(
                doc, _TOK, max_passage_chars=max_passage_chars, max_answers=max_answers,
                task=(None if chat_format == "plain" else chat_format),
            )
            if conv:
                rendered.append(conv)
        if not rendered:
            return []
        # add_special_tokens=False: the template already carries <|bos|>/<|eos|>
        return [np.asarray(e.ids, dtype=np.uint16)
                for e in _ENC.encode_batch(rendered, add_special_tokens=False)]
    # add_special_tokens=False -> the post_processor must not add its own
    # <|bos|>/<|eos|>; the boundary markers come from --doc-template below.
    # (Without this every document ends up with a doubled <|eos|>.)
    encs = _ENC.encode_batch(texts, add_special_tokens=False)
    out = []
    for e in encs:
        ids = np.asarray(e.ids, dtype=np.uint16)
        if bos_id is not None:
            ids = np.concatenate((np.asarray([bos_id], dtype=np.uint16), ids))
        if eos_id is not None:
            ids = np.concatenate((ids, np.asarray([eos_id], dtype=np.uint16)))
        out.append(ids)
    return out


def _init_worker(tokenizer_path: str, vocab_size: int, tokenizer_dir: str) -> None:
    global _ENC, _TOK
    from tokenizers import Tokenizer

    from wisp.tokenizer import WispTokenizer

    _TOK = WispTokenizer(tokenizer_dir)
    _ENC = Tokenizer.from_file(tokenizer_path)
    if _ENC.get_vocab_size(with_added_tokens=True) != vocab_size:
        raise RuntimeError(
            f"vocab mismatch: tokenizer.json has {_ENC.get_vocab_size(with_added_tokens=True)}, "
            f"expected {vocab_size}"
        )


def _tokenize_shard(job: Tuple[str, str, str, int, Optional[int], int, int, str, int, int]) -> Tuple[str, int, int, float, int]:
    """Worker: tokenize one parquet shard into a .part file."""
    (src, part_path, text_column, eos_id, bos_id, batch_rows, max_docs,
     chat_format, max_passage_chars, max_answers) = job
    import pyarrow.parquet as pq

    if os.path.exists(part_path) and os.path.getsize(part_path) > 0:
        # resumable: trust an existing part only if its sidecar says it finished
        stamp = part_path + ".done"
        if os.path.exists(stamp):
            with open(stamp, "r", encoding="utf-8") as fh:
                info = json.load(fh)
            return src, int(info["tokens"]), int(info["docs"]), 0.0, int(info.get("seen", info["docs"]))
    t0 = time.time()
    pf = pq.ParquetFile(src)
    cols = pf.schema_arrow.names
    col = text_column if text_column in cols else next((c for c in _TEXT_COLUMNS if c in cols), cols[0])
    n_tokens = n_docs = n_seen = 0
    pending: List[np.ndarray] = []
    pending_tokens = 0
    with open(part_path, "wb") as fh:
        for batch in pf.iter_batches(batch_size=batch_rows, columns=[col]):
            texts = [t for t in batch.column(col).to_pylist() if isinstance(t, str) and t]
            if max_docs and n_docs + len(texts) > max_docs:
                texts = texts[: max(0, max_docs - n_docs)]
            if not texts:
                break
            n_seen += len(texts)
            arrays = _encode_texts(texts, eos_id, bos_id, chat_format,
                                   max_passage_chars, max_answers)
            pending.extend(arrays)
            pending_tokens += sum(len(a) for a in arrays)
            n_docs += len(arrays)
            if max_docs and n_docs >= max_docs:
                break
            if pending_tokens >= (1 << 23):  # ~8M tokens per flush
                block = np.concatenate(pending)
                fh.write(block.tobytes())
                n_tokens += len(block)
                pending, pending_tokens = [], 0
        if pending:
            block = np.concatenate(pending)
            fh.write(block.tobytes())
            n_tokens += len(block)
    with open(part_path + ".done", "w", encoding="utf-8") as fh:
        json.dump({"tokens": int(n_tokens), "docs": int(n_docs), "seen": int(n_seen)}, fh)
    return src, int(n_tokens), int(n_docs), time.time() - t0, int(n_seen)


def collect_sources(root: str, limit_files: int = 0, pattern: str = ".parquet",
                    name_prefix: str = "") -> Dict[str, List[str]]:
    """Group parquet files by their relative directory (the corpus name)."""
    groups: Dict[str, List[str]] = {}
    for dirpath, _, filenames in os.walk(root):
        files = sorted(f for f in filenames if f.endswith(pattern))
        if not files:
            continue
        rel = os.path.relpath(dirpath, root)
        name = "all" if rel == "." else rel
        if name_prefix:
            name = f"{name_prefix.strip('/')}/{name}"
        groups.setdefault(name, []).extend(os.path.join(dirpath, f) for f in files)
    if limit_files:
        groups = {k: v[:limit_files] for k, v in groups.items()}
    return groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory tree with .parquet shards")
    ap.add_argument("--dst", required=True, help="output root (one subdir per corpus)")
    ap.add_argument("--tokenizer-dir", default=DEFAULT_TOKENIZER_DIR)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--batch-rows", type=int, default=1000)
    ap.add_argument("--limit-files", type=int, default=0, help="only the first N shards per corpus (testing)")
    ap.add_argument("--max-docs-per-shard", type=int, default=0, help="stop after N documents per shard (testing)")
    ap.add_argument("--doc-template", default="eos", choices=["eos", "bos_eos", "none"])
    ap.add_argument("--text-column", default="auto")
    ap.add_argument("--chat-format", default="", choices=["", "plain", "summarize", "rewrite", "polish", "expand"],
                    help="re-emit QA documents with the Wispv1 chat template "
                         "(optional value = task token to prefix)")
    ap.add_argument("--max-passage-chars", type=int, default=6000)
    ap.add_argument("--max-answers", type=int, default=12)
    ap.add_argument("--name-prefix", default="",
                    help="prefix for the corpus names, e.g. 'decay' -> decay/en, decay/zh "
                         "(needed when the source tree is a subdirectory)")
    ap.add_argument("--force", action="store_true", help="ignore existing .done markers")
    args = ap.parse_args()

    with open(os.path.join(args.tokenizer_dir, "tokenizer_config.json"), "r", encoding="utf-8") as fh:
        tconf = json.load(fh)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(args.tokenizer_dir, "tokenizer.json"))
    vocab_size = tok.get_vocab_size(with_added_tokens=True)
    eos_id = tok.token_to_id(tconf["eos_token"]) if args.doc_template != "none" else None
    bos_id = tok.token_to_id(tconf["bos_token"]) if args.doc_template == "bos_eos" else None
    del tok

    groups = collect_sources(args.src, args.limit_files, name_prefix=args.name_prefix)
    if not groups:
        print(f"no parquet files under {args.src}")
        return 2

    print(f"tokenizer: {args.tokenizer_dir} (vocab {vocab_size}, eos={eos_id}, "
          f"template={args.doc_template})")
    print(f"source:    {args.src}")
    print(f"dest:      {args.dst}")
    for name, files in sorted(groups.items()):
        print(f"  {name:22s} {len(files):4d} shards")

    os.makedirs(args.dst, exist_ok=True)
    grand_tokens = grand_docs = 0
    t_start = time.time()
    for name, files in sorted(groups.items()):
        out_dir = os.path.join(args.dst, name)
        parts_dir = os.path.join(out_dir, "parts")
        os.makedirs(parts_dir, exist_ok=True)
        text_column = None if args.text_column == "auto" else args.text_column

        jobs = []
        for src in files:
            base = os.path.splitext(os.path.basename(src))[0]
            part = os.path.join(parts_dir, base + ".bin")
            if args.force and os.path.exists(part + ".done"):
                os.remove(part + ".done")
            jobs.append((src, part, text_column or "auto", eos_id, bos_id,
                         args.batch_rows, args.max_docs_per_shard,
                         args.chat_format, args.max_passage_chars, args.max_answers))

        print(f"\n[{name}] tokenizing {len(jobs)} shards with {args.workers} workers ...", flush=True)
        n_tokens = n_docs = n_seen = 0
        t0 = time.time()
        with mp.Pool(args.workers, initializer=_init_worker,
                     initargs=(os.path.join(args.tokenizer_dir, "tokenizer.json"), vocab_size,
                               args.tokenizer_dir)) as pool:
            for k, (src, tk, dc, dt, sn) in enumerate(pool.imap_unordered(_tokenize_shard, jobs), 1):
                n_tokens += tk
                n_docs += dc
                n_seen += sn
                rate = n_tokens / max(time.time() - t0, 1e-6)
                print(f"  [{k:3d}/{len(jobs)}] {os.path.basename(src)[:44]:46s} "
                      f"{tk/1e6:7.2f}M tok  {rate/1e6:5.2f}M tok/s  eta "
                      f"{(len(jobs)-k)*max(time.time()-t0,1)/k/60:5.1f} min", flush=True)

        # concatenate the parts into one contiguous stream (best for random access)
        print(f"[{name}] concatenating {len(jobs)} parts -> tokens.bin ...", flush=True)
        bin_path = os.path.join(out_dir, "tokens.bin")
        written = 0
        with open(bin_path, "wb") as out:
            for src in files:
                base = os.path.splitext(os.path.basename(src))[0]
                part = os.path.join(parts_dir, base + ".bin")
                if not os.path.exists(part):
                    raise RuntimeError(f"missing part {part}")
                with open(part, "rb") as fh:
                    while True:
                        chunk = fh.read(1 << 24)
                        if not chunk:
                            break
                        out.write(chunk)
                        written += len(chunk)
        if written // 2 != n_tokens:
            raise RuntimeError(f"{name}: concatenated {written//2} tokens, workers reported {n_tokens}")
        meta = {
            "name": name,
            "num_tokens": int(n_tokens),
            "num_docs": int(n_docs),
            "dtype": "uint16",
            "bin_file": "tokens.bin",
            "tokenizer": os.path.abspath(args.tokenizer_dir),
            "vocab_size": vocab_size,
            "eos_token_id": eos_id,
            "bos_token_id": bos_id,
            "doc_template": args.doc_template,
            "chat_format": args.chat_format,
            "docs_seen": int(n_seen),
            "sources": [os.path.relpath(s, args.src) for s in files],
        }
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        shutil.rmtree(parts_dir, ignore_errors=True)
        print(f"[{name}] done: {n_tokens/1e9:.3f}B tokens, {n_docs:,} docs, "
              f"{os.path.getsize(bin_path)/1e9:.2f} GB in {(time.time()-t0)/60:.1f} min")
        if n_seen and n_docs < n_seen:
            drop = 1 - n_docs / n_seen
            note = ("  <-- looks like this pool has no 问题:/答案: pairs; "
                    "only use --chat-format on the qa pools") if drop > 0.3 else ""
            print(f"[{name}] WARNING: dropped {drop:.1%} of documents ({n_seen-n_docs:,}) "
                  f"that did not parse as QA{note}")
        grand_tokens += n_tokens
        grand_docs += n_docs

    print(f"\nALL DONE: {grand_tokens/1e9:.3f}B tokens, {grand_docs:,} docs, "
          f"{grand_tokens*2/1e9:.2f} GB on disk, {(time.time()-t_start)/60:.1f} min total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
