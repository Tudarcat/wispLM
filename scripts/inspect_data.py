#!/usr/bin/env python3
"""Look at the training data: raw parquet, chat-templated, and tokenized streams.

    python scripts/inspect_data.py                    # overview + one sample per pool
    python scripts/inspect_data.py summary
    python scripts/inspect_data.py raw     --pool pretrain/en --n 2 --chars 1200
    python scripts/inspect_data.py chat    --pool sft/zh --n 2
    python scripts/inspect_data.py tokens  --root /data/wispv1_tok --corpus decay/zh --blocks 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

# just mod this macro if you want to look at a different data root than the default
DEFAULT_DATA = "/media/ydwu/数据移动硬盘/Wispv1-Data"
TEXT_COL = "content"
RULE = "─" * 96


# ----------------------------------------------------------------------
def parquet_pools(root: str) -> dict:
    """{pool_name: [shard paths]} for a directory tree of parquet shards."""
    pools = {}
    for dirpath, _, files in os.walk(root):
        shards = sorted(os.path.join(dirpath, f) for f in files if f.endswith(".parquet"))
        if shards:
            pools[os.path.relpath(dirpath, root)] = shards
    return pools


def read_docs(shard: str, n: int, start: int = 0, column: str = TEXT_COL) -> List[str]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(shard)
    cols = pf.schema_arrow.names
    col = column if column in cols else next((c for c in ("content", "text", "raw_content") if c in cols), cols[0])
    out: List[str] = []
    for rg in range(pf.metadata.num_row_groups):
        batch = pf.read_row_group(rg, columns=[col]).column(col)
        for i in range(len(batch)):
            if len(out) < start:
                out.append(None)  # type: ignore[arg-type]
                continue
            if len(out) >= start + n:
                return [x for x in out if x is not None]
            out.append(batch[i].as_py())
    return [x for x in out if x is not None][:n]


def wrap(text: str, width: int = 96, indent: str = "  ") -> str:
    """Wrap text for display, preserving paragraph breaks."""
    import textwrap

    out = []
    for para in text.split("\n"):
        if not para.strip():
            out.append("")
            continue
        out.extend(textwrap.wrap(para, width=width) or [""])
    return "\n".join(indent + line for line in out)


def render_turns(text: str, tok, max_chars: int = 400) -> str:
    """Print a templated conversation one turn per block, markers made visible."""
    marker_ids = list(tok.role_token_ids.values()) + [tok.bos_token_id, tok.eos_token_id]
    names = sorted((tok.reserved_name(i) for i in marker_ids), key=len, reverse=True)
    out = []
    i = 0
    while i < len(text):
        hit = next((n for n in names if text.startswith(n, i)), None)
        if hit:
            out.append(("MARK", hit))
            i += len(hit)
        else:
            j = min((text.find(n, i) for n in names if text.find(n, i) != -1), default=len(text))
            out.append(("TEXT", text[i:j]))
            i = j
    lines = []
    for kind, val in out:
        if kind == "MARK":
            lines.append(f"\n  {val}")
        else:
            body = val.strip()
            if not body:
                continue
            if max_chars and len(body) > max_chars:
                body = body[:max_chars] + f" …[+{len(body)-max_chars}]"
            lines.append(wrap(body, width=92, indent="      "))
    return "\n".join(lines)


def show(text: str, chars: int, label: str) -> None:
    body = text if chars <= 0 or len(text) <= chars else text[:chars] + f" …[+{len(text)-chars} chars]"
    print(f"  {label} ({len(text):,} chars)")
    print(wrap(body))
    print()


# ----------------------------------------------------------------------
def cmd_summary(args) -> int:
    import pyarrow.parquet as pq

    if not os.path.isdir(args.data):
        print(f"[error] {args.data} not found")
        return 2
    pools = parquet_pools(args.data)
    if not pools:
        print(f"[error] no .parquet under {args.data}")
        return 2
    print(f"data root: {args.data}\n")
    print(f"{'pool':16s} {'shards':>7s} {'size':>9s} {'docs':>12s} {'docs/shard':>11s}")
    tot = [0, 0, 0]
    for name, shards in sorted(pools.items()):
        size = sum(os.path.getsize(s) for s in shards)
        docs = 0
        for s in shards[: args.full_count and len(shards) or 3]:
            docs += pq.ParquetFile(s).metadata.num_rows
        scaled = docs * len(shards) / min(len(shards), args.full_count and len(shards) or 3)
        print(f"{name:16s} {len(shards):7d} {size/1e9:8.2f}G {scaled:12,.0f} {scaled/max(len(shards),1):11,.0f}")
        tot[0] += len(shards)
        tot[1] += size
        tot[2] += scaled
    print(f"{'TOTAL':16s} {tot[0]:7d} {tot[1]/1e9:8.2f}G {tot[2]:12,.0f}")

    meas = os.path.join(args.data, "manifest", "tokenizer_measurement.json")
    if os.path.exists(meas):
        with open(meas, "r", encoding="utf-8") as fh:
            m = json.load(fh)
        print("\nmeasured tokens/byte (Wispv1 24k tokenizer):")
        for pool, r in sorted(m.items()):
            print(f"  {pool:16s} tok/byte={r['tokens_per_byte']:.4f}  "
                  f"tok/doc={r['tokens_per_doc']:.0f}  chars/token={r['chars_per_token']:.2f}")
    return 0


def cmd_raw(args) -> int:
    pools = parquet_pools(args.data)
    if args.pool not in pools:
        print(f"[error] pool {args.pool!r} not found; available: {sorted(pools)}")
        return 2
    shard = pools[args.pool][args.shard]
    docs = read_docs(shard, args.n, start=args.skip)
    print(f"{args.pool}  shard {args.shard}: {os.path.basename(shard)}")
    print(f"rows in shard: {_rows(shard):,}")
    print(RULE)
    for i, d in enumerate(docs):
        show(d, args.chars, f"doc #{args.skip + i}")
    return 0


def _rows(shard: str) -> int:
    import pyarrow.parquet as pq
    return pq.ParquetFile(shard).metadata.num_rows


def cmd_chat(args) -> int:
    from wisp.chat import format_conversation, parse_qa_document
    from wisp.tokenizer import WispTokenizer

    pools = parquet_pools(args.data)
    if args.pool not in pools:
        print(f"[error] pool {args.pool!r} not found; available: {sorted(pools)}")
        return 2
    tok = WispTokenizer()
    docs = read_docs(pools[args.pool][args.shard], args.n, start=args.skip)
    print(f"{args.pool}  ->  chat template   (tokenizer: vocab {tok.vocab_size}, "
          f"patched={tok.is_wispv1_patched})")
    print(RULE)
    for i, raw in enumerate(docs):
        doc = parse_qa_document(raw)
        conv = format_conversation(doc, tok)
        print(f"  doc #{i}: parsed {len(doc.pairs)} QA pairs, language={doc.language}, "
              f"passage={len(doc.passage)} chars")
        if not conv:
            print("  [not a QA document]\n" + RULE)
            continue
        n_raw = len(tok.encode(raw, add_special_tokens=False))
        n_new = len(tok.encode(conv, add_special_tokens=False))
        print(f"  tokens: raw={n_raw:,} -> templated={n_new:,} ({n_new/n_raw-1:+.1%})")
        print(render_turns(conv, tok, max_chars=args.chars if args.chars > 0 else 400))
        print(RULE)
    return 0


def cmd_tokens(args) -> int:
    from wisp.data import discover_corpora
    from wisp.tokenizer import WispTokenizer

    tok = WispTokenizer()
    corpora = discover_corpora(args.root, in_memory=False)
    names = sorted(corpora)
    if args.corpus and args.corpus not in corpora:
        print(f"[error] corpus {args.corpus!r} not found; available: {names}")
        return 2
    targets = [args.corpus] if args.corpus else names
    specials = {i: tok.reserved_name(i) for i in tok.special_token_ids}
    for name in targets:
        c = corpora[name]
        print(f"{name}: {c.num_tokens/1e9:.4f}B tokens, {c.num_docs:,} docs, {c.dtype}, "
              f"chat_format={c.meta.get('chat_format') or 'plain'}")
        print(RULE)
        for b in range(args.blocks):
            start = (args.block + b) * args.seq
            ids = np.asarray(c.tokens[start:start + args.seq])
            oneline = " ".join(
                f"[{specials[int(i)]}]" if int(i) in specials else tok._tok.id_to_token(int(i)) or "?"
                for i in ids
            )
            print(f"  block {args.block + b}  (tokens {start}..{start+len(ids)-1}, seq={args.seq})")
            print(wrap(oneline))
            print(f"  decoded: {tok.decode(ids.tolist(), skip_special_tokens=True)[:400]!r}")
            print(f"  special ids in block: "
                  f"{ {specials[int(i)]: int((ids == i).sum()) for i in np.unique(ids) if int(i) in specials} }")
            print()
        print(RULE)
    return 0


# ----------------------------------------------------------------------
def cmd_overview(args) -> int:
    cmd_summary(args)
    pools = parquet_pools(args.data)
    print("\n" + "=" * 96 + "\nRAW PRETRAIN SAMPLE\n" + "=" * 96)
    for pool in [p for p in sorted(pools) if p.startswith("pretrain")][:1]:
        args.pool = pool
        args.shard = args.n_shard
        args.n = 1
        args.skip = 0
        args.chars = args.chars
        cmd_raw(args)
    print("\n" + "=" * 96 + "\nRAW SFT SAMPLE\n" + "=" * 96)
    for pool in [p for p in sorted(pools) if p.startswith("sft")][:1]:
        args.pool = pool
        cmd_raw(args)
    print("\n" + "=" * 96 + "\nSAME SFT DOC AS A CHAT-TEMPLATE CONVERSATION\n" + "=" * 96)
    for pool in [p for p in sorted(pools) if p.startswith("sft")][:1]:
        args.pool = pool
        args.n = 1
        cmd_chat(args)
    return 0


def cmd_verify(args) -> int:
    """Check a tokenized corpus: file size, boundary markers, special-token census."""
    from wisp.data import discover_corpora
    from wisp.tokenizer import WispTokenizer

    tok = WispTokenizer()
    corpora = discover_corpora(args.root, in_memory=False)
    bad = 0
    print(f"{'corpus':14s} {'docs':>10s} {'tokens':>13s} {'size':>9s} {'raw?':>5s} "
          f"{'bos':>9s} {'eos':>9s} {'user':>9s} {'asst':>9s} {'unk%':>6s}")
    for name in sorted(corpora):
        c = corpora[name]
        size = os.path.getsize(c.bin_path)
        ok_size = size == c.num_tokens * c.dtype.itemsize
        arr = np.memmap(c.bin_path, dtype=c.dtype, mode="r")
        # census over an evenly spaced sample (a full scan of 7 GB is slow),
        # then scale the counts back up to the whole file
        step = max(1, len(arr) // 4_000_000)
        sample = np.asarray(arr[::step])
        scale = len(arr) / max(len(sample), 1)
        census = {i: int((sample == i).sum() * scale) for i in range(15)}
        census_s = {i: int((sample == i).sum()) for i in range(15)}
        unk_pct = 100.0 * census_s[0] / max(len(sample), 1)
        chat = c.meta.get("chat_format") or ""
        ok_bos = census[1] <= c.num_docs * 1.02 + 1
        ok_eos = census[2] <= c.num_docs * 1.02 + 1
        ok_pairs = (abs(census[5] - census[6]) <= 0.02 * max(census[5], 1)) if chat else True
        flag = "" if (ok_size and ok_bos and ok_eos and ok_pairs) else "  <-- CHECK"
        bad += 0 if (ok_size and ok_bos and ok_eos and ok_pairs) else 1
        print(f"{name:14s} {c.num_docs:10,} {c.num_tokens:13,} {size/1e9:8.2f}G "
              f"{chat[:5]:>5s} {census[1]:9,} {census[2]:9,} {census[5]:9,} {census[6]:9,} "
              f"{unk_pct:5.2f}%{flag}")
    del tok
    print("\nOK" if bad == 0 else f"\n{bad} corpus(es) need attention")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    def common(p, n=2):
        p.add_argument("--data", default=DEFAULT_DATA)
        p.add_argument("--chars", type=int, default=1200, help="truncate shown text (0 = full)")
        p.add_argument("--shard", type=int, default=0)
        p.add_argument("--skip", type=int, default=0)
        p.add_argument("--n", type=int, default=n)
        p.add_argument("--full-count", action="store_true", help="count rows in every shard (slow)")

    s = sub.add_parser("summary", help="per-pool inventory")
    common(s, n=0)
    s.set_defaults(func=cmd_summary)

    s = sub.add_parser("raw", help="raw parquet documents")
    common(s)
    s.add_argument("--pool", default="pretrain/en")
    s.set_defaults(func=cmd_raw)

    s = sub.add_parser("chat", help="QA documents rendered with the special-token template")
    common(s, n=1)
    s.add_argument("--pool", default="sft/zh")
    s.set_defaults(func=cmd_chat)

    s = sub.add_parser("tokens", help="decode blocks from a tokenized corpus")
    s.add_argument("--root", required=True, help="root passed to scripts/tokenize_data.py")
    s.add_argument("--corpus", default=None)
    s.add_argument("--blocks", type=int, default=1)
    s.add_argument("--block", type=int, default=0)
    s.add_argument("--seq", type=int, default=64)
    s.set_defaults(func=cmd_tokens)

    s = sub.add_parser("verify", help="check a tokenized corpus tree")
    s.add_argument("--root", required=True)
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("overview", help="summary + one sample of each kind (default)")
    common(s, n=1)
    s.set_defaults(func=cmd_overview)

    argv = sys.argv[1:]
    if not argv or argv[0].startswith("-"):
        argv = ["overview"] + argv
    args = ap.parse_args(argv)
    for name, default in (("data", DEFAULT_DATA), ("chars", 1200), ("shard", 0),
                          ("skip", 0), ("n", 1), ("full_count", False), ("n_shard", 0)):
        if not hasattr(args, name):
            setattr(args, name, default)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
