#!/usr/bin/env python3
"""Download a *random but reproducible* slice of Ultra-FineWeb-L3 (Wispv1 data plan).

The dataset (``openbmb/Ultra-FineWeb-L3``) has 1,590 parquet shards / ~1.7 TB:

    data/ultrafineweb_en_l3/multi_style   552 shards   594.8 GB   164B tok
    data/ultrafineweb_en_l3/qa            442 shards   475.9 GB   245B tok
    data/ultrafineweb_zh_l3/multi_style   286 shards   308.1 GB    82B tok
    data/ultrafineweb_zh_l3/qa            310 shards   333.1 GB   118B tok

We only need a Chinchilla-scale slice, so shards are **sampled uniformly at
random inside each pool** (seeded -> reproducible, and the same seed can be
re-run later for an exact superset).

Category mapping (see README):  ``multi_style`` -> pretrain, ``qa`` -> sft.

    python scripts/fetch_ultrafineweb.py plan                  # show the plan
    python scripts/fetch_ultrafineweb.py download --workers 4  # fetch it
    python scripts/fetch_ultrafineweb.py measure               # real tokens/byte
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

REPO = "openbmb/Ultra-FineWeb-L3"
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DEST = "/media/ydwu/数据移动硬盘/Wispv1-Data"

# pool -> repo path, actual inventory bytes, card token count (MiniCPM5 tokenizer)
# NOTE: sft/en has 616 shards / 663.2 GB (an early recursive listing was truncated
# at 1000 entries and under-counted it at 442 shards).
POOLS: Dict[str, Dict] = {
    "pretrain/en": {"path": "data/ultrafineweb_en_l3/multi_style", "bytes": 594.82e9, "tokens": 164e9, "shards": 552},
    "pretrain/zh": {"path": "data/ultrafineweb_zh_l3/multi_style", "bytes": 308.08e9, "tokens": 82e9, "shards": 286},
    "sft/en": {"path": "data/ultrafineweb_en_l3/qa", "bytes": 663.18e9, "tokens": 245e9, "shards": 616},
    "sft/zh": {"path": "data/ultrafineweb_zh_l3/qa", "bytes": 333.13e9, "tokens": 118e9, "shards": 310},
}


# ----------------------------------------------------------------------
# inventory
# ----------------------------------------------------------------------
def fetch_tree(path: str, cache_dir: str, refresh: bool = False) -> List[dict]:
    key = hashlib.sha1(path.encode()).hexdigest()[:12]
    cache = os.path.join(cache_dir, f"tree_{key}.json")
    if refresh or not os.path.exists(cache) or os.path.getsize(cache) < 100:
        os.makedirs(cache_dir, exist_ok=True)
        url = f"{ENDPOINT}/api/datasets/{REPO}/tree/main/{path}?recursive=true"
        print(f"  listing {path} ...", flush=True)
        subprocess.run(["curl", "-sSL", "--max-time", "300", url, "-o", cache], check=True)
    with open(cache, "r", encoding="utf-8") as fh:
        entries = json.load(fh)
    if not isinstance(entries, list):
        raise RuntimeError(f"bad tree response for {path}: {str(entries)[:200]}")
    files = [
        {"path": e["path"], "size": int(e.get("size", 0))}
        for e in entries
        if e.get("type") == "file" and e["path"].endswith(".parquet")
    ]
    files.sort(key=lambda x: x["path"])
    for i, f in enumerate(files):
        f["idx"] = i
    return files


def load_inventory(cache_dir: str, refresh: bool = False) -> Dict[str, List[dict]]:
    return {pool: fetch_tree(meta["path"], cache_dir, refresh) for pool, meta in POOLS.items()}


# ----------------------------------------------------------------------
# planning
# ----------------------------------------------------------------------
def measured_tokens_per_byte(dest: str) -> Dict[str, float]:
    path = os.path.join(dest, "manifest", "tokenizer_measurement.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {pool: r["tokens_per_byte"] for pool, r in data.items() if r.get("tokens_per_byte")}


def token_targets(args) -> Dict[str, float]:
    en, zh = [float(x) for x in args.lang_mix.split(",")]
    total = args.params * args.tokens_per_param
    out = {}
    for pool in POOLS:
        cat, lang = pool.split("/")
        frac = (1 - args.sft_frac) if cat == "pretrain" else args.sft_frac
        out[pool] = total * frac * (en if lang == "en" else zh)
    return out


def card_tokens_per_byte(inv: Dict[str, List[dict]]) -> Dict[str, float]:
    """Card token counts divided by the *actual* inventory size of each pool."""
    out = {}
    for pool, files in inv.items():
        total = sum(f["size"] for f in files)
        out[pool] = POOLS[pool]["tokens"] / total if total else 1.0
    return out


def build_plan(args, inv: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    """Pick random shards per pool until the token target is covered."""
    card = card_tokens_per_byte(inv)
    tpb = measured_tokens_per_byte(args.dest)
    if tpb and not args.use_card_ratio:
        print("  using MEASURED tokens/byte from tokenizer_measurement.json")
    else:
        tpb = card
    targets = token_targets(args)
    plan: Dict[str, List[dict]] = {}
    for pool, files in inv.items():
        need_bytes = targets[pool] / tpb[pool]
        n_by_size = need_bytes / (sum(f["size"] for f in files) / len(files))
        n = min(len(files), max(1, int(round(n_by_size))))
        # shuffle-then-take (not sample(k)) so that raising the budget later only
        # *extends* the selection instead of reshuffling it -> no wasted downloads
        order = list(range(len(files)))
        random.Random(f"{args.seed}:{pool}").shuffle(order)
        picked = sorted(order[:n])
        plan[pool] = [dict(files[i], rank=k) for k, i in enumerate(picked)]
    return plan


def print_plan(args, plan: Dict[str, List[dict]], inv: Dict[str, List[dict]]) -> None:
    tpb = measured_tokens_per_byte(args.dest) or card_tokens_per_byte(inv)
    targets = token_targets(args)
    print(f"  model {args.params/1e6:.2f}M params x {args.tokens_per_param:.0f} tok/param "
          f"= {sum(targets.values())/1e9:.2f}B tokens | sft {args.sft_frac:.0%} | "
          f"lang en/zh {args.lang_mix}")
    print(f"  {'pool':13s} {'shards':>7s} {'of pool':>8s} {'GB':>8s} {'est. tokens':>12s}  shard idx range")
    tn = tb = tt = 0
    for pool, files in plan.items():
        n = len(files)
        b = sum(f["size"] for f in files)
        t = b * tpb[pool]
        tn += n
        tb += b
        tt += t
        idx = [f["idx"] for f in files]
        print(f"  {pool:13s} {n:7d} {len(inv[pool]):8d} {b/1e9:8.2f} {t/1e9:11.2f}B  "
              f"{min(idx)}..{max(idx)}")
    print(f"  {'TOTAL':13s} {tn:7d} {sum(len(v) for v in inv.values()):8d} {tb/1e9:8.2f} {tt/1e9:11.2f}B")
    print(f"  destination: {args.dest}")


# ----------------------------------------------------------------------
# download
# ----------------------------------------------------------------------
def download_one(dest: str, pool: str, f: dict, retries: int = 4) -> tuple:
    out = os.path.join(dest, pool, os.path.basename(f["path"]))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out) and os.path.getsize(out) == f["size"]:
        return pool, f["path"], "skip", f["size"]
    url = f"{ENDPOINT}/datasets/{REPO}/resolve/main/{f['path']}"
    for attempt in range(retries):
        proc = subprocess.run(
            ["curl", "-sSL", "-C", "-", "--retry", "3", "--retry-delay", "5",
             "--connect-timeout", "30", "--max-time", "7200", url, "-o", out]
        )
        if proc.returncode == 0 and os.path.exists(out) and os.path.getsize(out) == f["size"]:
            return pool, f["path"], "ok", f["size"]
        time.sleep(3 * (attempt + 1))
    size = os.path.getsize(out) if os.path.exists(out) else 0
    return pool, f["path"], "FAIL", size


def cmd_download(args) -> int:
    inv = load_inventory(os.path.join(args.cache, "tree"), args.refresh)
    plan = build_plan(args, inv)
    print_plan(args, plan, inv)
    if args.dry_run:
        return 0

    os.makedirs(os.path.join(args.dest, "manifest"), exist_ok=True)
    manifest_path = os.path.join(args.dest, "manifest", "download_manifest.json")
    jobs = [(pool, f) for pool in POOLS for f in plan[pool]]
    done: List[dict] = []
    failed: List[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, args.dest, p, f): (p, f) for p, f in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            pool, path, status, size = fut.result()
            rec = {"pool": pool, "path": path, "status": status, "size": size}
            (failed if status == "FAIL" else done).append(rec)
            gb = sum(d["size"] for d in done) / 1e9
            print(f"[{k:3d}/{len(jobs)}] {status:4s} {pool:13s} {os.path.basename(path)[:44]:46s} "
                  f"{size/1e9:5.2f} GB  total={gb:6.1f} GB  {time.time()-t0:5.0f}s", flush=True)
            with open(manifest_path, "w", encoding="utf-8") as fh:
                json.dump({"repo": REPO, "endpoint": ENDPOINT, "seed": args.seed,
                           "params": args.params, "tokens_per_param": args.tokens_per_param,
                           "sft_frac": args.sft_frac, "lang_mix": args.lang_mix,
                           "files": done + failed}, fh, indent=2, ensure_ascii=False)
    print(f"\n{len(done)} ok/skip | {len(failed)} failed | {sum(d['size'] for d in done)/1e9:.1f} GB -> {args.dest}")
    for d in failed:
        print("  FAILED:", d["path"])
    return 1 if failed else 0


# ----------------------------------------------------------------------
# measurement (real tokens/byte for *our* 24k tokenizer)
# ----------------------------------------------------------------------
def cmd_measure(args) -> int:
    sys.path.insert(0, HERE)
    from wisp.tokenizer import WispTokenizer

    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("pyarrow is required for measure")
        return 1

    tok = WispTokenizer()
    rng = random.Random(args.seed)
    report: Dict[str, dict] = {}
    for pool in POOLS:
        root = os.path.join(args.dest, pool)
        if not os.path.isdir(root):
            continue
        names = sorted(n for n in os.listdir(root) if n.endswith(".parquet"))
        if not names:
            continue
        n_docs = n_tokens = n_bytes_text = 0
        total_rows = total_bytes = 0
        text_col_used = None
        for fn in names:
            fp = os.path.join(root, fn)
            total_bytes += os.path.getsize(fp)
            pf = pq.ParquetFile(fp)
            total_rows += pf.metadata.num_rows
            cols = pf.schema_arrow.names
            text_col = next((c for c in ("text", "content", "raw_content", "answer") if c in cols), cols[0])
            text_col_used = text_col
            # only decompress the first row group: a 1 GB shard's text column can
            # be several GB once decompressed
            n_rg = pf.metadata.row_group(0).num_rows
            col = pf.read_row_group(0, columns=[text_col]).column(text_col)
            take = min(args.docs_per_shard, n_rg)
            for i in sorted(rng.sample(range(n_rg), take)):
                s = col[i].as_py()
                if not isinstance(s, str):
                    s = json.dumps(s, ensure_ascii=False) if isinstance(s, (dict, list)) else str(s)
                n_docs += 1
                n_bytes_text += len(s.encode("utf-8"))
                n_tokens += len(tok.encode(s, add_special_tokens=False))
        # mean tokens per document x documents per compressed byte
        tokens_per_doc = n_tokens / max(n_docs, 1)
        docs_per_byte = total_rows / max(total_bytes, 1)
        report[pool] = {
            "shards": len(names),
            "compressed_bytes": total_bytes,
            "rows": total_rows,
            "docs_sampled": n_docs,
            "tokens_sampled": n_tokens,
            "text_column": text_col_used,
            "tokens_per_doc": tokens_per_doc,
            "docs_per_compressed_byte": docs_per_byte,
            "tokens_per_byte": tokens_per_doc * docs_per_byte,
            "utf8_bytes_per_doc": n_bytes_text / max(n_docs, 1),
            "chars_per_token": (n_bytes_text / 3) / max(n_tokens, 1),
            "card_tokens_per_byte": POOLS[pool]["tokens"] / POOLS[pool]["bytes"],
        }
        r = report[pool]
        print(f"  {pool:13s} shards={r['shards']:3d} rows={r['rows']:,} col='{r['text_column']}' "
              f"tok/doc={r['tokens_per_doc']:8.1f} tok/byte={r['tokens_per_byte']:.4f} "
              f"(card {r['card_tokens_per_byte']:.4f})")

    out = os.path.join(args.dest, "manifest", "tokenizer_measurement.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"  written: {out}")
    return 0


# ----------------------------------------------------------------------
# manifest + verification
# ----------------------------------------------------------------------
README_TEMPLATE = """# Wispv1 预训练 / SFT 数据（Ultra-FineWeb-L3 随机切片）

本目录是从 [`openbmb/Ultra-FineWeb-L3`](https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L3)
中**按固定随机种子均匀抽样**得到的分块（shard），用于 Wispv1（{params:.2f}M 参数）的训练。

## 实际内容（本目录）

| 路径 | 内容 | shards | 体积 | 估算 token |
|---|---|---|---|---|
| `pretrain/en` | multi-style 重写文档（英文） | {pe_n} | {pe_gb:.2f} GB | {pe_tok:.2f}B |
| `pretrain/zh` | multi-style 重写文档（中文） | {pz_n} | {pz_gb:.2f} GB | {pz_tok:.2f}B |
| `sft/en` | Q&A 问答对（英文） | {se_n} | {se_gb:.2f} GB | {se_tok:.2f}B |
| `sft/zh` | Q&A 问答对（中文） | {sz_n} | {sz_gb:.2f} GB | {sz_tok:.2f}B |
| **合计** | | **{total_n}** | **{total_gb:.2f} GB** | **~{total_tokens:.2f}B** |

* 实际约 **{real_tpp:.1f} token / 参数**（Chinchilla 最优 20×，本切片取「稍多」）
* 实际预训练 : SFT ≈ **{real_pretrain_pct:.0f}% : {real_sft_pct:.0f}%**
  （SFT 略高于计划的 {sft_pct:.0f}%，因为每个 SFT 池最少也要下 1 个 shard ≈ 1.08 GB）
* 语言比例 英文 **{en_pct:.0f}%** / 中文 **{zh_pct:.0f}%**（按 `Wispv1.md`）
* 约占全数据集（1,764 shards / 1.90 TB）的 **{pct:.2f}%**

## 抽样方案

| 项 | 值 |
|---|---|
| 目标 | {tpp:.0f} × {params:.2f}M 参数 = {target_tokens:.2f}B token |
| 随机种子 | `{seed}` |
| 抽样方式 | 每个子集内对所有 shard 索引做 seeded shuffle，取前 N 个 |
| HF 镜像 | `{endpoint}` |

因为在子集内部是**均匀随机**抽取 shard，而每个 shard 本身就是随机顺序的网页文本，
所以这个切片在整个语料上无偏，不会造成领域/知识缺失。同种子重跑会得到同一集合；
**加大预算只会追加新 shard，不会重排已下载的部分**。

## 文件格式

Parquet（snappy 压缩），每行一篇文档，列为 `uid / content / style`：

* `pretrain/*`（style = `multi_style`）：重写后的正文，四种风格
  （encyclopedia / textbook / blog / abstract）
* `sft/*`（style = `qa`）：原始正文 + 多组问答对（MiniCPM4 / Qwen3 合成）

## 使用

```bash
DATA="/media/ydwu/数据移动硬盘/Wispv1-Data"

# 预训练（train.py 直接支持 parquet 目录）
python train.py --config configs/wispv1_250m.json \\
    --data $DATA/pretrain --seq-len 4096 --micro-batch 4 --grad-accum 32

# 断点续传 / 扩大预算（同 seed，已下载的会自动 skip）
python scripts/fetch_ultrafineweb.py download --tokens-per-param 30 --workers 8

# 校验完整性（大小 + parquet footer + 行数）
python scripts/fetch_ultrafineweb.py verify
```
"""


def cmd_manifest(args) -> int:
    inv = load_inventory(os.path.join(args.cache, "tree"), args.refresh)
    plan = build_plan(args, inv)
    stats = on_disk_stats(args.dest, inv)
    tpb = measured_tokens_per_byte(args.dest) or card_tokens_per_byte(inv)
    targets = token_targets(args)
    en, zh = [float(x) for x in args.lang_mix.split(",")]

    entries = {}
    for pool, files in plan.items():
        entries[pool] = [
            {"rank": f["rank"], "idx": f["idx"], "file": os.path.basename(f["path"]),
             "size": f["size"], "repo_path": f["path"]}
            for f in files
        ]
    os.makedirs(os.path.join(args.dest, "manifest"), exist_ok=True)
    with open(os.path.join(args.dest, "manifest", "dataset_plan.json"), "w", encoding="utf-8") as fh:
        json.dump({"repo": REPO, "endpoint": ENDPOINT, "seed": args.seed,
                   "params": args.params, "tokens_per_param": args.tokens_per_param,
                   "sft_frac": args.sft_frac, "lang_mix": args.lang_mix,
                   "token_targets": targets, "tokens_per_byte": tpb,
                   "plan": entries, "on_disk": stats},
                  fh, indent=2, ensure_ascii=False)

    # the README describes what is really on disk. Units: gb[p] is GB,
    # tokens_b[p] is *billions* of tokens.
    gb = {p: stats[p]["bytes"] / 1e9 for p in POOLS}
    tokens_b = {p: gb[p] * 1e9 * tpb[p] / 1e9 for p in POOLS}
    n = {p: stats[p]["shards"] for p in POOLS}
    total_bytes = sum(stats[p]["bytes"] for p in POOLS)
    total_gb = total_bytes / 1e9
    total_n = sum(n.values())
    total_tokens_b = sum(tokens_b.values())
    pre_tokens_b = sum(tokens_b[p] for p in POOLS if p.startswith("pretrain"))
    readme = README_TEMPLATE.format(
        params=args.params / 1e6, total_tokens=total_tokens_b, tpp=args.tokens_per_param,
        target_tokens=sum(targets.values()) / 1e9,
        pretrain_pct=(1 - args.sft_frac) * 100, sft_pct=args.sft_frac * 100,
        en_pct=en * 100, zh_pct=zh * 100, seed=args.seed, endpoint=ENDPOINT,
        pe_n=n["pretrain/en"], pe_gb=gb["pretrain/en"], pe_tok=tokens_b["pretrain/en"],
        pz_n=n["pretrain/zh"], pz_gb=gb["pretrain/zh"], pz_tok=tokens_b["pretrain/zh"],
        se_n=n["sft/en"], se_gb=gb["sft/en"], se_tok=tokens_b["sft/en"],
        sz_n=n["sft/zh"], sz_gb=gb["sft/zh"], sz_tok=tokens_b["sft/zh"],
        total_n=total_n, total_gb=total_gb, pct=total_bytes / 1899.2e9 * 100,
        real_tpp=total_tokens_b * 1e9 / args.params,
        real_pretrain_pct=pre_tokens_b / total_tokens_b * 100,
        real_sft_pct=(1 - pre_tokens_b / total_tokens_b) * 100,
    )
    with open(os.path.join(args.dest, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(readme)
    print(f"  manifest/dataset_plan.json + README.md written to {args.dest}")
    print(f"  on disk: {total_n} shards / {total_gb:.2f} GB / ~{total_tokens_b:.2f}B tokens "
          f"({total_tokens_b*1e9/args.params:.1f} tok/param)")
    return 0


def on_disk_stats(dest: str, inv: Dict[str, List[dict]]) -> Dict[str, dict]:
    """What is actually on disk per pool (which may be a superset of the plan)."""
    by_name = {pool: {os.path.basename(f["path"]): f for f in files} for pool, files in inv.items()}
    out: Dict[str, dict] = {}
    for pool in POOLS:
        root = os.path.join(dest, pool)
        names = sorted(n for n in os.listdir(root) if n.endswith(".parquet")) if os.path.isdir(root) else []
        nbytes = sum(os.path.getsize(os.path.join(root, n)) for n in names)
        out[pool] = {
            "shards": len(names), "bytes": nbytes,
            "files": [n for n in names],
            "idx": sorted(by_name[pool][n]["idx"] for n in names if n in by_name[pool]),
            "in_plan": sum(1 for n in names if n in by_name[pool] and by_name[pool][n].get("rank") is not None),
        }
    return out


def cmd_verify(args) -> int:
    """Verify every shard that is actually on disk (size + parquet footer + rows)."""
    inv = load_inventory(os.path.join(args.cache, "tree"), args.refresh)
    by_name = {pool: {os.path.basename(f["path"]): f for f in files} for pool, files in inv.items()}
    import pyarrow.parquet as pq

    bad, ok, rows_total, bytes_total, unknown = [], 0, 0, 0, []
    for pool in POOLS:
        root = os.path.join(args.dest, pool)
        for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            if not name.endswith(".parquet"):
                continue
            fp = os.path.join(root, name)
            size = os.path.getsize(fp)
            ref = by_name[pool].get(name)
            if ref is None:
                unknown.append((pool, name))
            elif size != ref["size"]:
                bad.append((pool, name, f"SIZE {size} != {ref['size']}"))
                continue
            try:
                md = pq.ParquetFile(fp).metadata
                rows_total += md.num_rows
                bytes_total += size
                ok += 1
            except Exception as exc:  # pragma: no cover
                bad.append((pool, name, f"PARQUET {exc}"))
    print(f"  verified {ok} shards | {bytes_total/1e9:.2f} GB | {rows_total:,} documents")
    for pool, name in unknown:
        print(f"  note: {pool}/{name} is not in the repo inventory")
    for pool, path, why in bad:
        print(f"  BAD {pool:13s} {os.path.basename(path)}: {why}")
    if bad:
        print(f"  -> re-run `download` to repair ({len(bad)} file(s))")
    return 1 if bad else 0


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["plan", "download", "measure", "manifest", "verify"])
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--cache", default=os.path.join(HERE, ".cache"))
    ap.add_argument("--seed", type=int, default=20260514)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--docs-per-shard", type=int, default=150)
    ap.add_argument("--use-card-ratio", action="store_true",
                    help="ignore a previous measurement and use the dataset card ratio")
    ap.add_argument("--params", type=float, default=259.19e6)
    ap.add_argument("--tokens-per-param", type=float, default=25.0,
                    help="Chinchilla = 20; 'a bit more than optimal' -> 25")
    ap.add_argument("--lang-mix", default="0.55,0.45", help="en,zh")
    ap.add_argument("--sft-frac", type=float, default=0.08,
                    help="share of the token budget reserved for the SFT (QA) pool")
    args = ap.parse_args()

    if args.command == "plan":
        inv = load_inventory(os.path.join(args.cache, "tree"), args.refresh)
        print_plan(args, build_plan(args, inv), inv)
        return 0
    if args.command == "download":
        return cmd_download(args)
    if args.command == "manifest":
        return cmd_manifest(args)
    if args.command == "verify":
        return cmd_verify(args)
    return cmd_measure(args)


if __name__ == "__main__":
    raise SystemExit(main())
