#!/usr/bin/env python3
"""Download ``Morton-Li/QiYuanTokenizer-Small`` into ``<repo>/tokenizer``.

The main HF endpoint is not reachable from some networks; ``hf-mirror.com`` is
used by default.  Override with ``--endpoint`` or the ``HF_ENDPOINT`` env var.

    python scripts/download_tokenizer.py
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

REPO_ID = "Morton-Li/QiYuanTokenizer-Small"
FILES = [
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer.vocab",
    "tokenizer_config.json",
    "tokenizer.py",
    "README.md",
]

DEFAULT_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")


def download(repo_id: str, filename: str, endpoint: str, out_dir: str) -> str:
    url = f"{endpoint.rstrip('/')}/{repo_id}/resolve/main/{filename}"
    dst = os.path.join(out_dir, filename)
    tmp = dst + ".part"
    print(f"  -> {filename}")
    req = urllib.request.Request(url, headers={"User-Agent": "wispv1-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            fh.write(chunk)
    os.replace(tmp, dst)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tokenizer"),
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"downloading {args.repo_id} from {args.endpoint} -> {args.out_dir}")
    for f in FILES:
        download(args.repo_id, f, args.endpoint, args.out_dir)

    # sanity check
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from wisp.tokenizer import WispTokenizer

        tok = WispTokenizer(args.out_dir)
        info = tok.special_tokens_map()
        ids = tok.encode("你好，Wispv1！Hello world.")
        print("vocab_size:", info["vocab_size"])
        print("special ids:", info)
        print("sample ids:", ids)
        print("roundtrip:", tok.decode(ids))
    except Exception as exc:  # pragma: no cover
        print(f"[warn] sanity check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
