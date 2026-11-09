#!/usr/bin/env python3
"""Minimal interactive chat client for a Wispv1 checkpoint (talk to the SFT model).

    # interactive
    python scripts/chat.py --ckpt /media/ydwu/KINGSTON/sft/sft/final

    # one question, non-interactive
    python scripts/chat.py --ckpt ... --prompt "菌落有什么特征？"

    # raw text continuation instead of the chat template (compare with pre-SFT)
    python scripts/chat.py --ckpt ... --raw --prompt "The history of the Internet"

In-chat commands::

    /reset                  clear the conversation
    /system <text>          set (or clear) the system prompt
    /task summarize|rewrite|polish|expand|none
    /temp 0.8   /tokens 200  /rep 1.1
    /quit

Decoding defaults are chosen for a 250M model: `temperature 0.8` (greedy loops, see
docs/alpha_dynamics.md neighbours), `top_p 0.9`, and a mild repetition penalty.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from wisp.model import WispForCausalLM                       # noqa: E402
from wisp.tokenizer import WispTokenizer, load_tokenizer_near  # noqa: E402


# ----------------------------------------------------------------------
def build_prompt(tok: WispTokenizer, history: List[Tuple[str, str]],
                 system: Optional[str], task: Optional[str]) -> List[int]:
    """``<|bos|> [<|system|>…] [task] (<|user|>…<|end_of_turn|> <|assistant|>…<|end_of_turn|>)* <|assistant|>``"""
    eot = tok.reserved_name(tok.end_of_turn_token_id)
    parts: List[str] = [tok.bos_token]
    if system:
        parts += [tok.reserved_name(tok.system_token_id), system, eot]
    if task and task in tok.task_token_ids:
        parts.append(tok.reserved_name(tok.task_token_ids[task]))
    for user, assistant in history:
        parts += [tok.reserved_name(tok.user_token_id), user, eot]
        if assistant:
            parts += [tok.reserved_name(tok.assistant_token_id), assistant, eot]
    parts.append(tok.reserved_name(tok.assistant_token_id))
    return tok.encode("".join(parts), add_special_tokens=False)


@torch.no_grad()
def generate(model, tok: WispTokenizer, prompt_ids: List[int], max_new: int,
             temperature: float, top_k: int, top_p: float, rep_penalty: float,
             stop_ids: List[int], device: torch.device,
             max_context: int = 4096, stream: bool = True) -> Tuple[str, int, float]:
    """Token-by-token decoding with a KV cache, optional repetition penalty."""
    budget = max(16, max_context - max_new)
    if len(prompt_ids) > budget:                 # keep the newest context
        prompt_ids = prompt_ids[-budget:]
    cur = torch.tensor([prompt_ids], device=device)
    past = None
    out_ids: List[int] = []
    text = ""
    t0 = time.time()
    for _ in range(max_new):
        out = model(cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1].float()

        if rep_penalty != 1.0 and out_ids:
            recent = torch.tensor(sorted(set(out_ids[-64:])), device=logits.device)
            vals = logits[recent]
            logits[recent] = torch.where(vals > 0, vals / rep_penalty, vals * rep_penalty)

        if temperature <= 0:
            nxt = int(torch.argmax(logits))
        else:
            logits = logits / temperature
            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if 0 < top_p < 1.0:
                srt, idx = torch.sort(logits, descending=True)
                probs = torch.softmax(srt, dim=-1)
                drop = probs.cumsum(-1) - probs >= top_p
                srt = srt.masked_fill(drop, float("-inf"))
                logits = torch.empty_like(logits).scatter(-1, idx, srt)
            nxt = int(torch.multinomial(torch.softmax(logits, dim=-1), 1))

        if nxt in stop_ids:
            break
        out_ids.append(nxt)
        cur = torch.tensor([[nxt]], device=device)

        if stream:
            now = tok.decode(out_ids, skip_special_tokens=True)
            if len(now) > len(text):
                print(now[len(text):], end="", flush=True)
                text = now
    if not stream:
        text = tok.decode(out_ids, skip_special_tokens=True)
    return text, len(out_ids), time.time() - t0


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="directory with config.json + model.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16"])
    ap.add_argument("--system", default=None, help="system prompt")
    ap.add_argument("--task", default=None,
                    choices=["summarize", "rewrite", "polish", "expand"])
    ap.add_argument("--prompt", default=None, help="non-interactive: ask this and exit")
    ap.add_argument("--file", default=None,
                    help="prepend this file's content as the passage of the first user turn "
                         "(this is the format the model was SFT'd on: passage + question)")
    ap.add_argument("--raw", action="store_true",
                    help="plain continuation without the chat template")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--rep-penalty", type=float, default=1.1)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--no-stream", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else (args.device if args.device != "auto" else "cpu"))
    dtype = {"auto": torch.bfloat16 if device.type == "cuda" else torch.float32,
             "float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]

    tok = load_tokenizer_near(args.ckpt)
    model = WispForCausalLM.from_pretrained(args.ckpt).to(device=device, dtype=dtype).eval()
    n = sum(p.numel() for p in model.parameters())
    stop_ids = [i for i in (tok.eos_token_id, tok.end_of_turn_token_id) if i is not None]
    print(f"Wispv1 chat | {n/1e6:.1f}M params | {device} {str(dtype).replace('torch.', '')} "
          f"| vocab {tok.vocab_size} | stop on {stop_ids}")

    passage = ""
    if args.file:
        with open(args.file, "r", encoding="utf-8", errors="ignore") as fh:
            passage = fh.read().strip()
        print(f"(passage: {len(passage):,} chars from {args.file})")

    def ask(history: List[Tuple[str, str]], user: str, system, task) -> Tuple[str, int, float]:
        if passage and not history:            # the model expects passage + question in one turn
            user = f"{passage}\n\n{user}"
        ids = build_prompt(tok, history + [(user, "")], system, task)
        return generate(model, tok, ids, args.max_new_tokens, args.temperature,
                        args.top_k, args.top_p, args.rep_penalty, stop_ids, device,
                        args.max_context, stream=not args.no_stream)

    # ---- non-interactive ----
    if args.prompt is not None:
        if args.raw:
            ids = tok.encode(args.prompt, add_special_tokens=False)
            ids = [tok.bos_token_id] + ids
            print(f"续写: {args.prompt}", end="" if not args.no_stream else "\n")
            text, ntok, dt = generate(model, tok, ids, args.max_new_tokens, args.temperature,
                                      args.top_k, args.top_p, args.rep_penalty, stop_ids,
                                      device, args.max_context, stream=not args.no_stream)
        else:
            print(f"你: {args.prompt}\nWisp: ", end="", flush=True)
            text, ntok, dt = ask([], args.prompt, args.system, args.task)
        if args.no_stream:
            print(text)
        print(f"\n[{ntok} tok, {ntok/max(dt,1e-9):.1f} tok/s]")
        return 0

    # ---- interactive ----
    print("命令: /reset  /system <文本>  /task <summarize|rewrite|polish|expand|none>  "
          "/temp 0.8  /tokens 200  /rep 1.1  /quit")
    history: List[Tuple[str, str]] = []
    system, task = args.system, args.task
    while True:
        try:
            user = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit", "/q"):
            break
        if user == "/reset":
            history.clear()
            print("(已清空对话)")
            continue
        if user.startswith("/system"):
            system = user[7:].strip() or None
            print(f"(system = {system!r})")
            continue
        if user.startswith("/task"):
            val = user[5:].strip()
            task = None if val in ("", "none") else val
            print(f"(task = {task!r})")
            continue
        if user.startswith("/temp"):
            args.temperature = float(user[5:].strip())
            print(f"(temperature = {args.temperature})")
            continue
        if user.startswith("/tokens"):
            args.max_new_tokens = int(user[7:].strip())
            print(f"(max_new_tokens = {args.max_new_tokens})")
            continue
        if user.startswith("/rep"):
            args.rep_penalty = float(user[4:].strip())
            print(f"(rep_penalty = {args.rep_penalty})")
            continue

        print("Wisp: ", end="", flush=True)
        text, ntok, dt = ask(history, user, system, task)
        if args.no_stream:
            print(text, end="")
        print(f"\n[{ntok} tok, {ntok/max(dt,1e-9):.1f} tok/s]")
        history.append((user, text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
