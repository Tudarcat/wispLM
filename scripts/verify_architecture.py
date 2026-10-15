#!/usr/bin/env python3
"""Structural + numerical self-check for a Wispv1 config.

    python scripts/verify_architecture.py                        # configs/wispv1_250m.json
    python scripts/verify_architecture.py --config my.json --seq-len 256
    python scripts/verify_architecture.py --tiny --train-steps 30

Prints a report and exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time
from typing import List

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wisp.config import WispConfig  # noqa: E402
from wisp.model import WispForCausalLM, count_parameters, estimate_parameter_breakdown  # noqa: E402

OK, BAD = "  [ok]  ", " [FAIL] "
_failures: List[str] = []


def check(cond: bool, msg: str) -> None:
    print((OK if cond else BAD) + msg)
    if not cond:
        _failures.append(msg)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "wispv1_250m.json"))
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--train-steps", type=int, default=3)
    ap.add_argument("--tiny", action="store_true", help="use a tiny config (fast)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = (
        WispConfig(
            vocab_size=512, hidden_size=128, num_hidden_layers=8, num_attention_heads=4,
            num_key_value_heads=2, head_dim=32, intermediate_size=384, ffn_share_layers=4,
            max_position_embeddings=512,
        )
        if args.tiny
        else WispConfig.load(args.config)
    )
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    section("config")
    print(f"  {cfg.to_dict()}")

    model = WispForCausalLM(cfg).to(device)
    n = count_parameters(model)
    est = estimate_parameter_breakdown(cfg)
    section("parameters")
    print(f"  actual unique params : {n:,} ({n/1e6:.2f}M)")
    print(f"  analytic estimate    : {est['total']:.2f}M")
    print(f"  breakdown (M)        : {{{', '.join(f'{k}: {v:.2f}' for k, v in model.parameter_breakdown().items())}}}")
    check(abs(n / 1e6 - est["total"]) / est["total"] < 1e-3, "counted params match analytic estimate")

    # ---- depth / sharing -------------------------------------------
    section("FFN weight sharing")
    L, S = cfg.num_hidden_layers, cfg.ffn_share_layers
    layers = model.model.layers
    pairs_ok = all(layers[2 * i].ffn is layers[2 * i + 1].ffn for i in range(S // 2))
    distinct_ok = len({id(layers[2 * i].ffn) for i in range(S // 2)}) == S // 2
    free_ok = all(not layers[i].uses_shared_ffn for i in range(S, L))
    free_distinct = len({id(layers[i].ffn) for i in range(S, L)}) == L - S
    attn_distinct = len({id(l.self_attn) for l in layers}) == L
    print(f"  LayerPair region : layers 0..{S-1} -> {S//2} pairs, {S//2} distinct FFN modules")
    print(f"  free region      : layers {S}..{L-1} -> {L-S} distinct FFN modules")
    check(pairs_ok, "each pair shares one FFN module")
    check(distinct_ok, "different pairs have different FFNs")
    check(free_ok and free_distinct, "free layers have independent FFNs")
    check(attn_distinct, "attention (GQA) is never shared")
    total_ffn = len([k for k in model.canonical_state_dict() if ".ffn." in k]) // 3
    check(total_ffn == S // 2 + (L - S), f"checkpoint holds {total_ffn} FFN modules, not {S} copies")
    eff = 3 * cfg.hidden_size * cfg.intermediate_size * S / 2 / 1e6
    print(f"  params saved by sharing: {eff:.1f}M (~{eff*4:.0f} MB fp32)")

    # ---- alpha ------------------------------------------------------
    section("alpha-scaled residuals")
    shared_alphas = [l.residual_attn.alpha for l in layers[:S]]
    has = all(a is not None for a in shared_alphas) and all(l.residual_ffn.alpha is not None for l in layers[:S])
    init_ok = all(torch.allclose(a, torch.full_like(a, cfg.alpha_init)) for a in shared_alphas)
    plain_outside = all(l.residual_attn.alpha is None for l in layers[S:])
    seen, n_alpha = set(), 0
    for name, p in model.named_parameters():
        if "alpha" in name and id(p) not in seen:
            seen.add(id(p))
            n_alpha += p.numel()
    print(f"  alpha vectors: {len(seen)} x {cfg.hidden_size} = {n_alpha:,} params (mode={cfg.alpha_mode})")
    formula = ("g = f(RMSNorm(x)) + alpha * x" if cfg.alpha_target == "identity"
               else "g = RMSNorm(x) + alpha * f(RMSNorm(x))")
    print(f"  formula: {formula}   (Pre-LN; alpha_target={cfg.alpha_target})")
    check(has, f"layers 0..{S-1} carry alpha on both residual paths")
    check(init_ok, f"alpha initialised to {cfg.alpha_init}")
    check(plain_outside if cfg.alpha_mode == "shared" else True, f"layers {S}.. use a plain residual")

    # ---- exact formula check ---------------------------------------
    with torch.no_grad():
        layer = layers[0]
        x = torch.randn(2, 5, cfg.hidden_size, device=device)
        saved = (layer.residual_attn.alpha.clone(), layer.residual_ffn.alpha.clone())
        layer.residual_attn.alpha.zero_()
        layer.residual_ffn.alpha.zero_()
        out, _ = layer(x, model.model.rotary)
        h = layer.input_layernorm(x)
        expect = layer.ffn(layer.post_attention_layernorm(layer.self_attn(h, model.model.rotary)[0]))
        layer.residual_attn.alpha.copy_(saved[0])
        layer.residual_ffn.alpha.copy_(saved[1])
    if cfg.alpha_target == "identity":
        check(torch.allclose(out, expect, atol=1e-6), "alpha=0 -> layer output is exactly the sub-layer output")
    else:
        check(torch.allclose(out, x, atol=1e-6), "alpha=0 -> layer output is exactly the identity (ReZero)")

    # ---- tying -------------------------------------------------------
    section("weight tying")
    tied = model.lm_head.weight is model.model.embed_tokens.weight
    check(tied == cfg.tie_word_embeddings, f"embedding <-> LM head tied = {tied}")
    vo = cfg.hidden_size * cfg.vocab_size / 1e6
    print(f"  saving {vo:.1f}M params (~{vo*4:.0f} MB fp32)")

    # ---- forward / backward -----------------------------------------
    section(f"forward+backward  (batch={args.batch_size}, seq={args.seq_len}, device={device.type})")
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), device=device)
    t0 = time.time()
    out = model(ids, labels=ids)
    bwd_start = time.time()
    out.loss.backward()
    t1 = time.time()
    print(f"  loss {out.loss.item():.4f} | fwd {bwd_start-t0:.2f}s | bwd {t1-bwd_start:.2f}s")
    check(torch.isfinite(out.loss), "loss is finite")
    grads_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters() if p.requires_grad
    )
    check(grads_ok, "all parameter gradients are finite and present")
    # the shared FFN must have accumulated gradient from both layers of the pair
    g0 = layers[0].ffn.down_proj.weight.grad
    check(g0 is not None and g0.abs().sum() > 0, "shared FFN receives gradient")
    alpha_grad_ok = layers[0].residual_attn.alpha.grad is not None and layers[0].residual_attn.alpha.grad.abs().sum() > 0
    check(alpha_grad_ok, "alpha receives gradient")
    if device.type == "cpu":
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
        print(f"  peak RSS {rss:.2f} GB (fp32 weights + grads + activations)")

    # ---- save / load --------------------------------------------------
    section("checkpoint round trip")
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        size = os.path.getsize(os.path.join(d, "model.pt")) / 1e6
        loaded = WispForCausalLM.from_pretrained(d).to(device)
        model.eval()
        with torch.no_grad():
            a = model(ids).logits
            b = loaded(ids).logits
        print(f"  checkpoint size {size:.1f} MB fp32")
        check(torch.allclose(a, b, atol=1e-5), "loaded model reproduces logits")

    print("\n" + ("ALL CHECKS PASSED" if not _failures else f"{len(_failures)} CHECK(S) FAILED"))
    for f in _failures:
        print(f"  - {f}")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
