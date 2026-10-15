"""Unit tests for the Wispv1 architecture. Run: pytest -q"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

from wisp import WispConfig, WispForCausalLM, WispModel, count_parameters, estimate_parameter_breakdown


def tiny_config(**kw) -> WispConfig:
    base = dict(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        ffn_share_layers=4,
        max_position_embeddings=128,
        alpha_init=1e-4,
    )
    base.update(kw)
    return WispConfig(**base)


# ---------------------------------------------------------------- config
def test_config_validation():
    with pytest.raises(ValueError):
        tiny_config(ffn_share_layers=3)  # must be even
    with pytest.raises(ValueError):
        tiny_config(ffn_share_layers=8)  # > num_hidden_layers
    with pytest.raises(ValueError):
        tiny_config(num_attention_heads=4, num_key_value_heads=3)  # not a multiple
    cfg = tiny_config()
    assert cfg.num_shared_pairs == 2
    assert cfg.num_independent_layers == 2
    assert cfg.num_ffn_modules == 4


def test_estimate_matches_actual():
    cfg = tiny_config()
    model = WispForCausalLM(cfg)
    est = estimate_parameter_breakdown(cfg)["total"]
    actual = count_parameters(model) / 1e6
    assert abs(est - actual) / actual < 1e-6, (est, actual)


# ------------------------------------------------------------- structure
def test_ffn_pair_sharing_structure():
    cfg = tiny_config(ffn_share_layers=4, num_hidden_layers=6)
    model = WispForCausalLM(cfg)
    layers = model.model.layers
    # layers 0/1 form a pair, 2/3 form a pair
    assert layers[0].ffn is layers[1].ffn
    assert layers[2].ffn is layers[3].ffn
    assert layers[0].ffn is not layers[2].ffn
    # free layers have their own FFN
    assert layers[4].ffn is not layers[5].ffn
    assert layers[4].ffn not in [layers[0].ffn, layers[2].ffn]
    # attention is NEVER shared
    assert len({id(l.self_attn) for l in layers}) == 6
    # named_parameters() de-duplicates by identity: shared FFNs appear once
    ffn_params = [n for n, _ in model.named_parameters() if ".ffn." in n]
    assert len(ffn_params) == cfg.num_ffn_modules * 3, ffn_params
    assert all(n.startswith("model.layers.") for n in ffn_params), ffn_params
    # the FFN of a pair lives under its first layer: 0, 2 (shared) + 4, 5 (free)
    ffn_layers = sorted({int(n.split(".")[2]) for n in ffn_params})
    assert ffn_layers == [0, 2, 4, 5], ffn_layers
    # ... and the checkpoint stores each aliased tensor exactly once
    canon = model.canonical_state_dict()
    assert len([k for k in canon if ".ffn." in k]) == cfg.num_ffn_modules * 3
    assert len(canon) == len(dict(model.named_parameters()))
    # the naive state_dict would repeat the shared FFNs once per layer path
    assert len([k for k in model.state_dict() if ".ffn." in k]) == 6 * 3


def test_shared_ffn_receives_gradient_from_both_layers():
    torch.manual_seed(0)
    cfg = tiny_config(num_hidden_layers=2, ffn_share_layers=2, alpha_mode="none")
    model = WispForCausalLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(x, labels=x)
    out.loss.backward()
    g = model.model.shared_ffns[0].gate_proj.weight.grad
    assert g is not None and torch.isfinite(g).all()

    # zero the output of the pair's second layer's FFN path and compare grads:
    # both layers must be able to write into the shared FFN.
    model.zero_grad(set_to_none=True)
    hidden = model.model.embed_tokens(x).detach()
    h0, _ = model.model.layers[0](hidden, model.model.rotary)
    # layer 1 alone
    h1, _ = model.model.layers[1](hidden, model.model.rotary)
    h1.sum().backward(retain_graph=True)
    g_single = model.model.shared_ffns[0].down_proj.weight.grad.clone()
    model.zero_grad(set_to_none=True)
    h0.sum().backward()
    g_other = model.model.shared_ffns[0].down_proj.weight.grad.clone()
    assert g_single.abs().sum() > 0
    assert g_other.abs().sum() > 0
    assert not torch.allclose(g_single, g_other)


# ---------------------------------------------------------------- alpha
def test_alpha_placement_and_init():
    cfg = tiny_config(num_hidden_layers=6, ffn_share_layers=4, alpha_init=1e-4, alpha_mode="shared")
    model = WispForCausalLM(cfg)
    for i, layer in enumerate(model.model.layers):
        if i < 4:
            assert layer.residual_attn.alpha is not None
            assert layer.residual_ffn.alpha is not None
            assert torch.allclose(layer.residual_attn.alpha, torch.full((cfg.hidden_size,), 1e-4))
        else:
            assert layer.residual_attn.alpha is None
            assert layer.residual_ffn.alpha is None

    cfg_none = tiny_config(alpha_mode="none")
    m2 = WispForCausalLM(cfg_none)
    assert all(l.residual_attn.alpha is None for l in m2.model.layers)
    assert not any("alpha" in n for n, _ in m2.named_parameters())

    cfg_all = tiny_config(alpha_mode="all", num_hidden_layers=6, ffn_share_layers=4)
    m3 = WispForCausalLM(cfg_all)
    assert all(l.residual_attn.alpha is not None for l in m3.model.layers)
    # 6 layers x 2 residuals x 64 channels
    assert sum(p.numel() for n, p in m3.named_parameters() if "alpha" in n) == 6 * 2 * 64


def test_residual_formula_g_eq_f_plus_alpha_x():
    """alpha_target="identity": alpha=0 -> exactly the sub-layer output."""
    torch.manual_seed(0)
    cfg = tiny_config(num_hidden_layers=2, ffn_share_layers=2, alpha_target="identity")
    model = WispForCausalLM(cfg)
    layer = model.model.layers[0]
    with torch.no_grad():
        layer.residual_attn.alpha.zero_()
        layer.residual_ffn.alpha.zero_()
        x = torch.randn(2, 5, cfg.hidden_size)
        out, _ = layer(x, model.model.rotary)
        h = layer.input_layernorm(x)
        expect = layer.ffn(layer.post_attention_layernorm(layer.self_attn(h, model.model.rotary)[0]))
    assert torch.allclose(out, expect, atol=1e-6)

    # ... and with alpha == 1 it must equal the plain residual
    with torch.no_grad():
        layer.residual_attn.alpha.fill_(1.0)
        layer.residual_ffn.alpha.fill_(1.0)
        out1, _ = layer(x, model.model.rotary)
        a = layer.self_attn(layer.input_layernorm(x), model.model.rotary)[0]
        mid = x + a
        f = layer.ffn(layer.post_attention_layernorm(mid))
        expect1 = mid + f
    assert torch.allclose(out1, expect1, atol=1e-5)


def test_alpha_is_per_channel():
    cfg = tiny_config(alpha_mode="all")
    model = WispForCausalLM(cfg)
    a = model.model.layers[0].residual_attn.alpha
    assert a.shape == (cfg.hidden_size,)
    assert a.requires_grad


# ---------------------------------------------------------------- forward
def test_forward_shapes_and_loss():
    cfg = tiny_config()
    model = WispForCausalLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, 16))
    out = model(x)
    assert out.logits.shape == (3, 16, cfg.vocab_size)
    assert out.loss is None
    out = model(x, labels=x)
    assert out.loss.ndim == 0 and torch.isfinite(out.loss)


def test_labels_ignore_index():
    cfg = tiny_config()
    model = WispForCausalLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = x.clone()
    y[:, -4:] = -100
    loss_masked = model(x, labels=y).loss
    y2 = x.clone()
    loss_full = model(x, labels=y2).loss
    assert torch.isfinite(loss_masked)
    assert not torch.isclose(loss_masked, loss_full)


def test_causality_no_future_leak():
    torch.manual_seed(0)
    cfg = tiny_config(final_norm=True)
    model = WispForCausalLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        base = model(x).logits
        x2 = x.clone()
        x2[0, 7] = (x2[0, 7] + 13) % cfg.vocab_size  # change a future token
        alt = model(x2).logits
    assert torch.allclose(base[0, :7], alt[0, :7], atol=1e-5)
    assert not torch.allclose(base[0, 7:], alt[0, 7:], atol=1e-5)


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    cfg = tiny_config()
    model = WispForCausalLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 9))
    with torch.no_grad():
        full = model(x).logits
        out1 = model(x[:, :5], use_cache=True)
        past = out1.past_key_values
        out2 = model(x[:, 5:], past_key_values=past, use_cache=True)
        inc = torch.cat([out1.logits, out2.logits], dim=1)
    assert torch.allclose(full, inc, atol=1e-5)


def test_attention_mask_padding():
    torch.manual_seed(0)
    cfg = tiny_config()
    model = WispForCausalLM(cfg).eval()
    x = torch.randint(1, cfg.vocab_size, (2, 10))
    mask = torch.ones_like(x)
    mask[1, 6:] = 0
    x_pad = x.clone()
    x_pad[1, 6:] = cfg.pad_token_id
    with torch.no_grad():
        a = model(x, attention_mask=mask).logits
        b = model(x_pad, attention_mask=mask).logits
    assert torch.allclose(a[0], b[0], atol=1e-5)
    assert torch.allclose(a[1, :6], b[1, :6], atol=1e-5)


# ---------------------------------------------------------------- tying
def test_padding_row_is_zero():
    """Init order must not re-randomise the embedding when weights are tied."""
    for tied in (True, False):
        cfg = tiny_config(tie_word_embeddings=tied)
        model = WispForCausalLM(cfg)
        assert torch.all(model.model.embed_tokens.weight[cfg.pad_token_id] == 0)
        # a normal embedding row must not be all zeros
        assert model.model.embed_tokens.weight[7].abs().sum() > 0


def test_weight_tying_and_roundtrip():
    cfg = tiny_config()
    model = WispForCausalLM(cfg)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    x = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        ref = model(x).logits
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        loaded = WispForCausalLM.from_pretrained(d)
        with torch.no_grad():
            got = loaded(x).logits
    assert torch.allclose(ref, got, atol=1e-6)
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight


def test_untied_embeddings():
    cfg = tiny_config(tie_word_embeddings=False)
    model = WispForCausalLM(cfg)
    assert model.lm_head.weight is not model.model.embed_tokens.weight
    est = estimate_parameter_breakdown(cfg)["total"]
    assert abs(est - count_parameters(model) / 1e6) / est < 1e-6


# ---------------------------------------------------------------- misc
def test_gqa_shapes():
    cfg = tiny_config(num_attention_heads=8, num_key_value_heads=2, head_dim=8, hidden_size=64)
    assert cfg.num_query_groups == 4
    model = WispForCausalLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    assert model(x).logits.shape == (1, 4, cfg.vocab_size)


def test_positions_beyond_train_len_with_scaling():
    cfg = tiny_config(max_position_embeddings=32, rope_scaling={"type": "linear", "factor": 4.0})
    model = WispForCausalLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 64))  # 2x the "trained" window
    assert model(x).logits.shape == (1, 64, cfg.vocab_size)


def test_dynamic_ntk_rope():
    cfg = tiny_config(rope_scaling={"type": "dynamic", "factor": 8.0})
    model = WispForCausalLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 20))
    assert torch.isfinite(model(x).logits).all()


def test_parameter_breakdown_totals():
    cfg = tiny_config()
    model = WispForCausalLM(cfg)
    bd = model.parameter_breakdown()
    total = sum(bd.values())
    assert abs(total - count_parameters(model) / 1e6) < 1e-4


def test_250m_config_parameter_count():
    cfg = WispConfig()  # d=768, L=48, 12/3 heads, ffn 2048, first 24 layers shared
    est = estimate_parameter_breakdown(cfg)
    assert 0.24e3 < est["total"] < 0.27e3, est
    model = WispForCausalLM(cfg)
    n = count_parameters(model)
    assert abs(n / 1e6 - est["total"]) / est["total"] < 1e-3
    assert len(model.model.shared_ffns) == 12
    assert len(model.model.free_ffns) == 24


def test_alpha_target_branch_is_rezero():
    """alpha_target='branch' (the default) -> g = x + alpha * f(x)."""
    torch.manual_seed(0)
    cfg = tiny_config(num_hidden_layers=2, ffn_share_layers=2)
    assert cfg.alpha_target == "branch", "ReZero must be the default"
    model = WispForCausalLM(cfg)
    layer = model.model.layers[0]
    x = torch.randn(2, 4, cfg.hidden_size)
    with torch.no_grad():
        layer.residual_attn.alpha.zero_()
        layer.residual_ffn.alpha.zero_()
        out, _ = layer(x, model.model.rotary)
        assert torch.allclose(out, x, atol=1e-6), "alpha=0 on the branch -> identity"

        layer.residual_attn.alpha.fill_(1.0)
        layer.residual_ffn.alpha.fill_(1.0)
        out1, _ = layer(x, model.model.rotary)
        a = layer.self_attn(layer.input_layernorm(x), model.model.rotary)[0]
        mid = x + a
        expect = mid + layer.ffn(layer.post_attention_layernorm(mid))
    assert torch.allclose(out1, expect, atol=1e-5)


def test_alpha_target_validation():
    with pytest.raises(ValueError):
        tiny_config(alpha_target="nope")


def test_generation_smoke():
    torch.manual_seed(0)
    cfg = tiny_config()
    model = WispForCausalLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(x, max_new_tokens=5, temperature=0.0)
    assert out.shape == (1, 9)
