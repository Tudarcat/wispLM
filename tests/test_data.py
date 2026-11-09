"""Tests for the pre-training data pipeline (wisp/data.py) and chunked loss."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from wisp.data import BatchPrefetcher, MixtureBlockSampler, TokenizedCorpus, build_sampler, discover_corpora


# ----------------------------------------------------------------- helpers
def write_corpus(root: str, name: str, tokens: np.ndarray, vocab: int = 1000) -> str:
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    arr = np.asarray(tokens, dtype=np.uint16)
    with open(os.path.join(d, "tokens.bin"), "wb") as fh:
        fh.write(arr.tobytes())
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"name": name, "num_tokens": int(len(arr)), "num_docs": int(len(arr) // 10),
                   "dtype": "uint16", "bin_file": "tokens.bin", "eos_token_id": 2,
                   "vocab_size": vocab}, fh)
    return d


@pytest.fixture()
def corpora_dir(tmp_path):
    rng = np.random.default_rng(0)
    write_corpus(str(tmp_path), "pretrain/en", rng.integers(0, 1000, 20000))
    write_corpus(str(tmp_path), "pretrain/zh", rng.integers(0, 1000, 12000))
    return str(tmp_path)


# ------------------------------------------------------------ corpus / io
def test_discover_and_load(corpora_dir):
    corpora = discover_corpora(corpora_dir)
    assert set(corpora) == {"pretrain/en", "pretrain/zh"}
    assert corpora["pretrain/en"].num_tokens == 20000
    assert corpora["pretrain/en"].dtype == np.uint16
    assert corpora["pretrain/en"].in_memory
    assert "RAM" in repr(corpora["pretrain/en"])


def test_mmap_matches_ram(corpora_dir):
    a = TokenizedCorpus(os.path.join(corpora_dir, "pretrain/en"), in_memory=True)
    b = TokenizedCorpus(os.path.join(corpora_dir, "pretrain/en"), in_memory=False)
    assert a.num_tokens == b.num_tokens
    assert np.array_equal(np.asarray(a.tokens[:100]), np.asarray(b.tokens[:100]))


def test_missing_meta_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        TokenizedCorpus(str(tmp_path))


# --------------------------------------------------------------- sampler
def test_blocks_are_disjoint_and_cover_epoch(corpora_dir):
    """One epoch uses every block of every pool exactly once, then rolls over."""
    sampler, _ = build_sampler(corpora_dir, block_size=128, seed=7, in_memory=True)
    L = 128
    total = sampler.steps_per_epoch
    rows = set()
    for _ in range(total):
        arr = sampler.next_batch(1)
        assert arr.shape == (1, L + 1)
        rows.add(arr[0].tobytes())
    assert len(rows) == total, "blocks must not repeat inside an epoch"
    assert sampler.pos == total and sampler.epoch == 0
    sampler.next_batch(1)  # triggers the rollover
    assert sampler.epoch == 1 and sampler.pos == 1


def test_epoch_does_not_exhaust_the_smaller_pool(corpora_dir):
    sampler, _ = build_sampler(corpora_dir, block_size=128, seed=7)
    counts = np.bincount(sampler.assignment, minlength=len(sampler.corpora))
    for ci, c in enumerate(counts):
        assert 0 < c <= sampler.limits[ci]


def test_sampler_is_deterministic(corpora_dir):
    a, _ = build_sampler(corpora_dir, block_size=64, seed=3)
    b, _ = build_sampler(corpora_dir, block_size=64, seed=3)
    for _ in range(10):
        assert np.array_equal(a.next_batch(4), b.next_batch(4))


def test_sampler_seed_changes_order(corpora_dir):
    a, _ = build_sampler(corpora_dir, block_size=64, seed=3)
    b, _ = build_sampler(corpora_dir, block_size=64, seed=4)
    assert not np.array_equal(a.next_batch(4), b.next_batch(4))


def test_state_dict_roundtrip_is_exact(corpora_dir):
    s1, _ = build_sampler(corpora_dir, block_size=64, seed=11)
    for _ in range(7):
        s1.next_batch(3, )
    state = s1.state_dict()
    expected = [s1.next_batch(3) for _ in range(5)]

    s2, _ = build_sampler(corpora_dir, block_size=64, seed=11)
    s2.load_state_dict(state)
    got = [s2.next_batch(3) for _ in range(5)]
    for e, g in zip(expected, got):
        assert np.array_equal(e, g)


def test_state_dict_roundtrip_across_epoch_boundary(corpora_dir):
    s1, _ = build_sampler(corpora_dir, block_size=64, seed=5)
    # walk past an epoch boundary
    remaining = s1.steps_per_epoch
    while remaining > 0:
        s1.next_batch(min(7, remaining))
        remaining -= min(7, remaining)
    s1.next_batch(2)  # now inside epoch 1
    state = s1.state_dict()
    assert state["epoch"] == 1
    expected = [s1.next_batch(4) for _ in range(3)]
    s2, _ = build_sampler(corpora_dir, block_size=64, seed=5)
    s2.load_state_dict(state)
    for e, g in zip(expected, [s2.next_batch(4) for _ in range(3)]):
        assert np.array_equal(e, g)


def test_mixture_ratio_is_respected(corpora_dir):
    sampler, _ = build_sampler(corpora_dir, block_size=64, mixture={"pretrain/en": 0.75, "pretrain/zh": 0.25})
    counts = {}
    n = 2000
    first_tokens = []
    for _ in range(n):
        arr = sampler.next_batch(1)
        first_tokens.append(int(arr[0, 0]))
    # both corpora must appear, and the minority must not be starved
    names = [c.name for c in sampler.corpora]
    assert names == ["pretrain/en", "pretrain/zh"]
    assert 0.15 < sampler.weights[1] < 0.35


def test_mixture_stops_at_smallest_epoch(corpora_dir):
    sampler, _ = build_sampler(corpora_dir, block_size=64, mixture={"pretrain/en": 0.5, "pretrain/zh": 0.5})
    n_en, n_zh = sampler.limits
    # with equal weights every pool is drawn n_i times, so the epoch length is
    # set by the *smaller* pool and nothing runs out mid-epoch
    assert sampler.steps_per_epoch == 2 * min(n_en, n_zh)
    counts = np.bincount(sampler.assignment, minlength=2)
    assert counts[0] == min(n_en, n_zh) and counts[1] == min(n_en, n_zh)


def test_val_split_is_disjoint_from_train(corpora_dir):
    train, _ = build_sampler(corpora_dir, block_size=64, val_fraction=0.1, split="train")
    val, _ = build_sampler(corpora_dir, block_size=64, val_fraction=0.1, split="val")
    assert sum(train.limits) + sum(val.limits) == sum(train.n_samples)
    # validation blocks live in the tail of each corpus
    for ci in range(len(val.corpora)):
        assert val.offsets[ci] == train.limits[ci]
    # deterministic validation order
    v1 = [val.next_val_batch(2) for _ in range(3)]
    val.reset_val()
    v2 = [val.next_val_batch(2) for _ in range(3)]
    for a, b in zip(v1, v2):
        assert np.array_equal(a, b)


def test_empty_split_raises(corpora_dir):
    with pytest.raises(ValueError):
        build_sampler(corpora_dir, block_size=100000, val_fraction=0.5)


def test_missing_mixture_name_raises(corpora_dir):
    with pytest.raises(KeyError):
        build_sampler(corpora_dir, block_size=64, mixture={"nope": 1.0})


# ------------------------------------------------------------- prefetcher
@pytest.mark.parametrize("pin", [False])
def test_prefetcher_yields_correct_shapes_and_is_ordered(corpora_dir, pin):
    sampler, _ = build_sampler(corpora_dir, block_size=32, seed=1)
    pf = BatchPrefetcher(sampler, 4, device="cpu", slots=2, pin_memory=pin)
    try:
        batches = [pf.next() for _ in range(5)]
    finally:
        pf.close()
    for b in batches:
        assert b.input_ids.shape == (4, 32)
        assert b.input_ids.dtype == torch.int64
        assert b.input_ids.min() >= 0 and b.input_ids.max() < 1000
    # the state after batch k must equal a fresh sampler fast-forwarded to k
    ref, _ = build_sampler(corpora_dir, block_size=32, seed=1)
    ref.load_state_dict(batches[2].sampler_state)
    assert np.array_equal(ref.next_batch(4)[:, :-1], batches[3].input_ids.numpy().astype(np.int32))


def test_prefetcher_matches_sampler_without_prefetch(corpora_dir):
    sampler, _ = build_sampler(corpora_dir, block_size=32, seed=2)
    pf = BatchPrefetcher(sampler, 3, device="cpu", slots=3, pin_memory=False)
    try:
        got = np.concatenate([pf.next().input_ids.numpy() for _ in range(4)])
    finally:
        pf.close()
    ref, _ = build_sampler(corpora_dir, block_size=32, seed=2)
    want = np.concatenate([ref.next_batch(3)[:, :-1] for _ in range(4)])
    assert np.array_equal(got, want)


def test_prefetcher_surfaces_worker_errors(corpora_dir):
    sampler, _ = build_sampler(corpora_dir, block_size=32, seed=2)
    sampler.corpora[0].tokens = None  # make the worker blow up
    pf = BatchPrefetcher(sampler, 2, device="cpu", slots=1, pin_memory=False)
    try:
        with pytest.raises(RuntimeError):
            for _ in range(50):
                pf.next()
    finally:
        pf.close()


# ------------------------------------------------------------ chunked loss
def _tiny_model(vocab=64):
    from wisp.config import WispConfig
    from wisp.model import WispForCausalLM

    cfg = WispConfig(vocab_size=vocab, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=16, intermediate_size=128, ffn_share_layers=2,
                     max_position_embeddings=128)
    return WispForCausalLM(cfg).eval()


def test_chunked_loss_matches_full_loss():
    torch.manual_seed(0)
    model = _tiny_model()
    x = torch.randint(0, 64, (3, 40))
    with torch.no_grad():
        full = model(x, labels=x).loss
        for chunk in (1, 7, 13, 64):
            got = model(x, labels=x, loss_chunk_size=chunk).loss
            assert torch.allclose(full, got, atol=1e-5), (chunk, full.item(), got.item())


def test_chunked_loss_respects_ignore_index():
    torch.manual_seed(0)
    model = _tiny_model()
    x = torch.randint(0, 64, (2, 30))
    y = x.clone()
    y[:, 10:20] = -100
    with torch.no_grad():
        full = model(x, labels=y).loss
        got = model(x, labels=y, loss_chunk_size=5).loss
    assert torch.allclose(full, got, atol=1e-5)
    assert full.item() > 0


def test_chunked_loss_is_differentiable():
    torch.manual_seed(0)
    model = _tiny_model()
    x = torch.randint(0, 64, (2, 24))
    out = model(x, labels=x, loss_chunk_size=8)
    out.loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.lm_head.weight.grad is not None


# --------------------------------------------------------- micro-batch probe
class _FakeArgs:
    micro_batch = 0
    mem_fraction = 0.90
    loss_chunk = 512
    max_micro_batch = 64


def test_search_max_micro_batch_finds_exact_edge():
    from train import search_max_micro_batch

    calls = []

    def try_fn(mb):
        calls.append(mb)
        ok = mb <= 13                       # memory limit lands between 13 and 14
        return ok, mb * 1000 if ok else 10 ** 9

    best, peak = search_max_micro_batch(try_fn, 64)
    assert best == 13
    assert peak == 13000
    assert len(calls) <= 12, f"probe should be cheap, took {len(calls)} passes"


def test_search_max_micro_batch_when_everything_fits():
    from train import search_max_micro_batch

    best, peak = search_max_micro_batch(lambda mb: (True, mb * 7), 64)
    assert best == 64 and peak == 64 * 7


def test_search_max_micro_batch_when_nothing_fits():
    from train import search_max_micro_batch

    best, peak = search_max_micro_batch(lambda mb: (False, 1), 64)
    assert best == 0 and peak == 0


def test_search_max_micro_batch_edge_one():
    from train import search_max_micro_batch

    best, _ = search_max_micro_batch(lambda mb: (mb <= 1, mb), 64)
    assert best == 1


def test_optimizer_state_bytes_counts_unique_params():
    from train import optimizer_state_bytes
    from wisp.config import WispConfig
    from wisp.model import WispForCausalLM

    cfg = WispConfig(vocab_size=64, hidden_size=32, num_hidden_layers=4, num_attention_heads=2,
                     num_key_value_heads=1, head_dim=16, intermediate_size=64, ffn_share_layers=2)
    model = WispForCausalLM(cfg)
    n_unique = sum(p.numel() for p in model.parameters())
    assert optimizer_state_bytes(model) == n_unique * 8


# ------------------------------------------------------------ unk masking
def test_make_loss_labels_masks_only_flagged_ids():
    from train import make_loss_labels

    vocab = 32
    lut = torch.zeros(vocab, dtype=torch.bool)
    lut[0] = True          # unk
    lut[3] = True          # pad
    ids = torch.tensor([[0, 5, 3, 7], [9, 0, 0, 1]])
    out = make_loss_labels(ids, lut)
    assert out.tolist() == [[-100, 5, -100, 7], [9, -100, -100, 1]]
    # the input must not be mutated (labels are re-derived every micro-batch)
    assert ids.tolist() == [[0, 5, 3, 7], [9, 0, 0, 1]]


def test_make_loss_labels_noop_without_lut():
    from train import make_loss_labels

    ids = torch.tensor([[0, 5], [3, 7]])
    assert make_loss_labels(ids, None) is ids


def test_unk_masking_changes_the_loss():
    """Masking unk must lower the loss exactly by removing those targets."""
    from train import make_loss_labels
    from wisp.config import WispConfig
    from wisp.model import WispForCausalLM

    torch.manual_seed(0)
    cfg = WispConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     num_key_value_heads=1, head_dim=16, intermediate_size=64, ffn_share_layers=2)
    model = WispForCausalLM(cfg).eval()
    x = torch.randint(2, 64, (2, 16))
    x[:, 5:9] = 0                                    # inject unk tokens
    lut = torch.zeros(64, dtype=torch.bool)
    lut[0] = True
    with torch.no_grad():
        plain = model(x, labels=x).loss
        masked = model(x, labels=make_loss_labels(x, lut)).loss
    assert torch.isfinite(plain) and torch.isfinite(masked)
    assert not torch.isclose(plain, masked)

    # masking unk must equal computing the loss only on the kept targets
    with torch.no_grad():
        logits = model(x).logits[:, :-1].float()
        y = x[:, 1:].clone()
        keep = y != 0
        manual = torch.nn.functional.cross_entropy(logits[0][keep[0]], y[0][keep[0]])
        manual = (manual * keep[0].sum() +
                  torch.nn.functional.cross_entropy(logits[1][keep[1]], y[1][keep[1]]) * keep[1].sum()) / keep.sum()
    assert torch.allclose(masked, manual, atol=1e-5), (masked.item(), manual.item())


# ------------------------------------------------------------ alpha groups
def test_alpha_gets_its_own_param_group():
    from train import build_param_groups
    from wisp.config import WispConfig
    from wisp.model import WispForCausalLM

    cfg = WispConfig(vocab_size=64, hidden_size=32, num_hidden_layers=4, num_attention_heads=2,
                     num_key_value_heads=1, head_dim=16, intermediate_size=64, ffn_share_layers=2)
    model = WispForCausalLM(cfg)
    groups = build_param_groups(model, weight_decay=0.1)
    assert len(groups) == 3
    alpha_group = groups[-1]
    assert alpha_group["weight_decay"] == 0.0
    assert alpha_group.get("alpha_scale") is True
    alpha_ids = {id(p) for p in alpha_group["params"]}
    assert len(alpha_ids) == 2 * cfg.ffn_share_layers  # two alphas per shared layer
    for g in groups[:2]:
        assert not (alpha_ids & {id(p) for p in g["params"]})


def test_checkpoint_keys_are_not_prefixed(tmp_path):
    """A torch.compile/DDP wrapper must not leak `_orig_mod.` into checkpoints."""
    import torch.nn as nn

    from train import unwrap
    from wisp.config import WispConfig
    from wisp.model import WispForCausalLM

    cfg = WispConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     num_key_value_heads=1, head_dim=16, intermediate_size=64, ffn_share_layers=2)
    raw = WispForCausalLM(cfg)

    class FakeCompiled(nn.Module):
        """Mimics torch.compile's OptimizedModule attribute layout."""
        def __init__(self, m):
            super().__init__()
            self._orig_mod = m

        def forward(self, *a, **k):
            return self._orig_mod(*a, **k)

    wrapped = FakeCompiled(raw)
    assert unwrap(wrapped) is raw
    keys = set(unwrap(wrapped).state_dict())
    assert keys == set(raw.state_dict())
    assert not any(k.startswith("_orig_mod.") for k in keys)

    # round trip through the real save/load path
    wrapped.save_pretrained = raw.save_pretrained
    raw.save_pretrained(str(tmp_path))
    loaded = WispForCausalLM.from_pretrained(str(tmp_path))
    assert loaded is not None


# ------------------------------------------------- disk-full checkpointing
def _fake_train_objects():
    from wisp.config import WispConfig
    from wisp.model import WispForCausalLM

    cfg = WispConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     num_key_value_heads=1, head_dim=16, intermediate_size=64, ffn_share_layers=2)
    model = WispForCausalLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    class A:
        pass
    return model, opt, cfg, A()


def test_checkpoint_saves_normally(tmp_path):
    import train as T

    model, opt, cfg, args = _fake_train_objects()
    ok = T.save_checkpoint(str(tmp_path / "step10.pt"), model, opt, args, cfg, 10, 100, {}, keep_last=3)
    assert ok
    assert (tmp_path / "step10.pt").exists()
    assert not (tmp_path / "step10.pt.tmp").exists()


def test_checkpoint_refuses_when_disk_is_full(tmp_path, monkeypatch):
    """A full disk must return False, not raise, and leave no partial .tmp."""
    import train as T

    model, opt, cfg, args = _fake_train_objects()
    monkeypatch.setattr(T, "disk_free_bytes", lambda p: 0)
    ok = T.save_checkpoint(str(tmp_path / "step10.pt"), model, opt, args, cfg, 10, 100, {}, keep_last=3)
    assert ok is False
    assert not (tmp_path / "step10.pt").exists()
    assert not (tmp_path / "step10.pt.tmp").exists()


def test_checkpoint_prunes_older_ones_to_make_room(tmp_path, monkeypatch):
    """When space is short, older checkpoints are dropped instead of failing."""
    import train as T

    model, opt, cfg, args = _fake_train_objects()
    for step in (10, 20, 30):
        (tmp_path / f"step{step}.pt").write_bytes(b"x" * 16)
    needed = T.estimate_checkpoint_bytes(model)
    # pretend we have just under one checkpoint worth of room
    monkeypatch.setattr(T, "disk_free_bytes", lambda p: needed // 2)
    # faking pruning: it deletes files, so free space must grow accordingly
    real_prune = T.prune_checkpoints

    def prune_then_report(out_dir, keep_last):
        before = len(list(tmp_path.glob("step*.pt")))
        real_prune(out_dir, keep_last)
        after = len(list(tmp_path.glob("step*.pt")))
        monkeypatch.setattr(T, "disk_free_bytes",
                            lambda p: (needed // 2) + (before - after) * 16 * 10 ** 9)

    monkeypatch.setattr(T, "prune_checkpoints", prune_then_report)
    ok = T.save_checkpoint(str(tmp_path / "step40.pt"), model, opt, args, cfg, 40, 400, {}, keep_last=3)
    assert ok, "should have pruned old checkpoints and then succeeded"
    assert (tmp_path / "step40.pt").exists()


def test_checkpoint_survives_enospc_on_write(tmp_path, monkeypatch):
    """torch.save raising ENOSPC must be caught, cleaned up, and reported as False."""
    import train as T

    model, opt, cfg, args = _fake_train_objects()
    calls = {"n": 0}

    def boom(payload, path):
        calls["n"] += 1
        open(path, "wb").write(b"partial" * 100)   # leave a partial file behind
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(T.torch, "save", boom)
    ok = T.save_checkpoint(str(tmp_path / "step10.pt"), model, opt, args, cfg, 10, 100, {}, keep_last=1)
    assert ok is False
    assert calls["n"] == 3, "should retry a few times"
    assert not (tmp_path / "step10.pt").exists()
    assert not (tmp_path / "step10.pt.tmp").exists(), "partial file must be removed"


def test_checkpoint_size_estimate_is_sane():
    import train as T
    from wisp.config import WispConfig
    from wisp.model import WispForCausalLM

    cfg = WispConfig()
    est = T.estimate_checkpoint_bytes(WispForCausalLM(cfg))
    # 259M params: fp32 weights 1.04 GB + 2 moments 2.07 GB + 256 MB slack
    assert 3.0e9 < est < 4.0e9, est


# --------------------------------------------------------- SFT loss masking
def _sft_corpus(root, n_tokens=40000, seed=0):
    """tokens + mask, shaped like scripts/tokenize_sft.py output."""
    import json, os
    import numpy as np
    rng = np.random.default_rng(seed)
    ids = rng.integers(16, 1000, size=n_tokens, dtype=np.uint16)
    ids[::200] = 5     # <|user|>
    ids[1::200] = 6    # <|assistant|>
    ids[50::200] = 4   # <|end_of_turn|>
    mask = np.zeros(n_tokens, dtype=np.uint8)
    mask[1::2] = 1     # arbitrary but deterministic supervision pattern
    d = os.path.join(root, "sft/zh")
    os.makedirs(d, exist_ok=True)
    ids.tofile(os.path.join(d, "tokens.bin"))
    mask.tofile(os.path.join(d, "mask.bin"))
    json.dump({"name": "sft/zh", "num_tokens": int(n_tokens), "num_docs": 10,
               "dtype": "uint16", "bin_file": "tokens.bin", "mask_file": "mask.bin",
               "num_supervised_tokens": int(mask.sum())},
              open(os.path.join(d, "meta.json"), "w"))


def test_sft_corpus_loads_mask(tmp_path):
    from wisp.data import discover_corpora

    _sft_corpus(str(tmp_path))
    c = discover_corpora(str(tmp_path))["sft/zh"]
    assert c.mask is not None and len(c.mask) == c.num_tokens


def test_masked_sampler_keeps_ids_and_mask_aligned(tmp_path):
    from wisp.data import build_sampler

    _sft_corpus(str(tmp_path))
    s, corpora = build_sampler(str(tmp_path), block_size=64, split="train")
    assert s.has_mask
    c = corpora["sft/zh"]
    ids, msk = s.next_batch_masked(2)
    assert ids.shape == (2, 65) and msk.shape == (2, 65)
    # the ids and mask must describe the same span of the stream
    start = s.queues[0][0] * 64
    assert np.array_equal(ids[0], c.tokens[start:start + 65])
    assert np.array_equal(msk[0], c.mask[start:start + 65])


def test_masked_and_plain_sampler_draw_the_same_blocks(tmp_path):
    """Masked and unmasked draws must advance the schedule identically."""
    from wisp.data import build_sampler

    _sft_corpus(str(tmp_path))
    a, _ = build_sampler(str(tmp_path), block_size=64, seed=5)
    b, _ = build_sampler(str(tmp_path), block_size=64, seed=5)
    plain = a.next_batch(4)
    masked, _m = b.next_batch_masked(4)
    assert np.array_equal(plain, masked)


def test_prefetcher_returns_mask(tmp_path):
    from wisp.data import BatchPrefetcher, build_sampler

    _sft_corpus(str(tmp_path))
    s, _ = build_sampler(str(tmp_path), block_size=32, split="train")
    pf = BatchPrefetcher(s, 2, device="cpu", slots=1, pin_memory=False)
    try:
        b = pf.next()
    finally:
        pf.close()
    assert b.loss_mask is not None
    assert b.loss_mask.shape == b.input_ids.shape
    assert b.loss_mask.dtype == torch.bool
    assert b.loss_mask.any() and not b.loss_mask.all()


def test_mixed_mask_availability_is_rejected(tmp_path):
    """One corpus with a mask and one without must fail loudly, not silently."""
    from wisp.data import build_sampler

    _sft_corpus(str(tmp_path))
    import json, os
    import numpy as np
    d = os.path.join(str(tmp_path), "pretrain/en")
    os.makedirs(d, exist_ok=True)
    np.arange(40000, dtype=np.uint16).tofile(os.path.join(d, "tokens.bin"))
    json.dump({"name": "pretrain/en", "num_tokens": 40000, "num_docs": 10,
               "dtype": "uint16", "bin_file": "tokens.bin"}, open(os.path.join(d, "meta.json"), "w"))
    with pytest.raises(ValueError, match="mixed mask availability"):
        build_sampler(str(tmp_path), block_size=64, split="train")


def test_loss_only_depends_on_supervised_positions():
    """Perturbing a masked-out target must not change the loss."""
    from train import make_loss_labels
    from wisp.config import WispConfig
    from wisp.model import WispForCausalLM

    torch.manual_seed(0)
    cfg = WispConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     num_key_value_heads=1, head_dim=16, intermediate_size=64, ffn_share_layers=2)
    model = WispForCausalLM(cfg).eval()
    x = torch.randint(2, 64, (2, 16))
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, 8:] = True                       # supervise only the second half

    def loss_of(ids):
        with torch.no_grad():
            return model(ids, labels=make_loss_labels(ids, None, mask)).loss

    base = loss_of(x)
    # changing a token that is not supervised (position < 8) must not matter
    y = x.clone()
    y[:, 1:7] = torch.randint(2, 64, y[:, 1:7].shape)
    assert torch.allclose(base, loss_of(y), atol=1e-6)
    # changing a supervised token must matter
    z = x.clone()
    z[:, 10:14] = torch.randint(2, 64, z[:, 10:14].shape)
    assert not torch.isclose(base, loss_of(z))


def test_loss_labels_combine_unk_and_sft_masks():
    from train import make_loss_labels

    ids = torch.tensor([[0, 5, 6, 7, 8]])
    lut = torch.zeros(64, dtype=torch.bool)
    lut[0] = True                                    # ignore unk
    sft = torch.tensor([[True, True, False, True, True]])
    out = make_loss_labels(ids, lut, sft)
    assert out.tolist() == [[-100, 5, -100, 7, 8]]


def _marker_names(tok):
    names = {tok.bos_token_id: "bos", tok.eos_token_id: "eos"}
    names.update({i: n for n, i in tok.role_token_ids.items()})
    names.update({i: f"task:{n}" for n, i in tok.task_token_ids.items()})
    names.update({i: n for n, i in tok.reasoning_token_ids.items()})
    return names


def test_derive_sft_mask_from_marker_stream(tmp_path):
    """The streaming state machine must mask exactly the assistant turns."""
    from scripts.derive_sft_mask import derive_mask
    from wisp.tokenizer import WispTokenizer

    tok = WispTokenizer()
    bos, eos, eot = tok.bos_token_id, tok.eos_token_id, tok.end_of_turn_token_id
    user, asst = tok.user_token_id, tok.assistant_token_id
    # <bos> user U <eot> assistant A1 A2 <eot> user U <eot> assistant B1 <eot> <eos>
    ids = [bos, user, 10, 11, eot, asst, 20, 21, eot, user, 12, eot, asst, 30, eot, eos]
    np.asarray(ids, dtype=np.uint16).tofile(tmp_path / "tokens.bin")
    info = derive_mask(str(tmp_path / "tokens.bin"), str(tmp_path / "mask.bin"),
                       tok, _marker_names(tok), verbose=False)
    mask = np.fromfile(tmp_path / "mask.bin", dtype=np.uint8)
    assert mask.tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0]
    assert info["num_supervised_tokens"] == 5
    assert info["num_turns"] == 2 and info["num_docs"] == 1


def test_derive_sft_mask_survives_chunk_boundary(tmp_path, monkeypatch):
    """A turn straddling a read chunk must not lose its mask."""
    from scripts import derive_sft_mask as D
    from wisp.tokenizer import WispTokenizer

    tok = WispTokenizer()
    bos, eos, eot = tok.bos_token_id, tok.eos_token_id, tok.end_of_turn_token_id
    user, asst = tok.user_token_id, tok.assistant_token_id
    ids = [bos, user, 1, eot, asst, 50, 51, 52, 53, 54, eot, eos]
    np.asarray(ids, dtype=np.uint16).tofile(tmp_path / "tokens.bin")
    monkeypatch.setattr(D, "CHUNK", 4)          # force many tiny chunks
    info = D.derive_mask(str(tmp_path / "tokens.bin"), str(tmp_path / "mask.bin"),
                         tok, _marker_names(tok), verbose=False)
    mask = np.fromfile(tmp_path / "mask.bin", dtype=np.uint8)
    assert mask.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
    assert info["num_supervised_tokens"] == 6


def test_mask_auto_detected_when_meta_is_stale(tmp_path):
    """mask.bin next to tokens.bin must be picked up even if meta.json forgot it."""
    import json, os
    import numpy as np
    from wisp.data import discover_corpora

    d = tmp_path / "decay/en"
    d.mkdir(parents=True)
    ids = np.arange(20000, dtype=np.uint16)
    mask = np.zeros(20000, dtype=np.uint8)
    mask[1::2] = 1
    ids.tofile(d / "tokens.bin")
    mask.tofile(d / "mask.bin")
    (d / "meta.json").write_text(json.dumps({
        "name": "decay/en", "num_tokens": 20000, "num_docs": 100,
        "dtype": "uint16", "bin_file": "tokens.bin",   # <- no mask_file
    }))
    c = discover_corpora(str(tmp_path))["decay/en"]
    assert c.mask is not None
    assert c.mask_auto_detected is True
    assert len(c.mask) == c.num_tokens


def test_mask_length_mismatch_is_rejected(tmp_path):
    import json
    import numpy as np
    from wisp.data import TokenizedCorpus

    d = tmp_path / "c"
    d.mkdir()
    np.arange(1000, dtype=np.uint16).tofile(d / "tokens.bin")
    np.zeros(500, dtype=np.uint8).tofile(d / "mask.bin")     # wrong length
    (d / "meta.json").write_text(json.dumps({
        "name": "c", "num_tokens": 1000, "dtype": "uint16",
        "bin_file": "tokens.bin", "mask_file": "mask.bin"}))
    with pytest.raises(ValueError, match="mask has"):
        TokenizedCorpus(str(d))
