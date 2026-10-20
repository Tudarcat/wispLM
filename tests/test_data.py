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
