"""Store checks: whitener correctness on known covariance, hash
immutability, writer/reader round trip with exact token accounting,
whitener-hash rejection, and the write-time audit guards."""

import json

import pytest
import torch

from block_crosscoder_experiment.store import (
    ShardWriter,
    StoreReader,
    Whitener,
    WhitenerAccumulator,
)

S, D = 3, 16


def gaussian_batches(n_batches=40, batch=512, seed=0):
    """Site s has covariance scaled by (s+1)^2 and mean s."""
    gen = torch.Generator().manual_seed(seed)
    # Well-conditioned mixing (eigenvalues bounded away from 0) so the
    # ridge term stays negligible next to every eigendirection.
    A = torch.randn(S, D, D, generator=gen) / D**0.5 + 2 * torch.eye(D)
    out = []
    for _ in range(n_batches):
        z = torch.randn(batch, S, D, generator=gen)
        x = torch.einsum("sde,nse->nsd", A, z)
        x = x * torch.arange(1, S + 1).view(1, S, 1) + torch.arange(S).view(1, S, 1)
        out.append(x)
    return out, A


def fit_whitener(batches, ridge_scale=1e-3):
    acc = WhitenerAccumulator(S, D)
    for x in batches:
        acc.update(x)
    return acc.finalize(sites=list(range(S)), meta={"test": 1}, ridge_scale=ridge_scale)


def test_whitener_whitens():
    batches, _ = gaussian_batches()
    w = fit_whitener(batches)
    xw = torch.cat([w.apply(x) for x in batches])
    # Whitened data: mean ~0, covariance ~I per site (small ridge).
    assert xw.mean(dim=0).abs().max() < 0.05
    for s in range(S):
        cov = xw[:, s].T @ xw[:, s] / xw.shape[0]
        assert (cov - torch.eye(D)).abs().max() < 0.1


def test_site_rms_scalars_restore_unit_power():
    """F7 renorm arm: at production shrinkage (ridge_scale 1.0), sites
    with different anisotropy retain different whitened power; the RMS
    scalars restore ~unit mean per-dim power at every site."""
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(20_000, S, D, generator=gen)
    decay = torch.stack(
        [torch.arange(1, D + 1).float() ** (-(s + 1) / 2) for s in range(S)]
    )
    x = x * decay.view(1, S, D)
    acc = WhitenerAccumulator(S, D)
    acc.update(x)
    w = acc.finalize(sites=list(range(S)), meta={}, ridge_scale=1.0)
    xw = w.apply(x)
    raw = xw.pow(2).mean(dim=(0, 2))
    assert (raw.max() - raw.min()) > 0.05  # shrinkage leaves unequal site power
    power = (xw * w.site_rms_scalars().view(1, S, 1)).pow(2).mean(dim=(0, 2))
    assert torch.allclose(power, torch.ones(S), atol=0.02)


def test_production_whitener_folds_site_renorm_once():
    gen = torch.Generator().manual_seed(8)
    x = torch.randn(20_000, S, D, generator=gen)
    decay = torch.stack(
        [torch.arange(1, D + 1).float() ** (-(s + 1) / 2) for s in range(S)]
    )
    x = x * decay.view(1, S, D)
    acc = WhitenerAccumulator(S, D)
    acc.update(x)
    w = acc.finalize(
        sites=list(range(S)), meta={"campaign": "test"},
        ridge_scale=1.0, site_renorm=True,
    )
    power = w.apply(x).pow(2).mean(dim=(0, 2))
    assert torch.allclose(power, torch.ones(S), atol=0.02)
    assert w.meta["site_rms_renorm_folded"] is True
    assert len(w.meta["site_rms_scalars"]) == S
    assert torch.equal(w.site_rms_scalars(), torch.ones(S))


def test_whitener_roundtrip_and_hash(tmp_path):
    batches, _ = gaussian_batches(n_batches=5)
    w = fit_whitener(batches)
    x = batches[0]
    err = (w.unapply(w.apply(x)) - x).norm() / x.norm()
    assert err < 1e-4
    w.save(tmp_path / "w.pt")
    w2 = Whitener.load(tmp_path / "w.pt")
    assert w2.hash == w.hash
    assert torch.equal(w2.W, w.W)


def test_whitener_rejects_fp16():
    acc = WhitenerAccumulator(S, D)
    with pytest.raises(TypeError, match="fp16"):
        acc.update(torch.zeros(4, S, D, dtype=torch.float16))


def write_store(tmp_path, n_tokens=1000, tokens_per_shard=128, whitener_hash="abc"):
    gen = torch.Generator().manual_seed(1)
    writer = ShardWriter(
        tmp_path,
        "train",
        whitener_hash=whitener_hash,
        sites=[7, 10, 13],
        d_model=D,
        tokens_per_shard=tokens_per_shard,
        free_space_floor_frac=0.0,
    )
    # Distinct per-token payload so exact coverage is checkable: token i
    # is split across two channels as (i // 256, i % 256) + 1 — both
    # small enough to be bf16-exact.
    x = 0.01 * torch.randn(n_tokens, 3, D, generator=gen)
    ids = torch.arange(n_tokens)
    x[:, 0, 0] = (ids // 256).float() + 1
    x[:, 0, 1] = (ids % 256).float() + 1
    for chunk in x.split(300):
        writer.add(chunk)
    manifest = writer.close()
    return manifest, x


def token_ids(batch: torch.Tensor) -> torch.Tensor:
    """Recover the planted ids from a [n, S, D] batch."""
    return ((batch[:, 0, 0].float() - 1) * 256 + batch[:, 0, 1].float() - 1).long()


def test_writer_reader_round_trip(tmp_path):
    manifest, _ = write_store(tmp_path)
    assert manifest["n_tokens"] == 1000
    assert len(manifest["shards"]) == 8  # 7 full 128s + remainder 104
    reader = StoreReader(tmp_path, "train", expected_whitener_hash="abc")
    assert reader.verify() == 1000

    seq = torch.cat(list(reader.sequential_batches(64)), dim=0)
    assert torch.equal(token_ids(seq), torch.arange(1000))  # stored order kept

    # Shuffled epoch: batch shapes constant, no token repeated, coverage
    # near-complete (partial batches at buffer boundaries may drop).
    got = list(reader.shuffled_batches(64, seed=5, epochs=1, buffer_tokens=256))
    ids = torch.cat([token_ids(b) for b in got])
    assert all(b.shape == (64, 3, D) for b in got)
    assert ids.unique().numel() == ids.numel()
    assert ids.numel() >= 1000 - 64
    # Same seed -> identical order; different seed -> different order.
    again = torch.cat(
        [token_ids(b) for b in reader.shuffled_batches(64, seed=5, epochs=1, buffer_tokens=256)]
    )
    assert torch.equal(ids, again)
    other = torch.cat(
        [token_ids(b) for b in reader.shuffled_batches(64, seed=6, epochs=1, buffer_tokens=256)]
    )
    assert not torch.equal(ids, other)


def test_reader_rejects_wrong_whitener(tmp_path):
    write_store(tmp_path)
    with pytest.raises(ValueError, match="whitener"):
        StoreReader(tmp_path, "train", expected_whitener_hash="different")


def test_reader_detects_corruption(tmp_path):
    manifest, _ = write_store(tmp_path)
    shard = tmp_path / "train" / manifest["shards"][0]["file"]
    raw = bytearray(shard.read_bytes())
    raw[-1] ^= 0xFF  # flip one payload byte, header intact
    shard.write_bytes(bytes(raw))
    reader = StoreReader(tmp_path, "train")
    with pytest.raises(ValueError, match="checksum"):
        reader.verify()


def test_writer_audits(tmp_path):
    writer = ShardWriter(
        tmp_path, "train", whitener_hash="abc", sites=[0], d_model=D,
        tokens_per_shard=8, free_space_floor_frac=0.0,
    )
    bad = torch.randn(8, 1, D)
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        writer.add(bad)
    with pytest.raises(ValueError, match="zero-row"):
        writer.add(torch.zeros(8, 1, D))
    with pytest.raises(TypeError, match="fp16"):
        writer.add(torch.randn(4, 1, D).half())


def test_manifest_contents(tmp_path):
    write_store(tmp_path)
    manifest = json.loads((tmp_path / "train" / "split.json").read_text())
    assert manifest["whitener_hash"] == "abc"
    assert manifest["sites"] == [7, 10, 13]
    assert sum(s["n_tokens"] for s in manifest["shards"]) == manifest["n_tokens"]


def test_site_subset_view_matches_sliced_full_stream(tmp_path):
    """E4: at matched seed, the subset reader's shuffled stream is exactly
    the full reader's stream sliced to the requested sites — the
    factorial's matched-data guarantee."""
    write_store(tmp_path)
    full = StoreReader(tmp_path, "train", expected_whitener_hash="abc")
    sub = StoreReader(tmp_path, "train", expected_whitener_hash="abc", sites=[10])
    assert sub.n_sites == 1 and sub.sites == (10,)
    assert sub.n_tokens == full.n_tokens

    kw = dict(seed=5, epochs=1, buffer_tokens=256)
    for xf, xs in zip(full.shuffled_batches(64, **kw), sub.shuffled_batches(64, **kw)):
        assert xs.shape == (64, 1, D)
        assert torch.equal(xs, xf[:, 1:2])

    seq_full = torch.cat(list(full.sequential_batches(64)), dim=0)
    seq_sub = torch.cat(list(sub.sequential_batches(64)), dim=0)
    assert torch.equal(seq_sub, seq_full[:, 1:2])

    # Multi-site subsets keep stored order; verify() still checks full shards.
    pair = StoreReader(tmp_path, "train", sites=[7, 13])
    x = next(pair.sequential_batches(64))
    assert torch.equal(x, torch.cat(list(full.sequential_batches(64)))[:64][:, [0, 2]])
    assert pair.verify() == 1000


def test_site_subset_rejects_bad_requests(tmp_path):
    write_store(tmp_path)
    with pytest.raises(ValueError, match="not in store"):
        StoreReader(tmp_path, "train", sites=[99])
    with pytest.raises(ValueError, match="duplicate"):
        StoreReader(tmp_path, "train", sites=[10, 10])
    with pytest.raises(ValueError, match="stored order"):
        StoreReader(tmp_path, "train", sites=[13, 7])


def test_prefetch_preserves_order_and_reraises(tmp_path):
    """E5: prefetch is a transparent, order-preserving wrapper; worker
    exceptions surface at the consumption point."""
    from block_crosscoder_experiment.store import prefetch_batches

    write_store(tmp_path)
    reader = StoreReader(tmp_path, "train", expected_whitener_hash="abc")
    kw = dict(seed=5, epochs=1, buffer_tokens=256)
    plain = list(reader.shuffled_batches(64, **kw))
    fetched = list(prefetch_batches(reader.shuffled_batches(64, **kw), depth=3))
    assert len(plain) == len(fetched)
    assert all(torch.equal(a, b) for a, b in zip(plain, fetched))

    def boom():
        yield torch.zeros(2)
        raise RuntimeError("worker died")

    it = prefetch_batches(boom(), depth=2)
    next(it)
    with pytest.raises(RuntimeError, match="worker died"):
        next(it)


def test_merged_manifest_reads_across_splits(tmp_path):
    """E6: a merged manifest whose shard entries point into sibling split
    dirs (../train/...) reads as one logical split — the epochs-vs-fresh
    12M arm's access path."""
    write_store(tmp_path, n_tokens=500)
    gen = torch.Generator().manual_seed(2)
    w = ShardWriter(
        tmp_path, "train_ext", whitener_hash="abc", sites=[7, 10, 13],
        d_model=D, tokens_per_shard=128, free_space_floor_frac=0.0,
    )
    ext = 0.01 * torch.randn(300, 3, D, generator=gen)
    ids = torch.arange(500, 800)
    ext[:, 0, 0] = (ids // 256).float() + 1
    ext[:, 0, 1] = (ids % 256).float() + 1
    w.add(ext)
    ext_manifest = w.close()

    train_manifest = json.loads((tmp_path / "train" / "split.json").read_text())
    merged_dir = tmp_path / "train_all"
    merged_dir.mkdir()
    merged = {
        "split": "train_all",
        "whitener_hash": "abc",
        "sites": [7, 10, 13],
        "d_model": D,
        "n_tokens": 800,
        "shards": (
            [{"file": f"../train/{s['file']}", "n_tokens": s["n_tokens"]}
             for s in train_manifest["shards"]]
            + [{"file": f"../train_ext/{s['file']}", "n_tokens": s["n_tokens"]}
               for s in ext_manifest["shards"]]
        ),
        "meta": {},
    }
    (merged_dir / "split.json").write_text(json.dumps(merged))

    reader = StoreReader(tmp_path, "train_all", expected_whitener_hash="abc")
    assert reader.verify() == 800
    seq = torch.cat(list(reader.sequential_batches(50)), dim=0)
    assert torch.equal(token_ids(seq), torch.arange(800))
    got = torch.cat(
        [token_ids(b) for b in reader.shuffled_batches(50, seed=3, epochs=1, buffer_tokens=200)]
    )
    assert got.unique().numel() == got.numel()
