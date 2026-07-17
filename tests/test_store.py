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
