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
    cuda_prefetch_batches,
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
    """At the declared shrinkage (ridge_scale 1.0), sites
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
        sites=list(range(S)),
        meta={"campaign": "test"},
        ridge_scale=1.0,
        site_renorm=True,
    )
    power = w.apply(x).pow(2).mean(dim=(0, 2))
    assert torch.allclose(power, torch.ones(S), atol=0.02)
    assert w.meta["site_rms_renorm_folded"] is True
    assert len(w.meta["site_rms_scalars"]) == S
    assert torch.equal(w.site_rms_scalars(), torch.ones(S))


def test_rectangular_site_renorm_ignores_padded_coordinates():
    gen = torch.Generator().manual_seed(82)
    widths = (3, 7)
    x = torch.zeros(30_000, 2, max(widths))
    x[:, 0, : widths[0]] = torch.randn(
        x.shape[0], widths[0], generator=gen
    ) * torch.tensor([1.0, 0.2, 0.05])
    x[:, 1, : widths[1]] = torch.randn(
        x.shape[0], widths[1], generator=gen
    ) * torch.linspace(1.0, 0.05, widths[1])
    acc = WhitenerAccumulator(2, max(widths))
    acc.update(x)
    transform = acc.finalize(
        sites=(0, 1),
        meta={"site_dims": list(widths)},
        ridge_scale=1.0,
        site_renorm=True,
    )
    y = transform.apply(x)
    power = torch.stack(
        [y[:, site, :width].pow(2).mean() for site, width in enumerate(widths)]
    )
    assert torch.allclose(power, torch.ones(2), atol=0.02)
    assert torch.count_nonzero(y[:, 0, widths[0] :]) == 0


@pytest.mark.parametrize("mode", ["none", "scalar_rms", "layer", "whiten"])
def test_normalization_modes(mode):
    batches, _ = gaussian_batches(n_batches=8)
    acc = WhitenerAccumulator(S, D)
    for x in batches:
        acc.update(x)
    w = acc.finalize(
        sites=list(range(S)),
        meta={"campaign": "store-test"},
        ridge_scale=1e-3,
        mode=mode,
    )
    x = torch.cat(batches)
    y = w.apply(x)
    assert w.mode == mode
    assert torch.isfinite(y).all()
    if mode == "none":
        assert torch.equal(y, x.float())
    elif mode == "scalar_rms":
        assert y.mean(dim=0).abs().max() < 0.08
        assert torch.allclose(y.pow(2).mean(dim=(0, 2)), torch.ones(S), atol=0.08)
    elif mode == "layer":
        assert y.mean(dim=-1).abs().max() < 1e-5
        assert torch.allclose(y.pow(2).mean(dim=-1), torch.ones(y.shape[:2]), atol=1e-4)
        with pytest.raises(ValueError, match="not invertible"):
            w.unapply(y[:2])
    else:
        assert y.mean(dim=0).abs().max() < 0.08


@pytest.mark.parametrize("mode", ["none", "scalar_rms", "sqrt_d"])
def test_diagonal_modes_never_build_or_apply_dense_covariance(monkeypatch, mode):
    batches, _ = gaussian_batches(n_batches=4)
    x = torch.cat(batches)
    accumulator = WhitenerAccumulator(S, D, track_covariance=False)
    for batch in batches:
        accumulator.update(batch)
    centered_norm = None
    if mode == "sqrt_d":
        mean = x.double().mean(dim=0)
        centered_norm = (x.double() - mean).norm(dim=-1).mean(dim=0)
    transform = accumulator.finalize(
        sites=range(S),
        meta={"site_dims": [D] * S},
        mode=mode,
        mean_centered_norm=centered_norm,
    )
    assert accumulator.outer is None

    def forbidden(*args, **kwargs):
        raise AssertionError("diagonal normalization called a dense kernel")

    monkeypatch.setattr(torch, "einsum", forbidden)
    monkeypatch.setattr(torch.linalg, "inv", forbidden)
    normalized = transform.apply(x)
    assert torch.allclose(transform.unapply(normalized), x.float(), atol=2e-5)


def test_writer_resume_rebuilds_exact_ordered_stream(tmp_path):
    values = torch.arange(1, 13 * 2 * 3 + 1).reshape(13, 2, 3).float()
    row_ids = torch.stack((torch.arange(13), torch.arange(13) + 100), dim=1)
    writer = ShardWriter(
        tmp_path,
        "train",
        whitener_hash="resume",
        sites=(1, 2),
        d_model=3,
        tokens_per_shard=4,
        free_space_floor_frac=0,
    )
    writer.add(values[:8], row_ids[:8])
    writer.synchronize()
    assert writer.persisted_tokens == 8
    writer.abort()
    with pytest.raises(ValueError, match="incomplete"):
        StoreReader(tmp_path, "train")

    resumed = ShardWriter(
        tmp_path,
        "train",
        whitener_hash="resume",
        sites=(1, 2),
        d_model=3,
        tokens_per_shard=4,
        free_space_floor_frac=0,
        resume=True,
    )
    assert resumed.persisted_tokens == 8
    resumed.add(values[8:], row_ids[8:])
    manifest = resumed.close()
    reader = StoreReader(tmp_path, "train")
    observed_x = []
    observed_ids = []
    for x, ids in reader.sequential_batches_with_ids(5):
        observed_x.append(x)
        observed_ids.append(ids)
    assert torch.equal(torch.cat(observed_x), values.to(torch.bfloat16))
    assert torch.equal(torch.cat(observed_ids), row_ids)
    assert reader.verify() == 13
    assert manifest["format_version"] == 3
    assert manifest["row_ids_dtype"] == "int64"


def test_writer_rejects_non_int64_row_ids_and_duplicate_sites(tmp_path):
    with pytest.raises(ValueError, match="nonempty and unique"):
        ShardWriter(
            tmp_path,
            "bad",
            whitener_hash="x",
            sites=(1, 1),
            d_model=2,
        )
    writer = ShardWriter(
        tmp_path,
        "good",
        whitener_hash="x",
        sites=(1,),
        d_model=2,
        free_space_floor_frac=0,
    )
    with pytest.raises(TypeError, match="int64 exactly"):
        writer.add(torch.ones(2, 1, 2), torch.ones(2, 1, dtype=torch.int32))


def test_sqrt_d_uses_exact_centered_mean_norm():
    batches, _ = gaussian_batches(n_batches=8)
    x = torch.cat(batches)
    acc = WhitenerAccumulator(S, D)
    for batch in batches:
        acc.update(batch)
    mean = x.double().mean(dim=0)
    centered_mean_norm = (x.double() - mean).norm(dim=-1).mean(dim=0)
    sqrt_d = acc.finalize(
        sites=list(range(S)),
        meta={"site_dims": [D] * S},
        mode="sqrt_d",
        mean_centered_norm=centered_mean_norm,
    )
    transformed = sqrt_d.apply(x)
    assert torch.allclose(
        transformed.norm(dim=-1).mean(dim=0),
        torch.full((S,), D**0.5),
        atol=2e-4,
    )


def test_rectangular_layer_norm_ignores_padding():
    gen = torch.Generator().manual_seed(81)
    x = torch.zeros(128, 2, 5)
    x[:, 0, :3] = torch.randn(128, 3, generator=gen)
    x[:, 1, :5] = torch.randn(128, 5, generator=gen)
    acc = WhitenerAccumulator(2, 5)
    acc.update(x)
    transform = acc.finalize(sites=(0, 1), meta={"site_dims": [3, 5]}, mode="layer")
    y = transform.apply(x)
    assert y[:, 0, :3].mean(dim=-1).abs().max() < 1e-5
    assert y[:, 1, :5].mean(dim=-1).abs().max() < 1e-5
    assert torch.count_nonzero(y[:, 0, 3:]) == 0


def test_site_renorm_only_valid_with_whitening():
    batches, _ = gaussian_batches(n_batches=2)
    acc = WhitenerAccumulator(S, D)
    for x in batches:
        acc.update(x)
    with pytest.raises(ValueError, match="only for mode='whiten'"):
        acc.finalize(sites=list(range(S)), meta={}, mode="scalar_rms", site_renorm=True)


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


def test_transform_hash_covers_eigenvalues_and_fit_count():
    batches, _ = gaussian_batches(n_batches=3)
    w = fit_whitener(batches)
    altered_eigs = Whitener(
        mean=w.mean,
        W=w.W,
        ridge=w.ridge,
        eigenvalues=w.eigenvalues.clone(),
        sites=w.sites,
        n_fit_tokens=w.n_fit_tokens,
        meta=dict(w.meta),
    )
    altered_eigs.eigenvalues[0, 0] += 1
    assert altered_eigs.hash != w.hash
    altered_n = Whitener(
        mean=w.mean,
        W=w.W,
        ridge=w.ridge,
        eigenvalues=w.eigenvalues,
        sites=w.sites,
        n_fit_tokens=w.n_fit_tokens + 1,
        meta=dict(w.meta),
    )
    assert altered_n.hash != w.hash


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

    # Shuffled epoch: no token repeated and exact coverage, including the
    # final partial batch.
    got = list(reader.shuffled_batches(64, seed=5, epochs=1, buffer_tokens=256))
    ids = torch.cat([token_ids(b) for b in got])
    assert all(b.shape[1:] == (3, D) and 0 < b.shape[0] <= 64 for b in got)
    assert ids.unique().numel() == ids.numel()
    assert ids.numel() == 1000
    # Same seed -> identical order; different seed -> different order.
    again = torch.cat(
        [
            token_ids(b)
            for b in reader.shuffled_batches(64, seed=5, epochs=1, buffer_tokens=256)
        ]
    )
    assert torch.equal(ids, again)
    other = torch.cat(
        [
            token_ids(b)
            for b in reader.shuffled_batches(64, seed=6, epochs=1, buffer_tokens=256)
        ]
    )
    assert not torch.equal(ids, other)


def test_shuffled_reader_applies_exact_prefix_before_permutation(tmp_path):
    write_store(tmp_path, n_tokens=160, tokens_per_shard=37)
    reader = StoreReader(tmp_path, "train")
    batches = list(
        reader.shuffled_batches(
            16,
            seed=9,
            epochs=2,
            buffer_tokens=48,
            prefix_tokens=80,
        )
    )
    observed = torch.cat([token_ids(batch) for batch in batches])
    assert observed.numel() == 160
    assert int(observed.max()) < 80
    assert set(observed[:80].tolist()) == set(range(80))
    assert set(observed[80:].tolist()) == set(range(80))
    with pytest.raises(ValueError, match="prefix_tokens"):
        list(reader.shuffled_batches(16, seed=1, prefix_tokens=161))


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
        tmp_path,
        "train",
        whitener_hash="abc",
        sites=[0],
        d_model=D,
        tokens_per_shard=8,
        free_space_floor_frac=0.0,
    )
    bad = torch.randn(8, 1, D)
    bad[0, 0, 0] = float("nan")
    writer.add(bad)
    with pytest.raises(ValueError, match="non-finite"):
        writer.synchronize()
    assert not tuple((tmp_path / "train").glob("*.safetensors"))
    with pytest.raises(RuntimeError, match="poisoned"):
        writer.close()
    assert writer._executor is None

    zero_writer = ShardWriter(
        tmp_path,
        "zeros",
        whitener_hash="abc",
        sites=[0],
        d_model=D,
        tokens_per_shard=8,
        free_space_floor_frac=0.0,
    )
    zero_writer.add(torch.zeros(8, 1, D))
    with pytest.raises(ValueError, match="zero-row"):
        zero_writer.synchronize()
    zero_writer.abort()

    type_writer = ShardWriter(
        tmp_path,
        "type",
        whitener_hash="abc",
        sites=[0],
        d_model=D,
        free_space_floor_frac=0.0,
    )
    with pytest.raises(TypeError, match="fp16"):
        type_writer.add(torch.randn(4, 1, D).half())
    type_writer.abort()


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
        assert xs.shape == (xf.shape[0], 1, D)
        assert torch.equal(xs, xf[:, 1:2])

    seq_full = torch.cat(list(full.sequential_batches(64)), dim=0)
    seq_sub = torch.cat(list(sub.sequential_batches(64)), dim=0)
    assert torch.equal(seq_sub, seq_full[:, 1:2])

    # Multi-site subsets keep stored order; verify() still checks full shards.
    pair = StoreReader(tmp_path, "train", sites=[7, 13])
    x = next(pair.sequential_batches(64))
    assert torch.equal(x, torch.cat(list(full.sequential_batches(64)))[:64][:, [0, 2]])
    assert pair.verify() == 1000


def test_site_subset_avoids_materializing_contiguous_axes(tmp_path):
    write_store(tmp_path)
    acts = torch.randn(11, 3, D)
    storage = acts.untyped_storage().data_ptr()

    explicit_full = StoreReader(tmp_path, "train", sites=[7, 10, 13])
    full_view = explicit_full._subset(acts)
    assert full_view is acts

    contiguous = StoreReader(tmp_path, "train", sites=[10, 13])
    contiguous_view = contiguous._subset(acts)
    assert torch.equal(contiguous_view, acts[:, 1:3])
    assert contiguous_view.untyped_storage().data_ptr() == storage

    singleton = StoreReader(tmp_path, "train", sites=[10])
    singleton_view = singleton._subset(acts)
    assert torch.equal(singleton_view, acts[:, 1:2])
    assert singleton_view.untyped_storage().data_ptr() == storage

    noncontiguous = StoreReader(tmp_path, "train", sites=[7, 13])
    compact = noncontiguous._subset(acts)
    assert torch.equal(compact, acts[:, [0, 2]])
    assert compact.untyped_storage().data_ptr() != storage

    empty = StoreReader(tmp_path, "train", sites=[])
    assert empty._subset(acts).shape == (len(acts), 0, D)


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


def test_prefetch_close_cancels_early_exit():
    import threading

    from block_crosscoder_experiment.store import prefetch_batches

    source_closed = threading.Event()

    def forever():
        try:
            while True:
                yield torch.zeros(1)
        finally:
            source_closed.set()

    it = prefetch_batches(forever(), depth=1)
    next(it)
    it.close()
    assert source_closed.wait(timeout=2.0)


def test_cuda_prefetch_rejects_non_cuda_device_and_bad_depth():
    source = iter((torch.zeros(1),))
    with pytest.raises(ValueError, match="requires a CUDA device"):
        cuda_prefetch_batches(source, device="cpu")
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="depth must be positive"):
            cuda_prefetch_batches(iter((torch.zeros(1),)), device="cuda", depth=0)
        with pytest.raises(TypeError, match="dtype_policy"):
            cuda_prefetch_batches(
                iter((torch.zeros(1),)),
                device="cuda",
                dtype_policy="float32",  # type: ignore[arg-type]
            )
    else:
        with pytest.raises(RuntimeError, match="CUDA is unavailable"):
            cuda_prefetch_batches(iter((torch.zeros(1),)), device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_prefetch_nested_order_dtype_stream_and_depth_bound():
    device = torch.device("cuda", torch.cuda.current_device())
    consumer_stream = torch.cuda.current_stream(device)
    produced: list[int] = []
    policy_streams: list[int] = []

    def source():
        for index in range(6):
            produced.append(index)
            yield {
                "x": torch.full((128,), index, dtype=torch.bfloat16),
                "metadata": (
                    torch.tensor([index], dtype=torch.int64),
                    [torch.tensor([index % 2 == 0], dtype=torch.bool)],
                ),
            }

    def floating_fp32(tensor: torch.Tensor) -> torch.dtype | None:
        policy_streams.append(torch.cuda.current_stream(device).cuda_stream)
        return torch.float32 if tensor.is_floating_point() else None

    batches = cuda_prefetch_batches(
        source(),
        device=device,
        depth=2,
        dtype_policy=floating_fp32,
    )
    first = next(batches)
    # The current batch plus exactly two lookahead batches were requested.
    assert produced == [0, 1, 2]
    assert first["x"].device == device
    assert first["x"].dtype == torch.float32
    assert first["metadata"][0].dtype == torch.int64
    assert first["metadata"][1][0].dtype == torch.bool
    assert float(first["x"].sum()) == 0.0

    second = next(batches)
    assert produced == [0, 1, 2, 3]
    assert float(second["x"].sum()) == 128.0
    remaining = list(batches)
    assert [int(batch["metadata"][0]) for batch in remaining] == [2, 3, 4, 5]
    assert policy_streams
    assert all(stream != consumer_stream.cuda_stream for stream in policy_streams)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_prefetch_static_dtype_casts_only_floating_leaves():
    batch = (
        torch.arange(8, dtype=torch.bfloat16),
        torch.arange(8, dtype=torch.int64),
    )
    floating, integer = next(
        cuda_prefetch_batches(
            iter((batch,)),
            device="cuda",
            dtype_policy=torch.float32,
        )
    )
    assert floating.dtype == torch.float32
    assert integer.dtype == torch.int64
    assert torch.equal(floating.cpu(), batch[0].float())
    assert torch.equal(integer.cpu(), batch[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_prefetch_preserves_error_position_and_closes_source():
    closed = False

    def source():
        nonlocal closed
        try:
            yield torch.tensor([1.0])
            yield torch.tensor([2.0])
            raise RuntimeError("copy source died")
        finally:
            closed = True

    batches = cuda_prefetch_batches(source(), device="cuda", depth=4)
    assert float(next(batches)) == 1.0
    assert float(next(batches)) == 2.0
    with pytest.raises(RuntimeError, match="copy source died"):
        next(batches)
    assert closed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_prefetch_close_drains_and_closes_source():
    closed = False

    def source():
        nonlocal closed
        try:
            while True:
                yield torch.ones(32)
        finally:
            closed = True

    batches = cuda_prefetch_batches(source(), device="cuda", depth=2)
    next(batches)
    batches.close()
    assert closed
