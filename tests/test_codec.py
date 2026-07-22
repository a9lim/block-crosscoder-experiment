"""Offline tests for the preregistered R-D codec (tranche 3)."""

import copy
from dataclasses import replace
import weakref

from types import SimpleNamespace

import pytest
import torch

import block_crosscoder_experiment.codec as codec_module
from block_crosscoder_experiment.codec import (
    Codec,
    CodecSpec,
    _RDEvaluationInput,
    _artifact_digest,
    _decode_trusted_packet_events_q_chunks,
    _encode_batch_all_q_events,
    _evaluate_rd_stream,
    _packet_from_output,
    _rotate_multi_q_events,
    decode_batch,
    decode_batch_all_q,
    encode_batch,
    encode_batch_all_q,
    estimate_calibration_peak_bytes,
    evaluate_rd,
    fit_codec,
)
from block_crosscoder_experiment.model import BSCOutput, BlockCrosscoder, BSCConfig
from block_crosscoder_experiment.runtime_limits import (
    DECODED_ENERGY_EXACT_IMPLEMENTATION,
    DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION,
)

G, B, S, D = 8, 4, 3, 16


def make_model(seed=0, b=B, g=G, k=3.0, **overrides):
    cfg = BSCConfig(
        n_blocks=g,
        block_dim=b,
        n_sites=S,
        d_model=D,
        k=k,
        seed=seed,
        **overrides,
    )
    m = BlockCrosscoder(cfg)
    return m


def calibrated(m, x):
    m.fit_threshold_([x], m.cfg.k)
    return m


def batches_of(x, n=4):
    return list(x.split(x.shape[0] // n))


def rotate_blocks_(model, seed):
    """Represent the same untied model in an independently rotated block gauge."""
    generator = torch.Generator().manual_seed(seed)
    block_dim = model.cfg.block_dim
    with torch.no_grad():
        for block in range(model.cfg.n_blocks):
            q, r = torch.linalg.qr(
                torch.randn(block_dim, block_dim, generator=generator)
            )
            rotation = (q * torch.sign(torch.diagonal(r))).to(model.D.device)
            model.D[:, block] = torch.einsum("bc,scd->sbd", rotation, model.D[:, block])
            model.E[:, block] = torch.einsum("bc,scd->sbd", rotation, model.E[:, block])


def test_codec_fits_and_evaluates():
    torch.manual_seed(0)
    m = calibrated(make_model(), torch.randn(2048, S, D))
    x = torch.randn(4096, S, D)
    spec = CodecSpec(qs=(4, 8), floor=10, n_bootstrap=64)
    codec = fit_codec(m, batches_of(x), spec)
    assert codec.calib_tokens == 4096
    assert codec.n_included > 0
    res = evaluate_rd(m, codec, batches_of(torch.randn(2048, S, D)), row_len=128)
    assert res["n_rows"] == 16
    assert res["distortion_space"] == "transformed_activation_view"
    assert res["fvu_definition"] == "sse_over_centered_total_in_transformed_view"
    p4, p8 = res["points"]["4"], res["points"]["8"]
    # More levels: distortion no worse, amplitude bits higher.
    assert p8["fvu_pooled"] <= p4["fvu_pooled"] + 1e-9
    assert p8["amplitude_bits_per_token"] > p4["amplitude_bits_per_token"]
    assert p4["rate_bits_per_token"] > p4["amplitude_bits_per_token"]
    lo, hi = p4["fvu_ci95"]
    assert lo <= p4["fvu_pooled"] <= hi
    assert len(p4["fvu_per_site"]) == S
    assert res["rate_model"] == "fixed_width_decodable_payload_bits_v1"
    assert res["zero_rate"]["fvu_pooled"] == 1.0
    assert p4["rate_bits_per_token"] >= p4["amplitude_bits_per_token"]
    assert len(p4["rate_bits_ci95"]) == 2


def test_factorized_codec_pipeline_never_materializes_site_weights(monkeypatch):
    torch.manual_seed(1901)
    model = BlockCrosscoder(
        BSCConfig(
            n_blocks=16,
            block_dim=2,
            n_sites=3,
            d_model=12,
            k=3,
            decoder_constraint="free",
            site_rank=2,
        )
    )
    calibration = torch.randn(512, 3, 12)
    model.fit_threshold_(list(calibration.split(128)), 3, method="exact")

    def refuse_materialization():
        raise AssertionError("direct factorized codec must stay in rank space")

    monkeypatch.setattr(model, "decoder_tensor", refuse_materialization)
    monkeypatch.setattr(model, "encoder_tensor", refuse_materialization)
    spec = CodecSpec(qs=(4, 8), floor=1, n_bootstrap=8)
    codec = fit_codec(model, list(calibration.split(128)), spec)
    evaluation = torch.randn(256, 3, 12)
    packet = encode_batch(model, codec, evaluation[:64], q=4)
    decoded = decode_batch(model, codec, packet)
    assert decoded.shape == evaluation[:64].shape
    _, packets = encode_batch_all_q(model, codec, evaluation[:64])
    decoded_all = decode_batch_all_q(model, codec, packets)
    assert set(decoded_all) == {4, 8}
    result = evaluate_rd(model, codec, list(evaluation.split(64)), row_len=32)
    assert result["n_rows"] == 8


def test_codec_threshold_packets_preserve_stiefel_score_mode_after_reload(tmp_path):
    cfg_values = {
        "n_blocks": 16,
        "block_dim": 2,
        "n_sites": 2,
        "d_model": 6,
        "site_dims": (6, 4),
        "k": 3,
        "seed": 509,
        "selection": "token_topk",
        "encoder_mode": "tied",
        "encoder_fusion": "availability_rescaled_sum",
        "decoder_constraint": "gram",
        "selection_score": "decoded_energy",
    }
    exact = BlockCrosscoder(
        BSCConfig(
            **cfg_values,
            decoded_energy_implementation=DECODED_ENERGY_EXACT_IMPLEMENTATION,
        )
    ).eval()
    fast = BlockCrosscoder(
        BSCConfig(
            **cfg_values,
            decoded_energy_implementation=(
                DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION
            ),
        )
    ).eval()
    fast.load_state_dict(exact.state_dict())
    generator = torch.Generator().manual_seed(510)
    calibration = torch.randn(256, 2, 6, generator=generator)
    evaluation = torch.randn(64, 2, 6, generator=generator)
    exact.fit_threshold_([calibration], 3.0, method="exact")
    fast.fit_threshold_([calibration], 3.0, method="exact")
    assert (
        abs(float(fast.theta - exact.theta)) / max(abs(float(exact.theta)), 1e-12)
        <= 2e-5
    )

    exact_selection = exact(evaluation, mode="threshold")
    fast_selection = fast(evaluation, mode="threshold")
    assert torch.equal(fast_selection.mask, exact_selection.mask)
    codec = fit_codec(
        fast,
        list(calibration.split(64)),
        CodecSpec(qs=(4,), floor=1, n_bootstrap=2),
    )
    assert codec.meta["model_cfg"]["decoded_energy_implementation"] == (
        DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION
    )
    path = tmp_path / "specialized-codec.pt"
    codec.save(path)
    reloaded = type(codec).load(path)
    restored = BlockCrosscoder(BSCConfig(**reloaded.meta["model_cfg"])).eval()
    restored.load_state_dict(fast.state_dict())
    restored.validate_decoded_energy_implementation()

    exact_packet = encode_batch(exact, reloaded, evaluation, q=4)
    fast_packet = encode_batch(fast, reloaded, evaluation, q=4)
    restored_packet = encode_batch(restored, reloaded, evaluation, q=4)
    for field in ("counts", "block_ids", "amplitude_symbols"):
        assert torch.equal(getattr(fast_packet, field), getattr(exact_packet, field))
        assert torch.equal(
            getattr(restored_packet, field),
            getattr(fast_packet, field),
        )


def test_codec_calibration_memory_ceiling_fails_without_sampling():
    torch.manual_seed(101)
    model = calibrated(make_model(), torch.randn(128, S, D))
    spec = CodecSpec(qs=(4,), floor=1, n_bootstrap=2, max_calibration_event_bytes=1)
    with pytest.raises(MemoryError, match="memory ceiling"):
        fit_codec(model, [torch.randn(16, S, D)], spec)


def test_codec_calibration_memory_estimator_caps_moment_workspace():
    boundary = 262_144
    at_boundary = estimate_calibration_peak_bytes(boundary, 4)
    assert at_boundary == boundary * (32 + 24 * 4 + 8 * 4 * 4 + 8 * 4)
    assert estimate_calibration_peak_bytes(boundary + 1, 4) - at_boundary == (
        32 + 24 * 4
    )


def test_high_q_approaches_unquantized():
    torch.manual_seed(1)
    m = calibrated(make_model(), torch.randn(2048, S, D))
    x = torch.randn(4096, S, D)
    spec = CodecSpec(qs=(12,), floor=10, n_bootstrap=8)
    codec = fit_codec(m, batches_of(x), spec)
    ev = torch.randn(2048, S, D)
    res = evaluate_rd(m, codec, batches_of(ev), row_len=128)

    # Reference: unquantized threshold-mode FVU with the same exclusion
    # and the same calib-mean centering.
    with torch.no_grad():
        err = tot = 0.0
        mu = codec.calib_mean.float()
        for xb in batches_of(ev):
            z = m.encode(xb)
            mask = m.select(z, mode="threshold") & codec.included.unsqueeze(0)
            xhat = m.decode(z * mask.unsqueeze(-1))
            err += float((xb - xhat).double().pow(2).sum())
            tot += float((xb - mu).double().pow(2).sum())
    assert abs(res["points"]["12"]["fvu_pooled"] - err / tot) < 5e-3


def test_sequence_bootstrap_uses_stored_ids_across_batch_boundaries():
    torch.manual_seed(102)
    model = calibrated(make_model(), torch.randn(256, S, D))
    codec = fit_codec(
        model,
        batches_of(torch.randn(256, S, D)),
        CodecSpec(qs=(4,), floor=1, n_bootstrap=8),
    )
    x = torch.randn(12, S, D)
    sequence_ids = torch.tensor([7] * 3 + [8] * 5 + [11] * 4)
    row_ids = torch.stack((sequence_ids, torch.arange(12)), dim=1)
    pairs = [
        (x[:4], row_ids[:4]),
        (x[4:9], row_ids[4:9]),
        (x[9:], row_ids[9:]),
    ]
    result = evaluate_rd(model, codec, pairs)
    assert result["n_rows"] == 3
    assert result["sequence_grouping"] == "stored_sequence_ids"
    assert result["row_len"] is None

    bad = [(x[:6], row_ids[:6]), (x[6:], row_ids[6:].flip(0))]
    with pytest.raises(ValueError, match="strictly increasing"):
        evaluate_rd(model, codec, bad)


def test_gauge_rotation_invariance():
    """Rotating a block's decoder/encoder/code gauge must not move
    the codec's R-D point — the canonical orientation absorbs it."""
    torch.manual_seed(2)
    calib = torch.randn(4096, S, D)
    ev = torch.randn(2048, S, D)
    spec = CodecSpec(qs=(4,), floor=10, n_bootstrap=8)

    m1 = calibrated(make_model(seed=3), calib[:2048])
    m2 = calibrated(make_model(seed=3), calib[:2048])
    rotate_blocks_(m2, seed=99)
    # Same function represented in a rotated gauge: outputs identical.
    with torch.no_grad():
        assert torch.allclose(m1(ev[:64]).xhat, m2(ev[:64]).xhat, atol=1e-4)

    r1 = evaluate_rd(
        m1, fit_codec(m1, batches_of(calib), spec), batches_of(ev), row_len=128
    )
    r2 = evaluate_rd(
        m2, fit_codec(m2, batches_of(calib), spec), batches_of(ev), row_len=128
    )
    f1, f2 = r1["points"]["4"]["fvu_pooled"], r2["points"]["4"]["fvu_pooled"]
    assert abs(f1 - f2) < 5e-3, (f1, f2)
    assert abs(r1["support_bits_per_token"] - r2["support_bits_per_token"]) < 1e-6


def test_floor_exclusion_reported_and_enforced():
    torch.manual_seed(4)
    m = calibrated(make_model(), torch.randn(2048, S, D))
    x = torch.randn(2048, S, D)
    # Impossible floor: everything excluded -> no bits, FVU = ratio to mean.
    spec = CodecSpec(qs=(4,), floor=10**9, n_bootstrap=8)
    codec = fit_codec(m, batches_of(x), spec)
    assert codec.n_included == 0
    assert codec.meta["n_excluded"] == G
    res = evaluate_rd(m, codec, batches_of(x), row_len=128)
    assert res["avg_count"] == 0.0
    assert res["support_bits_per_token"] == 0.0
    assert res["bernoulli_bits_per_token"] == 0.0
    assert res["points"]["4"]["amplitude_bits_per_token"] == 0.0


def test_codec_serialization_roundtrip(tmp_path):
    torch.manual_seed(44)
    m = calibrated(make_model(), torch.randn(2048, S, D))
    codec = fit_codec(
        m,
        batches_of(torch.randn(2048, S, D)),
        CodecSpec(qs=(4, 6), floor=10, n_bootstrap=8),
    )
    codec.meta["binding"] = {"whitener_hash": "abc"}
    path = tmp_path / "codec.pt"
    codec.save(path)
    loaded = type(codec).load(path)
    assert loaded.spec == codec.spec
    assert loaded.calib_tokens == codec.calib_tokens
    assert loaded.meta == codec.meta
    for name in (
        "included",
        "rank_to_block",
        "rotation",
        "lo",
        "hi",
        "count_log2p",
        "bernoulli_log2p",
        "bernoulli_log2q",
        "calib_events",
        "calib_mean",
    ):
        assert torch.equal(getattr(loaded, name), getattr(codec, name))


def test_rehashed_codec_bytes_still_require_semantic_validity():
    torch.manual_seed(144)
    model = calibrated(make_model(), torch.randn(256, S, D))
    codec = fit_codec(
        model,
        [torch.randn(256, S, D)],
        CodecSpec(qs=(4,), floor=1, n_bootstrap=2),
    )
    pristine = codec.to_payload()

    def authenticated(mutated):
        unsigned = {
            key: value for key, value in mutated.items() if key != "artifact_sha256"
        }
        mutated["artifact_sha256"] = _artifact_digest(unsigned)
        return mutated

    extra = copy.deepcopy(pristine)
    extra["ignored_future_field"] = 1
    extra = authenticated(extra)
    with pytest.raises(ValueError, match="payload keys mismatch"):
        type(codec).from_payload(extra)

    nonorthogonal = copy.deepcopy(pristine)
    nonorthogonal["rotation"][0].zero_()
    with pytest.raises(ValueError, match="not orthonormal"):
        type(codec).from_payload(authenticated(nonorthogonal))

    inverted_range = copy.deepcopy(pristine)
    inverted_range["hi"][0, 0] = inverted_range["lo"][0, 0] - 1
    with pytest.raises(ValueError, match="ceiling is below"):
        type(codec).from_payload(authenticated(inverted_range))

    bad_probability = copy.deepcopy(pristine)
    bad_probability["count_log2p"].zero_()
    with pytest.raises(ValueError, match="not normalized"):
        type(codec).from_payload(authenticated(bad_probability))

    wrong_dtype = copy.deepcopy(pristine)
    wrong_dtype["lo"] = wrong_dtype["lo"].double()
    with pytest.raises(TypeError, match="lo dtype"):
        type(codec).from_payload(authenticated(wrong_dtype))


def test_count_model_is_fit_after_floor_exclusion():
    class StubModel:
        cfg = SimpleNamespace(n_blocks=3, block_dim=1, n_sites=1, d_model=1)

        def __call__(self, x, mode):
            masks = torch.tensor(
                [[1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                dtype=torch.bool,
                device=x.device,
            )
            z = torch.ones(4, 3, 1, device=x.device)
            z_selected = z * masks.unsqueeze(-1)
            return SimpleNamespace(
                xhat=torch.zeros_like(x),
                z=z,
                z_selected=z_selected,
                scores=z.squeeze(-1),
                mask=masks,
            )

    codec = fit_codec(
        StubModel(),
        [torch.arange(4, dtype=torch.float32).view(4, 1, 1)],
        CodecSpec(qs=(4,), floor=2, n_bootstrap=8),
    )
    assert codec.included.tolist() == [True, True, False]
    probs = codec.count_log2p.exp2()
    # Included counts are [2,1,1,0]. The complete legal alphabet is 0..2,
    # so add-one-smoothed masses are exactly 2,3,2 (no tail-clamp aliases).
    assert probs[0] == pytest.approx(2 / 7)
    assert probs[1] == pytest.approx(3 / 7)
    assert probs[2] == pytest.approx(2 / 7)


def test_packet_compacts_noncontiguous_included_block_ids():
    torch.manual_seed(104)
    model = calibrated(make_model(b=1, g=16, k=2.0), torch.randn(256, S, D))
    codec = fit_codec(
        model,
        [torch.randn(256, S, D)],
        CodecSpec(qs=(4,), floor=1, n_bootstrap=2),
    )
    included = torch.zeros(16, dtype=torch.bool)
    included[[0, 15]] = True
    codec = replace(
        codec,
        included=included,
        rank_to_block=torch.tensor([0, 15], dtype=torch.long),
        count_log2p=torch.zeros(3, dtype=torch.float64),
    )
    mask = included.view(1, -1)
    z = torch.full((1, 16, 1), 0.5)
    out = SimpleNamespace(mask=mask, z_selected=z * mask.unsqueeze(-1))
    packet = _packet_from_output(model, codec, out, q=4)
    assert packet.block_ids.tolist() == [0, 1]
    assert codec.rank_to_block[packet.block_ids.long()].tolist() == [0, 15]
    # Two compact IDs require one bit each. Pricing raw dictionary IDs would
    # incorrectly need four bits and is not the packet this codec decodes.
    assert (codec.n_included - 1).bit_length() == 1
    assert decode_batch(model, codec, packet).shape == (1, S, D)

    bad = replace(packet, block_ids=torch.tensor([0, 2], dtype=torch.int32))
    with pytest.raises(ValueError, match="block rank"):
        decode_batch(model, codec, bad)


def test_scalar_b1_path():
    torch.manual_seed(5)
    m = calibrated(make_model(b=1, g=32, k=12.0), torch.randn(2048, S, D))
    x = torch.randn(4096, S, D)
    spec = CodecSpec(qs=(6,), floor=10, n_bootstrap=8)
    codec = fit_codec(m, batches_of(x), spec)
    res = evaluate_rd(m, codec, batches_of(torch.randn(2048, S, D)), row_len=128)
    p = res["points"]["6"]
    # b=1: amplitude bits = q * realized count.
    assert abs(p["amplitude_bits_per_token"] - 6 * res["avg_count"]) < 1e-9


def test_count_model_prices_every_legal_count_and_rejects_impossible_counts():
    torch.manual_seed(6)
    m = calibrated(make_model(), torch.randn(2048, S, D))
    spec = CodecSpec(qs=(4,), floor=1, n_bootstrap=8)
    codec = fit_codec(m, batches_of(torch.randn(2048, S, D)), spec)
    k = torch.arange(codec.n_included + 1)
    lp = codec.log2_count_prob(k)
    assert torch.isfinite(lp).all()  # smoothing: no -inf anywhere
    with pytest.raises(ValueError, match="outside"):
        codec.log2_count_prob(torch.tensor([codec.n_included + 1]))


def test_explicit_sparse_packet_round_trip():
    cfg = BSCConfig(
        n_blocks=6,
        block_dim=2,
        n_sites=2,
        d_model=5,
        k=2,
    )
    model = BlockCrosscoder(cfg)
    x = torch.randn(96, 2, 5)
    model.fit_threshold_([x[:48]], target_avg_blocks=2)
    codec = fit_codec(model, [x[:48]], CodecSpec(qs=(4,), floor=1, n_bootstrap=4))
    packet = encode_batch(model, codec, x[48:], 4)
    decoded = decode_batch(model, codec, packet)
    assert decoded.shape == x[48:].shape
    assert torch.isfinite(decoded).all()
    assert packet.amplitude_symbols.dtype == torch.int32
    with pytest.raises(ValueError, match="length"):
        decode_batch(model, codec, replace(packet, block_ids=packet.block_ids[:-1]))
    if packet.block_ids.numel():
        bad_symbols = packet.amplitude_symbols.clone()
        bad_symbols[0, 0] = 1 << packet.q
        with pytest.raises(ValueError, match="alphabet"):
            decode_batch(model, codec, replace(packet, amplitude_symbols=bad_symbols))


@pytest.mark.parametrize("block_dim", (1, 2, 4, 8))
def test_multi_q_event_rotation_preserves_cpu_reduction(block_dim):
    generator = torch.Generator().manual_seed(1701 + block_dim)
    event_rotation = torch.randn(97, block_dim, block_dim, generator=generator)
    canonical_codes = torch.randn(6, 97, block_dim, generator=generator)
    expected = torch.einsum(
        "eji,qej->qei",
        event_rotation,
        canonical_codes,
    )

    actual = _rotate_multi_q_events(event_rotation, canonical_codes)

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_multi_q_scalar_cuda_rotation_is_exact_and_skips_matmul(monkeypatch):
    generator = torch.Generator(device="cuda").manual_seed(1702)
    event_rotation = torch.randn(65_536, 1, 1, device="cuda", generator=generator)
    canonical_codes = torch.randn(6, 65_536, 1, device="cuda", generator=generator)
    expected = torch.einsum(
        "eji,qej->qei",
        event_rotation,
        canonical_codes,
    )

    def unexpected_matmul(*args, **kwargs):
        raise AssertionError("scalar multi-q rotation must not launch matmul")

    monkeypatch.setattr(torch, "matmul", unexpected_matmul)
    actual = _rotate_multi_q_events(event_rotation, canonical_codes)

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("block_dim", (2, 4, 6, 8))
def test_multi_q_cuda_rotation_stays_within_release_bound(monkeypatch, block_dim):
    generator = torch.Generator(device="cuda").manual_seed(1701 + block_dim)
    event_rotation = torch.randn(
        65_536,
        block_dim,
        block_dim,
        device="cuda",
        generator=generator,
    )
    canonical_codes = torch.randn(
        2,
        65_536,
        block_dim,
        device="cuda",
        generator=generator,
    )
    expected = torch.einsum(
        "eji,qej->qei",
        event_rotation,
        canonical_codes,
    )
    original_matmul = torch.matmul
    matmul_calls = 0

    def counted_matmul(*args, **kwargs):
        nonlocal matmul_calls
        matmul_calls += 1
        return original_matmul(*args, **kwargs)

    monkeypatch.setattr(torch, "matmul", counted_matmul)
    actual = _rotate_multi_q_events(event_rotation, canonical_codes)

    difference = actual - expected
    relative_l2 = difference.norm() / expected.norm().clamp_min(1e-30)
    assert matmul_calls == 1
    assert difference.abs().max().item() <= 5e-6
    assert relative_l2.item() <= 3e-7


@pytest.mark.parametrize("encoder_mode", ("untied", "tied"))
def test_all_q_encoding_runs_one_selection_and_matches_packets(
    monkeypatch, encoder_mode
):
    model = make_model(g=12, b=2, k=3, encoder_mode=encoder_mode)
    x = torch.randn(96, S, D)
    model.fit_threshold_([x[:48]], target_avg_blocks=3)
    codec = fit_codec(
        model,
        [x[:48]],
        CodecSpec(qs=(2, 4, 6), floor=1, n_bootstrap=4),
    )
    original = model.select_with_materialized
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "select_with_materialized", counted)
    original_decode = model.decode
    decode_calls = 0

    def counted_decode(*args, **kwargs):
        nonlocal decode_calls
        decode_calls += 1
        return original_decode(*args, **kwargs)

    monkeypatch.setattr(model, "decode", counted_decode)
    out, events, packets = _encode_batch_all_q_events(model, codec, x[48:])
    assert calls == 1
    assert decode_calls == 0
    sparse_mm = torch.sparse.mm
    sparse_calls = 0

    def counted_sparse_mm(*args, **kwargs):
        nonlocal sparse_calls
        sparse_calls += 1
        return sparse_mm(*args, **kwargs)

    monkeypatch.setattr(torch.sparse, "mm", counted_sparse_mm)
    decoded = decode_batch_all_q(model, codec, packets)
    assert sparse_calls == 1
    for q, packet in packets.items():
        expected = _packet_from_output(model, codec, out, q)
        assert torch.equal(packet.counts, expected.counts)
        assert torch.equal(packet.block_ids, expected.block_ids)
        assert torch.equal(packet.amplitude_symbols, expected.amplitude_symbols)
        torch.testing.assert_close(
            decoded[q],
            decode_batch(model, codec, packet),
            rtol=1e-6,
            atol=1e-6,
        )

    public_out, public_packets = encode_batch_all_q(model, codec, x[48:])
    assert calls == 2
    assert decode_calls == 1
    assert isinstance(public_out, BSCOutput)
    assert torch.equal(
        public_out.xhat,
        original_decode(public_out.z_selected),
    )
    assert public_packets.keys() == packets.keys()

    alternate_decoder = model.decoder_tensor().clone()
    alternate_decoder[0, 0, 0, 0] += 1.0
    alternate_out, _ = encode_batch_all_q(
        model,
        codec,
        x[48:],
        _decoder=alternate_decoder,
    )
    with torch.no_grad():
        expected_alternate = model.forward_with_materialized(
            x[48:],
            mode="threshold",
            _decoder=alternate_decoder,
            _score_geometry=model._frozen_score_geometry(alternate_decoder),
        )[0]
    for actual, expected in zip(alternate_out, expected_alternate, strict=True):
        assert torch.equal(actual, expected)

    sparse_calls = 0
    tensor_on = codec._tensor_on
    rotation_lookups = 0

    def counted_tensor_on(name, *args, **kwargs):
        nonlocal rotation_lookups
        if name == "rotation":
            rotation_lookups += 1
        return tensor_on(name, *args, **kwargs)

    monkeypatch.setattr(codec, "_tensor_on", counted_tensor_on)
    trusted = {}
    for chunk in _decode_trusted_packet_events_q_chunks(
        model,
        codec,
        events,
        packets,
        q_chunk_size=2,
    ):
        trusted.update(chunk)
    assert sparse_calls == 2
    assert rotation_lookups == 1
    monkeypatch.setattr(codec, "_tensor_on", tensor_on)
    rows = torch.repeat_interleave(
        torch.arange(events.n_tokens, device=events.counts.device),
        events.counts.long(),
    )
    for q, packet in packets.items():
        torch.testing.assert_close(trusted[q], decoded[q], rtol=1e-6, atol=1e-6)
        levels = (1 << q) - 1
        lo = codec._tensor_on("lo", events.original_ids.device)[events.original_ids]
        span = (
            codec._tensor_on("hi", events.original_ids.device)[events.original_ids] - lo
        ).clamp_min(1e-12)
        canonical = lo + packet.amplitude_symbols.float() / levels * span
        values = torch.einsum(
            "eji,ej->ei",
            codec._tensor_on("rotation", events.original_ids.device)[
                events.original_ids
            ],
            canonical,
        )
        dense_code = torch.zeros(
            events.n_tokens,
            model.cfg.n_blocks,
            model.cfg.block_dim,
            device=values.device,
        )
        dense_code[rows, events.original_ids] = values
        torch.testing.assert_close(
            trusted[q],
            model.decode(dense_code),
            rtol=1e-6,
            atol=1e-6,
        )

    lazy_trusted = {}
    for chunk in _decode_trusted_packet_events_q_chunks(
        model,
        codec,
        events,
        qs=tuple(packets),
        q_chunk_size=2,
    ):
        lazy_trusted.update(chunk)
    for q in packets:
        torch.testing.assert_close(
            lazy_trusted[q],
            trusted[q],
            rtol=0,
            atol=0,
        )

    support_mutation = dict(packets)
    support_mutation[4] = replace(
        packets[4],
        block_ids=packets[4].block_ids.clone(),
    )
    with pytest.raises(ValueError, match="support is not event-bound"):
        list(
            _decode_trusted_packet_events_q_chunks(
                model,
                codec,
                events,
                support_mutation,
            )
        )

    previous_result = None

    def lifetime_checked_sparse_mm(*args, **kwargs):
        nonlocal previous_result
        assert previous_result is None or previous_result() is None
        result = sparse_mm(*args, **kwargs)
        previous_result = weakref.ref(result)
        return result

    monkeypatch.setattr(torch.sparse, "mm", lifetime_checked_sparse_mm)
    reduced = {}
    for chunk in _decode_trusted_packet_events_q_chunks(
        model,
        codec,
        events,
        packets,
        q_chunk_size=2,
    ):
        for q, prediction in chunk.items():
            reduced[q] = prediction.sum().item()
        del prediction, chunk
    assert reduced.keys() == packets.keys()

    mismatched = dict(packets)
    changed_counts = packets[4].counts.clone()
    changed_counts[0] += 1
    mismatched[4] = replace(packets[4], counts=changed_counts)
    with pytest.raises(ValueError, match="identical support counts"):
        decode_batch_all_q(model, codec, mismatched)


def test_codec_device_cache_refreshes_after_tensor_mutation():
    model = make_model(g=6, b=2, k=2)
    x = torch.randn(64, S, D)
    model.fit_threshold_([x[:32]], target_avg_blocks=2)
    codec = fit_codec(
        model,
        [x[:32]],
        CodecSpec(qs=(4,), floor=1, n_bootstrap=4),
    )
    first = codec._tensor_on("lo", "cpu", dtype=torch.float64).clone()
    codec.lo.add_(1.0)
    second = codec._tensor_on("lo", "cpu", dtype=torch.float64)
    assert torch.equal(second, first + 1.0)


class _RecordingRDEvaluationObserver:
    def __init__(self) -> None:
        self.batch_contexts: list[object | None] = []
        self.packet_events: list[object] = []
        self.chunk_qs: list[tuple[int, ...]] = []
        self.predictions: dict[int, list[torch.Tensor]] = {}
        self.ended = 0

    def begin_batch(self, batch) -> None:
        self.batch_contexts.append(batch.context)
        self.packet_events.append(batch.packet_events)
        assert batch.sequence_ids.device.type == "cpu"
        assert batch.sequence_ids.dtype == torch.int64

    def consume_decoded_chunk(self, batch, decoded_chunk) -> None:
        assert batch.packet_events is self.packet_events[-1]
        self.chunk_qs.append(tuple(decoded_chunk))
        for q, prediction in decoded_chunk.items():
            self.predictions.setdefault(q, []).append(prediction.detach().cpu().clone())

    def end_batch(self, batch) -> None:
        assert batch.packet_events is self.packet_events[-1]
        self.ended += 1


def _joint_rd_fixture(
    *,
    qs: tuple[int, ...],
    decoder_bias: bool = True,
    site_dims: tuple[int, ...] | None = None,
) -> tuple[BlockCrosscoder, Codec, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(811)
    model = make_model(
        seed=812,
        b=2,
        g=10,
        k=3.0,
        decoder_bias=decoder_bias,
        site_dims=site_dims,
    )
    calibration = torch.randn(96, S, D, generator=generator)
    if site_dims is not None:
        calibration = calibration * model.coordinate_mask[:, 0, 0].cpu()
    calibrated(model, calibration)
    if decoder_bias:
        with torch.no_grad():
            model.c.copy_(
                torch.randn(model.c.shape, generator=generator)
                * model.coordinate_mask[:, 0, 0].cpu()
            )
    codec = fit_codec(
        model,
        list(calibration.split(24)),
        CodecSpec(qs=qs, floor=1, n_bootstrap=8),
    )
    evaluation = torch.randn(19, S, D, generator=generator)
    if site_dims is not None:
        evaluation = evaluation * model.coordinate_mask[:, 0, 0].cpu()
    sequence_ids = torch.tensor([3] * 4 + [8] * 7 + [19] * 8)
    row_ids = torch.stack((sequence_ids, torch.arange(len(sequence_ids))), dim=1)
    return model, codec, evaluation, row_ids


def test_joint_rd_stream_preserves_public_payload_and_reuses_one_packet_traversal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qs = (8, 2, 12, 4, 6)
    model, codec, evaluation, row_ids = _joint_rd_fixture(qs=qs)
    slices = (slice(0, 5), slice(5, 13), slice(13, 19))
    public_batches = [(evaluation[sl], row_ids[sl]) for sl in slices]
    expected = evaluate_rd(model, codec, public_batches)

    observer = _RecordingRDEvaluationObserver()
    contexts = [object() for _ in slices]
    joint_batches = [
        _RDEvaluationInput(evaluation[sl], row_ids[sl], context)
        for sl, context in zip(slices, contexts, strict=True)
    ]
    threshold_calls = 0
    decode_calls = 0
    original_threshold = codec_module._threshold_select
    original_decode = codec_module._decode_trusted_packet_events_q_chunks

    def counted_threshold(*args, **kwargs):
        nonlocal threshold_calls
        threshold_calls += 1
        return original_threshold(*args, **kwargs)

    def counted_decode(*args, **kwargs):
        nonlocal decode_calls
        decode_calls += 1
        yield from original_decode(*args, **kwargs)

    monkeypatch.setattr(codec_module, "_threshold_select", counted_threshold)
    monkeypatch.setattr(
        codec_module,
        "_decode_trusted_packet_events_q_chunks",
        counted_decode,
    )
    actual = _evaluate_rd_stream(model, codec, joint_batches, observer=observer)

    assert actual == expected
    assert threshold_calls == len(slices)
    assert decode_calls == len(slices)
    assert observer.batch_contexts == contexts
    assert observer.ended == len(slices)
    assert observer.chunk_qs == [
        chunk for _ in slices for chunk in ((8, 2), (12, 4), (6,))
    ]
    assert tuple(observer.predictions) == qs
    assert all(len(observer.predictions[q]) == len(slices) for q in qs)
    for events, sl in zip(observer.packet_events, slices, strict=True):
        assert events.n_tokens == len(evaluation[sl])
        assert int(events.counts.sum()) == len(events.block_ids)


@pytest.mark.parametrize("decoder_bias", (False, True))
@pytest.mark.parametrize("padded", (False, True))
@pytest.mark.parametrize("zero_support", (False, True))
def test_joint_rd_stream_matches_public_for_bias_padding_and_zero_support(
    decoder_bias: bool,
    padded: bool,
    zero_support: bool,
) -> None:
    qs = (2, 4, 6)
    site_dims = (D, D - 3, D - 7) if padded else None
    model, codec, evaluation, row_ids = _joint_rd_fixture(
        qs=qs,
        decoder_bias=decoder_bias,
        site_dims=site_dims,
    )
    if zero_support:
        codec = replace(
            codec,
            included=torch.zeros_like(codec.included),
            rank_to_block=torch.empty(0, dtype=torch.long),
            count_log2p=torch.zeros(1, dtype=torch.float64),
        )
    slices = (slice(0, 6), slice(6, 14), slice(14, 19))
    expected = evaluate_rd(
        model,
        codec,
        [(evaluation[sl], row_ids[sl]) for sl in slices],
    )
    observer = _RecordingRDEvaluationObserver()
    actual = _evaluate_rd_stream(
        model,
        codec,
        [
            _RDEvaluationInput(
                evaluation[sl],
                row_ids[sl],
                {"raw": evaluation[sl]},
            )
            for sl in slices
        ],
        observer=observer,
    )

    assert actual == expected
    assert observer.ended == len(slices)
    assert observer.chunk_qs == [chunk for _ in slices for chunk in ((2, 4), (6,))]
    if zero_support:
        assert actual["avg_count"] == 0.0
        assert actual["support_bits_per_token"] == 0.0
        assert all(int(events.counts.sum()) == 0 for events in observer.packet_events)
    if padded:
        coordinate_mask = model.coordinate_mask[:, 0, 0].cpu()
        for predictions in observer.predictions.values():
            for prediction in predictions:
                assert torch.equal(
                    prediction * ~coordinate_mask,
                    torch.zeros_like(prediction),
                )
