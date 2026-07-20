"""Offline tests for the preregistered R-D codec (tranche 3)."""

from types import SimpleNamespace

import pytest
import torch

from block_crosscoder_experiment.codec import CodecSpec, evaluate_rd, fit_codec
from block_crosscoder_experiment.model import BlockCrosscoder, BSCConfig

G, B, S, D = 8, 4, 3, 16


def make_model(seed=0, b=B, g=G, k=3.0):
    cfg = BSCConfig(n_blocks=g, block_dim=b, n_sites=S, d_model=D, k=k, seed=seed)
    m = BlockCrosscoder(cfg)
    return m


def calibrated(m, x):
    m.fit_threshold_([x], m.cfg.k)
    return m


def batches_of(x, n=4):
    return list(x.split(x.shape[0] // n))


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
    p4, p8 = res["points"]["4"], res["points"]["8"]
    # More levels: distortion no worse, amplitude bits higher.
    assert p8["fvu_pooled"] <= p4["fvu_pooled"] + 1e-9
    assert p8["amplitude_bits_per_token"] > p4["amplitude_bits_per_token"]
    assert p4["rate_bits_per_token"] > p4["amplitude_bits_per_token"]
    lo, hi = p4["fvu_ci95"]
    assert lo <= p4["fvu_pooled"] <= hi
    assert len(p4["fvu_per_site"]) == S


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


def test_gauge_rotation_invariance():
    """R13: rotating a block's decoder/encoder/code gauge must not move
    the codec's R-D point — the canonical orientation absorbs it."""
    from block_crosscoder_experiment.battery import rotate_blocks_

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

    r1 = evaluate_rd(m1, fit_codec(m1, batches_of(calib), spec),
                     batches_of(ev), row_len=128)
    r2 = evaluate_rd(m2, fit_codec(m2, batches_of(calib), spec),
                     batches_of(ev), row_len=128)
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
        "included", "rotation", "lo", "hi", "count_log2p",
        "bernoulli_log2p", "bernoulli_log2q", "calib_events", "calib_mean",
    ):
        assert torch.equal(getattr(loaded, name), getattr(codec, name))


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
                xhat=torch.zeros_like(x), z=z, z_selected=z_selected,
                scores=z.squeeze(-1), mask=masks,
            )

    codec = fit_codec(
        StubModel(),
        [torch.arange(4, dtype=torch.float32).view(4, 1, 1)],
        CodecSpec(qs=(4,), floor=2, n_bootstrap=8),
    )
    assert codec.included.tolist() == [True, True, False]
    probs = codec.count_log2p.exp2()
    # Included counts are [2,1,1,0], so add-one-smoothed masses are 2,3,2.
    assert probs[0] == pytest.approx(2 / 17)
    assert probs[1] == pytest.approx(3 / 17)
    assert probs[2] == pytest.approx(2 / 17)


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


def test_count_model_prices_unseen_counts():
    torch.manual_seed(6)
    m = calibrated(make_model(), torch.randn(2048, S, D))
    spec = CodecSpec(qs=(4,), floor=1, n_bootstrap=8)
    codec = fit_codec(m, batches_of(torch.randn(2048, S, D)), spec)
    k = torch.tensor([0, 1, 10**6])
    lp = codec.log2_count_prob(k)
    assert torch.isfinite(lp).all()  # smoothing: no -inf anywhere
