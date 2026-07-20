import torch

from block_crosscoder_experiment.cli.verify_store import _verify_round_trip
from block_crosscoder_experiment.store import Whitener


class _Reader:
    def __init__(self, batches):
        self.batches = batches

    def sequential_batches(self, _batch_size):
        yield from self.batches


def test_round_trip_uses_configured_transform_and_aligns_chunk_boundaries():
    torch.manual_seed(0)
    raw = torch.randn(9, 2, 5, dtype=torch.bfloat16)
    whitener = Whitener(
        mean=torch.zeros(2, 5),
        W=torch.eye(5).expand(2, 5, 5).clone(),
        ridge=torch.zeros(2),
        eigenvalues=torch.ones(2, 5),
        sites=(1, 2),
        n_fit_tokens=10,
        meta={"normalization": "layer", "layer_norm_eps": 1e-5},
    )
    stored = whitener.apply(raw).to(torch.bfloat16)
    n, rel, exact = _verify_round_trip(
        whitener,
        _Reader([raw[:2], raw[2:7], raw[7:]]),
        _Reader([stored[:4], stored[4:5], stored[5:]]),
        device="cpu",
        batch_size=3,
    )
    assert n == raw.shape[0]
    assert rel == 0.0
    assert exact == 1.0


def test_round_trip_fails_if_stored_prefix_is_shorter():
    raw = torch.zeros(3, 1, 2, dtype=torch.bfloat16)

    class Identity:
        @staticmethod
        def apply(x):
            return x.float()

    try:
        _verify_round_trip(
            Identity(), _Reader([raw]), _Reader([raw[:2]]), device="cpu"
        )
    except ValueError as exc:
        assert "shorter" in str(exc)
    else:
        raise AssertionError("short stored prefix was accepted")
