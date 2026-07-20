import json
from types import SimpleNamespace

import pytest
import torch

from block_crosscoder_experiment.cli.train_bsc import (
    _calibration_extent,
    _prepare_run_dir,
    _raw_reconstruct,
    _validate_raw_alignment,
)
from block_crosscoder_experiment.store import Whitener


def test_calibration_extent_includes_final_partial_batch():
    assert _calibration_extent(8193, 4096, None) == (3, 8193)
    assert _calibration_extent(8193, 4096, 2) == (2, 8192)
    with pytest.raises(ValueError, match="cap"):
        _calibration_extent(8193, 4096, 0)


def _aligned_reader(*, n_tokens=10, model="m"):
    meta = {
        "model": model,
        "model_revision": "rev",
        "corpus": "data",
        "corpus_config": "cfg",
        "corpus_revision": "data-rev",
        "corpus_split": "train",
        "context_size": 8,
        "prepend_bos": True,
        "dropped_positions": 2,
        "pack_convention": "concat-no-boundary",
        "hook_names": ["blocks.1.hook_resid_post"],
    }
    return SimpleNamespace(
        n_tokens=n_tokens,
        sites=(1,),
        d_model=4,
        manifest={"meta": meta},
    )


def test_raw_store_alignment_is_fail_closed():
    normalized = _aligned_reader()
    raw = _aligned_reader()
    normalized_whitener = SimpleNamespace(mode="whiten", meta={"model": "m"})
    raw_whitener = SimpleNamespace(mode="none", meta={"model": "m"})
    _validate_raw_alignment(
        normalized, raw, normalized_whitener, raw_whitener
    )
    raw.n_tokens += 1
    with pytest.raises(ValueError, match="not token/site/model aligned"):
        _validate_raw_alignment(
            normalized, raw, normalized_whitener, raw_whitener
        )


def test_raw_reconstruction_matches_whitener_convention():
    raw = torch.randn(5, 1, 4)
    eye = torch.eye(4).unsqueeze(0)
    none = Whitener(
        mean=torch.zeros(1, 4),
        W=eye,
        ridge=torch.zeros(1),
        eigenvalues=torch.ones(1, 4),
        sites=(1,),
        n_fit_tokens=5,
        meta={"normalization": "none"},
    )
    normalized_none = none.apply(raw)
    assert torch.allclose(_raw_reconstruct(none, normalized_none, raw), raw)
    inverse = (torch.linalg.inv(none.W.double()).float(), none.mean)
    assert torch.allclose(
        _raw_reconstruct(
            none, normalized_none, raw, linear_inverse=inverse
        ),
        raw,
    )
    layer = SimpleNamespace(
        mode="layer", meta={"layer_norm_eps": 1e-5}
    )
    normalized = torch.nn.functional.layer_norm(raw, (4,), eps=1e-5)
    assert torch.allclose(
        _raw_reconstruct(layer, normalized, raw), raw, atol=1e-5
    )


def test_run_directory_refuses_fresh_log_contamination(tmp_path):
    run = tmp_path / "run"
    manifest = {"format_version": 1, "binding": {"whitener_hash": "abc"}}
    _prepare_run_dir(run, manifest, resume=False)
    assert json.loads((run / "run_manifest.json").read_text()) == manifest
    (run / "steps.jsonl").write_text("old run\n")
    with pytest.raises(SystemExit, match="refuses non-empty"):
        _prepare_run_dir(run, manifest, resume=False)


def test_run_directory_resume_requires_exact_manifest(tmp_path):
    run = tmp_path / "run"
    manifest = {
        "format_version": 1,
        "binding": {"whitener_hash": "abc", "betas": (0.9, 0.999)},
    }
    _prepare_run_dir(run, manifest, resume=False)
    assert json.loads((run / "run_manifest.json").read_text())["binding"][
        "betas"
    ] == [0.9, 0.999]
    _prepare_run_dir(run, manifest, resume=True)
    with pytest.raises(SystemExit, match="manifest mismatch"):
        _prepare_run_dir(
            run,
            {"format_version": 1, "binding": {"whitener_hash": "different"}},
            resume=True,
        )


def test_run_directory_resume_rejects_legacy_directory(tmp_path):
    run = tmp_path / "legacy"
    run.mkdir()
    (run / "latest.pt").write_bytes(b"legacy")
    with pytest.raises(SystemExit, match="legacy/unbound"):
        _prepare_run_dir(run, {"format_version": 1}, resume=True)
