import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from block_crosscoder_experiment.cli.sweep_manifolds import (
    _cell_fingerprint,
    _diagnostic_health,
    _verify_or_initialize_fingerprint,
)


def _write_log(tmp_path, rows):
    run = tmp_path / "run"
    run.mkdir()
    (run / "steps.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def test_diagnostic_health_accepts_finite_improving_run(tmp_path):
    _write_log(tmp_path, [
        {
            "step": 0, "rec": 1.0, "total": 1.0, "grad_norm": 0.1,
            "gram_residual_postcast": 1e-4, "floor_hits": 0,
            "dead_frac_window": 0.0,
        },
        {
            "step": 499, "rec": 0.7, "total": 0.7, "grad_norm": 0.2,
            "gram_residual_postcast": 2e-4, "floor_hits": 0,
            "dead_frac_window": 0.1,
        },
    ])
    healthy, detail = _diagnostic_health(tmp_path)
    assert healthy
    assert detail["reason"] == "healthy"


def test_diagnostic_health_rejects_excess_guard_skips(tmp_path):
    _write_log(tmp_path, [
        {
            "step": 0, "rec": 1.0, "total": 1.0, "grad_norm": 0.1,
            "gram_residual_postcast": 1e-4, "floor_hits": 0,
            "dead_frac_window": 0.0,
        },
        {"guard_event": {"skipped": True}},
        {
            "step": 499, "rec": 0.7, "total": 0.7, "grad_norm": 0.2,
            "gram_residual_postcast": 2e-4, "floor_hits": 0,
            "dead_frac_window": 0.1,
        },
    ])
    healthy, detail = _diagnostic_health(tmp_path)
    assert not healthy
    assert detail["skip_rate"] > 0.001


def test_cell_fingerprint_changes_with_horizon_and_store():
    args = SimpleNamespace(
        lam=1e-3, epochs=2, train_split="train", warmup_steps=1000,
        store=Path("/data/store"),
    )
    cell = {"G": 4096, "b": 4, "k": 32, "lr": 3e-4}
    screen = _cell_fingerprint(args, "center", cell, 0)
    args.epochs = 4
    finalist = _cell_fingerprint(args, "center", cell, 0)
    assert screen != finalist
    assert screen["epochs"] == 2
    assert finalist["epochs"] == 4


def test_fingerprint_refuses_legacy_artifacts(tmp_path):
    cell_root = tmp_path / "cell" / "seed0"
    run_root = cell_root / "run"
    run_root.mkdir(parents=True)
    (run_root / "report.json").write_text("{}\n")

    with pytest.raises(SystemExit, match="artifact-bearing cell"):
        _verify_or_initialize_fingerprint(
            cell_root / "config.json", {"epochs": 4}, dry_run=False
        )
    assert not (cell_root / "config.json").exists()


def test_fingerprint_initializes_empty_cell_and_verifies_exact_match(tmp_path):
    path = tmp_path / "cell" / "seed0" / "config.json"
    expected = {"epochs": 2, "G": 4096}
    _verify_or_initialize_fingerprint(path, expected, dry_run=False)
    _verify_or_initialize_fingerprint(path, expected, dry_run=False)
    assert json.loads(path.read_text()) == expected

    with pytest.raises(SystemExit, match="configuration mismatch"):
        _verify_or_initialize_fingerprint(
            path, {"epochs": 4, "G": 4096}, dry_run=False
        )
