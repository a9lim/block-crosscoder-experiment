import numpy as np

from block_crosscoder_experiment.analysis.render_sweep_frames import (
    _config_summary,
    _heldout_class_means,
    _upsert_figure,
)


def test_heldout_class_means_use_only_split_b():
    acts = np.arange(8 * 2 * 3, dtype=np.float32).reshape(8, 2, 3)
    rows = np.arange(8)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    folds = np.array([0, 1, 1, 2, 0, 1, 1, 2], dtype=np.int8)

    means, counts = _heldout_class_means(acts, rows, labels, folds, 2)

    np.testing.assert_array_equal(counts, [2, 2])
    np.testing.assert_allclose(means[:, 0], acts[[1, 2]].mean(0))
    np.testing.assert_allclose(means[:, 1], acts[[5, 6]].mean(0))


def test_config_summary_names_the_comparison_axes():
    summary = _config_summary({
        "G": 4096,
        "b": 4,
        "k": 32.0,
        "lr": 3e-4,
        "lambda_rank": 1e-3,
        "site_renorm": True,
        "optimizer_tokens": 11_993_088,
        "seed": 0,
    })

    assert summary == (
        "G4096 b4 k32; lr 0.0003; λ 0.001; renorm; 12M; seed 0"
    )


def test_upsert_figure_replaces_only_matching_run_family():
    manifest = {"figures": [
        {"run": "default", "family": "month", "file": "old"},
        {"run": "default", "family": "weekday", "file": "weekday"},
    ]}

    _upsert_figure(
        manifest,
        {"run": "default", "family": "month", "file": "new"},
    )

    assert manifest["figures"] == [
        {"run": "default", "family": "weekday", "file": "weekday"},
        {"run": "default", "family": "month", "file": "new"},
    ]
