import numpy as np

from block_crosscoder_experiment.analysis.manifold_metrics import (
    cycle_harmonic,
    cycle_permutation_null,
    cyclic_train_eval_metrics,
    fit_cycle_harmonic,
    native_roundness,
    score_cycle_centroids,
    sequence_folds,
    stratified_folds,
)


def sample_cycle(n_classes=12, per_class=100, noise=0.03, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(n_classes), per_class)
    harmonic = cycle_harmonic(n_classes)[labels]
    embed = np.array([[1.2, -0.4, 0.2, 0.7], [0.4, 1.2, -0.7, 0.2]])
    points = harmonic @ embed + noise * rng.standard_normal((len(labels), 4))
    return points, labels


def test_perfect_held_out_cycle_scores_high():
    train_x, train_y = sample_cycle(seed=1)
    eval_x, eval_y = sample_cycle(seed=2)
    metrics = cyclic_train_eval_metrics(train_x, train_y, eval_x, eval_y, 12)
    assert metrics["shape_r2"] > 0.99
    assert metrics["roundness_fit"] > 0.99
    assert metrics["token_cosine"] > 0.99
    assert metrics["chord_corr"] > 0.99


def test_roundness_distinguishes_ellipse_from_order():
    harmonic = cycle_harmonic(12)
    circle = fit_cycle_harmonic(harmonic)
    ellipse = fit_cycle_harmonic(harmonic * np.array([1.0, 0.1]))
    np.testing.assert_allclose(native_roundness(circle), 1.0, atol=1e-12)
    assert native_roundness(ellipse) < 0.21
    # Both remain perfectly harmonically ordered; roundness supplies the
    # information adjacency hits and shape R2 do not.
    assert score_cycle_centroids(harmonic * [1.0, 0.1], ellipse)["shape_r2"] > 0.999


def test_roundness_is_orthogonal_gauge_invariant():
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    base = cycle_harmonic(9) @ np.array([[2.0, 0.0, 0.3, 0.0], [0.0, 0.7, 0.0, 0.2]])
    assert np.isclose(
        native_roundness(fit_cycle_harmonic(base)),
        native_roundness(fit_cycle_harmonic(base @ q)),
    )


def test_cycle_beats_random_phase_permutations():
    train_x, train_y = sample_cycle(per_class=20, seed=31)
    eval_x, eval_y = sample_cycle(per_class=20, seed=32)
    train_mu = np.stack([train_x[train_y == c].mean(0) for c in range(12)])
    eval_mu = np.stack([eval_x[eval_y == c].mean(0) for c in range(12)])
    null = cycle_permutation_null(
        train_mu, eval_mu, n_permutations=999, seed=33, batch_size=111
    )
    assert null["observed_topology_floor"] > 0.99
    assert null["permutation_p_topology"] <= 0.01


def test_noncyclic_order_fails_held_out_shape():
    train_x, train_y = sample_cycle(seed=4)
    eval_x, eval_y = sample_cycle(seed=5)
    shuffled = np.random.default_rng(6).permutation(12)
    metrics = cyclic_train_eval_metrics(
        train_x, train_y, eval_x, shuffled[eval_y], 12
    )
    assert metrics["shape_r2"] < 0.5
    assert metrics["token_cosine"] < 0.5


def test_stratified_folds_cover_every_class():
    labels = np.repeat(np.arange(7), np.arange(3, 10))
    folds = stratified_folds(labels, seed=7)
    for fold in range(3):
        assert set(labels[folds == fold]) == set(range(7))
    assert np.array_equal(folds, stratified_folds(labels, seed=7))


def test_sequence_folds_never_split_a_packed_row():
    labels = np.tile(np.arange(4), 12)
    groups = np.repeat(np.arange(12), 4)
    folds = sequence_folds(groups, labels, seed=9)
    for group in np.unique(groups):
        assert len(set(folds[groups == group])) == 1
    for fold in range(3):
        assert set(labels[folds == fold]) == set(range(4))
