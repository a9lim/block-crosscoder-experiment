"""Held-out, continuous metrics for ordered manifold fidelity.

The descriptive figure pipeline historically used cyclic-adjacency hits.  That
is a useful topology gate, but it saturates quickly (seven weekdays have only
seven possible hits) and cannot distinguish a round native circle from a thin
ellipse.  The routines here are deliberately projection-free at the model
boundary: a fixed semantic harmonic is fitted in the block's native code
space, then scored on a disjoint sample.

All token-level reductions are class-balanced.  This matters for the calendar
probe, where ``May`` is much rarer after the capitalization filter than most
other months.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cycle_harmonic(n_classes: int) -> np.ndarray:
    """Return the fixed semantic cycle ``[cos(2πc/C), sin(2πc/C)]``."""

    phase = 2.0 * np.pi * np.arange(n_classes, dtype=np.float64) / n_classes
    return np.column_stack((np.cos(phase), np.sin(phase)))


def class_means(
    points: np.ndarray, labels: np.ndarray, n_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Class-balanced sufficient statistics with an explicit missing check."""

    x = np.asarray(points, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(y, minlength=n_classes)
    if len(counts) != n_classes or np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"missing classes in manifold split: {missing}")
    means = np.stack([x[y == c].mean(axis=0) for c in range(n_classes)])
    return means, counts


@dataclass(frozen=True)
class HarmonicFit:
    """Affine map from the fixed two-dimensional cycle into native code."""

    intercept: np.ndarray  # [d]
    matrix: np.ndarray  # [2, d]

    def predict(self, n_classes: int) -> np.ndarray:
        return self.intercept + cycle_harmonic(n_classes) @ self.matrix


def fit_cycle_harmonic(centroids: np.ndarray) -> HarmonicFit:
    """Fit ``mu_c = a + h_c A`` by balanced least squares."""

    mu = np.asarray(centroids, dtype=np.float64)
    design = np.column_stack((np.ones(len(mu)), cycle_harmonic(len(mu))))
    coef, *_ = np.linalg.lstsq(design, mu, rcond=None)
    return HarmonicFit(intercept=coef[0], matrix=coef[1:])


def native_roundness(fit: HarmonicFit) -> float:
    """Gauge-invariant circularity from the two singular values of ``A``.

    ``1`` is an isotropically embedded circle and ``0`` is rank one.  Right
    multiplication by any legitimate within-block orthogonal gauge leaves the
    singular values, and therefore this score, unchanged.
    """

    singular = np.linalg.svd(fit.matrix, compute_uv=False)
    if len(singular) < 2:
        return 0.0
    s1, s2 = map(float, singular[:2])
    denom = s1 * s1 + s2 * s2
    return 0.0 if denom <= 0.0 else 2.0 * s1 * s2 / denom


def _off_diagonal_distances(points: np.ndarray) -> np.ndarray:
    delta = points[:, None, :] - points[None, :, :]
    distance = np.linalg.norm(delta, axis=-1)
    return distance[np.triu_indices(len(points), 1)]


def chord_distance_correlation(centroids: np.ndarray) -> float:
    """Correlation of native centroid distances with ideal cyclic chords."""

    actual = _off_diagonal_distances(np.asarray(centroids, dtype=np.float64))
    ideal = _off_diagonal_distances(cycle_harmonic(len(centroids)))
    if actual.std() <= 1e-12 or ideal.std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(actual, ideal)[0, 1])


def score_cycle_centroids(
    centroids: np.ndarray, fit: HarmonicFit
) -> dict[str, float]:
    """Score held-out centroids against a harmonic map frozen elsewhere."""

    mu = np.asarray(centroids, dtype=np.float64)
    pred = fit.predict(len(mu))
    residual = float(np.square(mu - pred).sum())
    total = float(np.square(mu - mu.mean(axis=0, keepdims=True)).sum())
    r2 = 1.0 - residual / max(total, 1e-30)
    eval_fit = fit_cycle_harmonic(mu)
    return {
        "shape_r2": r2,
        "roundness_fit": native_roundness(fit),
        "roundness_eval": native_roundness(eval_fit),
        "chord_corr": chord_distance_correlation(mu),
    }


def cycle_permutation_null(
    train_centroids: np.ndarray,
    eval_centroids: np.ndarray,
    *,
    n_permutations: int = 20_000,
    seed: int = 0,
    batch_size: int = 1_000,
) -> dict[str, float | int]:
    """Permutation p-value for the held-out cyclic topology floor.

    The null randomly assigns the declared phases to semantic classes, fits
    on the training centroids, and scores on the evaluation centroids. Gram
    identities keep the calculation in class space, so the standing 20k null
    remains cheap even for 2560-dimensional residual-stream activations.
    """

    train = np.asarray(train_centroids, dtype=np.float64)
    evaluate = np.asarray(eval_centroids, dtype=np.float64)
    if train.shape != evaluate.shape or train.ndim != 2:
        raise ValueError("train/eval centroids must have the same [class, dim] shape")
    n_classes = len(train)
    fit = fit_cycle_harmonic(train)
    observed_metrics = score_cycle_centroids(evaluate, fit)
    observed = min(
        np.clip(observed_metrics["shape_r2"], 0.0, 1.0),
        observed_metrics["roundness_fit"],
        np.clip(observed_metrics["chord_corr"], 0.0, 1.0),
    )

    # For any phase permutation h, h.T h = C/2 I and sum(h)=0.  Express
    # residual energy and the two singular values of the fitted harmonic via
    # CxC Gram matrices instead of materializing predicted [C,d] arrays.
    centered_eval = evaluate - train.mean(axis=0, keepdims=True)
    k_train = train @ train.T
    k_cross = train @ centered_eval.T
    eval_total = float(np.square(evaluate - evaluate.mean(axis=0)).sum())
    eval_about_train = float(np.square(centered_eval).sum())
    upper = np.triu_indices(n_classes, 1)
    actual_chords = _off_diagonal_distances(evaluate)
    actual_chords = actual_chords - actual_chords.mean()
    actual_norm = float(np.linalg.norm(actual_chords))

    base_harmonic = cycle_harmonic(n_classes)
    rng = np.random.default_rng(seed)
    exceed = 0
    done = 0
    while done < n_permutations:
        size = min(batch_size, n_permutations - done)
        phases = np.stack([
            base_harmonic[rng.permutation(n_classes)] for _ in range(size)
        ])
        q_train = np.einsum("nci,cd,ndj->nij", phases, k_train, phases)
        q_cross = np.einsum("nci,cd,ndj->nij", phases, k_cross, phases)
        residual = (
            eval_about_train
            - (4.0 / n_classes) * np.trace(q_cross, axis1=1, axis2=2)
            + (2.0 / n_classes) * np.trace(q_train, axis1=1, axis2=2)
        )
        shape = 1.0 - residual / max(eval_total, 1e-30)

        # A A.T = 4/C^2 h.T K_train h. Its eigenvalues are squared singular
        # values, and the closed form below is the roundness definition.
        gram2 = (4.0 / n_classes**2) * q_train
        trace = np.trace(gram2, axis1=1, axis2=2)
        determinant = gram2[:, 0, 0] * gram2[:, 1, 1] - gram2[:, 0, 1] ** 2
        roundness = 2.0 * np.sqrt(np.maximum(determinant, 0.0)) / np.maximum(
            trace, 1e-30
        )

        ideal = np.linalg.norm(
            phases[:, upper[0]] - phases[:, upper[1]], axis=2
        )
        ideal -= ideal.mean(axis=1, keepdims=True)
        chord = (ideal @ actual_chords) / np.maximum(
            np.linalg.norm(ideal, axis=1) * actual_norm, 1e-30
        )
        topology = np.minimum.reduce((
            np.clip(shape, 0.0, 1.0),
            roundness,
            np.clip(chord, 0.0, 1.0),
        ))
        exceed += int(np.count_nonzero(topology >= observed - 1e-12))
        done += size

    return {
        "observed_topology_floor": float(observed),
        "permutation_p_topology": float((exceed + 1) / (n_permutations + 1)),
        "n_permutations": int(n_permutations),
        "exceedances": int(exceed),
    }


@dataclass(frozen=True)
class CircularDecoder:
    """Class-balanced affine decoder from native code to semantic phase."""

    intercept: np.ndarray  # [2]
    matrix: np.ndarray  # [d, 2]

    def predict(self, points: np.ndarray) -> np.ndarray:
        return self.intercept + np.asarray(points, dtype=np.float64) @ self.matrix


def fit_circular_decoder(
    points: np.ndarray, labels: np.ndarray, n_classes: int, *, ridge: float = 1e-6
) -> CircularDecoder:
    """Fit a balanced ridge decoder ``code -> (cos phase, sin phase)``."""

    x = np.asarray(points, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    target = cycle_harmonic(n_classes)[y]
    counts = np.bincount(y, minlength=n_classes)
    if np.any(counts == 0):
        raise ValueError("circular decoder fit split is missing a class")
    # sqrt weights make every class contribute unit total squared weight.
    weight = 1.0 / np.sqrt(counts[y])
    design = np.column_stack((np.ones(len(x)), x))
    xw = design * weight[:, None]
    yw = target * weight[:, None]
    gram = xw.T @ xw
    penalty = np.eye(gram.shape[0]) * ridge
    penalty[0, 0] = 0.0  # do not penalize the intercept
    coef = np.linalg.solve(gram + penalty, xw.T @ yw)
    return CircularDecoder(intercept=coef[0], matrix=coef[1:])


def score_circular_decoder(
    points: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    decoder: CircularDecoder,
) -> dict[str, float]:
    """Held-out balanced cosine and angular-error summaries."""

    x = np.asarray(points, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    pred = decoder.predict(x)
    pred_norm = np.linalg.norm(pred, axis=1)
    unit = pred / np.maximum(pred_norm[:, None], 1e-12)
    target = cycle_harmonic(n_classes)[y]
    cosine = np.einsum("nd,nd->n", unit, target)
    cosine[pred_norm <= 1e-12] = 0.0
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    per_class_cos = np.array([cosine[y == c].mean() for c in range(n_classes)])
    per_class_angle = np.array([np.median(angle[y == c]) for c in range(n_classes)])
    return {
        "token_cosine": float(per_class_cos.mean()),
        "token_cosine_min_class": float(per_class_cos.min()),
        "angular_error_deg_median": float(np.median(per_class_angle)),
        "angular_error_deg_p90_class": float(np.quantile(per_class_angle, 0.9)),
    }


def cyclic_train_eval_metrics(
    train_points: np.ndarray,
    train_labels: np.ndarray,
    eval_points: np.ndarray,
    eval_labels: np.ndarray,
    n_classes: int,
) -> dict[str, float]:
    """Fit on one split and return the continuous held-out cyclic battery."""

    train_mu, _ = class_means(train_points, train_labels, n_classes)
    eval_mu, _ = class_means(eval_points, eval_labels, n_classes)
    harmonic = fit_cycle_harmonic(train_mu)
    decoder = fit_circular_decoder(train_points, train_labels, n_classes)
    return {
        **score_cycle_centroids(eval_mu, harmonic),
        **score_circular_decoder(eval_points, eval_labels, n_classes, decoder),
    }


def stratified_folds(
    labels: np.ndarray, *, n_folds: int = 3, seed: int = 0
) -> np.ndarray:
    """Deterministic class-stratified folds for legacy artifacts.

    New captures carry sequence ids and should be split at sequence level.
    This fallback exists so the substantial Phase-0 checkpoint archive can be
    re-scored now; callers must label the resulting evidence sample-split.
    """

    y = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    fold = np.full(len(y), -1, dtype=np.int8)
    for c in np.unique(y):
        index = np.flatnonzero(y == c)
        rng.shuffle(index)
        for f, chunk in enumerate(np.array_split(index, n_folds)):
            fold[chunk] = f
    if np.any(fold < 0):
        raise AssertionError("fold assignment left samples unassigned")
    return fold


def sequence_folds(
    sequence_ids: np.ndarray,
    labels: np.ndarray,
    *,
    n_folds: int = 3,
    seed: int = 0,
) -> np.ndarray:
    """Assign whole packed sequences to deterministic folds.

    Raises instead of silently changing the split when a fold loses a class;
    the remedy is a larger capture or an explicitly recorded seed change.
    """

    groups = np.asarray(sequence_ids)
    y = np.asarray(labels, dtype=np.int64)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    assignment = {int(group): i % n_folds for i, group in enumerate(unique)}
    folds = np.array([assignment[int(group)] for group in groups], dtype=np.int8)
    expected = set(np.unique(y))
    for fold in range(n_folds):
        if set(y[folds == fold]) != expected:
            raise ValueError(
                "sequence split leaves a class absent; increase the capture "
                "or choose a different split seed"
            )
    return folds


__all__ = [
    "CircularDecoder",
    "HarmonicFit",
    "chord_distance_correlation",
    "class_means",
    "cycle_harmonic",
    "cyclic_train_eval_metrics",
    "cycle_permutation_null",
    "fit_circular_decoder",
    "fit_cycle_harmonic",
    "native_roundness",
    "score_circular_decoder",
    "score_cycle_centroids",
    "sequence_folds",
    "stratified_folds",
]
