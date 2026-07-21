"""Deterministic, declarative Phase-1 synthetic protocols.

This module makes the paper bridges and the block-crosscoder synthesis ladder
content-addressable data protocols.  Every stream is a pure function of
its configuration, split, presentation index, and stored ground-truth tensors.
Repeated presentations of one unique example are therefore exactly equal, and
batch boundaries cannot change the generated examples.

Two protocol families are implemented:

* FelSyntheticConfig: the single-site additive manifold mixture from
  arXiv:2606.25234.
* LadderSyntheticConfig: the project's novel one-DGP-delta-at-a-time
  shared-support/shared-coordinate block-crosscoder ladder, with orthogonal
  truth-known site-map-rank, site-span, frequency, coactivation, coordinate-
  amplitude, and inter-factor-subspace-overlap controls.

Generated contributions are stored sparsely by event.  For event e,
event_example[e] is a row in the batch, event_factor[e] identifies the planted
factor, coordinates[e] contains its per-site intrinsic coordinate, and
contributions[e] is its clean contribution at every padded site.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Literal

import numpy as np
import torch

__all__ = [
    "FactorMetadata",
    "FelDataset",
    "FelSyntheticConfig",
    "LADDER_STEPS",
    "LadderDataset",
    "LadderSyntheticConfig",
    "Phase1Batch",
    "Phase1Dataset",
    "make_fel_dataset",
    "make_ladder_dataset",
]

Split = Literal["train", "eval"]

_UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
_SM64_A = np.uint64(0x9E3779B97F4A7C15)
_SM64_B = np.uint64(0xBF58476D1CE4E5B9)
_SM64_C = np.uint64(0x94D049BB133111EB)
_ID_MIX = np.uint64(0xD2B74407B1CE6E93)
_COLUMN_MIX = np.uint64(0xCA5A826395121157)
_STREAM_MIX = np.uint64(0xA24BAED4963EE407)


def _validate_split(split: str) -> Split:
    if split not in {"train", "eval"}:
        raise ValueError("split must be 'train' or 'eval'")
    return split  # type: ignore[return-value]


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Vectorized SplitMix64 finalizer with deliberate uint64 wraparound."""
    with np.errstate(over="ignore"):
        z = (values + _SM64_A) & _UINT64_MASK
        z = ((z ^ (z >> np.uint64(30))) * _SM64_B) & _UINT64_MASK
        z = ((z ^ (z >> np.uint64(27))) * _SM64_C) & _UINT64_MASK
        return z ^ (z >> np.uint64(31))


def _affine_permutation_parameters(
    size: int, *, seed: int, epoch: int
) -> tuple[int, int]:
    """Return a seeded O(1)-memory bijection of ``range(size)``.

    A full ``randperm`` is not viable for the large declared synthetic pools.
    The SplitMix-derived affine map is a genuine epoch permutation whenever
    its multiplier is coprime to ``size``; the explicit method is therefore
    reproducible, batch-boundary independent, and auditable as the project's
    adapted presentation-order choice.
    """
    if size <= 0 or epoch < 0:
        raise ValueError("permutation size must be positive and epoch nonnegative")
    if size == 1:
        return 0, 0
    with np.errstate(over="ignore"):
        counter = np.array(
            [
                (np.uint64(seed) * _SM64_A + np.uint64(epoch) * _STREAM_MIX)
                & _UINT64_MASK,
                (np.uint64(seed) * _COLUMN_MIX + np.uint64(epoch + 1) * _ID_MIX)
                & _UINT64_MASK,
            ],
            dtype=np.uint64,
        )
    mixed = _splitmix64(counter)
    multiplier = int(mixed[0] % np.uint64(size))
    if multiplier == 0:
        multiplier = 1
    while math.gcd(multiplier, size) != 1:
        multiplier = (multiplier + 1) % size
        if multiplier == 0:
            multiplier = 1
    offset = int(mixed[1] % np.uint64(size))
    return multiplier, offset


def _uniform(
    example_ids: torch.Tensor,
    columns: int,
    *,
    seed: int,
    stream: int,
) -> torch.Tensor:
    """Stateless U(0,1) values indexed by example, column, seed, and stream."""
    if columns < 0:
        raise ValueError("columns must be nonnegative")
    if columns == 0:
        return torch.empty((len(example_ids), 0), dtype=torch.float32)
    ids = example_ids.detach().cpu().numpy().astype(np.uint64, copy=False)[:, None]
    cols: np.ndarray = np.arange(columns, dtype=np.uint64)[None, :]
    with np.errstate(over="ignore"):
        counters = (
            ids * _ID_MIX
            + cols * _COLUMN_MIX
            + np.uint64(seed) * _SM64_A
            + np.uint64(stream) * _STREAM_MIX
        ) & _UINT64_MASK
    bits = _splitmix64(counters)
    # The half-unit offset keeps Box-Muller away from exactly zero.
    values = ((bits >> np.uint64(11)).astype(np.float64) + 0.5) / float(1 << 53)
    return torch.from_numpy(values.astype(np.float32, copy=False))


def _normal(
    example_ids: torch.Tensor,
    columns: int,
    *,
    seed: int,
    stream: int,
) -> torch.Tensor:
    """Stateless standard normals, using fixed-counter Box-Muller draws."""
    u1 = _uniform(example_ids, columns, seed=seed, stream=2 * stream)
    u2 = _uniform(example_ids, columns, seed=seed, stream=2 * stream + 1)
    radius = torch.sqrt(-2.0 * torch.log(u1.clamp_min(torch.finfo(u1.dtype).tiny)))
    return radius * torch.cos(2.0 * math.pi * u2)


def _standardized_student_t_df3(
    example_ids: torch.Tensor,
    columns: int,
    *,
    seed: int,
    stream: int,
) -> torch.Tensor:
    """Stateless multivariate Student-t draws with unit marginal variance.

    One chi-squared denominator is shared across the coordinates of an event,
    so this is an elliptical multivariate t rather than a product of unrelated
    univariate scales.  If ``g ~ N(0, I)`` and ``v ~ chi2(3)``, then
    ``g / sqrt(v)`` is ``t_3 / sqrt(3)`` coordinatewise and therefore has
    finite unit marginal variance.  The denominator uses a disjoint counter
    stream while preserving the Gaussian numerator draw used by the control.
    """
    if columns < 0:
        raise ValueError("columns must be nonnegative")
    if columns == 0:
        return torch.empty((len(example_ids), 0), dtype=torch.float32)
    numerator = _normal(
        example_ids,
        columns,
        seed=seed,
        stream=stream,
    )
    denominator_normals = _normal(
        example_ids,
        3,
        seed=seed,
        stream=5_000_000 + stream,
    )
    denominator = denominator_normals.square().sum(dim=1, keepdim=True).sqrt()
    return numerator / denominator.clamp_min(torch.finfo(numerator.dtype).tiny)


def _factor_coordinates(
    example_ids: torch.Tensor,
    columns: int,
    *,
    law: str,
    seed: int,
    stream: int,
) -> torch.Tensor:
    """Draw standardized coordinates without perturbing the Gaussian path."""
    if law == "gaussian":
        return _normal(example_ids, columns, seed=seed, stream=stream)
    if law == "student_t_df3":
        return _standardized_student_t_df3(
            example_ids,
            columns,
            seed=seed,
            stream=stream,
        )
    raise ValueError(f"unknown factor-coordinate amplitude law: {law}")


def _elementary_symmetric(weights: torch.Tensor, order: int) -> torch.Tensor:
    """Elementary symmetric polynomial coefficients through ``order``.

    The conditional-Bernoulli support law used below assigns a size-k subset
    ``S`` probability proportional to ``prod(weights[S])``.  Its normalizer
    and exact marginal inclusion probabilities are elementary symmetric
    polynomials, so no Monte Carlo estimate is needed in the truth record.
    """
    values = weights.detach().cpu().to(torch.float64).flatten()
    if values.numel() == 0:
        if order == 0:
            return torch.ones(1, dtype=torch.float64)
        raise ValueError("cannot form a positive-order polynomial without weights")
    if order < 0 or order > values.numel():
        raise ValueError("order must be in [0, number of weights]")
    if not torch.isfinite(values).all() or bool((values <= 0).any()):
        raise ValueError("conditional-Bernoulli weights must be finite and positive")
    coefficients = torch.zeros(order + 1, dtype=torch.float64)
    coefficients[0] = 1.0
    degree = 0
    for weight in values:
        degree = min(degree + 1, order)
        for index in range(degree, 0, -1):
            coefficients[index] += weight * coefficients[index - 1]
    return coefficients


def _conditional_bernoulli_marginals(weights: torch.Tensor, count: int) -> torch.Tensor:
    """Exact inclusion probabilities for a fixed-cardinality weighted law."""
    values = weights.detach().cpu().to(torch.float64).flatten()
    if not 0 <= count <= values.numel():
        raise ValueError("count must be in [0, number of weights]")
    if count == 0:
        return torch.zeros_like(values)
    if count == values.numel():
        return torch.ones_like(values)
    normalizer = _elementary_symmetric(values, count)[count]
    marginals = torch.empty_like(values)
    for index, weight in enumerate(values):
        without = torch.cat((values[:index], values[index + 1 :]))
        complement = _elementary_symmetric(without, count - 1)[count - 1]
        marginals[index] = weight * complement / normalizer
    return marginals


def _conditional_bernoulli_sample(
    example_ids: torch.Tensor,
    weights: torch.Tensor,
    count: int,
    *,
    seed: int,
    stream: int,
) -> torch.Tensor:
    """Stateless exact-k samples with mass proportional to weight products.

    A suffix dynamic program gives each next inclusion probability conditional
    on the choices already made.  The draw is indexed only by example id,
    factor position, seed, and stream, so slicing or rebatching the stream
    cannot change a support.
    """
    values = weights.detach().cpu().to(torch.float64).flatten()
    n_factors = int(values.numel())
    if not 0 <= count <= n_factors:
        raise ValueError("count must be in [0, number of weights]")
    if not torch.isfinite(values).all() or bool((values <= 0).any()):
        raise ValueError("conditional-Bernoulli weights must be finite and positive")
    selected = torch.zeros(len(example_ids), n_factors, dtype=torch.bool)
    if count == 0:
        return selected
    if count == n_factors:
        selected.fill_(True)
        return selected

    # suffix[i, r] = e_r(weights[i:]).
    suffix = torch.zeros(n_factors + 1, count + 1, dtype=torch.float64)
    suffix[:, 0] = 1.0
    for index in range(n_factors - 1, -1, -1):
        suffix[index] = suffix[index + 1]
        upper = min(count, n_factors - index)
        suffix[index, 1 : upper + 1] += values[index] * suffix[index + 1, :upper]

    uniforms = _uniform(
        example_ids,
        n_factors,
        seed=seed,
        stream=stream,
    ).to(torch.float64)
    remaining = torch.full((len(example_ids),), count, dtype=torch.long)
    for index in range(n_factors):
        eligible = remaining > 0
        if not bool(eligible.any()):
            break
        candidates_left = n_factors - index
        forced = eligible & (remaining == candidates_left)
        chosen = forced.clone()
        stochastic = eligible & ~forced
        if bool(stochastic.any()):
            r = remaining[stochastic]
            numerator = values[index] * suffix[index + 1, r - 1]
            denominator = suffix[index, r]
            probability = numerator / denominator
            chosen[stochastic] = uniforms[stochastic, index] < probability
        selected[chosen, index] = True
        remaining[chosen] -= 1
    if bool((remaining != 0).any()):
        raise RuntimeError("conditional-Bernoulli support accounting mismatch")
    return selected


def _categorical(
    example_ids: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    seed: int,
    stream: int,
) -> torch.Tensor:
    """Stateless categorical draw from a finite, explicitly normalized law."""
    values = probabilities.detach().cpu().to(torch.float64).flatten()
    if values.numel() == 0:
        raise ValueError("categorical distribution must be nonempty")
    if not torch.isfinite(values).all() or bool((values < 0).any()):
        raise ValueError("categorical probabilities must be finite and nonnegative")
    total = values.sum()
    if total <= 0:
        raise ValueError("categorical probabilities must have positive mass")
    cumulative = torch.cumsum(values / total, dim=0)
    cumulative[-1] = 1.0
    draws = _uniform(example_ids, 1, seed=seed, stream=stream)[:, 0].to(torch.float64)
    return torch.searchsorted(cumulative, draws, right=False)


def _orthonormal(rows: int, columns: int, *, seed: int) -> torch.Tensor:
    if not 1 <= columns <= rows:
        raise ValueError("orthonormal matrix requires 1 <= columns <= rows")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(rows, columns, generator=gen, dtype=torch.float64)
    q, r = torch.linalg.qr(raw, mode="reduced")
    signs = torch.sign(torch.diagonal(r))
    signs[signs == 0] = 1
    return (q * signs).to(torch.float32)


def _unit_vector(dimension: int, *, seed: int) -> torch.Tensor:
    return _orthonormal(dimension, 1, seed=seed)[:, 0]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _state_digest(protocol: dict[str, Any], tensors: dict[str, torch.Tensor]) -> str:
    """Digest the declaration and all tensors that determine the stream."""
    h = hashlib.sha256()
    h.update(b"block-crosscoder-phase1-stream-v1\0")
    h.update(_canonical_json(protocol))
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(str(tensor.dtype).encode("ascii"))
        h.update(b"\0")
        h.update(_canonical_json(list(tensor.shape)))
        h.update(b"\0")
        h.update(tensor.numpy().tobytes(order="C"))
        h.update(b"\0")
    return h.hexdigest()


def _add_events_by_factor_(
    target: torch.Tensor,
    event_example: torch.Tensor,
    event_factor: torch.Tensor,
    contributions: torch.Tensor,
    *,
    n_factors: int,
) -> None:
    """Accumulate in fixed factor order, independent of batch partitioning.

    Each factor contributes at most once to an example.  Advanced indexed
    assignment is therefore unambiguous, and the per-row floating-point
    addition order depends only on factor id rather than an index_add kernel's
    batching strategy.
    """
    for factor in range(n_factors):
        event_rows = torch.nonzero(event_factor == factor, as_tuple=False).flatten()
        if len(event_rows) == 0:
            continue
        examples = event_example[event_rows]
        target[examples] = target[examples] + contributions[event_rows]


@dataclass(frozen=True)
class FactorMetadata:
    index: int
    name: str
    family: str
    category: str
    coordinate_dim: int
    intrinsic_dim: int
    frequency: float
    active_sites: tuple[int, ...]

    def to_protocol(self) -> dict[str, Any]:
        value = asdict(self)
        value["active_sites"] = list(self.active_sites)
        return value


@dataclass(frozen=True)
class Phase1Batch:
    """One presentation batch and its sparse planted-contribution events."""

    x: torch.Tensor
    clean_x: torch.Tensor
    active: torch.Tensor
    observed: torch.Tensor
    example_ids: torch.Tensor
    presentation_ids: torch.Tensor
    event_example: torch.Tensor
    event_factor: torch.Tensor
    coordinates: torch.Tensor
    contributions: torch.Tensor

    @property
    def n_events(self) -> int:
        return int(self.event_factor.numel())


class Phase1Dataset:
    """Common indexed-stream interface for all Phase-1 DGPs."""

    protocol: dict[str, Any]
    factors: tuple[FactorMetadata, ...]
    contribution_maps: torch.Tensor
    valid_dimensions: torch.Tensor
    factor_categories: tuple[str, ...]
    category_names: tuple[str, ...]
    category_index: torch.Tensor

    def __init__(
        self,
        *,
        split: Split,
        unique_examples: int,
        presentations: int,
        split_seed: int,
        presentation_order: str,
    ) -> None:
        if unique_examples <= 0:
            raise ValueError("unique_examples must be positive")
        if presentations <= 0:
            raise ValueError("presentations must be positive")
        self.split = split
        self.unique_examples = int(unique_examples)
        self.presentations = int(presentations)
        self.split_seed = int(split_seed)
        if presentation_order not in {
            "deterministic_epoch_shuffle",
            "cyclic_unshuffled",
        }:
            raise ValueError("unsupported presentation_order")
        self.presentation_order = presentation_order

    def __len__(self) -> int:
        return self.presentations

    @property
    def stream_digest(self) -> str:
        return str(self.protocol["stream_digest"])

    @property
    def ground_truth(self) -> dict[str, Any]:
        """Contribution/category metadata consumed by recovery gates."""
        truth = {
            "factors": self.factors,
            "factor_categories": self.factor_categories,
            "category_names": self.category_names,
            "category_index": self.category_index,
            "contribution_maps": self.contribution_maps,
            "valid_dimensions": self.valid_dimensions,
        }
        truth.update(getattr(self, "_ground_truth_extra", {}))
        return truth

    def protocol_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.protocol)

    def sample(self, count: int, *, start: int = 0) -> Phase1Batch:
        if count <= 0:
            raise ValueError("count must be positive")
        if start < 0 or start + count > self.presentations:
            raise ValueError("requested presentations are outside the stream")
        presentation_ids = torch.arange(start, start + count, dtype=torch.long)
        positions = presentation_ids.remainder(self.unique_examples)
        if self.presentation_order == "cyclic_unshuffled":
            example_ids = positions
        else:
            example_ids = torch.empty_like(positions)
            epochs = torch.div(
                presentation_ids, self.unique_examples, rounding_mode="floor"
            )
            for epoch_tensor in torch.unique_consecutive(epochs):
                epoch = int(epoch_tensor)
                selected = epochs == epoch
                multiplier, offset = _affine_permutation_parameters(
                    self.unique_examples,
                    seed=self.split_seed,
                    epoch=epoch,
                )
                example_ids[selected] = (
                    multiplier * positions[selected] + offset
                ).remainder(self.unique_examples)
        return self._generate(example_ids, presentation_ids)

    def batches(
        self,
        batch_size: int,
        *,
        start: int = 0,
        stop: int | None = None,
        drop_last: bool = False,
    ) -> Iterator[Phase1Batch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        end = self.presentations if stop is None else stop
        if not 0 <= start <= end <= self.presentations:
            raise ValueError("invalid presentation interval")
        cursor = start
        while cursor < end:
            count = min(batch_size, end - cursor)
            if drop_last and count < batch_size:
                return
            yield self.sample(count, start=cursor)
            cursor += count

    def _generate(
        self, example_ids: torch.Tensor, presentation_ids: torch.Tensor
    ) -> Phase1Batch:
        raise NotImplementedError

    def _finish_protocol(
        self,
        protocol: dict[str, Any],
        *,
        digest_tensors: dict[str, torch.Tensor],
    ) -> None:
        protocol["counts"] = {
            "unique_examples": self.unique_examples,
            "presentations": self.presentations,
            "presentation_order": self.presentation_order,
            "shuffle_algorithm": (
                "splitmix64_seeded_affine_bijection_v1"
                if self.presentation_order == "deterministic_epoch_shuffle"
                else "identity"
            ),
            "repeat_semantics": "presentations with the same example_id are exact",
        }
        protocol["split"] = self.split
        protocol["split_seed"] = self.split_seed
        protocol["stream_digest_kind"] = "sha256-canonical-declaration-and-state-v1"
        protocol["stream_digest"] = _state_digest(protocol, digest_tensors)
        self.protocol = protocol

    def _set_categories(self) -> None:
        self.factor_categories = tuple(f.category for f in self.factors)
        self.category_names = tuple(dict.fromkeys(self.factor_categories))
        lookup = {name: i for i, name in enumerate(self.category_names)}
        self.category_index = torch.tensor(
            [lookup[name] for name in self.factor_categories], dtype=torch.long
        )


# ---------------------------------------------------------------------------
# Fel additive manifold mixture


_FEL_MANIFOLD_FAMILIES = (
    "circle",
    "disk",
    "sphere",
    "torus",
    "mobius",
    "swiss_roll",
    "helix",
)
_FEL_COORDINATE_DIMS = {
    "scalar": 1,
    "circle": 2,
    "disk": 2,
    "sphere": 3,
    "torus": 3,
    "mobius": 3,
    "swiss_roll": 3,
    "helix": 3,
}
_FEL_INTRINSIC_DIMS = {
    "scalar": 1,
    "circle": 1,
    "disk": 2,
    "sphere": 2,
    "torus": 2,
    "mobius": 2,
    "swiss_roll": 2,
    "helix": 1,
}


@dataclass(frozen=True)
class FelSyntheticConfig:
    ambient_dim: int = 128
    n_factors: int = 128
    active_per_example: int = 4
    calibration_examples: int = 50_000
    train_unique_examples: int = 300_000
    train_presentations: int = 300_000
    eval_unique_examples: int = 100_000
    eval_presentations: int = 100_000
    structure_seed: int = 26_060_252
    train_seed: int = 1_260_625_234
    eval_seed: int = 2_260_625_234
    presentation_order: str = "deterministic_epoch_shuffle"

    def __post_init__(self) -> None:
        if self.ambient_dim < 3:
            raise ValueError("ambient_dim must be at least three")
        if self.n_factors < 2 or self.n_factors % 2:
            raise ValueError("n_factors must be positive and even")
        if not 1 <= self.active_per_example <= self.n_factors:
            raise ValueError("active_per_example must be in [1, n_factors]")
        if self.calibration_examples < 2:
            raise ValueError("calibration_examples must be at least two")
        if self.train_seed == self.eval_seed:
            raise ValueError("train and eval seeds must differ")


def _fel_raw(
    family: str,
    example_ids: torch.Tensor,
    *,
    seed: int,
    stream: int,
) -> torch.Tensor:
    if family == "scalar":
        return _normal(example_ids, 1, seed=seed, stream=stream)
    if family == "circle":
        angle = 2 * math.pi * _uniform(example_ids, 1, seed=seed, stream=stream)
        return torch.cat((torch.cos(angle), torch.sin(angle)), dim=1)
    if family == "disk":
        uv = _uniform(example_ids, 2, seed=seed, stream=stream)
        radius = torch.sqrt(uv[:, :1])
        angle = 2 * math.pi * uv[:, 1:]
        return torch.cat((radius * torch.cos(angle), radius * torch.sin(angle)), dim=1)
    if family == "sphere":
        xyz = _normal(example_ids, 3, seed=seed, stream=stream)
        return xyz / xyz.norm(dim=1, keepdim=True).clamp_min(1e-12)
    if family == "torus":
        uv = 2 * math.pi * _uniform(example_ids, 2, seed=seed, stream=stream)
        major, minor = 2.0, 0.5
        ring = major + minor * torch.cos(uv[:, 1:])
        return torch.cat(
            (
                ring * torch.cos(uv[:, :1]),
                ring * torch.sin(uv[:, :1]),
                minor * torch.sin(uv[:, 1:]),
            ),
            dim=1,
        )
    if family == "mobius":
        uv = _uniform(example_ids, 2, seed=seed, stream=stream)
        angle = 2 * math.pi * uv[:, :1]
        width = 2 * uv[:, 1:] - 1
        radial = 1 + 0.5 * width * torch.cos(angle / 2)
        return torch.cat(
            (
                radial * torch.cos(angle),
                radial * torch.sin(angle),
                0.5 * width * torch.sin(angle / 2),
            ),
            dim=1,
        )
    if family == "swiss_roll":
        uv = _uniform(example_ids, 2, seed=seed, stream=stream)
        turn = (1.5 + 3.0 * uv[:, :1]) * math.pi
        height = 2 * uv[:, 1:] - 1
        return torch.cat(
            (turn * torch.cos(turn), height, turn * torch.sin(turn)), dim=1
        )
    if family == "helix":
        turn = (4 * _uniform(example_ids, 1, seed=seed, stream=stream) - 2) * math.pi
        return torch.cat(
            (torch.cos(turn), torch.sin(turn), turn / (2 * math.pi)), dim=1
        )
    raise ValueError(f"unknown Fel family: {family}")


class FelDataset(Phase1Dataset):
    def __init__(self, config: FelSyntheticConfig, split: Split) -> None:
        unique = (
            config.train_unique_examples
            if split == "train"
            else config.eval_unique_examples
        )
        presentations = (
            config.train_presentations
            if split == "train"
            else config.eval_presentations
        )
        split_seed = config.train_seed if split == "train" else config.eval_seed
        super().__init__(
            split=split,
            unique_examples=unique,
            presentations=presentations,
            split_seed=split_seed,
            presentation_order=config.presentation_order,
        )
        self.config = config
        self.n_sites = 1
        self.site_dims = (config.ambient_dim,)
        self.padded_dim = config.ambient_dim
        self.max_coordinate_dim = 3
        self.valid_dimensions = torch.ones(1, config.ambient_dim, dtype=torch.bool)

        n_scalar = config.n_factors // 2
        n_manifold = config.n_factors - n_scalar
        manifold_families = [
            _FEL_MANIFOLD_FAMILIES[i % len(_FEL_MANIFOLD_FAMILIES)]
            for i in range(n_manifold)
        ]
        families = ["scalar"] * n_scalar + manifold_families

        means = torch.zeros(config.n_factors, self.max_coordinate_dim)
        rms = torch.zeros(config.n_factors)
        maps = torch.zeros(
            config.n_factors,
            1,
            config.ambient_dim,
            self.max_coordinate_dim,
        )
        calibration_center_residual = torch.zeros(config.n_factors)
        calibration_rms_after = torch.zeros(config.n_factors)
        calibration_ids = torch.arange(config.calibration_examples, dtype=torch.long)
        factors: list[FactorMetadata] = []
        for factor, family in enumerate(families):
            coordinate_dim = _FEL_COORDINATE_DIMS[family]
            raw = _fel_raw(
                family,
                calibration_ids,
                seed=config.structure_seed,
                stream=10_000 + 17 * factor,
            )
            mean = raw.mean(dim=0)
            centered = raw - mean
            factor_rms = torch.sqrt(centered.square().sum(dim=1).mean())
            if not torch.isfinite(factor_rms) or factor_rms <= 0:
                raise RuntimeError(f"degenerate Fel calibration for factor {factor}")
            normalized = centered / factor_rms
            means[factor, :coordinate_dim] = mean
            rms[factor] = factor_rms
            calibration_center_residual[factor] = normalized.mean(dim=0).norm()
            calibration_rms_after[factor] = torch.sqrt(
                normalized.square().sum(dim=1).mean()
            )
            maps[factor, 0, :, :coordinate_dim] = _orthonormal(
                config.ambient_dim,
                coordinate_dim,
                seed=config.structure_seed + 100_003 * (factor + 1),
            )
            factors.append(
                FactorMetadata(
                    index=factor,
                    name=f"fel_{family}_{factor:03d}",
                    family=family,
                    category="scalar" if family == "scalar" else "manifold",
                    coordinate_dim=coordinate_dim,
                    intrinsic_dim=_FEL_INTRINSIC_DIMS[family],
                    frequency=config.active_per_example / config.n_factors,
                    active_sites=(0,),
                )
            )

        self.factors = tuple(factors)
        self.calibration_mean = means
        self.calibration_rms = rms
        self.calibration_center_residual = calibration_center_residual
        self.calibration_rms_after = calibration_rms_after
        self.contribution_maps = maps
        self._set_categories()

        family_counts = {
            family: sum(f.family == family for f in self.factors)
            for family in ("scalar", *_FEL_MANIFOLD_FAMILIES)
        }
        protocol = {
            "schema": "block-crosscoder.phase1.synthetic.v1",
            "name": "fel_additive_manifold_mixture",
            "lineage": {
                "classification": "adapted",
                "paper": "Structuring Sparsity: Block-Sparse Featurizers Capture Visual Concept Manifolds",
                "citation": "arXiv:2606.25234",
                "url": "https://arxiv.org/abs/2606.25234",
                "exact_scope": "factor families, counts, calibration, additive support, and noiseless primary toy",
                "engineering_scope": "stateless generation and deterministic epoch presentation order",
            },
            "rationale": (
                "Paper-faithful known-factor bridge for testing whether a method "
                "recovers additive scalar and manifold factors rather than only FVU."
            ),
            "configuration": asdict(config),
            "sites": {
                "dimensions": [config.ambient_dim],
                "padded_dimension": config.ambient_dim,
            },
            "sampling": {
                "ambient_dimension": config.ambient_dim,
                "factor_count": config.n_factors,
                "family_counts": family_counts,
                "manifold_balance_rule": "round-robin; counts differ by at most one",
                "calibration_examples_per_factor": config.calibration_examples,
                "normalization": "factor centered and total-RMS normalized on calibration split",
                "mixture_scale": "each normalized contribution divided by sqrt(active_per_example)",
                "active_without_replacement": config.active_per_example,
                "noise_std": 0.0,
            },
            "seeds": {
                "structure": config.structure_seed,
                "train": config.train_seed,
                "eval": config.eval_seed,
            },
            "factors": [factor.to_protocol() for factor in self.factors],
        }
        self._finish_protocol(
            protocol,
            digest_tensors={
                "calibration_mean": self.calibration_mean,
                "calibration_rms": self.calibration_rms,
                "contribution_maps": self.contribution_maps,
            },
        )

    def _generate(
        self, example_ids: torch.Tensor, presentation_ids: torch.Tensor
    ) -> Phase1Batch:
        n = len(example_ids)
        n_factors = len(self.factors)
        scores = _uniform(example_ids, n_factors, seed=self.split_seed, stream=1)
        selected = scores.topk(
            self.config.active_per_example, dim=1, largest=True, sorted=False
        ).indices
        active = torch.zeros(n, n_factors, dtype=torch.bool)
        active.scatter_(1, selected, True)
        n_events = n * self.config.active_per_example
        event_example = torch.empty(n_events, dtype=torch.long)
        event_factor = torch.empty(n_events, dtype=torch.long)
        coordinates = torch.zeros(n_events, 1, self.max_coordinate_dim)
        contributions = torch.zeros(n_events, 1, self.padded_dim)

        cursor = 0
        scale = 1 / math.sqrt(self.config.active_per_example)
        for factor, metadata in enumerate(self.factors):
            rows = torch.nonzero(active[:, factor], as_tuple=False).flatten()
            count = len(rows)
            if count == 0:
                continue
            ids = example_ids[rows]
            raw = _fel_raw(
                metadata.family,
                ids,
                seed=self.split_seed,
                stream=10_000 + 17 * factor,
            )
            dim = metadata.coordinate_dim
            coord = (
                (raw - self.calibration_mean[factor, :dim])
                / self.calibration_rms[factor]
                * scale
            )
            sl = slice(cursor, cursor + count)
            event_example[sl] = rows
            event_factor[sl] = factor
            coordinates[sl, 0, :dim] = coord
            # Compute in fp64, then round once.  This removes BLAS
            # microkernel-dependent fp32 differences between batch sizes.
            contributions[sl, 0] = (
                coord.to(torch.float64)
                @ self.contribution_maps[factor, 0, :, :dim].to(torch.float64).T
            ).to(torch.float32)
            cursor += count
        if cursor != n_events:
            raise RuntimeError("Fel event accounting mismatch")

        clean_x = torch.zeros(n, 1, self.padded_dim)
        _add_events_by_factor_(
            clean_x,
            event_example,
            event_factor,
            contributions,
            n_factors=n_factors,
        )
        observed = torch.ones(n, 1, dtype=torch.bool)
        return Phase1Batch(
            x=clean_x.clone(),
            clean_x=clean_x,
            active=active,
            observed=observed,
            example_ids=example_ids.clone(),
            presentation_ids=presentation_ids.clone(),
            event_example=event_example,
            event_factor=event_factor,
            coordinates=coordinates,
            contributions=contributions,
        )


def make_fel_dataset(
    config: FelSyntheticConfig | None = None, *, split: Split = "train"
) -> FelDataset:
    return FelDataset(config or FelSyntheticConfig(), _validate_split(split))


# ---------------------------------------------------------------------------
# Project-native one-delta block-crosscoder ladder


LADDER_STEPS = (
    "baseline",
    "shared_support",
    "site_rotation",
    "site_scale",
    "noise",
    "rank_heterogeneity",
)


@dataclass(frozen=True)
class LadderSyntheticConfig:
    step: str = "baseline"
    n_sites: int = 3
    d_model: int = 32
    n_factors: int = 12
    block_dim: int = 4
    base_rank: int = 2
    active_per_example: int = 3
    scale_ratio: float = 2.0
    noise_std: float = 0.1
    site_map_rank_family: Literal["rank1", "rank2", "independent"] = "rank1"
    site_presence_span: Literal["one", "two", "all"] = "all"
    feature_frequency: Literal["uniform", "zipf_alpha_1"] = "uniform"
    coactivation_probability: float = 0.0
    coordinate_amplitude_law: Literal["gaussian", "student_t_df3"] = "gaussian"
    factor_subspace_overlap: Literal["uncontrolled", "paired_30deg"] = "uncontrolled"
    train_unique_examples: int = 100_000
    train_presentations: int = 100_000
    eval_unique_examples: int = 20_000
    eval_presentations: int = 20_000
    structure_seed: int = 31_415_926
    train_seed: int = 1_314_159_265
    eval_seed: int = 2_314_159_265
    presentation_order: str = "deterministic_epoch_shuffle"

    def __post_init__(self) -> None:
        if self.step not in LADDER_STEPS:
            raise ValueError(f"step must be one of {LADDER_STEPS}")
        if self.n_sites < 2:
            raise ValueError("n_sites must be at least two")
        if self.d_model <= 0 or self.block_dim <= 0:
            raise ValueError("dimensions must be positive")
        if self.block_dim > self.d_model:
            raise ValueError("block_dim must not exceed d_model")
        if not 1 <= self.base_rank <= self.block_dim:
            raise ValueError("base_rank must be in [1, block_dim]")
        if not 1 <= self.active_per_example <= self.n_factors:
            raise ValueError("active_per_example must be in [1, n_factors]")
        if self.scale_ratio <= 0:
            raise ValueError("scale_ratio must be positive")
        if self.noise_std < 0:
            raise ValueError("noise_std must be nonnegative")
        if self.site_map_rank_family not in {"rank1", "rank2", "independent"}:
            raise ValueError(
                "site_map_rank_family must be 'rank1', 'rank2', or 'independent'"
            )
        if self.site_presence_span not in {"one", "two", "all"}:
            raise ValueError("site_presence_span must be 'one', 'two', or 'all'")
        if self.feature_frequency not in {"uniform", "zipf_alpha_1"}:
            raise ValueError("feature_frequency must be 'uniform' or 'zipf_alpha_1'")
        if self.coordinate_amplitude_law not in {"gaussian", "student_t_df3"}:
            raise ValueError(
                "coordinate_amplitude_law must be 'gaussian' or 'student_t_df3'"
            )
        if self.factor_subspace_overlap not in {
            "uncontrolled",
            "paired_30deg",
        }:
            raise ValueError(
                "factor_subspace_overlap must be 'uncontrolled' or 'paired_30deg'"
            )
        if self.factor_subspace_overlap == "paired_30deg":
            if self.n_factors < 2:
                raise ValueError(
                    "paired_30deg factor_subspace_overlap requires at least two factors"
                )
            if self.d_model < 2 * self.block_dim:
                raise ValueError(
                    "paired_30deg factor_subspace_overlap requires "
                    "d_model >= 2 * block_dim"
                )
        if self.coactivation_probability not in {0.0, 0.5, 0.9}:
            raise ValueError("coactivation_probability must be one of 0.0, 0.5, or 0.9")
        if self.coactivation_probability > 0:
            if self.n_factors % 2:
                raise ValueError(
                    "positive coactivation_probability requires an even n_factors"
                )
            if self.active_per_example < 2:
                raise ValueError(
                    "positive coactivation_probability requires active_per_example >= 2"
                )
        if self.train_seed == self.eval_seed:
            raise ValueError("train and eval seeds must differ")


_LADDER_DELTA: dict[str, dict[str, Any] | None] = {
    "baseline": None,
    "shared_support": {
        "field": "coordinate_truth",
        "baseline": "shared_coordinate",
        "value": "shared_support_site_specific_coordinates",
    },
    "site_rotation": {
        "field": "site_rotation",
        "baseline": "identity",
        "value": "independent_orthogonal",
    },
    "site_scale": {
        "field": "site_scale",
        "baseline": "all_one",
        "value": "geometric_scale_imbalance",
    },
    "noise": {
        "field": "noise_std",
        "baseline": 0.0,
        "value": "configured",
    },
    "rank_heterogeneity": {
        "field": "factor_rank",
        "baseline": "uniform_base_rank",
        "value": "cycle_1_to_block_dim",
    },
}


class LadderDataset(Phase1Dataset):
    def __init__(self, config: LadderSyntheticConfig, split: Split) -> None:
        unique = (
            config.train_unique_examples
            if split == "train"
            else config.eval_unique_examples
        )
        presentations = (
            config.train_presentations
            if split == "train"
            else config.eval_presentations
        )
        split_seed = config.train_seed if split == "train" else config.eval_seed
        super().__init__(
            split=split,
            unique_examples=unique,
            presentations=presentations,
            split_seed=split_seed,
            presentation_order=config.presentation_order,
        )
        self.config = config
        self.n_sites = config.n_sites
        self.site_dims = (config.d_model,) * config.n_sites
        self.padded_dim = config.d_model
        self.max_coordinate_dim = config.block_dim
        self.valid_dimensions = torch.ones(
            config.n_sites, config.d_model, dtype=torch.bool
        )

        # Site-axis map families are a truth-known analogue of factorizing a
        # decoder across layers.  ``independent`` removes low-rank structure
        # from the site axis while retaining the defining shared support and
        # shared coordinate; it is therefore a hard factorization stress, not
        # a negative control for cross-layer feature existence.
        overlap_generator = torch.Generator(device="cpu").manual_seed(
            config.structure_seed + 60_000_019
        )
        overlap_order = torch.randperm(config.n_factors, generator=overlap_generator)
        paired_count = config.n_factors - config.n_factors % 2
        factor_overlap_pairs = overlap_order[:paired_count].reshape(-1, 2)
        if config.factor_subspace_overlap == "uncontrolled":
            # Keep this construction exactly as it predates the overlap axis;
            # adding an explicit Gaussian/uncontrolled control must not change
            # any generated tensor.
            map_bases = torch.stack(
                [
                    torch.stack(
                        [
                            _orthonormal(
                                config.d_model,
                                config.block_dim,
                                seed=(
                                    config.structure_seed
                                    + 100_003 * (factor + 1)
                                    + 10_000_019 * basis
                                ),
                            )
                            for basis in range(config.n_sites)
                        ]
                    )
                    for factor in range(config.n_factors)
                ]
            )
        else:
            map_bases = torch.empty(
                config.n_factors,
                config.n_sites,
                config.d_model,
                config.block_dim,
            )
            angle = math.radians(30.0)
            for pair_index, pair in enumerate(factor_overlap_pairs):
                first, second = int(pair[0]), int(pair[1])
                joint_frame = _orthonormal(
                    config.d_model,
                    2 * config.block_dim,
                    seed=config.structure_seed + 90_000_059 * (pair_index + 1),
                )
                first_frame = joint_frame[:, : config.block_dim]
                complement_frame = joint_frame[:, config.block_dim :]
                second_frame = (
                    math.cos(angle) * first_frame + math.sin(angle) * complement_frame
                )
                for basis in range(config.n_sites):
                    coordinate_rotation = (
                        torch.eye(config.block_dim)
                        if basis == 0
                        else _orthonormal(
                            config.block_dim,
                            config.block_dim,
                            seed=(
                                config.structure_seed
                                + 91_000_063 * (pair_index + 1)
                                + 10_000_019 * basis
                            ),
                        )
                    )
                    map_bases[first, basis] = first_frame @ coordinate_rotation
                    map_bases[second, basis] = second_frame @ coordinate_rotation
            if config.n_factors % 2:
                factor = int(overlap_order[-1])
                map_bases[factor] = torch.stack(
                    [
                        _orthonormal(
                            config.d_model,
                            config.block_dim,
                            seed=(
                                config.structure_seed
                                + 100_003 * (factor + 1)
                                + 10_000_019 * basis
                            ),
                        )
                        for basis in range(config.n_sites)
                    ]
                )
        base_frames = map_bases[:, 0]
        realized_pair_principal_angles_degrees = torch.empty(
            len(factor_overlap_pairs), config.block_dim, dtype=torch.float64
        )
        for pair_index, pair in enumerate(factor_overlap_pairs):
            first, second = int(pair[0]), int(pair[1])
            singular_values = torch.linalg.svdvals(
                base_frames[first].to(torch.float64).T
                @ base_frames[second].to(torch.float64)
            ).clamp(0.0, 1.0)
            realized_pair_principal_angles_degrees[pair_index] = (
                torch.rad2deg(torch.acos(singular_values)).sort().values
            )
        site_map_loadings = torch.zeros(config.n_sites, config.n_sites)
        if config.site_map_rank_family == "rank1":
            site_map_loadings[:, 0] = 1.0
        elif config.site_map_rank_family == "rank2":
            site_map_loadings[:, 0] = 1.0
            site_map_loadings[:, 1] = torch.linspace(-1.0, 1.0, config.n_sites)
            site_map_loadings[:, :2] /= site_map_loadings[:, :2].norm(
                dim=1, keepdim=True
            )
        else:
            site_map_loadings.copy_(torch.eye(config.n_sites))

        maps = torch.einsum(
            "sq,fqdr->fsdr",
            site_map_loadings.to(torch.float64),
            map_bases.to(torch.float64),
        ).to(torch.float32)
        rotations = torch.eye(config.d_model).repeat(config.n_sites, 1, 1)
        if config.step == "site_rotation":
            for site in range(1, config.n_sites):
                rotations[site] = _orthonormal(
                    config.d_model,
                    config.d_model,
                    seed=config.structure_seed + 1_000_003 * (site + 1),
                )
        site_scales = torch.ones(config.n_sites)
        if config.step == "site_scale":
            site_scales = torch.logspace(
                0.0,
                math.log10(config.scale_ratio),
                config.n_sites,
            )
        ranks = torch.full((config.n_factors,), config.base_rank, dtype=torch.long)
        if config.step == "rank_heterogeneity":
            ranks = 1 + torch.arange(config.n_factors).remainder(config.block_dim)

        factor_site_mask = torch.zeros(
            config.n_factors, config.n_sites, dtype=torch.bool
        )
        for factor in range(config.n_factors):
            if config.site_presence_span == "all":
                factor_site_mask[factor].fill_(True)
            elif config.site_presence_span == "one":
                factor_site_mask[factor, factor % config.n_sites] = True
            else:
                first = factor % config.n_sites
                factor_site_mask[factor, first] = True
                factor_site_mask[factor, (first + 1) % config.n_sites] = True

        for factor in range(config.n_factors):
            rank = int(ranks[factor])
            maps[factor, :, :, rank:] = 0.0
            for site in range(config.n_sites):
                maps[factor, site] = (
                    rotations[site] @ maps[factor, site]
                ) * site_scales[site]
                if not bool(factor_site_mask[factor, site]):
                    maps[factor, site].zero_()

        frequency_generator = torch.Generator(device="cpu").manual_seed(
            config.structure_seed + 70_000_033
        )
        frequency_order = torch.randperm(
            config.n_factors, generator=frequency_generator
        )
        frequency_rank = torch.empty(config.n_factors, dtype=torch.long)
        frequency_rank[frequency_order] = torch.arange(
            1, config.n_factors + 1, dtype=torch.long
        )
        if config.feature_frequency == "uniform":
            sampling_weights = torch.ones(config.n_factors, dtype=torch.float64)
        else:
            sampling_weights = frequency_rank.to(torch.float64).reciprocal()

        pair_generator = torch.Generator(device="cpu").manual_seed(
            config.structure_seed + 80_000_047
        )
        pair_order = torch.randperm(config.n_factors, generator=pair_generator)
        paired_count = config.n_factors - config.n_factors % 2
        coactivation_groups = pair_order[:paired_count].reshape(-1, 2)
        if len(coactivation_groups):
            pair_weights = sampling_weights[coactivation_groups].prod(dim=1)
            coactivation_group_probabilities = pair_weights / pair_weights.sum()
        else:
            coactivation_group_probabilities = torch.empty(dtype=torch.float64)

        base_inclusion = _conditional_bernoulli_marginals(
            sampling_weights, config.active_per_example
        )
        inclusion_probabilities = base_inclusion.clone()
        if config.coactivation_probability > 0:
            enforced_inclusion = torch.zeros(config.n_factors, dtype=torch.float64)
            all_indices = torch.arange(config.n_factors)
            for group_index, pair in enumerate(coactivation_groups):
                conditional = torch.zeros(config.n_factors, dtype=torch.float64)
                conditional[pair] = 1.0
                remaining_mask = torch.ones(config.n_factors, dtype=torch.bool)
                remaining_mask[pair] = False
                remaining_indices = all_indices[remaining_mask]
                remaining_count = config.active_per_example - 2
                if remaining_count:
                    conditional[remaining_indices] = _conditional_bernoulli_marginals(
                        sampling_weights[remaining_indices], remaining_count
                    )
                enforced_inclusion += (
                    coactivation_group_probabilities[group_index] * conditional
                )
            rho = config.coactivation_probability
            inclusion_probabilities = (
                1.0 - rho
            ) * base_inclusion + rho * enforced_inclusion

        factors: list[FactorMetadata] = []
        distribution_family = (
            "gaussian"
            if config.coordinate_amplitude_law == "gaussian"
            else "student_t_df3"
        )
        for factor in range(config.n_factors):
            rank = int(ranks[factor])
            active_sites = tuple(
                torch.nonzero(factor_site_mask[factor], as_tuple=False)
                .flatten()
                .tolist()
            )
            if config.site_presence_span == "one":
                category = "single_site_control"
                family = f"{distribution_family}_subspace_single_site_control"
            elif config.site_map_rank_family == "independent":
                category = "shared_high_rank_site_map"
                family = f"{distribution_family}_subspace_high_rank_site_map"
            elif config.site_presence_span == "two":
                category = "partial_site"
                family = f"{distribution_family}_subspace_partial_site"
            else:
                category = "shared"
                family = f"{distribution_family}_subspace"
            factors.append(
                FactorMetadata(
                    index=factor,
                    name=f"ladder_factor_{factor:03d}",
                    family=family,
                    category=category,
                    coordinate_dim=rank,
                    intrinsic_dim=rank,
                    frequency=float(inclusion_probabilities[factor]),
                    active_sites=active_sites,
                )
            )

        realized_site_map_ranks = torch.empty(config.n_factors, dtype=torch.long)
        for factor in range(config.n_factors):
            rank = int(ranks[factor])
            site_rows = maps[factor, :, :, :rank].reshape(config.n_sites, -1)
            realized_site_map_ranks[factor] = torch.linalg.matrix_rank(site_rows)

        self.factors = tuple(factors)
        self.factor_ranks = ranks
        self.base_frames = base_frames
        self.site_map_bases = map_bases
        self.site_map_loadings = site_map_loadings
        self.site_rotations = rotations
        self.site_scales = site_scales
        self.factor_site_mask = factor_site_mask
        self.realized_site_map_ranks = realized_site_map_ranks
        self.factor_frequency_rank = frequency_rank
        self.factor_sampling_weights = sampling_weights
        self.factor_inclusion_probabilities = inclusion_probabilities
        self.coactivation_groups = coactivation_groups
        self.coactivation_group_probabilities = coactivation_group_probabilities
        self.factor_overlap_pairs = factor_overlap_pairs
        self.realized_pair_principal_angles_degrees = (
            realized_pair_principal_angles_degrees
        )
        self.contribution_maps = maps
        self._contribution_maps_float64 = maps.to(torch.float64)
        self._ground_truth_extra = {
            "coordinate_truth": (
                "shared_support_site_specific_coordinates"
                if config.step == "shared_support"
                else "shared_coordinate"
            ),
            "site_map_rank_family": config.site_map_rank_family,
            "site_presence_span": config.site_presence_span,
            "feature_frequency": config.feature_frequency,
            "coactivation_probability": config.coactivation_probability,
            "factor_ranks": self.factor_ranks,
            "base_frames": self.base_frames,
            "site_map_bases": self.site_map_bases,
            "site_map_loadings": self.site_map_loadings,
            "factor_site_mask": self.factor_site_mask,
            "realized_site_map_ranks": self.realized_site_map_ranks,
            "factor_frequency_rank": self.factor_frequency_rank,
            "factor_sampling_weights": self.factor_sampling_weights,
            "factor_inclusion_probabilities": self.factor_inclusion_probabilities,
            "coactivation_groups": self.coactivation_groups,
            "coactivation_group_probabilities": (self.coactivation_group_probabilities),
            "coordinate_amplitude_law": config.coordinate_amplitude_law,
            "coordinate_standardization": (
                "unit_marginal_variance_before_inverse_sqrt_rank_scaling"
            ),
            "student_t_degrees_of_freedom": (
                3 if config.coordinate_amplitude_law == "student_t_df3" else None
            ),
            "factor_subspace_overlap": config.factor_subspace_overlap,
            "factor_overlap_pairs": self.factor_overlap_pairs,
            "target_pair_principal_angle_degrees": (
                30.0 if config.factor_subspace_overlap == "paired_30deg" else None
            ),
            "principal_angle_reference": (
                "full_block_dim_canonical_base_frames_before_rank_truncation"
            ),
            "realized_pair_principal_angles_degrees": (
                self.realized_pair_principal_angles_degrees
            ),
            "shared_feature_claim_eligible": torch.tensor(
                config.step != "shared_support" and config.site_presence_span != "one",
                dtype=torch.bool,
            ),
        }
        self._set_categories()
        delta = copy.deepcopy(_LADDER_DELTA[config.step])
        if delta is not None and delta["value"] == "configured":
            delta["value"] = config.noise_std
        protocol = {
            "schema": "block-crosscoder.phase1.synthetic.v1",
            "name": "block_crosscoder_synthesis_ladder",
            "lineage": {
                "classification": "novel",
                "citation": None,
                "nearest_sources": [
                    "arXiv:2606.25234",
                    "arXiv:2606.06333",
                    "https://transformer-circuits.pub/2024/crosscoders/index.html",
                ],
            },
            "rationale": (
                "Identify whether sharing support, then sharing vector "
                "coordinates, remains recoverable under declared map-rank, "
                "site-span, frequency, coactivation, amplitude-tail, and "
                "factor-subspace-overlap controls."
            ),
            "configuration": asdict(config),
            "design": {
                "baseline": {
                    "coordinate_truth": "shared_coordinate",
                    "site_rotation": "identity",
                    "site_scale": "all_one",
                    "noise_std": 0.0,
                    "factor_rank": config.base_rank,
                },
                "step": config.step,
                "delta_from_baseline": delta,
                "one_delta_enforced_by": "step enum",
                "orthogonal_truth_axes": {
                    "site_map_rank_family": config.site_map_rank_family,
                    "site_presence_span": config.site_presence_span,
                    "feature_frequency": config.feature_frequency,
                    "coactivation_probability": config.coactivation_probability,
                    "coordinate_amplitude_law": config.coordinate_amplitude_law,
                    "factor_subspace_overlap": config.factor_subspace_overlap,
                },
                "independent_map_interpretation": (
                    "shared_coordinate_high_rank_site_map_factorization_stress"
                    if config.site_map_rank_family == "independent"
                    else "not_applicable"
                ),
            },
            "sites": {
                "dimensions": list(self.site_dims),
                "padded_dimension": config.d_model,
            },
            "sampling": {
                "factor_count": config.n_factors,
                "active_without_replacement": config.active_per_example,
                "coordinate_truth": (
                    "shared_support_site_specific_coordinates"
                    if config.step == "shared_support"
                    else "shared_coordinate"
                ),
                "noise_std": config.noise_std if config.step == "noise" else 0.0,
                "factor_ranks": ranks.tolist(),
                "site_scales": site_scales.tolist(),
                "site_map_rank_family": config.site_map_rank_family,
                "site_map_loadings": site_map_loadings.tolist(),
                "realized_site_map_ranks": realized_site_map_ranks.tolist(),
                "site_presence_span": config.site_presence_span,
                "factor_active_sites": [
                    list(factor.active_sites) for factor in self.factors
                ],
                "feature_frequency": config.feature_frequency,
                "zipf_alpha": (
                    1.0 if config.feature_frequency == "zipf_alpha_1" else None
                ),
                "support_law": (
                    "fixed_cardinality_conditional_bernoulli_with_mass_"
                    "proportional_to_product_of_factor_weights"
                ),
                "factor_frequency_rank": frequency_rank.tolist(),
                "factor_sampling_weights": sampling_weights.tolist(),
                "factor_inclusion_probabilities": inclusion_probabilities.tolist(),
                "coactivation_probability": config.coactivation_probability,
                "coactivation_semantics": (
                    "mixture_probability_of_forcing_one_planted_factor_pair;_"
                    "not_claimed_as_the_binary_pairwise_correlation"
                ),
                "coactivation_groups": coactivation_groups.tolist(),
                "coactivation_group_probabilities": (
                    coactivation_group_probabilities.tolist()
                ),
                "coordinate_amplitude_law": config.coordinate_amplitude_law,
                "coordinate_standardization": (
                    "unit_marginal_variance_before_inverse_sqrt_rank_scaling"
                ),
                "student_t_degrees_of_freedom": (
                    3 if config.coordinate_amplitude_law == "student_t_df3" else None
                ),
                "student_t_construction": (
                    "elliptical_normal_over_sqrt_chi_squared_df3"
                    if config.coordinate_amplitude_law == "student_t_df3"
                    else None
                ),
                "factor_subspace_overlap": config.factor_subspace_overlap,
                "factor_overlap_pairs": factor_overlap_pairs.tolist(),
                "factor_overlap_pairing": (
                    "structure_seed_plus_60000019_local_torch_randperm"
                ),
                "target_pair_principal_angle_degrees": (
                    30.0 if config.factor_subspace_overlap == "paired_30deg" else None
                ),
                "realized_pair_principal_angles_degrees": (
                    realized_pair_principal_angles_degrees.tolist()
                ),
                "principal_angle_reference": (
                    "full_block_dim_canonical_base_frames_before_rank_truncation"
                ),
            },
            "seeds": {
                "structure": config.structure_seed,
                "train": config.train_seed,
                "eval": config.eval_seed,
            },
            "factors": [factor.to_protocol() for factor in self.factors],
        }
        self._finish_protocol(
            protocol,
            digest_tensors={
                "base_frames": self.base_frames,
                "contribution_maps": self.contribution_maps,
                "factor_ranks": self.factor_ranks,
                "factor_site_mask": self.factor_site_mask,
                "site_map_bases": self.site_map_bases,
                "site_map_loadings": self.site_map_loadings,
                "realized_site_map_ranks": self.realized_site_map_ranks,
                "factor_frequency_rank": self.factor_frequency_rank,
                "factor_sampling_weights": self.factor_sampling_weights,
                "factor_inclusion_probabilities": (self.factor_inclusion_probabilities),
                "coactivation_groups": self.coactivation_groups,
                "coactivation_group_probabilities": (
                    self.coactivation_group_probabilities
                ),
                "factor_overlap_pairs": self.factor_overlap_pairs,
                "realized_pair_principal_angles_degrees": (
                    self.realized_pair_principal_angles_degrees
                ),
                "site_rotations": self.site_rotations,
                "site_scales": self.site_scales,
            },
        )

    def _draw_support(self, example_ids: torch.Tensor) -> torch.Tensor:
        """Draw the declared exact-cardinality support and pair mixture."""
        active = _conditional_bernoulli_sample(
            example_ids,
            self.factor_sampling_weights,
            self.config.active_per_example,
            seed=self.split_seed,
            stream=1,
        )
        rho = self.config.coactivation_probability
        if rho == 0.0:
            return active

        enforce = (
            _uniform(example_ids, 1, seed=self.split_seed, stream=20_003)[:, 0] < rho
        )
        if not bool(enforce.any()):
            return active
        enforced_rows = torch.nonzero(enforce, as_tuple=False).flatten()
        group_draw = _categorical(
            example_ids[enforced_rows],
            self.coactivation_group_probabilities,
            seed=self.split_seed,
            stream=20_009,
        )
        all_indices = torch.arange(len(self.factors))
        for group_index, pair in enumerate(self.coactivation_groups):
            local = torch.nonzero(group_draw == group_index, as_tuple=False).flatten()
            if len(local) == 0:
                continue
            rows = enforced_rows[local]
            active[rows] = False
            active[rows[:, None], pair[None, :]] = True
            remaining_count = self.config.active_per_example - 2
            if remaining_count == 0:
                continue
            remaining_mask = torch.ones(len(self.factors), dtype=torch.bool)
            remaining_mask[pair] = False
            remaining_indices = all_indices[remaining_mask]
            remainder = _conditional_bernoulli_sample(
                example_ids[rows],
                self.factor_sampling_weights[remaining_indices],
                remaining_count,
                seed=self.split_seed,
                stream=30_001 + 101 * group_index,
            )
            active[rows[:, None], remaining_indices[None, :]] = remainder
        if not torch.equal(
            active.sum(dim=1),
            torch.full((len(example_ids),), self.config.active_per_example),
        ):
            raise RuntimeError("coactivation support accounting mismatch")
        return active

    def _generate(
        self, example_ids: torch.Tensor, presentation_ids: torch.Tensor
    ) -> Phase1Batch:
        n, n_factors = len(example_ids), len(self.factors)
        active = self._draw_support(example_ids)
        n_events = n * self.config.active_per_example
        factor_events = active.transpose(0, 1).nonzero(as_tuple=False)
        if len(factor_events) != n_events:
            raise RuntimeError("ladder event accounting mismatch")
        event_factor = factor_events[:, 0].contiguous()
        event_example = factor_events[:, 1].contiguous()
        factor_offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.long),
                torch.bincount(event_factor, minlength=n_factors).cumsum(dim=0),
            )
        ).tolist()
        coordinates = torch.zeros(n_events, self.n_sites, self.max_coordinate_dim)
        contributions = torch.zeros(n_events, self.n_sites, self.padded_dim)
        clean_x = torch.zeros(n, self.n_sites, self.padded_dim)

        for factor, metadata in enumerate(self.factors):
            start, stop = factor_offsets[factor : factor + 2]
            if start == stop:
                continue
            sl = slice(start, stop)
            rows = event_example[sl]
            ids = example_ids[rows]
            rank = metadata.coordinate_dim
            shared = _factor_coordinates(
                ids,
                rank,
                law=self.config.coordinate_amplitude_law,
                seed=self.split_seed,
                stream=1_000 + 17 * factor,
            ) / math.sqrt(rank)
            for site in metadata.active_sites:
                if self.config.step == "shared_support":
                    if site == 0:
                        site_coordinate = shared
                    else:
                        site_coordinate = _factor_coordinates(
                            ids,
                            rank,
                            law=self.config.coordinate_amplitude_law,
                            seed=self.split_seed,
                            stream=1_001 + 17 * factor + site,
                        ) / math.sqrt(rank)
                else:
                    site_coordinate = shared
                coordinates[sl, site, :rank] = site_coordinate
            contributions[sl] = torch.einsum(
                "esr,sdr->esd",
                coordinates[sl].to(torch.float64),
                self._contribution_maps_float64[factor],
            ).to(torch.float32)
            clean_x[rows] = clean_x[rows] + contributions[sl]
        x = clean_x.clone()
        if self.config.step == "noise":
            x = x + self.config.noise_std * _normal(
                example_ids,
                self.n_sites * self.padded_dim,
                seed=self.split_seed,
                stream=900_000,
            ).reshape(n, self.n_sites, self.padded_dim)
        observed = torch.ones(n, self.n_sites, dtype=torch.bool)
        return Phase1Batch(
            x=x,
            clean_x=clean_x,
            active=active,
            observed=observed,
            example_ids=example_ids.clone(),
            presentation_ids=presentation_ids.clone(),
            event_example=event_example,
            event_factor=event_factor,
            coordinates=coordinates,
            contributions=contributions,
        )


def make_ladder_dataset(
    config: LadderSyntheticConfig | None = None, *, split: Split = "train"
) -> LadderDataset:
    return LadderDataset(config or LadderSyntheticConfig(), _validate_split(split))
