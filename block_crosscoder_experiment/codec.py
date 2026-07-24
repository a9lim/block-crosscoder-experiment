"""The frozen rate–distortion codec specified in ``docs/design.md``.

Everything here is fit on the CALIBRATION split and frozen before the
evaluation split is touched. Selection runs in threshold mode — the
codec prices the deployed inference path, variable per-token counts
included. Both arms (block and scalar) flow through the
identical code path; b=1 makes the orientation trivial and the
amplitude obligation one coordinate.

Pipeline, per model:

1. **Calibration pass** (`fit_codec`): stream the calib split once,
   collecting every selected block-event (code vector + block id) plus
   the per-token count histogram and per-block firing frequencies.
2. **Active-count floor**: blocks with fewer than `floor` calib events
   are EXCLUDED from the codec — zeroed at decode, mask-stripped before
   counting, paying no bits — identically in both arms; exclusions and their
   calibration/evaluation usage shares are reported openly, with split size
   and floor bound in every cell rather than inherited from a campaign default.
3. **Canonical orientation**: per block, order the calib active-code
   second-moment spectral subspaces descending. Separated one-dimensional
   subspaces retain their principal axes; repeated or near-repeated clusters
   use projected Gram--Schmidt on active-code observations in immutable
   stream order. This construction is equivariant under the residual O(b)
   gauge; frozen thereafter. Without it, an arbitrary gauge rotation changes
   componentwise clipping while the model is unchanged (tested, including an
   exactly repeated two-dimensional eigenspace).
4. **Quantizer**: per canonical coordinate, clip to the calib
   0.1%/99.9% quantiles, then 2^q uniform levels spanning the range
   (endpoints included: xhat = lo + round(t*(2^q-1)) * (hi-lo)/(2^q-1));
   out-of-range saturates. q swept per spec.
5. **Operational support bits/token**: a fixed-width active-count field plus
   one fixed-width compact block-ID field per active event.  A frozen sorted
   compact-ID-to-dictionary-ID table is part of the priced codec artifact, so
   noncontiguous floor exclusions cannot make the packet undecodable or its
   support cost optimistic.  This exact packet is validated by
   ``encode_batch``/``decode_batch`` and is the only support rate used for
   selection.  Ideal count-plus-enumerative and
   independent-Bernoulli support rates are reported alongside as explicitly
   non-operational sensitivity analyses; neither is substituted for packet
   cost.
6. **Amplitude bits/token**: q * b * k_t — each selected block carries
   the obligation to transmit b coordinates; the scalar arm pays q * l_t
   for its own realized l_t.
7. **Distortion**: declared-coordinate FVU through the quantized codes, per site
   and pooled, centering by the CALIB-fit per-site mean (no eval-fit
   parameters anywhere).
8. **Uncertainty**: bootstrap over immutable stored sequence IDs,
   never tokens. Fixed-length grouping exists only as an explicitly labelled
   synthetic/test fallback.
"""

from __future__ import annotations

import math
import os
import tempfile
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import NamedTuple, Protocol

import torch

from .durability import durable_mkdir, fsync_directory
from .model import BSCOutput, BSCSelection
from .runtime_limits import TRUSTED_DECODE_Q_CHUNK
from .serialization import TYPED_PAYLOAD_DIGEST_CONTRACT, typed_payload_digest

__all__ = [
    "CodecSpec",
    "Codec",
    "EncodedBatch",
    "fit_codec",
    "evaluate_rd",
    "encode_batch",
    "encode_batch_all_q",
    "decode_batch",
    "decode_batch_all_q",
    "estimate_calibration_peak_bytes",
]


_CODEC_PAYLOAD_KEYS = {
    "format_version",
    "artifact_digest_contract",
    "spec",
    "included",
    "rank_to_block",
    "rotation",
    "lo",
    "hi",
    "count_log2p",
    "bernoulli_log2p",
    "bernoulli_log2q",
    "calib_events",
    "calib_tokens",
    "calib_mean",
    "meta",
    "artifact_sha256",
}

# Bounds the fp64 second-moment workspace independently of the number of
# selected calibration events.  At the publication block width (b=4), this is
# a 32 MiB outer-product slab plus a small code-conversion buffer.
_CALIBRATION_MOMENT_CHUNK = 262_144
# Batched ragged quantiles replace thousands of tiny torch.quantile calls while
# bounding the NaN-padded workspace to 32 MiB of fp32 values. A single unusually
# frequent group may exceed this cap only by its own unavoidable event payload.
_CALIBRATION_QUANTILE_PAD_MAX_ELEMENTS = 8 << 20
# Eigenvectors are not an identified frame inside a repeated eigenspace.  A
# relative cluster tolerance also keeps an almost-degenerate frame from being
# chosen by backend-level roundoff.  The complete value is recorded in every
# codec artifact.
_CANONICAL_EIGENSPACE_RELATIVE_TOLERANCE = 1e-6
# A covariance-null direction has no data-defined frame.  Eigh roundoff can
# lift an exact null eigenvalue by O(eps * lambda_max), so this narrow bound
# distinguishes numerical nullity from merely low-variance active content.
_CANONICAL_NULL_EIGENVALUE_RELATIVE_TOLERANCE = 512.0 * torch.finfo(torch.float64).eps


def estimate_calibration_peak_bytes(selected_events: int, block_dim: int) -> int:
    """Conservative host-memory estimate shared by preflight and fitting."""
    if selected_events < 0 or block_dim <= 0:
        raise ValueError(
            "calibration events must be nonnegative and block_dim positive"
        )
    moment_events = min(selected_events, _CALIBRATION_MOMENT_CHUNK)
    return selected_events * (32 + 24 * block_dim) + moment_events * (
        8 * block_dim * block_dim + 8 * block_dim
    )


def _canonical_second_moment_frames(
    moment: torch.Tensor,
    mean_code: torch.Tensor,
    codes: torch.Tensor,
    ids: torch.Tensor,
    included: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Return gauge-equivariant descending second-moment frames.

    A simple ``eigh`` orientation is not defined when eigenvalues repeat:
    different but functionally identical O(b) gauges may receive unrelated
    bases inside the repeated eigenspace.  We retain the second-moment
    spectral cluster subspaces, but choose the basis *within* each cluster by
    projected modified Gram--Schmidt over ``mean, event_0, event_1, ...``.
    Event order is the immutable calibration stream order and is independent
    of the code coordinates.  Every candidate and every spectral projector
    transforms covariantly, hence a gauge ``O`` gives
    ``frame' = frame @ O.T``.

    A covariance-null subspace has no data-defined frame.  Any orthonormal
    completion is harmless only if those coordinates are dropped exactly at
    quantization; the returned null mask therefore requires clip bounds
    ``lo == hi == 0``.  Non-null directions must still be identified from
    active content or calibration fails closed.
    """

    if moment.ndim != 3 or moment.shape[-1] != moment.shape[-2]:
        raise ValueError("codec second moments must be square per block")
    groups, block_dim, _ = moment.shape
    if mean_code.shape != (groups, block_dim):
        raise ValueError("codec active-code means disagree with second moments")
    if codes.ndim != 2 or codes.shape[1] != block_dim or ids.shape != (len(codes),):
        raise ValueError("codec event codes and block IDs disagree")
    if included.shape != (groups,):
        raise ValueError("codec inclusion mask disagrees with second moments")

    eye = torch.eye(block_dim, dtype=torch.float64)
    safe_moment = torch.where(
        included.view(-1, 1, 1),
        moment,
        eye.expand(groups, block_dim, block_dim),
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(safe_moment)
    eigenvalues = eigenvalues.flip(-1)
    eigenvectors = eigenvectors.flip(-1)  # descending, columns

    # Stable sorting groups the observations without changing their immutable
    # order inside a block.  That order is the tie-break used to identify a
    # symmetric (+/- pair) active-code distribution.
    event_order = torch.argsort(ids, stable=True)
    boundaries = torch.cat(
        (
            torch.zeros(1, dtype=torch.long),
            torch.bincount(ids, minlength=groups).cumsum(0),
        )
    )

    frames = eye.expand(groups, block_dim, block_dim).clone()
    null_coordinates = torch.zeros(groups, block_dim, dtype=torch.bool)
    relative_gaps: list[float] = []
    near_degenerate_groups = 0
    near_degenerate_block_ids: list[int] = []
    null_block_ids: list[int] = []
    null_dimensions = [0] * groups
    null_coordinate_count = 0
    max_null_dimension = 0
    max_cluster = 1
    eps = torch.finfo(torch.float64).eps

    for group in included.nonzero(as_tuple=False).flatten().tolist():
        values = eigenvalues[group]
        if block_dim > 1:
            scales = torch.maximum(values[:-1].abs(), values[1:].abs()).clamp_min(
                torch.finfo(torch.float64).tiny
            )
            gaps = ((values[:-1] - values[1:]).abs() / scales).tolist()
            relative_gaps.extend(float(gap) for gap in gaps)
        else:
            gaps = []

        clusters: list[tuple[int, int]] = []
        start = 0
        for index, gap in enumerate(gaps):
            if gap > _CANONICAL_EIGENSPACE_RELATIVE_TOLERANCE:
                clusters.append((start, index + 1))
                start = index + 1
        clusters.append((start, block_dim))
        group_max_cluster = max(stop - start for start, stop in clusters)
        max_cluster = max(max_cluster, group_max_cluster)
        near_degenerate_groups += int(group_max_cluster > 1)
        if group_max_cluster > 1:
            near_degenerate_block_ids.append(group)

        first = int(boundaries[group])
        last = int(boundaries[group + 1])
        group_codes = codes[event_order[first:last]]
        group_mean = mean_code[group].double()
        canonical_columns: list[torch.Tensor] = []
        for cluster_start, cluster_stop in clusters:
            dimension = cluster_stop - cluster_start
            spectral_basis = eigenvectors[group, :, cluster_start:cluster_stop]
            largest_eigenvalue = values[0].abs()
            null_threshold = (
                largest_eigenvalue * _CANONICAL_NULL_EIGENVALUE_RELATIVE_TOLERANCE
            ).clamp_min(torch.finfo(torch.float64).tiny)
            cluster_is_null = bool(
                values[cluster_start:cluster_stop].abs().max() <= null_threshold
            )
            if cluster_is_null:
                # No calibration observation distinguishes orientations in
                # this subspace.  Keep eigh's orthonormal completion solely as
                # a storage carrier; exact zero clip bounds below make its
                # decoded contribution identically zero in every gauge.
                for spectral_column in spectral_basis.unbind(dim=1):
                    vector = spectral_column
                    for _ in range(2):
                        for previous in canonical_columns:
                            vector = vector - previous * torch.dot(previous, vector)
                    norm = vector.norm()
                    if not bool(norm > (256.0 * eps)):
                        raise RuntimeError(
                            "codec canonical null-frame completion lost rank"
                        )
                    canonical_columns.append(vector / norm)
                null_coordinates[group, cluster_start:cluster_stop] = True
                null_coordinate_count += dimension
                null_dimensions[group] += dimension
                max_null_dimension = max(max_null_dimension, dimension)
                if not null_block_ids or null_block_ids[-1] != group:
                    null_block_ids.append(group)
                continue
            projector = spectral_basis @ spectral_basis.T
            cluster_columns: list[torch.Tensor] = []
            for candidate_index in range(len(group_codes) + 1):
                candidate = (
                    group_mean
                    if candidate_index == 0
                    else group_codes[candidate_index - 1].double()
                )
                candidate_norm = candidate.norm()
                if not bool(candidate_norm > 0):
                    continue
                vector = projector @ candidate
                # Eigh may lose more than 1e-9 inter-cluster orthogonality for
                # an ill-conditioned moment even when every eigenvalue is
                # separated. Two global MGS passes retain the identified
                # spectral directions while making the complete frame, not
                # merely each repeated-eigenspace cluster, orthonormal.
                for _ in range(2):
                    for previous in canonical_columns:
                        vector = vector - previous * torch.dot(previous, vector)
                norm = vector.norm()
                if bool(norm > (256.0 * eps * candidate_norm)):
                    column = vector / norm
                    cluster_columns.append(column)
                    canonical_columns.append(column)
                    if len(cluster_columns) == dimension:
                        break
            if len(cluster_columns) != dimension:
                raise ValueError(
                    "codec canonical frame is not identifiable from active "
                    f"calibration codes for included block {group}: "
                    f"eigenspace dimension {dimension}, identified "
                    f"{len(cluster_columns)}"
                )

        basis = torch.stack(canonical_columns, dim=1)
        gram = basis.T @ basis
        if not torch.allclose(gram, eye, rtol=1e-9, atol=1e-9):
            raise RuntimeError("codec canonical-frame construction lost orthogonality")
        frames[group] = basis.T

    diagnostics: dict[str, object] = {
        "canonical_orientation": "second_moment_ordered_event_frame_v3",
        "canonical_eigenspace_relative_tolerance": (
            _CANONICAL_EIGENSPACE_RELATIVE_TOLERANCE
        ),
        "canonical_near_degenerate_groups": near_degenerate_groups,
        "canonical_near_degenerate_block_ids": near_degenerate_block_ids,
        "canonical_null_eigenvalue_relative_tolerance": (
            _CANONICAL_NULL_EIGENVALUE_RELATIVE_TOLERANCE
        ),
        "canonical_null_coordinate_count": null_coordinate_count,
        "canonical_null_block_ids": null_block_ids,
        "canonical_null_dimensions": null_dimensions,
        "canonical_max_null_dimension": max_null_dimension,
        "canonical_max_eigenspace_cluster": max_cluster,
        "canonical_min_relative_eigengap": (
            min(relative_gaps) if relative_gaps else 1.0
        ),
    }
    return frames, null_coordinates, diagnostics


def _grouped_coordinate_quantiles(
    sorted_codes: torch.Tensor,
    boundaries: torch.Tensor,
    groups: torch.Tensor,
    quantiles: torch.Tensor,
    *,
    max_pad_elements: int = _CALIBRATION_QUANTILE_PAD_MAX_ELEMENTS,
) -> torch.Tensor:
    """Exact per-group quantiles through bounded ragged batches."""

    if sorted_codes.ndim != 2 or boundaries.ndim != 1 or groups.ndim != 1:
        raise ValueError("grouped quantile tensors have invalid rank")
    if max_pad_elements <= 0:
        raise ValueError("grouped quantile workspace bound must be positive")
    block_dim = sorted_codes.shape[1]
    group_ids = [int(group) for group in groups]
    result = torch.empty(
        len(quantiles),
        len(group_ids),
        block_dim,
        dtype=sorted_codes.dtype,
        device=sorted_codes.device,
    )
    cursor = 0
    while cursor < len(group_ids):
        stop = cursor
        max_count = 0
        while stop < len(group_ids):
            group = group_ids[stop]
            count = int(boundaries[group + 1] - boundaries[group])
            if count <= 0:
                raise ValueError("quantile group has no calibration events")
            candidate_max = max(max_count, count)
            candidate_elements = (stop - cursor + 1) * candidate_max * block_dim
            if stop > cursor and candidate_elements > max_pad_elements:
                break
            max_count = candidate_max
            stop += 1

        chunk_groups = group_ids[cursor:stop]
        padded = torch.full(
            (len(chunk_groups), max_count, block_dim),
            float("nan"),
            dtype=sorted_codes.dtype,
            device=sorted_codes.device,
        )
        for local_index, group in enumerate(chunk_groups):
            start = int(boundaries[group])
            end = int(boundaries[group + 1])
            padded[local_index, : end - start] = sorted_codes[start:end]
        result[:, cursor:stop] = torch.nanquantile(
            padded,
            quantiles.to(device=sorted_codes.device, dtype=sorted_codes.dtype),
            dim=1,
        )
        cursor = stop
    return result


def _artifact_digest(payload: dict) -> str:
    return typed_payload_digest(payload)


def _normalized_quantizer_position(
    values: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return clipped quantizer positions and the exact serialized span.

    A zero clip span is a constant coordinate, not a numerical interval that
    may be widened for division.  In particular, canonical-null coordinates
    are serialized as ``[0, 0]`` and must emit symbol zero and reconstruct
    exact zero even when evaluation presents out-of-calibration content.  A
    positive but tiny span likewise retains its actual endpoints.
    """

    span = hi - lo
    safe_denominator = torch.where(span > 0, span, torch.ones_like(span))
    normalized = ((values - lo) / safe_denominator).clamp(0.0, 1.0)
    normalized = torch.where(span > 0, normalized, torch.zeros_like(normalized))
    return normalized, span


@dataclass
class CodecSpec:
    qs: tuple[int, ...] = (4, 6, 8)
    clip_lo: float = 0.001  # 0.1% quantile
    clip_hi: float = 0.999  # 99.9% quantile
    floor: int = 1000  # min calib active events for codec inclusion
    n_bootstrap: int = 1000
    bootstrap_seed: int = 0
    max_calibration_event_bytes: int = 2 * 1024**3

    def __post_init__(self) -> None:
        self.qs = tuple(int(q) for q in self.qs)
        if not self.qs or len(set(self.qs)) != len(self.qs):
            raise ValueError("codec qs must be nonempty and unique")
        if any(q <= 0 or q > 24 for q in self.qs):
            raise ValueError("codec q values must lie in [1, 24]")
        if not 0.0 <= self.clip_lo < self.clip_hi <= 1.0:
            raise ValueError("codec clipping quantiles are invalid")
        if self.floor < 0 or self.n_bootstrap <= 0:
            raise ValueError("codec floor must be nonnegative and bootstrap positive")
        if self.max_calibration_event_bytes <= 0:
            raise ValueError("codec calibration memory ceiling must be positive")


@dataclass
class EncodedBatch:
    """Logical sparse packet used for encode/decode round-trip validation."""

    q: int
    n_tokens: int
    counts: torch.Tensor  # [n] int32
    block_ids: torch.Tensor  # [events] int32 compact included-block IDs, token-major
    amplitude_symbols: torch.Tensor  # [events,b] int32


@dataclass
class _PacketEvents:
    """Q-independent sparse event stream for a single model output."""

    n_tokens: int
    counts: torch.Tensor  # [n] int32 CPU
    block_ids: torch.Tensor  # [events] compact IDs, int32 CPU
    original_ids: torch.Tensor  # [events] dictionary IDs on model device
    canonical_codes: torch.Tensor  # [events,b] on model device


@dataclass(frozen=True)
class _PacketSupport:
    """Q-independent support tensors shared by rate and packet accounting."""

    mask: torch.Tensor  # [tokens,groups] bool on model device
    counts: torch.Tensor  # [tokens] int64 on model device
    rows: torch.Tensor  # [events] int64 on model device
    original_ids: torch.Tensor  # [events] dictionary IDs on model device


@dataclass(frozen=True)
class _RDEvaluationInput:
    """One transformed R-D batch plus executor-owned paired context.

    Public :func:`evaluate_rd` callers retain the historical tensor or
    ``(tensor, row_ids)`` surface.  The cell executor can instead wrap the
    transformed input in this private carrier and attach its paired raw-space
    tensors without teaching the codec how normalization is serialized.
    """

    transformed: torch.Tensor
    row_ids: torch.Tensor | None = None
    context: object | None = None


class _RDEvaluationSelection(NamedTuple):
    """Full-view threshold geometry reused solely for sparse packet events.

    Packetization gathers only positions selected by ``mask``. At those
    positions the raw and post-selection codes are bit-identical, so carrying
    another dense masked ``[tokens, blocks, block_dim]`` tensor would only
    inflate the fused evaluator's peak memory.
    """

    z: torch.Tensor
    scores: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True)
class _RDEvaluationBatch:
    """Trusted batch state shared with an optional joint endpoint observer."""

    transformed: torch.Tensor
    sequence_ids: torch.Tensor  # [tokens] int64 CPU
    row_ids: torch.Tensor | None
    packet_events: _PacketEvents
    context: object | None
    decoder: torch.Tensor | None
    decoder_matrix: torch.Tensor | None


class _RDEvaluationObserver(Protocol):
    """Executor hook for consuming the codec's one trusted packet traversal.

    Implementations must consume decoded tensors synchronously.  The q-chunk
    mapping and all prediction aliases are released as soon as
    ``consume_decoded_chunk`` returns.
    """

    def begin_batch(self, batch: _RDEvaluationBatch) -> None: ...

    def consume_decoded_chunk(
        self,
        batch: _RDEvaluationBatch,
        decoded_chunk: Mapping[int, torch.Tensor],
    ) -> None: ...

    def end_batch(self, batch: _RDEvaluationBatch) -> None: ...


@dataclass
class Codec:
    """Frozen codec metadata — everything fit on calibration."""

    spec: CodecSpec
    included: torch.Tensor  # [G] bool
    rank_to_block: torch.Tensor  # [G_included] sorted original dictionary IDs
    rotation: torch.Tensor  # [G, b, b] canonical frames (row-major: z_can = R z)
    lo: torch.Tensor  # [G, b] clip floor, canonical coords
    hi: torch.Tensor  # [G, b] clip ceiling, canonical coords
    count_log2p: torch.Tensor  # [K_max+1] log2 of smoothed count model
    bernoulli_log2p: torch.Tensor  # [G] log2 p_hat (smoothed firing freq)
    bernoulli_log2q: torch.Tensor  # [G] log2 (1 - p_hat)
    calib_events: torch.Tensor  # [G] active-event counts (reporting)
    calib_tokens: int
    calib_mean: torch.Tensor  # [S, d] fp64 per-site mean (FVU centering)
    meta: dict = field(default_factory=dict)
    _device_cache: dict[tuple[str, str, torch.dtype], tuple[int, torch.Tensor]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        tensor_fields = {
            "included": self.included,
            "rank_to_block": self.rank_to_block,
            "rotation": self.rotation,
            "lo": self.lo,
            "hi": self.hi,
            "count_log2p": self.count_log2p,
            "bernoulli_log2p": self.bernoulli_log2p,
            "bernoulli_log2q": self.bernoulli_log2q,
            "calib_events": self.calib_events,
            "calib_mean": self.calib_mean,
        }
        non_tensors = [
            name for name, value in tensor_fields.items() if not torch.is_tensor(value)
        ]
        if non_tensors:
            raise TypeError(
                "codec tensor fields are not tensors: " + ", ".join(non_tensors)
            )
        if not isinstance(self.spec, CodecSpec):
            raise TypeError("codec spec must be a CodecSpec")
        if not isinstance(self.meta, dict):
            raise TypeError("codec meta must be a mapping")
        if (
            not isinstance(self.calib_tokens, int)
            or isinstance(self.calib_tokens, bool)
            or self.calib_tokens < 0
        ):
            raise ValueError("codec calib_tokens must be a nonnegative integer")
        if self.included.dtype != torch.bool or self.included.ndim != 1:
            raise TypeError("codec included must be a one-dimensional bool tensor")
        groups = int(self.included.numel())
        if self.rotation.ndim != 3 or self.rotation.shape[0] != groups:
            raise ValueError("codec rotation must have shape [groups, block, block]")
        block = int(self.rotation.shape[1])
        if block <= 0 or self.rotation.shape[2] != block:
            raise ValueError("codec rotation matrices must be nonempty and square")
        expected_shapes = {
            "lo": (groups, block),
            "hi": (groups, block),
            "bernoulli_log2p": (groups,),
            "bernoulli_log2q": (groups,),
            "calib_events": (groups,),
        }
        for name, shape in expected_shapes.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"codec {name} shape must be {shape}")
        if self.rank_to_block.ndim != 1:
            raise ValueError("codec rank_to_block must be one-dimensional")
        if self.count_log2p.ndim != 1:
            raise ValueError("codec count_log2p must be one-dimensional")
        if self.calib_mean.ndim != 2:
            raise ValueError("codec calib_mean must have shape [sites, width]")
        expected = self.included.nonzero(as_tuple=False).flatten().to(torch.long)
        observed = self.rank_to_block.to(torch.long)
        if observed.ndim != 1 or not torch.equal(observed.cpu(), expected.cpu()):
            raise ValueError(
                "codec rank_to_block must be the sorted IDs selected by included"
            )

    def _validate_serialized_semantics(self) -> None:
        """Fail closed on authenticated-but-semantically-invalid bytes.

        The outer SHA-256 proves byte identity, not that tensor dtypes, shapes,
        probability tables, or canonical frames form a usable codec.  Saved
        artifacts pass this stronger check before any consumer operation.
        """

        expected_dtypes = {
            "included": torch.bool,
            "rank_to_block": torch.int64,
            "rotation": torch.float32,
            "lo": torch.float32,
            "hi": torch.float32,
            "count_log2p": torch.float64,
            "bernoulli_log2p": torch.float32,
            "bernoulli_log2q": torch.float32,
            "calib_events": torch.int64,
            "calib_mean": torch.float64,
        }
        for name, dtype in expected_dtypes.items():
            if getattr(self, name).dtype != dtype:
                raise TypeError(f"codec {name} dtype must be {dtype}")
        groups = int(self.included.numel())
        block = int(self.rotation.shape[1])
        if self.calib_tokens <= 0:
            raise ValueError("serialized codec calibration split is empty")
        if self.count_log2p.shape != (self.n_included + 1,):
            raise ValueError("codec count model does not span its legal alphabet")
        for name in (
            "rotation",
            "lo",
            "hi",
            "count_log2p",
            "bernoulli_log2p",
            "bernoulli_log2q",
            "calib_mean",
        ):
            if not bool(torch.isfinite(getattr(self, name)).all()):
                raise ValueError(f"codec {name} contains nonfinite values")
        if bool((self.calib_events < 0).any()) or bool(
            (self.calib_events > self.calib_tokens).any()
        ):
            raise ValueError("codec calibration event counts are impossible")
        if bool((self.hi < self.lo).any()):
            raise ValueError("codec quantizer ceiling is below its floor")
        expected_included = self.calib_events >= self.spec.floor
        if not torch.equal(self.included, expected_included):
            raise ValueError("codec inclusion mask disagrees with its event floor")
        identity = torch.eye(block, dtype=torch.float32).expand(groups, block, block)
        gram = torch.einsum("gij,gkj->gik", self.rotation, self.rotation)
        if not torch.allclose(gram, identity, rtol=5e-4, atol=5e-4):
            raise ValueError("codec canonical rotations are not orthonormal")
        count_probabilities = self.count_log2p.exp2()
        if bool((self.count_log2p > 1e-10).any()) or not torch.allclose(
            count_probabilities.sum(),
            torch.tensor(1.0, dtype=torch.float64),
            rtol=1e-9,
            atol=1e-9,
        ):
            raise ValueError("codec count probability model is not normalized")
        p = self.bernoulli_log2p.double().exp2()
        q = self.bernoulli_log2q.double().exp2()
        if bool((p <= 0).any() or (q <= 0).any()) or not torch.allclose(
            p + q,
            torch.ones_like(p),
            rtol=2e-6,
            atol=2e-6,
        ):
            raise ValueError("codec Bernoulli probability model is inconsistent")
        expected_p = (self.calib_events.double() + 1.0) / (self.calib_tokens + 2.0)
        if not torch.allclose(p, expected_p, rtol=2e-6, atol=2e-7):
            raise ValueError("codec Bernoulli model disagrees with calibration counts")

        required_meta = {
            "n_blocks",
            "block_dim",
            "count_alphabet_max",
            "n_excluded",
            "calibration_selected_events",
            "excluded_calib_event_share",
            "theta",
            "model_cfg",
            "canonical_orientation",
            "canonical_eigenspace_relative_tolerance",
            "canonical_near_degenerate_groups",
            "canonical_near_degenerate_block_ids",
            "canonical_null_eigenvalue_relative_tolerance",
            "canonical_null_coordinate_count",
            "canonical_null_block_ids",
            "canonical_null_dimensions",
            "canonical_max_null_dimension",
            "canonical_max_eigenspace_cluster",
            "canonical_min_relative_eigengap",
        }
        missing_meta = sorted(required_meta - set(self.meta))
        if missing_meta:
            raise ValueError("codec metadata lacks: " + ", ".join(missing_meta))
        model_cfg = self.meta["model_cfg"]
        if not isinstance(model_cfg, dict):
            raise TypeError("codec model_cfg metadata must be a mapping")
        sites = model_cfg.get("n_sites")
        width = model_cfg.get("d_model")
        if (
            self.meta["n_blocks"] != groups
            or self.meta["block_dim"] != block
            or model_cfg.get("n_blocks") != groups
            or model_cfg.get("block_dim") != block
            or not isinstance(sites, int)
            or not isinstance(width, int)
            or self.calib_mean.shape != (sites, width)
            or self.meta["count_alphabet_max"] != self.n_included
            or self.meta["n_excluded"] != groups - self.n_included
            or self.meta["calibration_selected_events"] != int(self.calib_events.sum())
        ):
            raise ValueError("codec metadata disagrees with its tensor contract")
        theta = self.meta["theta"]
        if not isinstance(theta, (int, float)) or not math.isfinite(float(theta)):
            raise ValueError("codec metadata has no finite deployment threshold")
        expected_excluded_share = float(
            self.calib_events[~self.included].sum()
            / max(1, int(self.calib_events.sum()))
        )
        if not math.isclose(
            float(self.meta["excluded_calib_event_share"]),
            expected_excluded_share,
            rel_tol=1e-7,
            abs_tol=1e-9,
        ):
            raise ValueError("codec excluded-event share disagrees with its counts")
        if (
            self.meta["canonical_orientation"] != "second_moment_ordered_event_frame_v3"
            or self.meta["canonical_eigenspace_relative_tolerance"]
            != _CANONICAL_EIGENSPACE_RELATIVE_TOLERANCE
            or not isinstance(self.meta["canonical_near_degenerate_groups"], int)
            or isinstance(self.meta["canonical_near_degenerate_groups"], bool)
            or not 0 <= self.meta["canonical_near_degenerate_groups"] <= self.n_included
            or not isinstance(self.meta["canonical_near_degenerate_block_ids"], list)
            or self.meta["canonical_null_eigenvalue_relative_tolerance"]
            != _CANONICAL_NULL_EIGENVALUE_RELATIVE_TOLERANCE
            or not isinstance(self.meta["canonical_null_coordinate_count"], int)
            or isinstance(self.meta["canonical_null_coordinate_count"], bool)
            or not 0
            <= self.meta["canonical_null_coordinate_count"]
            <= self.n_included * block
            or not isinstance(self.meta["canonical_null_block_ids"], list)
            or not isinstance(self.meta["canonical_null_dimensions"], list)
            or not isinstance(self.meta["canonical_max_null_dimension"], int)
            or isinstance(self.meta["canonical_max_null_dimension"], bool)
            or not 0 <= self.meta["canonical_max_null_dimension"] <= block
            or not isinstance(self.meta["canonical_max_eigenspace_cluster"], int)
            or isinstance(self.meta["canonical_max_eigenspace_cluster"], bool)
            or not 1 <= self.meta["canonical_max_eigenspace_cluster"] <= block
            or not isinstance(
                self.meta["canonical_min_relative_eigengap"], (int, float)
            )
            or isinstance(self.meta["canonical_min_relative_eigengap"], bool)
            or not math.isfinite(float(self.meta["canonical_min_relative_eigengap"]))
            or float(self.meta["canonical_min_relative_eigengap"]) < 0
        ):
            raise ValueError("codec canonical-orientation metadata is invalid")
        near_degenerate_ids = self.meta["canonical_near_degenerate_block_ids"]
        if (
            len(near_degenerate_ids) != self.meta["canonical_near_degenerate_groups"]
            or any(
                not isinstance(block_id, int)
                or isinstance(block_id, bool)
                or block_id < 0
                or block_id >= groups
                or not bool(self.included[block_id])
                for block_id in near_degenerate_ids
            )
            or near_degenerate_ids != sorted(set(near_degenerate_ids))
        ):
            raise ValueError("codec near-degenerate block IDs are invalid")
        null_block_ids = self.meta["canonical_null_block_ids"]
        null_dimensions = self.meta["canonical_null_dimensions"]
        if (
            any(
                not isinstance(block_id, int)
                or isinstance(block_id, bool)
                or block_id < 0
                or block_id >= groups
                or not bool(self.included[block_id])
                for block_id in null_block_ids
            )
            or null_block_ids != sorted(set(null_block_ids))
            or len(null_dimensions) != groups
            or any(
                not isinstance(dimension, int)
                or isinstance(dimension, bool)
                or dimension < 0
                or dimension > block
                for dimension in null_dimensions
            )
            or [
                block_id
                for block_id, dimension in enumerate(null_dimensions)
                if dimension
            ]
            != null_block_ids
            or sum(null_dimensions) != self.meta["canonical_null_coordinate_count"]
            or max(null_dimensions, default=0)
            != self.meta["canonical_max_null_dimension"]
            or bool(null_block_ids)
            != bool(self.meta["canonical_null_coordinate_count"])
            or bool(null_block_ids) != bool(self.meta["canonical_max_null_dimension"])
        ):
            raise ValueError("codec null-space block IDs are invalid")
        for block_id, null_dimension in enumerate(null_dimensions):
            if null_dimension and (
                bool(self.lo[block_id, block - null_dimension :].count_nonzero())
                or bool(self.hi[block_id, block - null_dimension :].count_nonzero())
            ):
                raise ValueError("codec null-space clip bounds must be exactly zero")

    @property
    def n_included(self) -> int:
        return int(self.rank_to_block.numel())

    def _tensor_on(
        self,
        name: str,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return one frozen codec tensor, cached on its consumer device."""

        source = getattr(self, name)
        target_device = torch.device(device)
        target_dtype = source.dtype if dtype is None else dtype
        key = (name, str(target_device), target_dtype)
        cached = self._device_cache.get(key)
        # Tensor._version changes on in-place mutation, so even diagnostic
        # callers cannot observe stale cached bytes.
        if cached is None or cached[0] != source._version:
            value = source.to(device=target_device, dtype=target_dtype)
            self._device_cache[key] = (source._version, value)
            return value
        return cached[1]

    def block_to_rank(self, *, device: torch.device | str = "cpu") -> torch.Tensor:
        mapping = torch.full(
            (self.included.numel(),), -1, dtype=torch.long, device=device
        )
        if self.n_included:
            mapping[self._tensor_on("rank_to_block", device, dtype=torch.long)] = (
                torch.arange(self.n_included, device=device)
            )
        return mapping

    def log2_count_prob(self, k: torch.Tensor) -> torch.Tensor:
        if bool((k < 0).any()) or bool((k > self.n_included).any()):
            raise ValueError("active count lies outside the frozen support alphabet")
        if self.count_log2p.numel() != self.n_included + 1:
            raise ValueError("count model does not cover every legal support count")
        return self.count_log2p[k]

    def quantize(self, z_can: torch.Tensor, q: int) -> torch.Tensor:
        """z_can: [n, G, b] canonical-frame codes -> quantized, same frame."""
        levels = (1 << q) - 1
        lo = self._tensor_on("lo", z_can.device)
        hi = self._tensor_on("hi", z_can.device)
        t, span = _normalized_quantizer_position(z_can, lo, hi)
        return lo + torch.round(t * levels) / levels * span

    def quantize_indices(self, z_can: torch.Tensor, q: int) -> torch.Tensor:
        """Integer amplitude symbols for an actual round-trip packet."""
        levels = (1 << q) - 1
        lo = self._tensor_on("lo", z_can.device)
        hi = self._tensor_on("hi", z_can.device)
        normalized, _ = _normalized_quantizer_position(z_can, lo, hi)
        return torch.round(normalized * levels).to(torch.int32)

    def dequantize_indices(self, symbols: torch.Tensor, q: int) -> torch.Tensor:
        levels = (1 << q) - 1
        lo = self._tensor_on("lo", symbols.device)
        hi = self._tensor_on("hi", symbols.device)
        return lo + symbols.float() / levels * (hi - lo)

    def save(self, path: str | Path) -> None:
        """Atomically serialize every calibration-fit codec parameter."""
        payload = self.to_payload()
        path = Path(path)
        durable_mkdir(path.parent, parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            # Codec artifacts are immutable.  Atomic create-if-absent closes
            # the exists-check/replace race and can never clobber another
            # publisher's durable bytes.
            os.link(temporary, path)
            temporary.unlink()
            temporary = None
            fsync_directory(path.parent)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def to_payload(self) -> dict:
        """Return the complete, internally authenticated consumer payload."""

        self._validate_serialized_semantics()
        payload = {
            "format_version": 3,
            "artifact_digest_contract": TYPED_PAYLOAD_DIGEST_CONTRACT,
            "spec": asdict(self.spec),
            "included": self.included,
            "rank_to_block": self.rank_to_block,
            "rotation": self.rotation,
            "lo": self.lo,
            "hi": self.hi,
            "count_log2p": self.count_log2p,
            "bernoulli_log2p": self.bernoulli_log2p,
            "bernoulli_log2q": self.bernoulli_log2q,
            "calib_events": self.calib_events,
            "calib_tokens": self.calib_tokens,
            "calib_mean": self.calib_mean,
            "meta": self.meta,
        }
        payload["artifact_sha256"] = _artifact_digest(payload)
        return payload

    @classmethod
    def from_payload(cls, value: dict, *, source: str = "codec payload") -> "Codec":
        """Validate and reconstruct a codec without consulting another file."""

        if set(value) != _CODEC_PAYLOAD_KEYS:
            missing = sorted(_CODEC_PAYLOAD_KEYS - set(value))
            extra = sorted(set(value) - _CODEC_PAYLOAD_KEYS)
            raise ValueError(
                f"codec payload keys mismatch in {source}: "
                f"missing={missing}, extra={extra}"
            )
        payload = dict(value)
        if (
            payload.get("format_version") != 3
            or payload.get("artifact_digest_contract") != TYPED_PAYLOAD_DIGEST_CONTRACT
        ):
            raise ValueError(f"unsupported codec format in {source}")
        claimed = payload.pop("artifact_sha256", None)
        if claimed is None or claimed != _artifact_digest(payload):
            raise ValueError(f"codec artifact hash mismatch in {source}")
        spec_dict = dict(payload["spec"])
        spec_dict["qs"] = tuple(spec_dict["qs"])
        codec = cls(
            spec=CodecSpec(**spec_dict),
            included=payload["included"],
            rank_to_block=payload["rank_to_block"],
            rotation=payload["rotation"],
            lo=payload["lo"],
            hi=payload["hi"],
            count_log2p=payload["count_log2p"],
            bernoulli_log2p=payload["bernoulli_log2p"],
            bernoulli_log2q=payload["bernoulli_log2q"],
            calib_events=payload["calib_events"],
            calib_tokens=int(payload["calib_tokens"]),
            calib_mean=payload["calib_mean"],
            meta=dict(payload["meta"]),
        )
        codec._validate_serialized_semantics()
        return codec

    @classmethod
    def load(cls, path: str | Path) -> "Codec":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError(f"expected codec mapping in {path}")
        return cls.from_payload(payload, source=str(path))


def _log2_binom(n: int, k: torch.Tensor) -> torch.Tensor:
    """log2 C(n, k), elementwise over integer tensor k (values > n clamp)."""
    kf = k.clamp(max=n).double()
    nf = float(n)
    return (
        torch.lgamma(torch.tensor(nf + 1.0)).double()
        - torch.lgamma(kf + 1.0)
        - torch.lgamma(nf - kf + 1.0)
    ) / math.log(2.0)


def _materialized_model_tensors(
    model,
    decoder: torch.Tensor | None = None,
    encoder: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not hasattr(model, "decoder_tensor") or not hasattr(
        model, "forward_with_materialized"
    ):
        return decoder, encoder
    if getattr(model, "uses_direct_factorized_execution", False):
        if decoder is not None or encoder is not None:
            raise ValueError(
                "direct factorized codec execution refuses materialized weights"
            )
        return None, None
    if decoder is None:
        decoder = model.decoder_tensor()
    if encoder is None:
        encoder = (
            model._tied_encoder_tensor(decoder)
            if model.cfg.encoder_mode == "tied"
            else model.encoder_tensor()
        )
    return decoder, encoder


def _threshold_select(
    model,
    x: torch.Tensor,
    decoder: torch.Tensor | None,
    encoder: torch.Tensor | None,
    score_geometry=None,
):
    if hasattr(model, "select_with_materialized"):
        kwargs = {}
        if decoder is not None:
            kwargs["_decoder"] = decoder
        if encoder is not None:
            kwargs["_encoder"] = encoder
        if score_geometry is not None:
            kwargs["_score_geometry"] = score_geometry
        return model.select_with_materialized(x, mode="threshold", **kwargs)[0]
    if decoder is None or encoder is None:
        return model(x, mode="threshold")
    # Preserve the duck-typed codec surface for external reference models.
    return model.forward_with_materialized(
        x,
        mode="threshold",
        _decoder=decoder,
        _encoder=encoder,
        _score_geometry=score_geometry,
    )[0]


@torch.no_grad()
def _packet_from_output(model, codec: Codec, out, q: int) -> EncodedBatch:
    """Build the one canonical packet representation from a model output."""
    events = _packet_events_from_output(model, codec, out)
    return _packet_from_events(codec, events, q)


@torch.no_grad()
def _packet_support(mask: torch.Tensor) -> _PacketSupport:
    events = mask.nonzero(as_tuple=False)
    return _PacketSupport(
        mask=mask,
        counts=mask.sum(dim=1),
        rows=events[:, 0],
        original_ids=events[:, 1],
    )


@torch.no_grad()
def _packet_events_from_output(
    model,
    codec: Codec,
    out,
    *,
    support: _PacketSupport | None = None,
    _selected_code: torch.Tensor | None = None,
) -> _PacketEvents:
    """Extract support and rotate only selected events.

    The previous path rotated and quantized a dense ``[tokens, groups, block]``
    tensor before discarding almost every entry.  Deployment support is sparse
    by construction, so all q-independent work is performed on its actual
    event stream once.
    """
    device = next(model.parameters()).device
    if support is None:
        included = codec._tensor_on("included", device)
        support = _packet_support(out.mask & included.unsqueeze(0))
    original_ids = support.original_ids
    selected_source = out.z_selected if _selected_code is None else _selected_code
    selected = selected_source[support.rows, original_ids]
    canonical = torch.einsum(
        "eij,ej->ei",
        codec._tensor_on("rotation", device)[original_ids],
        selected,
    )
    rank_to_block = codec._tensor_on("rank_to_block", device, dtype=torch.long)
    compact_ranks = torch.searchsorted(rank_to_block, original_ids)
    return _PacketEvents(
        n_tokens=out.mask.shape[0],
        counts=support.counts.to(torch.int32),
        block_ids=compact_ranks.to(torch.int32),
        original_ids=original_ids,
        canonical_codes=canonical,
    )


@torch.no_grad()
def _packet_from_events(codec: Codec, events: _PacketEvents, q: int) -> EncodedBatch:
    if q not in codec.spec.qs:
        raise ValueError(f"q={q} is not in the frozen codec spec")
    levels = (1 << q) - 1
    lo = codec._tensor_on("lo", events.canonical_codes.device)[events.original_ids]
    hi = codec._tensor_on("hi", events.canonical_codes.device)[events.original_ids]
    normalized, _ = _normalized_quantizer_position(
        events.canonical_codes,
        lo,
        hi,
    )
    symbols = torch.round(normalized * levels).to(torch.int32)
    return EncodedBatch(
        q=q,
        n_tokens=events.n_tokens,
        counts=events.counts,
        block_ids=events.block_ids,
        amplitude_symbols=symbols,
    )


@torch.no_grad()
def encode_batch(model, codec: Codec, x: torch.Tensor, q: int) -> EncodedBatch:
    """Encode a batch into explicit support and integer amplitude symbols."""
    device = next(model.parameters()).device
    x = x.to(device, torch.float32)
    decoder, encoder = _materialized_model_tensors(model)
    score_geometry = None if decoder is None else model._frozen_score_geometry(decoder)
    out = _threshold_select(model, x, decoder, encoder, score_geometry)
    return _packet_from_output(model, codec, out, q)


@torch.no_grad()
def _encode_batch_events(
    model,
    codec: Codec,
    x: torch.Tensor,
    *,
    _decoder: torch.Tensor | None = None,
    _encoder: torch.Tensor | None = None,
    _score_geometry=None,
) -> tuple[object, _PacketEvents]:
    """Run threshold inference once and retain its trusted sparse event stream."""
    device = next(model.parameters()).device
    x = x.to(device, torch.float32, non_blocking=True)
    if _decoder is None or _encoder is None:
        _decoder, _encoder = _materialized_model_tensors(model, _decoder, _encoder)
    if _score_geometry is None and _decoder is not None:
        _score_geometry = model._frozen_score_geometry(_decoder)
    out = _threshold_select(
        model,
        x,
        _decoder,
        _encoder,
        _score_geometry,
    )
    events = _packet_events_from_output(model, codec, out)
    return out, events


@torch.no_grad()
def _encode_batch_all_q_events(
    model,
    codec: Codec,
    x: torch.Tensor,
    qs: tuple[int, ...] | None = None,
    *,
    _decoder: torch.Tensor | None = None,
    _encoder: torch.Tensor | None = None,
    _score_geometry=None,
) -> tuple[object, _PacketEvents, dict[int, EncodedBatch]]:
    """Run threshold inference once and materialize public packets for all q."""
    out, events = _encode_batch_events(
        model,
        codec,
        x,
        _decoder=_decoder,
        _encoder=_encoder,
        _score_geometry=_score_geometry,
    )
    requested = codec.spec.qs if qs is None else tuple(qs)
    packets = {q: _packet_from_events(codec, events, q) for q in requested}
    return out, events, packets


@torch.no_grad()
def encode_batch_all_q(
    model,
    codec: Codec,
    x: torch.Tensor,
    qs: tuple[int, ...] | None = None,
    *,
    _decoder: torch.Tensor | None = None,
    _encoder: torch.Tensor | None = None,
) -> tuple[object, dict[int, EncodedBatch]]:
    """Run threshold inference once and emit the full output plus every packet."""
    if _decoder is None or _encoder is None:
        _decoder, _encoder = _materialized_model_tensors(model, _decoder, _encoder)
    selection, _, packets = _encode_batch_all_q_events(
        model,
        codec,
        x,
        qs,
        _decoder=_decoder,
        _encoder=_encoder,
    )
    if not isinstance(selection, BSCSelection):
        return selection, packets
    xhat = model.decode(selection.z_selected, _decoder=_decoder)
    return BSCOutput(xhat, *selection), packets


def _decode_sparse_rows(
    model,
    sparse_code: torch.Tensor,
    *,
    _decoder: torch.Tensor | None = None,
    _decoder_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode CSR rows through either the full-site or direct rank carrier."""

    if (
        getattr(model, "uses_direct_factorized_execution", False)
        and _decoder is None
        and _decoder_matrix is None
    ):
        cfg = model.cfg
        assert cfg.site_rank is not None
        assert model.D_site is not None and model.D_core is not None
        rank_output = torch.sparse.mm(
            sparse_code,
            model._decoder_factor_core_map(),
        ).reshape(-1, cfg.site_rank, cfg.d_model)
        xhat = torch.matmul(
            rank_output.transpose(1, 2),
            model.D_site.transpose(0, 1),
        ).transpose(1, 2)
    else:
        if _decoder_matrix is None:
            decoder = model.decoder_tensor() if _decoder is None else _decoder
            decoder_matrix = decoder.permute(1, 2, 0, 3).reshape(
                model.cfg.n_blocks * model.cfg.block_dim,
                model.cfg.n_sites * model.cfg.d_model,
            )
        else:
            decoder_matrix = _decoder_matrix
        xhat = torch.sparse.mm(sparse_code, decoder_matrix).reshape(
            -1,
            model.cfg.n_sites,
            model.cfg.d_model,
        )
    if model.cfg.decoder_bias:
        xhat = xhat + model.c.unsqueeze(0)
    if model._has_padded_coordinates:
        xhat = xhat * model.coordinate_mask[:, 0, 0].to(xhat.dtype)
    return xhat


@torch.no_grad()
def decode_batch(
    model,
    codec: Codec,
    packet: EncodedBatch,
    *,
    _decoder: torch.Tensor | None = None,
    _decoder_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode an explicit packet without access to the source activations."""
    device = next(model.parameters()).device
    G, b = model.cfg.n_blocks, model.cfg.block_dim
    if packet.q not in codec.spec.qs:
        raise ValueError(f"packet q={packet.q} is not in the frozen codec spec")
    if packet.n_tokens < 0 or packet.counts.shape != (packet.n_tokens,):
        raise ValueError("packet counts must have one entry per token")
    if packet.counts.dtype not in {torch.int32, torch.int64}:
        raise TypeError("packet counts must be an integer tensor")
    packet_devices = {
        packet.counts.device,
        packet.block_ids.device,
        packet.amplitude_symbols.device,
    }
    validation_device = (
        packet.counts.device if len(packet_devices) == 1 else torch.device("cpu")
    )
    counts = packet.counts.detach().to(device=validation_device, dtype=torch.long)
    block_ids = packet.block_ids.detach().to(device=validation_device, dtype=torch.long)
    amplitude_symbols = packet.amplitude_symbols.detach().to(
        device=validation_device,
        dtype=torch.long,
    )
    count_sum = counts.sum()
    valid_count_bounds = (counts >= 0).all() & (counts <= codec.n_included).all()
    if not bool(valid_count_bounds):
        raise ValueError("packet has an impossible support count")
    n_events = int(count_sum)
    if packet.block_ids.shape != (n_events,):
        raise ValueError("packet block_ids length disagrees with support counts")
    if packet.amplitude_symbols.shape != (n_events, b):
        raise ValueError(
            "packet amplitude symbol shape disagrees with events/block width"
        )
    if packet.block_ids.dtype not in {torch.int32, torch.int64}:
        raise TypeError("packet block_ids must be an integer tensor")
    if packet.amplitude_symbols.dtype not in {torch.int32, torch.int64}:
        raise TypeError("packet amplitude symbols must be integers")
    levels = (1 << packet.q) - 1
    if n_events:
        # Each support is a set. Duplicate IDs within one token would be an
        # ambiguous packet. Sort the sparse keys once as part of validation;
        # CSR then receives canonical column order regardless of wire order.
        starts = torch.repeat_interleave(
            torch.arange(packet.n_tokens, device=validation_device), counts
        )
        keys = starts * max(1, codec.n_included) + block_ids
        order = torch.argsort(keys)
        sorted_keys = keys[order]
        valid_values = (
            (block_ids >= 0).all()
            & (block_ids < codec.n_included).all()
            & (amplitude_symbols >= 0).all()
            & (amplitude_symbols <= levels).all()
        )
        duplicate = (sorted_keys[1:] == sorted_keys[:-1]).any()
        if not bool(valid_values & ~duplicate):
            if bool(duplicate):
                raise ValueError("packet repeats a block id within one token")
            if not bool((block_ids >= 0).all() & (block_ids < codec.n_included).all()):
                raise ValueError(
                    "packet block rank is outside the frozen support alphabet"
                )
            raise ValueError("packet amplitude symbol is outside the q-bit alphabet")
        block_ids = block_ids[order]
        amplitude_symbols = amplitude_symbols[order]
    else:
        valid_values = (amplitude_symbols >= 0).all() & (
            amplitude_symbols <= levels
        ).all()
        if not bool(valid_values):
            raise ValueError("packet amplitude symbol is outside the q-bit alphabet")

    # Decode the actual sparse event stream with a CSR x dense product.  This
    # is algebraically the same block sum as ``model.decode`` but scales with
    # selected events instead of the full dictionary width.
    ranks = block_ids.to(device=device, non_blocking=True)
    ids = codec._tensor_on("rank_to_block", device, dtype=torch.long)[ranks]
    symbols = amplitude_symbols.to(device=device, non_blocking=True)
    lo = codec._tensor_on("lo", device)[ids]
    hi = codec._tensor_on("hi", device)[ids]
    z_can = lo + symbols.float() / levels * (hi - lo)
    z_events = torch.einsum(
        "eji,ej->ei",
        codec._tensor_on("rotation", device)[ids],
        z_can,
    )
    crow = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=device),
            (counts.to(device=device, non_blocking=True) * b).cumsum(dim=0),
        )
    )
    columns = (
        ids.unsqueeze(1) * b
        + torch.arange(b, dtype=torch.long, device=device).unsqueeze(0)
    ).reshape(-1)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sparse CSR tensor support is in beta state.*",
            category=UserWarning,
        )
        sparse_code = torch.sparse_csr_tensor(
            crow,
            columns,
            z_events.reshape(-1),
            size=(packet.n_tokens, G * b),
            device=device,
            check_invariants=False,
        )
    return _decode_sparse_rows(
        model,
        sparse_code,
        _decoder=_decoder,
        _decoder_matrix=_decoder_matrix,
    )


def _rotate_multi_q_events(
    event_rotation: torch.Tensor,
    canonical_codes: torch.Tensor,
) -> torch.Tensor:
    """Rotate ``[q, event, b]`` row codes through ``[event, b, b]`` frames.

    CUDA's broadcast matmul is substantially faster than the equivalent
    three-operand einsum at the campaign event counts.  The scalar case has no
    reduction to optimize, while CPU retains the existing reduction order.
    """
    if canonical_codes.shape[-1] == 1:
        return canonical_codes * event_rotation[:, 0, 0].view(1, -1, 1)
    if canonical_codes.is_cuda:
        return torch.matmul(
            canonical_codes.unsqueeze(-2),
            event_rotation.unsqueeze(0),
        ).squeeze(-2)
    return torch.einsum("eji,qej->qei", event_rotation, canonical_codes)


@torch.no_grad()
def decode_batch_all_q(
    model,
    codec: Codec,
    packets: dict[int, EncodedBatch],
    *,
    _decoder: torch.Tensor | None = None,
    _decoder_matrix: torch.Tensor | None = None,
) -> dict[int, torch.Tensor]:
    """Validate common support once and decode every quantizer in one SpMM.

    ``encode_batch_all_q`` emits one immutable support stream shared by all
    amplitude precisions.  Revalidating and sorting that support, rebuilding
    the same CSR structure, and launching one sparse product per q adds no
    packet evidence.  Stacking q on the sparse row axis retains the same
    packet arithmetic while issuing one sparse matrix product.
    """
    if not packets:
        raise ValueError("multi-q decode requires at least one packet")
    requested = tuple(packets)
    if len(set(requested)) != len(requested):  # defensive for mapping-like callers
        raise ValueError("multi-q decode requires unique q values")
    for q, packet in packets.items():
        if q != packet.q:
            raise ValueError("packet mapping key disagrees with packet q")
        if q not in codec.spec.qs:
            raise ValueError(f"packet q={q} is not in the frozen codec spec")

    first = packets[requested[0]]
    n_tokens = first.n_tokens
    if n_tokens < 0 or first.counts.shape != (n_tokens,):
        raise ValueError("packet counts must have one entry per token")
    if first.counts.dtype not in {torch.int32, torch.int64}:
        raise TypeError("packet counts must be an integer tensor")
    if first.block_ids.dtype not in {torch.int32, torch.int64}:
        raise TypeError("packet block_ids must be an integer tensor")

    packet_devices = {
        tensor.device
        for packet in packets.values()
        for tensor in (packet.counts, packet.block_ids, packet.amplitude_symbols)
    }
    validation_device = (
        next(iter(packet_devices)) if len(packet_devices) == 1 else torch.device("cpu")
    )
    counts = first.counts.detach().to(device=validation_device, dtype=torch.long)
    block_ids = first.block_ids.detach().to(
        device=validation_device,
        dtype=torch.long,
    )
    count_sum = counts.sum()
    valid_count_bounds = (counts >= 0).all() & (counts <= codec.n_included).all()
    if not bool(valid_count_bounds):
        raise ValueError("packet has an impossible support count")
    n_events = int(count_sum)
    if first.block_ids.shape != (n_events,):
        raise ValueError("packet block_ids length disagrees with support counts")

    def same_tensor(left: torch.Tensor, right: torch.Tensor) -> bool:
        if left is right:
            return True
        return (
            left.device == right.device
            and left.dtype == right.dtype
            and left.shape == right.shape
            and left.stride() == right.stride()
            and left.storage_offset() == right.storage_offset()
            and left.data_ptr() == right.data_ptr()
        )

    symbols_by_q: dict[int, torch.Tensor] = {}
    for q, packet in packets.items():
        if packet.n_tokens != n_tokens or packet.counts.shape != (n_tokens,):
            raise ValueError("multi-q packets do not share one token/count shape")
        if packet.counts.dtype not in {torch.int32, torch.int64}:
            raise TypeError("packet counts must be an integer tensor")
        if packet.block_ids.dtype not in {torch.int32, torch.int64}:
            raise TypeError("packet block_ids must be an integer tensor")
        if packet.block_ids.shape != (n_events,):
            raise ValueError("packet block_ids length disagrees with support counts")
        if packet.amplitude_symbols.shape != (n_events, model.cfg.block_dim):
            raise ValueError(
                "packet amplitude symbol shape disagrees with events/block width"
            )
        if packet.amplitude_symbols.dtype not in {torch.int32, torch.int64}:
            raise TypeError("packet amplitude symbols must be integers")
        if not same_tensor(packet.counts, first.counts) and not torch.equal(
            packet.counts.detach().to(validation_device), counts
        ):
            raise ValueError("multi-q packets do not share identical support counts")
        if not same_tensor(packet.block_ids, first.block_ids) and not torch.equal(
            packet.block_ids.detach().to(validation_device, dtype=torch.long),
            block_ids,
        ):
            raise ValueError("multi-q packets do not share identical block IDs")
        symbols_by_q[q] = packet.amplitude_symbols.detach().to(
            device=validation_device,
            dtype=torch.long,
        )

    if n_events:
        starts = torch.repeat_interleave(
            torch.arange(n_tokens, device=validation_device),
            counts,
        )
        keys = starts * max(1, codec.n_included) + block_ids
        order = torch.argsort(keys)
        sorted_keys = keys[order]
        valid_ids = (block_ids >= 0).all() & (block_ids < codec.n_included).all()
        duplicate = (sorted_keys[1:] == sorted_keys[:-1]).any()
        if not bool(valid_ids & ~duplicate):
            if bool(duplicate):
                raise ValueError("packet repeats a block id within one token")
            raise ValueError("packet block rank is outside the frozen support alphabet")
        block_ids = block_ids[order]
        symbols_by_q = {q: symbols[order] for q, symbols in symbols_by_q.items()}

    valid_amplitudes = torch.stack(
        [
            (symbols >= 0).all() & (symbols <= (1 << q) - 1).all()
            for q, symbols in symbols_by_q.items()
        ]
    ).all()
    if not bool(valid_amplitudes):
        raise ValueError("packet amplitude symbol is outside the q-bit alphabet")

    device = next(model.parameters()).device
    G, b = model.cfg.n_blocks, model.cfg.block_dim
    ranks = block_ids.to(device=device, non_blocking=True)
    ids = codec._tensor_on("rank_to_block", device, dtype=torch.long)[ranks]
    lo = codec._tensor_on("lo", device)[ids]
    span = codec._tensor_on("hi", device)[ids] - lo
    symbol_stack = torch.stack(
        [symbols_by_q[q].to(device=device, non_blocking=True) for q in requested]
    )
    levels = torch.tensor(
        [(1 << q) - 1 for q in requested],
        dtype=torch.float32,
        device=device,
    ).view(-1, 1, 1)
    z_can = lo.unsqueeze(0) + symbol_stack.float() / levels * span.unsqueeze(0)
    event_rotation = codec._tensor_on("rotation", device)[ids]
    z_events = _rotate_multi_q_events(
        event_rotation,
        z_can,
    )
    expanded_counts = counts.to(device=device, non_blocking=True) * b
    crow = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=device),
            expanded_counts.repeat(len(requested)).cumsum(dim=0),
        )
    )
    event_columns = (
        ids.unsqueeze(1) * b
        + torch.arange(b, dtype=torch.long, device=device).unsqueeze(0)
    ).reshape(-1)
    columns = event_columns.repeat(len(requested))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sparse CSR tensor support is in beta state.*",
            category=UserWarning,
        )
        sparse_code = torch.sparse_csr_tensor(
            crow,
            columns,
            z_events.reshape(-1),
            size=(len(requested) * n_tokens, G * b),
            device=device,
            check_invariants=False,
        )
    predictions = _decode_sparse_rows(
        model,
        sparse_code,
        _decoder=_decoder,
        _decoder_matrix=_decoder_matrix,
    ).reshape(
        len(requested),
        n_tokens,
        model.cfg.n_sites,
        model.cfg.d_model,
    )
    return {q: predictions[index] for index, q in enumerate(requested)}


@torch.no_grad()
def _decode_trusted_packet_events_q_chunks(
    model,
    codec: Codec,
    events: _PacketEvents,
    packets: dict[int, EncodedBatch] | None = None,
    *,
    qs: tuple[int, ...] | None = None,
    _decoder: torch.Tensor | None = None,
    _decoder_matrix: torch.Tensor | None = None,
    q_chunk_size: int = TRUSTED_DECODE_Q_CHUNK,
):
    """Yield bounded multi-q decodes for the encoder's own packet stream.

    Public decode entry points must authenticate arbitrary external packets.
    Here the support, ordering, and integer alphabets were produced moments
    earlier by ``_packet_events_from_output`` and ``_packet_from_events``.
    Carrying that trusted event stream forward avoids a global event sort and
    all device-to-host validation synchronizations on every evaluation batch.
    """
    if q_chunk_size <= 0:
        raise ValueError("trusted decode q_chunk_size must be positive")
    requested = (
        tuple(packets)
        if packets is not None
        else (codec.spec.qs if qs is None else tuple(qs))
    )
    if not requested:
        return
    if len(requested) != len(set(requested)) or any(
        q not in codec.spec.qs for q in requested
    ):
        raise ValueError("trusted decode q binding is invalid")
    for q, packet in (packets or {}).items():
        if q != packet.q:
            raise ValueError("trusted packet q binding is invalid")
        if packet.n_tokens != events.n_tokens:
            raise ValueError("trusted packet token count is not event-bound")
        if (
            packet.counts is not events.counts
            or packet.block_ids is not events.block_ids
        ):
            raise ValueError("trusted packet support is not event-bound")
        if packet.amplitude_symbols.shape != (
            len(events.block_ids),
            model.cfg.block_dim,
        ):
            raise ValueError("trusted packet amplitude shape is invalid")
    device = next(model.parameters()).device
    G, b = model.cfg.n_blocks, model.cfg.block_dim
    # This is the encoder's own trusted event stream: ``original_ids`` already
    # carries the dictionary-space IDs from the selection mask.  Avoid mapping
    # the compact packet IDs back through ``rank_to_block`` on every decode.
    ids = events.original_ids
    counts = events.counts
    lo = codec._tensor_on("lo", device)[ids]
    hi = codec._tensor_on("hi", device)[ids]
    normalized_codes, span = _normalized_quantizer_position(
        events.canonical_codes,
        lo,
        hi,
    )
    event_rotation = codec._tensor_on("rotation", device)[ids]
    if packets is not None:
        normalized_codes = None
    expanded_counts = counts.to(dtype=torch.long) * b
    event_columns = (
        ids.unsqueeze(1) * b
        + torch.arange(b, dtype=torch.long, device=device).unsqueeze(0)
    ).reshape(-1)
    max_chunk = min(q_chunk_size, len(requested))
    all_levels = torch.tensor(
        [(1 << q) - 1 for q in requested],
        dtype=torch.float32,
        device=device,
    ).view(-1, 1, 1)
    crow_max = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=device),
            expanded_counts.repeat(max_chunk).cumsum(dim=0),
        )
    )
    columns_max = event_columns.repeat(max_chunk)
    for start in range(0, len(requested), q_chunk_size):
        chunk_qs = requested[start : start + q_chunk_size]
        chunk_len = len(chunk_qs)
        levels = all_levels[start : start + chunk_len]
        if packets is None:
            assert normalized_codes is not None
            symbols = torch.round(normalized_codes.unsqueeze(0) * levels).to(
                torch.int32
            )
        else:
            symbols = torch.stack([packets[q].amplitude_symbols for q in chunk_qs])
        z_can = lo.unsqueeze(0) + symbols / levels * span.unsqueeze(0)
        z_events = _rotate_multi_q_events(
            event_rotation,
            z_can,
        )
        crow = crow_max[: chunk_len * events.n_tokens + 1]
        columns = columns_max[: chunk_len * event_columns.numel()]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Sparse CSR tensor support is in beta state.*",
                category=UserWarning,
            )
            sparse_code = torch.sparse_csr_tensor(
                crow,
                columns,
                z_events.reshape(-1),
                size=(len(chunk_qs) * events.n_tokens, G * b),
                device=device,
                check_invariants=False,
            )
        predictions = _decode_sparse_rows(
            model,
            sparse_code,
            _decoder=_decoder,
            _decoder_matrix=_decoder_matrix,
        ).reshape(
            len(chunk_qs),
            events.n_tokens,
            model.cfg.n_sites,
            model.cfg.d_model,
        )
        decoded_chunk = {q: predictions[index] for index, q in enumerate(chunk_qs)}
        yield decoded_chunk
        # The consumer deletes its chunk before requesting the next one.  Drop
        # the generator frame's aliases as soon as it resumes so two SpMM
        # outputs and CSR workspaces cannot overlap in lifetime.
        del decoded_chunk, predictions, sparse_code, columns, crow
        del z_events, z_can, symbols, levels


@torch.no_grad()
def fit_codec(model, batches, spec: CodecSpec, *, device: str = "cpu") -> Codec:
    """One calibration pass. `batches` yields [B, S, d] (CPU, any float
    dtype except fp16); selection in threshold mode against the model's
    frozen theta."""
    G, b = model.cfg.n_blocks, model.cfg.block_dim
    S, d = model.cfg.n_sites, model.cfg.d_model

    ev_codes: list[torch.Tensor] = []
    ev_ids: list[torch.Tensor] = []
    ev_tokens: list[torch.Tensor] = []
    accumulation_device = torch.device(device)
    block_events = torch.zeros(G, dtype=torch.long, device=accumulation_device)
    mean_acc = torch.zeros(S, d, dtype=torch.float64)
    n_tokens = 0
    selected_events = 0
    estimated_peak_bytes = 0
    materialized_decoder, materialized_encoder = _materialized_model_tensors(model)
    score_geometry = (
        None
        if materialized_decoder is None
        else model._frozen_score_geometry(materialized_decoder)
    )

    for raw_x in batches:
        x = raw_x.to(device, torch.float32, non_blocking=True)
        out = _threshold_select(
            model,
            x,
            materialized_decoder,
            materialized_encoder,
            score_geometry,
        )
        mask = out.mask
        z_sel = out.z_selected
        nz = mask.nonzero()
        selected_events += int(nz.shape[0])
        # Conservative bound for list storage, concatenation overlap,
        # canonical-code workspace, sort indices and per-event IDs.  The
        # ceiling fails closed; calibration never samples or truncates events.
        estimated_peak_bytes = estimate_calibration_peak_bytes(selected_events, b)
        if estimated_peak_bytes > spec.max_calibration_event_bytes:
            raise MemoryError(
                "exact codec calibration exceeds its resolved event-memory ceiling: "
                f"estimated {estimated_peak_bytes} > "
                f"{spec.max_calibration_event_bytes} bytes"
            )
        ev_codes.append(z_sel[nz[:, 0], nz[:, 1]].float().cpu())
        event_indices = nz.to(torch.int32).cpu()
        ev_ids.append(event_indices[:, 1])
        ev_tokens.append(event_indices[:, 0] + n_tokens)
        block_events += mask.sum(dim=0)
        mean_acc += torch.sum(raw_x, dim=0, dtype=torch.float64).cpu()
        n_tokens += x.shape[0]

    codes = torch.cat(ev_codes) if ev_codes else torch.zeros(0, b)
    ids = torch.cat(ev_ids).long() if ev_ids else torch.zeros(0, dtype=torch.long)
    token_ids = (
        torch.cat(ev_tokens).long() if ev_tokens else torch.zeros(0, dtype=torch.long)
    )
    # torch.cat allocates consolidated storage; release the per-batch tensors
    # before the orientation/quantile pass so production calibration does not
    # retain a second copy of every event.
    ev_codes.clear()
    ev_ids.clear()
    ev_tokens.clear()
    block_events = block_events.cpu()
    included = block_events >= spec.floor
    rank_to_block = included.nonzero(as_tuple=False).flatten().to(torch.long)

    # The deployed support is stripped of excluded blocks. Fit its count
    # model after the floor is known; fitting raw counts and pricing stripped
    # counts assigns bits to events the codec cannot transmit.
    if included.any():
        kept_tokens = token_ids[included[ids]]
        included_counts = torch.bincount(kept_tokens, minlength=n_tokens)
        count_hist = torch.bincount(included_counts)
        del kept_tokens
    else:
        included_counts = torch.zeros(n_tokens, dtype=torch.long)
        count_hist = torch.tensor([n_tokens], dtype=torch.long)
    del included_counts, token_ids

    # Canonical orientation: batched second moments via index_add, with a
    # gauge-equivariant ordered-event frame inside repeated/near-repeated
    # eigenspaces.
    M = torch.zeros(G, b, b, dtype=torch.float64)
    mean_code = torch.zeros(G, b, dtype=torch.float64)
    for start in range(0, len(codes), _CALIBRATION_MOMENT_CHUNK):
        stop = start + _CALIBRATION_MOMENT_CHUNK
        codes64 = codes[start:stop].double()
        chunk_ids = ids[start:stop]
        M.index_add_(
            0,
            chunk_ids,
            codes64.unsqueeze(2) * codes64.unsqueeze(1),
        )
        mean_code.index_add_(0, chunk_ids, codes64)
    denom = block_events.clamp_min(1).double()
    M /= denom.view(-1, 1, 1)
    mean_code /= denom.view(-1, 1)
    R, canonical_null_coordinates, canonical_diagnostics = (
        _canonical_second_moment_frames(
            M,
            mean_code,
            codes,
            ids,
            included,
        )
    )

    # Clip quantiles per canonical coordinate.
    codes_can = torch.einsum("nij,nj->ni", R[ids].float(), codes)
    lo = torch.zeros(G, b)
    hi = torch.ones(G, b)
    order = torch.argsort(ids)
    sorted_ids = ids[order]
    sorted_codes = codes_can[order]
    boundaries = torch.searchsorted(sorted_ids, torch.arange(G + 1, dtype=torch.long))
    qs = torch.tensor([spec.clip_lo, spec.clip_hi])
    quantile_groups = (included & (block_events > 0)).nonzero().flatten()
    grouped_quantiles = _grouped_coordinate_quantiles(
        sorted_codes,
        boundaries,
        quantile_groups,
        qs,
    )
    lo[quantile_groups] = grouped_quantiles[0]
    hi[quantile_groups] = grouped_quantiles[1]
    # A null-space basis is mathematically unidentifiable, but also carries no
    # calibration signal.  Exact zero ranges make its packet symbols and
    # arbitrary orthonormal completion decode to the same zero contribution.
    lo[canonical_null_coordinates] = 0.0
    hi[canonical_null_coordinates] = 0.0
    # Count model: add-one smoothing over the *entire legal alphabet*
    # [0, G_included].  Tail clamping is not a code: an unseen but legal
    # count must still have a distinct decodable symbol.
    k_max_obs = int(count_hist.nonzero().max()) if count_hist.sum() else 0
    if included.any():
        K_max = int(included.sum())
        smoothed = torch.ones(K_max + 1, dtype=torch.float64)
        smoothed[: count_hist.numel()] += count_hist.double()
        count_log2p = torch.log2(smoothed / smoothed.sum())
    else:
        # Empty support has one possible count and requires no count code.
        count_log2p = torch.zeros(1, dtype=torch.float64)

    # Bernoulli support-entropy sensitivity model.
    p_hat = (block_events.double() + 1.0) / (n_tokens + 2.0)
    bernoulli_log2p = torch.log2(p_hat)
    bernoulli_log2q = torch.log2(1.0 - p_hat)

    return Codec(
        spec=spec,
        included=included,
        rank_to_block=rank_to_block,
        rotation=R.float(),
        lo=lo,
        hi=hi,
        count_log2p=count_log2p,
        bernoulli_log2p=bernoulli_log2p.float(),
        bernoulli_log2q=bernoulli_log2q.float(),
        calib_events=block_events,
        calib_tokens=n_tokens,
        calib_mean=mean_acc / max(n_tokens, 1),
        meta={
            "n_blocks": G,
            "block_dim": b,
            "k_max_obs": k_max_obs,
            "count_alphabet_max": int(included.sum()),
            "theta": float(getattr(model, "theta", float("nan"))),
            "model_cfg": (
                asdict(model.cfg) if is_dataclass(model.cfg) else vars(model.cfg)
            ),
            "n_excluded": int((~included).sum()),
            "calibration_selected_events": selected_events,
            "calibration_estimated_peak_bytes": estimated_peak_bytes,
            "calibration_event_memory_ceiling_bytes": (
                spec.max_calibration_event_bytes
            ),
            "excluded_calib_event_share": float(
                block_events[~included].sum() / max(1, block_events.sum())
            ),
            **canonical_diagnostics,
        },
    )


def _rd_evaluation_coroutine(
    model,
    codec: Codec,
    *,
    row_len: int | None = None,
    device: str = "cpu",
    observer: _RDEvaluationObserver | None = None,
    materialized_decoder: torch.Tensor | None = None,
    materialized_encoder: torch.Tensor | None = None,
    score_geometry=None,
) -> object:
    """Coroutine implementing one incremental threshold-packet traversal.

    Send ``(item, selection)`` pairs, where ``selection`` may be ``None`` to
    execute the codec-owned threshold selection. Sending ``None`` finalizes
    and returns the payload. The incremental surface lets the executor feed
    the exact full-view threshold selection already produced by the joint
    selector/shared-code evaluator without retaining an evaluation split.

    The codec remains the sole owner of threshold selection, trusted packet
    events, q-chunk decoding, rate arithmetic, transformed SSE, sequence
    grouping, and bootstrap order.  An observer can synchronously consume the
    very same event stream and decoded chunks for raw-space endpoints without
    another encode/decode traversal.
    """
    spec = codec.spec
    b = model.cfg.block_dim
    S = model.cfg.n_sites
    inc = codec._tensor_on("included", device)
    mu = codec._tensor_on("calib_mean", device, dtype=torch.float32)
    log2_1mq_total = float(codec.bernoulli_log2q[codec.included].double().sum())

    rows_err = {q: [] for q in spec.qs}  # per-row sq err (pooled over sites)
    rows_err_site = {q: [] for q in spec.qs}  # per-row [S]
    rows_tot: list[float] = []
    rows_tot_site: list[torch.Tensor] = []
    rows_bits_sup: list[float] = []
    rows_bits_bern: list[float] = []
    rows_counts: list[float] = []
    rows_n: list[int] = []

    # Rolling sequence assembly across batch boundaries.
    pend = {
        "err": {q: torch.zeros(S, dtype=torch.float64) for q in spec.qs},
        "tot": torch.zeros(S, dtype=torch.float64),
        "sup": 0.0,
        "bern": 0.0,
        "cnt": 0.0,
        "n": 0,
    }

    def close_row() -> None:
        if pend["n"] <= 0:
            raise RuntimeError("cannot close an empty sequence")
        for q in spec.qs:
            rows_err[q].append(float(pend["err"][q].sum()))
            rows_err_site[q].append(pend["err"][q].clone())
            pend["err"][q].zero_()
        rows_tot.append(float(pend["tot"].sum()))
        rows_tot_site.append(pend["tot"].clone())
        pend["tot"].zero_()
        rows_bits_sup.append(pend["sup"])
        rows_bits_bern.append(pend["bern"])
        rows_counts.append(pend["cnt"])
        rows_n.append(pend["n"])
        pend["sup"] = pend["bern"] = pend["cnt"] = 0.0
        pend["n"] = 0

    excluded_events = torch.zeros((), dtype=torch.int64, device=device)
    total_events = torch.zeros((), dtype=torch.int64, device=device)
    sequence_mode: str | None = None
    current_sequence: int | None = None
    fallback_token_offset = 0
    materialized_decoder, materialized_encoder = _materialized_model_tensors(
        model,
        materialized_decoder,
        materialized_encoder,
    )
    if score_geometry is None and materialized_decoder is not None:
        score_geometry = model._frozen_score_geometry(materialized_decoder)
    materialized_decoder_matrix = (
        None
        if materialized_decoder is None
        else materialized_decoder.permute(1, 2, 0, 3).reshape(
            model.cfg.n_blocks * model.cfg.block_dim,
            model.cfg.n_sites * model.cfg.d_model,
        )
    )
    while True:
        driven = yield
        if driven is None:
            break
        item, out = driven
        observer_context: object | None = None
        source_row_ids: torch.Tensor | None = None
        if isinstance(item, _RDEvaluationInput):
            x = item.transformed
            source_row_ids = item.row_ids
            observer_context = item.context
        elif isinstance(item, tuple):
            if len(item) != 2:
                raise ValueError("R-D batches must be x or (x, row_ids)")
            x, source_row_ids = item
        else:
            x = item

        if source_row_ids is not None:
            if sequence_mode == "fixed_length_fallback":
                raise ValueError("cannot mix stored IDs and fixed-length fallback")
            sequence_mode = "stored_sequence_ids"
            if (
                source_row_ids.ndim != 2
                or source_row_ids.shape[0] != x.shape[0]
                or source_row_ids.shape[1] < 1
            ):
                raise ValueError("row_ids must have shape [tokens, >=1]")
            sequence_ids = source_row_ids[:, 0].to(device="cpu", dtype=torch.int64)
        else:
            if sequence_mode == "stored_sequence_ids":
                raise ValueError("cannot mix stored IDs and fixed-length fallback")
            sequence_mode = "fixed_length_fallback"
            if row_len is None or row_len <= 0:
                raise ValueError("tensor-only R-D batches require positive row_len")
            sequence_ids = (
                torch.arange(
                    fallback_token_offset,
                    fallback_token_offset + x.shape[0],
                    dtype=torch.int64,
                )
                // row_len
            )
            fallback_token_offset += x.shape[0]
        x = x.to(device, torch.float32, non_blocking=True)
        if out is None:
            out = _threshold_select(
                model,
                x,
                materialized_decoder,
                materialized_encoder,
                score_geometry,
            )
        elif (
            not isinstance(out, (BSCSelection, _RDEvaluationSelection))
            or out.z.shape != (x.shape[0], model.cfg.n_blocks, b)
            or out.scores.shape != out.mask.shape
            or out.mask.shape != (x.shape[0], model.cfg.n_blocks)
            or out.z.device != x.device
            or out.scores.device != x.device
            or out.mask.device != x.device
            or (
                isinstance(out, BSCSelection)
                and (
                    out.z_selected.shape != out.z.shape
                    or out.z_selected.device != x.device
                )
            )
        ):
            raise ValueError("precomputed R-D threshold selection is misbound")
        raw_mask = out.mask
        mask = raw_mask & inc.unsqueeze(0)
        support = _packet_support(mask)
        counts = support.counts
        raw_event_count = raw_mask.sum()
        total_events += raw_event_count
        excluded_events += raw_event_count - counts.sum()

        # Non-operational support-rate sensitivities, per token.  The exact
        # fixed-width packet rate is assembled below from count and ID widths.
        if codec.n_included:
            mask_fp32 = mask.float()
            act_p = (
                (codec._tensor_on("bernoulli_log2p", device) * mask_fp32)
                .sum(dim=1)
                .double()
            )
            act_q = (
                (codec._tensor_on("bernoulli_log2q", device) * mask_fp32)
                .sum(dim=1)
                .double()
            )
        else:
            act_p = torch.zeros(x.shape[0], dtype=torch.float64, device=x.device)
            act_q = torch.zeros_like(act_p)

        if isinstance(out, _RDEvaluationSelection):
            packet_events = _packet_events_from_output(
                model,
                codec,
                out,
                support=support,
                _selected_code=out.z,
            )
        else:
            packet_events = _packet_events_from_output(
                model,
                codec,
                out,
                support=support,
            )
        del out, raw_mask, mask, support
        if codec.n_included:
            del mask_fp32
        batch = _RDEvaluationBatch(
            transformed=x,
            sequence_ids=sequence_ids,
            row_ids=source_row_ids,
            packet_events=packet_events,
            context=observer_context,
            decoder=materialized_decoder,
            decoder_matrix=materialized_decoder_matrix,
        )
        if observer is not None:
            observer.begin_batch(batch)
        err_site_device = {}
        for decoded_chunk in _decode_trusted_packet_events_q_chunks(
            model,
            codec,
            packet_events,
            qs=spec.qs,
            _decoder=materialized_decoder,
            _decoder_matrix=materialized_decoder_matrix,
        ):
            for q, xhat in decoded_chunk.items():
                # Distortion uses the exact integer packet a saved artifact
                # will decode. Only redundant validation is elided here.
                err_site_device[q] = (x - xhat).double().pow(2).sum(dim=2)
            if observer is not None:
                observer.consume_decoded_chunk(batch, decoded_chunk)
            del xhat, decoded_chunk
        tot_site_device = (x - mu).double().pow(2).sum(dim=2)

        # Queue every q/raw consumer before the one blocking D2H transfer. The
        # packed fp64 matrix preserves each existing reduction and transfers
        # counts exactly while collapsing Q+4 synchronization points to one.
        metric_host = torch.cat(
            (
                counts.double().unsqueeze(1),
                act_p.unsqueeze(1),
                act_q.unsqueeze(1),
                *(err_site_device[q] for q in spec.qs),
                tot_site_device,
            ),
            dim=1,
        ).cpu()
        counts_host = metric_host[:, 0].to(torch.int64)
        if codec.n_included:
            sup_bits = -codec.log2_count_prob(counts_host).double() + _log2_binom(
                codec.n_included, counts_host
            )
            bern_bits = -(metric_host[:, 1] + (log2_1mq_total - metric_host[:, 2]))
        else:
            sup_bits = torch.zeros(x.shape[0], dtype=torch.float64)
            bern_bits = torch.zeros(x.shape[0], dtype=torch.float64)
        metric_offset = 3
        err_site = {}
        for q in spec.qs:
            err_site[q] = metric_host[:, metric_offset : metric_offset + S]
            metric_offset += S
        tot_site = metric_host[:, metric_offset : metric_offset + S]

        if sequence_mode == "fixed_length_fallback" and row_len == 1:
            for q in spec.qs:
                rows_err[q].extend(err_site[q].sum(dim=1).tolist())
                rows_err_site[q].extend(err_site[q].unbind(dim=0))
            rows_tot.extend(tot_site.sum(dim=1).tolist())
            rows_tot_site.extend(tot_site.unbind(dim=0))
            rows_bits_sup.extend(sup_bits.tolist())
            rows_bits_bern.extend(bern_bits.tolist())
            rows_counts.extend(counts_host.tolist())
            rows_n.extend([1] * x.shape[0])
        else:
            # Assemble exact stored sequences (or the labelled synthetic fallback).
            unique_sequences, run_counts = torch.unique_consecutive(
                sequence_ids, return_counts=True
            )
            start = 0
            for sequence_tensor, run_count_tensor in zip(
                unique_sequences, run_counts, strict=True
            ):
                sequence = int(sequence_tensor)
                run_count = int(run_count_tensor)
                if current_sequence is None:
                    current_sequence = sequence
                elif sequence != current_sequence:
                    if sequence <= current_sequence:
                        raise ValueError(
                            "sequence IDs must be contiguous and strictly increasing"
                        )
                    close_row()
                    current_sequence = sequence
                sl = slice(start, start + run_count)
                for q in spec.qs:
                    pend["err"][q] += err_site[q][sl].sum(dim=0)
                pend["tot"] += tot_site[sl].sum(dim=0)
                pend["sup"] += float(sup_bits[sl].sum())
                pend["bern"] += float(bern_bits[sl].sum())
                pend["cnt"] += float(counts_host[sl].sum())
                pend["n"] += run_count
                start += run_count
        if observer is not None:
            observer.end_batch(batch)
        # A suspended coroutine retains every live local. Release all
        # batch-owned CPU/GPU carriers before yielding control back to the
        # selector/shared-code driver so no batch crosses the callback seam.
        del (
            driven,
            item,
            x,
            source_row_ids,
            observer_context,
            sequence_ids,
            counts,
            raw_event_count,
            act_p,
            act_q,
            packet_events,
            batch,
            err_site_device,
            tot_site_device,
            metric_host,
            counts_host,
            sup_bits,
            bern_bits,
            err_site,
            tot_site,
        )
    if current_sequence is not None:
        close_row()

    n_rows = len(rows_tot)
    if n_rows == 0:
        raise ValueError("R-D evaluation stream is empty")
    tot = torch.tensor(rows_tot, dtype=torch.float64)
    tot_site_t = torch.stack(rows_tot_site)  # [rows, S]
    n_tok = torch.tensor(rows_n, dtype=torch.float64)
    sup = torch.tensor(rows_bits_sup, dtype=torch.float64)
    bern = torch.tensor(rows_bits_bern, dtype=torch.float64)
    cnt = torch.tensor(rows_counts, dtype=torch.float64)
    generator = torch.Generator().manual_seed(spec.bootstrap_seed)
    site_denominator = tot_site_t.sum(dim=0)
    site_fvu_defined = site_denominator > 0

    def site_fvu_payload(numerator: torch.Tensor) -> list[float | None]:
        values = numerator / site_denominator.clamp_min(1e-30)
        return [
            float(value) if bool(defined) else None
            for value, defined in zip(values, site_fvu_defined, strict=True)
        ]

    def boot(num: torch.Tensor, den: torch.Tensor) -> tuple[float, float]:
        if spec.n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be positive")
        ratios: list[torch.Tensor] = []
        remaining = spec.n_bootstrap
        # Keep Phase-3 uncertainty bounded in memory: ordinary sequence
        # bootstrap, chunked only over replicate rows.
        while remaining:
            replicates = min(8, remaining)
            idx = torch.randint(
                0,
                n_rows,
                (replicates, n_rows),
                generator=generator,
            )
            ratios.append(num[idx].sum(dim=1) / den[idx].sum(dim=1).clamp_min(1e-30))
            remaining -= replicates
        r = torch.cat(ratios)
        lo_v, hi_v = torch.quantile(r, torch.tensor([0.025, 0.975], dtype=r.dtype))
        return float(lo_v), float(hi_v)

    count_width = 0 if codec.n_included == 0 else codec.n_included.bit_length()
    id_width = 0 if codec.n_included <= 1 else (codec.n_included - 1).bit_length()
    fixed_support = n_tok * count_width + cnt * id_width
    results: dict = {
        "rate_model": "fixed_width_decodable_payload_bits_v1",
        "sensitivity_rate_model": "ideal_enumerative_and_bernoulli_logical_bits_v1",
        # Codec-local FVU diagnoses quantization in the bound transformed
        # activation view. Real-model selection is computed separately after
        # applying the deployable inverse to the original activation space.
        "distortion_space": "transformed_activation_view",
        "fvu_definition": "sse_over_centered_total_in_transformed_view",
        "packet_roundtrip_validated": True,
        "n_rows": n_rows,
        "n_tokens": int(n_tok.sum()),
        "sequence_grouping": sequence_mode,
        "row_len": row_len if sequence_mode == "fixed_length_fallback" else None,
        "avg_count": float(cnt.sum() / n_tok.sum()),
        "codec_meta": dict(codec.meta),
        "eval_excluded_event_share": float(excluded_events / total_events.clamp_min(1)),
        "support_count_width_bits": count_width,
        "support_id_width_bits": id_width,
        "support_bits_per_token": float(fixed_support.sum() / n_tok.sum()),
        "support_bits_ci95": boot(fixed_support, n_tok),
        "ideal_enumerative_support_bits_per_token": float(sup.sum() / n_tok.sum()),
        "ideal_enumerative_support_bits_ci95": boot(sup, n_tok),
        "bernoulli_bits_per_token": float(bern.sum() / n_tok.sum()),
        "bernoulli_bits_ci95": boot(bern, n_tok),
        "zero_rate": {
            "reconstruction": "calibration_fit_per_site_mean",
            "fvu_pooled": 1.0,
            "fvu_per_site": [
                1.0 if bool(defined) else None for defined in site_fvu_defined
            ],
            "fvu_per_site_defined": site_fvu_defined.tolist(),
            "payload_bits_per_token": 0.0,
        },
        "points": {},
    }
    for q in spec.qs:
        err = torch.tensor(rows_err[q], dtype=torch.float64)
        err_site_t = torch.stack(rows_err_site[q])  # [rows, S]
        amp_bits = float(q * b * cnt.sum() / n_tok.sum())
        fvu_lo, fvu_hi = boot(err, tot)
        rate_numerator = fixed_support + q * b * cnt
        ideal_rate = results["ideal_enumerative_support_bits_per_token"] + amp_bits
        rate = float(rate_numerator.sum() / n_tok.sum())
        results["points"][str(q)] = {
            "q": q,
            "fvu_pooled": float(err.sum() / tot.sum()),
            "fvu_ci95": [fvu_lo, fvu_hi],
            "fvu_per_site": site_fvu_payload(err_site_t.sum(dim=0)),
            "fvu_per_site_defined": site_fvu_defined.tolist(),
            "amplitude_bits_per_token": amp_bits,
            "rate_bits_per_token": rate,
            "rate_bits_ci95": list(boot(rate_numerator, n_tok)),
            "ideal_enumerative_rate_bits_per_token": ideal_rate,
            "rate_bits_bernoulli": results["bernoulli_bits_per_token"] + amp_bits,
        }
    return results


class _RDEvaluationSession:
    """Single-owner incremental facade over the exact R-D reduction stream."""

    @torch.no_grad()
    def __init__(
        self,
        model,
        codec: Codec,
        *,
        row_len: int | None = None,
        device: str = "cpu",
        observer: _RDEvaluationObserver | None = None,
        materialized_decoder: torch.Tensor | None = None,
        materialized_encoder: torch.Tensor | None = None,
        score_geometry=None,
    ) -> None:
        self._coroutine = _rd_evaluation_coroutine(
            model,
            codec,
            row_len=row_len,
            device=device,
            observer=observer,
            materialized_decoder=materialized_decoder,
            materialized_encoder=materialized_encoder,
            score_geometry=score_geometry,
        )
        self._finished = False
        next(self._coroutine)

    @torch.no_grad()
    def consume(
        self,
        item,
        *,
        threshold_selection: BSCSelection | _RDEvaluationSelection | None = None,
    ) -> None:
        if self._finished:
            raise RuntimeError("R-D evaluation session is already finalized")
        self._coroutine.send((item, threshold_selection))

    @torch.no_grad()
    def finalize(self) -> dict:
        if self._finished:
            raise RuntimeError("R-D evaluation session is already finalized")
        self._finished = True
        try:
            self._coroutine.send(None)
        except StopIteration as stopped:
            return stopped.value
        raise RuntimeError("R-D evaluation coroutine did not terminate")

    def close(self) -> None:
        """Release an unfinished stream after an upstream evaluation failure."""

        if not self._finished:
            self._finished = True
            self._coroutine.close()


@torch.no_grad()
def _evaluate_rd_stream(
    model,
    codec: Codec,
    batches,
    *,
    row_len: int | None = None,
    device: str = "cpu",
    observer: _RDEvaluationObserver | None = None,
) -> dict:
    """Traverse threshold packets once for transformed and observed endpoints.

    Real-data callers must yield ``(x, row_ids)`` pairs from the sequential
    store reader; column zero is the immutable sequence ID. Tensor-only
    batches require ``row_len`` and are labelled as the fixed-length
    synthetic/test fallback. ``_RDEvaluationInput`` additionally lets the
    executor attach paired raw-space state. Mixing stored-ID and fallback
    contracts is rejected.
    """

    session = _RDEvaluationSession(
        model,
        codec,
        row_len=row_len,
        device=device,
        observer=observer,
    )
    for item in batches:
        session.consume(item)
    return session.finalize()


@torch.no_grad()
def evaluate_rd(
    model,
    codec: Codec,
    batches,
    *,
    row_len: int | None = None,
    device: str = "cpu",
) -> dict:
    """Evaluate transformed-space packet rate and distortion.

    This focused public surface intentionally exposes no observer.  The
    executor uses :func:`_evaluate_rd_stream` when it also needs paired
    raw-space endpoints from the identical trusted traversal.
    """

    return _evaluate_rd_stream(
        model,
        codec,
        batches,
        row_len=row_len,
        device=device,
    )
