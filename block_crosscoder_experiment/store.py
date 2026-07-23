"""Content-addressed raw and normalized activation stores.

Three pieces are shared by every real-data phase:

- ``WhitenerAccumulator`` / ``Whitener`` — the *training-side harvest-fit*
  whitener (not saklas's consumer-side neutral-fit ``LayerWhitener``; the
  two are never interchangeable). Per site: mean and covariance accumulated
  in fp64 from fp32 batches, ridge per the saklas convention
  (λ_s = mean-diag(Σ̂_s) × DEFAULT_RIDGE_SCALE), eigendecomposition in
  fp64, frozen W_s = (Σ_s + λ_s I)^{-1/2}. Immutable once fit: the exact
  μ, W, ridge, site list, and source manifest are hashed, and that hash
  rides in every shard header; mismatches are rejected at load.
- ``ShardWriter`` — atomic safetensors shards ([token, site, d] bf16,
  sequence-contiguous), whitener hash + content checksum in each header,
  free-space abort *before* every write, per-shard finiteness/zero-row
  audit at write time. fp16 is forbidden in this path.
- ``StoreReader`` — the only sanctioned access pattern: per-epoch
  shard-level shuffle, contiguous chunk reads into a RAM shuffle buffer,
  batches mixed within the buffer, permutation seed recorded (and shared
  by BSC and baseline runs). Token-random mmap access is deliberately
  not implemented. ``sequential_batches`` streams a split in stored order
  for eval (the eval store is never assumed RAM-resident).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch

from .serialization import TYPED_PAYLOAD_DIGEST_CONTRACT, tensor_payload_digest
from .durability import durable_mkdir

__all__ = [
    "cuda_prefetch_batches",
    "DEFAULT_RIDGE_SCALE",
    "NORMALIZATION_MODES",
    "Whitener",
    "WhitenerAccumulator",
    "ShardWriter",
    "StoreReader",
    "prefetch_batches",
]

DEFAULT_RIDGE_SCALE = 1.0
NORMALIZATION_MODES = (
    "none",
    "sqrt_d",
    "scalar_rms",
    "layer",
    "whiten",
)
FORBIDDEN_DTYPES = (torch.float16,)
STORE_DTYPE = torch.bfloat16
MANIFEST_NAME = "split.json"
STORE_FORMAT_VERSION = 3
ROW_IDS_DTYPE = torch.int64
ROW_IDS_DTYPE_NAME = "int64"
WHITENER_ARTIFACT_SCHEMA = "bsc-whitener-artifact-v1"
WHITENER_CONTENT_SCHEMA = "bsc-whitener-content-v3"
WHITENER_PAYLOAD_KEYS = frozenset(
    {
        "schema",
        "digest_contract",
        "mean",
        "W",
        "ridge",
        "eigenvalues",
        "sites",
        "n_fit_tokens",
        "meta",
        "hash",
    }
)


def _fsync_file(path: Path) -> None:
    """Flush a completed temporary artifact before publishing its name."""
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes after an atomic replacement."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _tensor_byte_view(tensor: torch.Tensor) -> memoryview:
    """Zero-copy byte view for hashing a contiguous CPU tensor."""
    return memoryview(tensor.contiguous().view(torch.uint8).numpy())


@dataclass(frozen=True, slots=True)
class _ShardWriteContract:
    """Immutable inputs shared with the one-deep persistence worker."""

    directory: Path
    split: str
    whitener_hash: str
    sites: tuple[int, ...]
    d_model: int
    meta_json: str
    free_space_floor_frac: float
    max_zero_row_frac: float


@dataclass(frozen=True, slots=True)
class _ShardWriteResult:
    """Immutable durable-file evidence returned by the worker."""

    file: str
    index: int
    n_tokens: int
    content_sha256: str
    row_ids_sha256: str
    row_id_width: int


@dataclass(slots=True)
class _PendingShardWrite:
    """One worker future and its private candidate ordered-stream states."""

    future: Future[_ShardWriteResult]
    content_stream_hasher: Any
    row_stream_hasher: Any


def _persist_shard(
    contract: _ShardWriteContract,
    index: int,
    acts: torch.Tensor,
    row_ids: torch.Tensor,
    content_stream_hasher: Any,
    row_stream_hasher: Any,
) -> _ShardWriteResult:
    """Audit and durably publish one detached shard without writer mutation."""

    # This is deliberately the worker's first operation. Bad tensors may be
    # detected one producer batch later, but they can never reach a published
    # path or an ordered-stream digest installed by the live writer.
    if not bool(torch.isfinite(acts).all()):
        raise ValueError("non-finite activations reached the shard writer")
    zero_rows = (~(acts != 0).any(dim=(1, 2))).float().mean()
    if float(zero_rows) > contract.max_zero_row_frac:
        raise ValueError(
            f"zero-row fraction {float(zero_rows):.2e} exceeds "
            f"{contract.max_zero_row_frac:.0e} — suspect the capture path"
        )

    from safetensors.torch import save_file

    nbytes = (
        acts.numel() * acts.element_size() + row_ids.numel() * row_ids.element_size()
    )
    usage = shutil.disk_usage(contract.directory)
    if usage.free - nbytes < contract.free_space_floor_frac * usage.total:
        raise RuntimeError(
            f"write would breach the {contract.free_space_floor_frac:.0%} "
            f"free-space floor ({usage.free / 1e9:.1f} GB free, "
            f"shard {nbytes / 1e9:.2f} GB)"
        )

    content_bytes = _tensor_byte_view(acts)
    row_bytes = _tensor_byte_view(row_ids)
    checksum = hashlib.sha256(content_bytes).hexdigest()
    row_checksum = hashlib.sha256(row_bytes).hexdigest()
    # Candidate states belong exclusively to this worker. They are installed
    # by the producer only after both the shard and its directory entry are
    # durable; on any failure the candidates are simply discarded.
    content_stream_hasher.update(content_bytes)
    row_stream_hasher.update(row_bytes)
    header = {
        "whitener_hash": contract.whitener_hash,
        "split": contract.split,
        "shard_index": str(index),
        "n_tokens": str(acts.shape[0]),
        "sites": json.dumps(list(contract.sites)),
        "d_model": str(contract.d_model),
        "dtype": "bfloat16",
        "content_sha256": checksum,
        "row_ids_sha256": row_checksum,
        "row_id_width": str(row_ids.shape[1]),
        "row_ids_dtype": ROW_IDS_DTYPE_NAME,
        "meta": contract.meta_json,
    }
    path = contract.directory / f"shard_{index:05d}.safetensors"
    tmp = path.with_suffix(".tmp")
    save_file(
        {"acts": acts, "row_ids": row_ids},
        tmp,
        metadata=header,
    )
    _fsync_file(tmp)
    os.replace(tmp, path)
    _fsync_directory(contract.directory)
    return _ShardWriteResult(
        file=path.name,
        index=index,
        n_tokens=int(acts.shape[0]),
        content_sha256=checksum,
        row_ids_sha256=row_checksum,
        row_id_width=int(row_ids.shape[1]),
    )


class WhitenerAccumulator:
    """fp64 sufficient statistics for the per-site whitener.

    Batches arrive as [n, S, d] (any float dtype except fp16); statistics
    are accumulated per batch in fp64. Covariance GEMMs run in fp64, which sidesteps
    TF32 entirely.
    """

    def __init__(
        self,
        n_sites: int,
        d_model: int,
        device: torch.device | str = "cpu",
        *,
        track_covariance: bool = True,
    ) -> None:
        if n_sites <= 0 or d_model <= 0:
            raise ValueError("normalization dimensions must be positive")
        self.n = 0
        self.sum = torch.zeros(n_sites, d_model, dtype=torch.float64, device=device)
        self.sum_squares = torch.zeros(
            n_sites, d_model, dtype=torch.float64, device=device
        )
        self.track_covariance = bool(track_covariance)
        self.outer = (
            torch.zeros(n_sites, d_model, d_model, dtype=torch.float64, device=device)
            if self.track_covariance
            else None
        )

    def update(self, x: torch.Tensor) -> None:
        """x: [n, S, d] raw activations."""
        if x.dtype in FORBIDDEN_DTYPES:
            raise TypeError("fp16 is forbidden in the harvest/store path")
        if x.ndim != 3 or tuple(x.shape[1:]) != tuple(self.sum.shape):
            raise ValueError(
                f"normalization batch must have shape [n, {self.sum.shape[0]}, "
                f"{self.sum.shape[1]}]"
            )
        if x.shape[0] <= 0:
            raise ValueError("normalization batch must be nonempty")
        x64 = x.to(device=self.sum.device, dtype=torch.float64)
        self.n += x64.shape[0]
        self.sum += x64.sum(dim=0)
        self.sum_squares += x64.square().sum(dim=0)
        if self.outer is not None:
            self.outer += torch.einsum("nsd,nse->sde", x64, x64)

    def merge(self, other: "WhitenerAccumulator") -> "WhitenerAccumulator":
        if (
            self.sum.shape != other.sum.shape
            or self.track_covariance != other.track_covariance
        ):
            raise ValueError("normalization accumulators have incompatible contracts")
        out = WhitenerAccumulator(
            self.sum.shape[0],
            self.sum.shape[1],
            self.sum.device,
            track_covariance=self.track_covariance,
        )
        out.n = self.n + other.n
        out.sum = self.sum + other.sum.to(self.sum.device)
        out.sum_squares = self.sum_squares + other.sum_squares.to(self.sum.device)
        if self.outer is not None:
            assert other.outer is not None and out.outer is not None
            out.outer = self.outer + other.outer.to(self.outer.device)
        return out

    def finalize(
        self,
        *,
        sites: Sequence[int],
        meta: dict,
        ridge_scale: float = DEFAULT_RIDGE_SCALE,
        site_renorm: bool = False,
        mode: str = "whiten",
        mean_centered_norm: torch.Tensor | None = None,
    ) -> "Whitener":
        """Fit and freeze one declared normalization transform in fp64.

        ``none`` stores raw activations. ``sqrt_d`` mean-centres and scales the
        dataset mean norm to ``sqrt(d_s)``. ``scalar_rms`` mean-centres and
        applies one RMS scalar per site. ``layer``
        applies token-wise LayerNorm over each site's valid coordinates.
        ``whiten`` uses the shrinkage-covariance transform.
        ``site_renorm`` is a separate, whiten-only factor which folds the
        calibration-fit per-site RMS scalar into ``W``.
        """
        if mode not in NORMALIZATION_MODES:
            raise ValueError(f"mode must be one of {NORMALIZATION_MODES}, got {mode!r}")
        resolved_sites = tuple(int(site) for site in sites)
        if len(resolved_sites) != self.sum.shape[0] or len(set(resolved_sites)) != len(
            resolved_sites
        ):
            raise ValueError("sites must uniquely identify every normalization site")
        if ridge_scale < 0:
            raise ValueError("ridge_scale cannot be negative")
        if site_renorm and mode != "whiten":
            raise ValueError("site_renorm is defined only for mode='whiten'")
        if self.n < 2:
            raise ValueError("normalization fit needs at least 2 tokens")
        fitted_mean = (self.sum / self.n).cpu()
        coordinate_variance = (
            self.sum_squares.cpu() / self.n - fitted_mean.square()
        ).clamp_min(0.0)
        n_sites, d = fitted_mean.shape
        site_dims = tuple(int(v) for v in meta.get("site_dims", (d,) * n_sites))
        if len(site_dims) != n_sites or any(v <= 0 or v > d for v in site_dims):
            raise ValueError("meta.site_dims must describe every padded site")
        eye = torch.eye(d, dtype=torch.float64).expand(n_sites, d, d).clone()
        # ``eigenvalues`` is retained in the artifact schema for whitening
        # diagnostics.  Diagonal modes store their exact coordinate variances;
        # they never materialize or diagonalize a covariance matrix.
        diagnostic_values = torch.ones(n_sites, d, dtype=torch.float64)
        for s, width in enumerate(site_dims):
            diagnostic_values[s, :width] = coordinate_variance[s, :width]

        if mode == "whiten":
            if self.outer is None:
                raise ValueError("whiten requires track_covariance=True")
            cov = self.outer.cpu() / self.n - torch.einsum(
                "sd,se->sde", fitted_mean, fitted_mean
            )
            mean = fitted_mean
            ridge = torch.zeros(n_sites, dtype=torch.float64)
            W = eye.clone()
            eigs = torch.ones(n_sites, d, dtype=torch.float64)
            for s, width in enumerate(site_dims):
                active_cov = cov[s, :width, :width]
                ridge[s] = active_cov.diagonal().mean() * ridge_scale
                e, V = torch.linalg.eigh(
                    active_cov + ridge[s] * torch.eye(width, dtype=torch.float64)
                )
                eigs[s, :width] = e
                W[s, :width, :width] = (V * e.clamp_min(1e-12).rsqrt()) @ V.T
        elif mode == "scalar_rms":
            mean = fitted_mean
            ridge = torch.zeros(n_sites, dtype=torch.float64)
            eigs = diagnostic_values
            W = eye.clone()
            for s, width in enumerate(site_dims):
                rms = coordinate_variance[s, :width].mean().clamp_min(1e-12).sqrt()
                W[s, :width, :width] *= rms.reciprocal()
        elif mode == "sqrt_d":
            if mean_centered_norm is None:
                raise ValueError(
                    "sqrt_d requires an exact second-pass mean_centered_norm"
                )
            mean_centered_norm = mean_centered_norm.double().cpu()
            if mean_centered_norm.shape != (n_sites,) or not torch.all(
                mean_centered_norm > 0
            ):
                raise ValueError(
                    "mean_centered_norm must be positive with shape [sites]"
                )
            mean = fitted_mean
            ridge = torch.zeros(n_sites, dtype=torch.float64)
            eigs = diagnostic_values
            W = eye.clone()
            for s, width in enumerate(site_dims):
                W[s, :width, :width] *= math.sqrt(width) / mean_centered_norm[s]
        else:
            # Raw and LayerNorm stores are deliberately not dataset-centred.
            # LayerNorm performs token-local centring/scaling in ``apply``.
            mean = torch.zeros_like(fitted_mean)
            ridge = torch.zeros(n_sites, dtype=torch.float64)
            eigs = diagnostic_values
            W = eye
        frozen_meta = dict(meta)
        frozen_meta["normalization"] = mode
        frozen_meta["normalization_fit_tokens"] = self.n
        frozen_meta["site_dims"] = list(site_dims)
        frozen_meta["layer_norm_eps"] = 1e-5
        if mode == "sqrt_d":
            frozen_meta["mean_centered_norm"] = mean_centered_norm.tolist()
        if site_renorm:
            # Padded coordinates are bookkeeping, not observed dimensions.
            # Including their placeholder eigenvalues dilutes the shrinkage
            # correction for rectangular multi-site stores.
            retained = torch.stack(
                [
                    ((eigs[s, :width] - ridge[s]) / eigs[s, :width].clamp_min(1e-30))
                    .clamp_min(0.0)
                    .mean()
                    for s, width in enumerate(site_dims)
                ]
            )
            scalars = retained.clamp_min(1e-12).rsqrt()
            W *= scalars[:, None, None]
            frozen_meta["site_rms_renorm_folded"] = True
            frozen_meta["site_rms_scalars"] = scalars.tolist()
        return Whitener(
            mean=mean.float(),
            W=W.float(),
            ridge=ridge.float(),
            eigenvalues=eigs.float(),
            sites=resolved_sites,
            n_fit_tokens=self.n,
            meta=frozen_meta,
        )


@dataclass
class Whitener:
    """Frozen activation transform."""

    mean: torch.Tensor  # [S, d] fp32
    W: torch.Tensor  # [S, d, d] fp32
    ridge: torch.Tensor  # [S] fp32
    eigenvalues: torch.Tensor  # [S, d] fp32, regularized-covariance spectrum
    sites: tuple[int, ...]
    n_fit_tokens: int
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate_canonical()

    @staticmethod
    def _validate_json_value(value: Any, *, path: str) -> None:
        """Require one injective JSON-like representation for metadata."""

        if value is None or isinstance(value, (str, bool)):
            return
        if type(value) is int:
            return
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError(f"{path} must not contain non-finite floats")
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                Whitener._validate_json_value(child, path=f"{path}[{index}]")
            return
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise TypeError(f"{path} keys must be strings")
            for key, child in value.items():
                Whitener._validate_json_value(child, path=f"{path}.{key}")
            return
        raise TypeError(
            f"{path} contains unsupported {type(value).__module__}."
            f"{type(value).__qualname__}"
        )

    def _validate_canonical(self) -> None:
        tensors = {
            "mean": self.mean,
            "W": self.W,
            "ridge": self.ridge,
            "eigenvalues": self.eigenvalues,
        }
        for name, tensor in tensors.items():
            if not torch.is_tensor(tensor):
                raise TypeError(f"whitener {name} must be a tensor")
            if tensor.layout != torch.strided:
                raise TypeError(f"whitener {name} must be a dense strided tensor")
            if tensor.device.type != "cpu":
                raise TypeError(f"whitener {name} must be stored on CPU")
            if tensor.dtype != torch.float32:
                raise TypeError(f"whitener {name} must have dtype torch.float32")
            if not tensor.is_contiguous():
                raise TypeError(f"whitener {name} must be contiguous")
            if tensor.requires_grad:
                raise TypeError(f"whitener {name} must not require gradients")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"whitener {name} must contain only finite values")

        if self.mean.ndim != 2 or self.mean.shape[0] <= 0 or self.mean.shape[1] <= 0:
            raise ValueError("whitener mean must have nonempty shape [sites, d_model]")
        n_sites, d_model = (int(value) for value in self.mean.shape)
        expected_shapes = {
            "W": (n_sites, d_model, d_model),
            "ridge": (n_sites,),
            "eigenvalues": (n_sites, d_model),
        }
        for name, expected in expected_shapes.items():
            if tuple(tensors[name].shape) != expected:
                raise ValueError(
                    f"whitener {name} has shape {tuple(tensors[name].shape)}, "
                    f"expected {expected}"
                )

        if not isinstance(self.sites, tuple) or len(self.sites) != n_sites:
            raise TypeError("whitener sites must be one tuple entry per site")
        if (
            any(type(site) is not int or site < 0 for site in self.sites)
            or len(set(self.sites)) != len(self.sites)
        ):
            raise ValueError("whitener sites must be unique non-negative integers")
        if type(self.n_fit_tokens) is not int or self.n_fit_tokens <= 0:
            raise ValueError("whitener n_fit_tokens must be a positive integer")
        if not isinstance(self.meta, dict):
            raise TypeError("whitener meta must be a JSON object")
        self._validate_json_value(self.meta, path="whitener meta")
        if self.mode not in NORMALIZATION_MODES:
            raise ValueError(f"unsupported whitener normalization mode {self.mode!r}")
        raw_site_dims = self.meta.get("site_dims")
        if (
            not isinstance(raw_site_dims, list)
            or len(raw_site_dims) != n_sites
            or any(type(width) is not int or width <= 0 or width > d_model for width in raw_site_dims)
        ):
            raise ValueError(
                "whitener meta.site_dims must give one positive padded width per site"
            )

    @property
    def mode(self) -> str:
        return str(self.meta.get("normalization", "whiten"))

    @property
    def site_dims(self) -> tuple[int, ...]:
        return tuple(
            int(v)
            for v in self.meta.get(
                "site_dims", (int(self.mean.shape[1]),) * len(self.sites)
            )
        )

    @property
    def hash(self) -> str:
        """Unambiguous typed digest of every transform-defining field."""

        self._validate_canonical()
        return tensor_payload_digest(
            {
                "schema": WHITENER_CONTENT_SCHEMA,
                "digest_contract": TYPED_PAYLOAD_DIGEST_CONTRACT,
                "mean": self.mean,
                "W": self.W,
                "ridge": self.ridge,
                "eigenvalues": self.eigenvalues,
                "sites": self.sites,
                "n_fit_tokens": self.n_fit_tokens,
                "meta": self.meta,
            }
        )

    def site_rms_scalars(self) -> torch.Tensor:
        """Additional site-RMS scaling required at load time.

        Transforms fit with ``site_renorm=True`` fold the scalar into ``W``
        and return ones here. Other whitening transforms derive scaling from the
        shrinkage spectrum: the whitened per-dim variance
        prediction is (e_j − λ_s)/e_j for e = eig(Σ+λI), so scaling site s
        by 1/sqrt(mean_j retained_j) restores ~unit mean per-dim power —
        directional rogue-dim suppression kept, equal total site power
        restored. Returns [S] fp32.
        """
        if self.mode != "whiten" or self.meta.get("site_rms_renorm_folded"):
            return torch.ones(len(self.sites), dtype=torch.float32)
        values = []
        for s, width in enumerate(self.site_dims):
            e = self.eigenvalues[s, :width].double()
            lam = self.ridge[s].double()
            retained = ((e - lam) / e).clamp_min(0.0).mean()
            values.append(retained.clamp_min(1e-12).rsqrt())
        return torch.stack(values).float()

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """x: [n, S, d] raw -> configured normalized coordinates in fp32."""
        if x.dtype in FORBIDDEN_DTYPES:
            raise TypeError("fp16 is forbidden in the harvest/store path")
        if self.mode == "layer":
            out = torch.zeros_like(x, dtype=torch.float32)
            for s, width in enumerate(self.site_dims):
                out[:, s, :width] = torch.nn.functional.layer_norm(
                    x[:, s, :width].float(),
                    (width,),
                    eps=float(self.meta.get("layer_norm_eps", 1e-5)),
                )
            return out
        mean = self.mean.to(x.device)
        W = self.W.to(x.device)
        if self.mode in {"none", "scalar_rms", "sqrt_d"}:
            scale = torch.diagonal(W, dim1=-2, dim2=-1).unsqueeze(0)
            return (x.float() - mean) * scale
        return torch.einsum("sde,nse->nsd", W, x.float() - mean)

    def unapply(self, xw: torch.Tensor) -> torch.Tensor:
        """Invert a fixed linear transform; token LayerNorm is non-invertible."""
        if self.mode == "layer":
            raise ValueError("token-wise LayerNorm is not invertible")
        if self.mode in {"none", "scalar_rms", "sqrt_d"}:
            scale = torch.diagonal(self.W.to(xw.device), dim1=-2, dim2=-1).unsqueeze(0)
            return xw.float() / scale.clamp_min(1e-30) + self.mean.to(xw.device)
        Winv = torch.linalg.inv(self.W.double()).float().to(xw.device)
        return torch.einsum("sde,nse->nsd", Winv, xw.float()) + self.mean.to(xw.device)

    def payload(self) -> dict[str, Any]:
        """Return the sole current serialized Whitener representation."""

        return {
            "schema": WHITENER_ARTIFACT_SCHEMA,
            "digest_contract": TYPED_PAYLOAD_DIGEST_CONTRACT,
            "mean": self.mean,
            "W": self.W,
            "ridge": self.ridge,
            "eigenvalues": self.eigenvalues,
            "sites": list(self.sites),
            "n_fit_tokens": self.n_fit_tokens,
            "meta": self.meta,
            "hash": self.hash,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.payload(), tmp)
        _fsync_file(tmp)
        os.replace(tmp, path)
        _fsync_directory(path.parent)

    @classmethod
    def load(cls, path: str | Path) -> "Whitener":
        p = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(p, dict) or set(p) != WHITENER_PAYLOAD_KEYS:
            raise ValueError(f"whitener payload has a noncanonical field set in {path}")
        if p.get("schema") != WHITENER_ARTIFACT_SCHEMA:
            raise ValueError(f"whitener payload has the wrong schema in {path}")
        if p.get("digest_contract") != TYPED_PAYLOAD_DIGEST_CONTRACT:
            raise ValueError(f"whitener payload has the wrong digest contract in {path}")
        if not isinstance(p.get("sites"), list):
            raise TypeError(f"whitener payload sites must be a list in {path}")
        w = cls(
            mean=p["mean"],
            W=p["W"],
            ridge=p["ridge"],
            eigenvalues=p["eigenvalues"],
            sites=tuple(p["sites"]),
            n_fit_tokens=p["n_fit_tokens"],
            meta=p["meta"],
        )
        if w.hash != p["hash"]:
            raise ValueError(f"whitener hash mismatch in {path} — file corrupted?")
        return w


class ShardWriter:
    """Sequence-contiguous whitened-bf16 shards with audited atomic writes.

    Layout: ``root/<split>/shard_00000.safetensors`` + a ``split.json``
    manifest. Every shard's safetensors metadata carries the whitener
    hash, token count, site list, and a sha256 content checksum.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        whitener_hash: str,
        sites: Sequence[int],
        d_model: int,
        meta: dict | None = None,
        tokens_per_shard: int = 150_000,
        free_space_floor_frac: float = 0.15,
        max_zero_row_frac: float = 1e-4,
        resume: bool = False,
        on_durable_shard: Callable[[int], None] | None = None,
    ) -> None:
        if not split or Path(split).name != split:
            raise ValueError("split must be one nonempty path component")
        resolved_sites = tuple(int(s) for s in sites)
        if not resolved_sites or len(set(resolved_sites)) != len(resolved_sites):
            raise ValueError("sites must be nonempty and unique")
        if d_model <= 0 or tokens_per_shard <= 0:
            raise ValueError("d_model and tokens_per_shard must be positive")
        if not 0.0 <= free_space_floor_frac < 1.0:
            raise ValueError("free_space_floor_frac must be in [0, 1)")
        if not 0.0 <= max_zero_row_frac <= 1.0:
            raise ValueError("max_zero_row_frac must be in [0, 1]")
        self.dir = Path(root) / split
        durable_mkdir(self.dir, parents=True, exist_ok=True)
        self.split = split
        self.whitener_hash = whitener_hash
        self.sites = resolved_sites
        self.d_model = d_model
        self.meta = dict(meta or {})
        self.tokens_per_shard = tokens_per_shard
        self.free_floor = free_space_floor_frac
        self.max_zero_row_frac = max_zero_row_frac
        self._on_durable_shard = on_durable_shard
        self._buffer: torch.Tensor | None = None
        self._row_id_buffer: torch.Tensor | None = None
        self._buffered = 0
        self._row_id_width: int | None = None
        self.shards: list[dict] = []
        self._content_stream_hasher = hashlib.sha256()
        self._row_stream_hasher = hashlib.sha256()
        self._next_row_id = 0
        self._executor: ThreadPoolExecutor | None = None
        self._pending: _PendingShardWrite | None = None
        self._poison: BaseException | None = None
        self._closed = False
        self._write_contract = _ShardWriteContract(
            directory=self.dir,
            split=self.split,
            whitener_hash=self.whitener_hash,
            sites=self.sites,
            d_model=self.d_model,
            meta_json=json.dumps(self.meta, sort_keys=True),
            free_space_floor_frac=self.free_floor,
            max_zero_row_frac=self.max_zero_row_frac,
        )
        existing = tuple(self.dir.iterdir())
        if resume:
            self._resume_existing_split()
        elif existing:
            raise ValueError(f"refusing nonempty split directory {self.dir}")

    @property
    def persisted_tokens(self) -> int:
        """Rows installed in the fsynced manifest, excluding pending I/O."""

        return sum(int(shard["n_tokens"]) for shard in self.shards)

    def _ensure_open(self) -> None:
        if self._poison is not None:
            raise RuntimeError("shard writer is poisoned by an earlier failure") from (
                self._poison
            )
        if self._closed:
            raise RuntimeError("shard writer is closed")

    def _worker(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"bsc-shard-{self.split}",
            )
        return self._executor

    def _shutdown_worker(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def _resume_existing_split(self) -> None:
        manifest_path = self.dir / MANIFEST_NAME
        if not manifest_path.is_file():
            entries = tuple(sorted(path.name for path in self.dir.iterdir()))
            first_shard = self.dir / "shard_00000.safetensors"
            if entries != (first_shard.name,):
                raise ValueError(
                    f"resumable split {self.dir} lacks its per-shard manifest; "
                    "recovery requires exactly one canonical first orphan shard"
                )
            self._adopt_orphan_shard(first_shard)
            return
        reader = StoreReader(
            self.dir.parent,
            self.split,
            allow_incomplete=True,
            _allow_unmanifested_tail=True,
        )
        reader.verify()
        manifest = reader.manifest
        expected = {
            "split": self.split,
            "whitener_hash": self.whitener_hash,
            "sites": list(self.sites),
            "d_model": self.d_model,
            "meta": self.meta,
            "tokens_per_shard": self.tokens_per_shard,
            "row_ids_dtype": ROW_IDS_DTYPE_NAME,
        }
        mismatches = {
            key: {"expected": value, "actual": manifest.get(key)}
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "resumable split contract changed: "
                + json.dumps(mismatches, sort_keys=True)
            )
        if manifest.get("complete") is True:
            raise ValueError(f"split {self.dir} is already complete")
        self.shards = [dict(item) for item in manifest["shards"]]
        self._row_id_width = int(manifest["row_id_width"])
        for record in self.shards:
            acts, row_ids = reader._shard_payload(record, verify=True)
            self._content_stream_hasher.update(_tensor_byte_view(acts))
            self._row_stream_hasher.update(_tensor_byte_view(row_ids))
        self._next_row_id = self.persisted_tokens
        expected_files = {record["file"] for record in self.shards}
        orphan_paths = tuple(
            sorted(
                (
                    path
                    for path in self.dir.glob("*.safetensors")
                    if path.name not in expected_files
                ),
                key=lambda path: path.name,
            )
        )
        if orphan_paths:
            # StoreReader's private recovery mode has already proved that this
            # is exactly the next canonical file and that no other shard files
            # are present.  Re-validate the complete header and payload before
            # making it part of the durable manifest.
            self._adopt_orphan_shard(orphan_paths[0])

    def _adopt_orphan_shard(self, path: Path) -> None:
        """Verify and manifest one shard published just before process death.

        The persistence worker publishes and fsyncs the safetensors file before advancing
        ``split.json``.  A crash in that narrow interval therefore leaves one
        canonical tail file.  Recovery accepts only that exact shape: the next
        shard index, the frozen writer contract, exact tensor/header sets, and
        checksums all have to agree.  The newly advanced incomplete manifest is
        itself atomically replaced and directory-fsynced by ``_write_manifest``.
        """

        from safetensors import safe_open

        index = len(self.shards)
        expected_name = f"shard_{index:05d}.safetensors"
        if path.parent != self.dir or path.name != expected_name:
            raise ValueError(
                f"orphan shard is not the next canonical tail: {path}; "
                f"expected {expected_name}"
            )
        expected_files = {record["file"] for record in self.shards} | {path.name}
        actual_files = {candidate.name for candidate in self.dir.glob("*.safetensors")}
        if actual_files != expected_files:
            raise ValueError(
                "orphan recovery requires exactly one canonical durable tail: "
                f"expected={sorted(expected_files)}, actual={sorted(actual_files)}"
            )
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                header = dict(handle.metadata())
                keys = set(handle.keys())
                if keys != {"acts", "row_ids"}:
                    raise ValueError(
                        f"orphan shard tensor set mismatch in {path}: {sorted(keys)}"
                    )
                acts = handle.get_tensor("acts")
                row_ids = handle.get_tensor("row_ids")
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"cannot read orphan shard {path}: {exc}") from exc

        header_keys = {
            "whitener_hash",
            "split",
            "shard_index",
            "n_tokens",
            "sites",
            "d_model",
            "dtype",
            "content_sha256",
            "row_ids_sha256",
            "row_id_width",
            "row_ids_dtype",
            "meta",
        }
        if set(header) != header_keys:
            raise ValueError(
                f"orphan shard header set mismatch in {path}: "
                f"expected={sorted(header_keys)}, actual={sorted(header)}"
            )
        try:
            n_tokens = int(header["n_tokens"])
            row_id_width = int(header["row_id_width"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"orphan shard has non-integer dimensions in {path}"
            ) from exc
        if (
            n_tokens <= 0
            or n_tokens > self.tokens_per_shard
            or header["n_tokens"] != str(n_tokens)
            or row_id_width <= 0
            or header["row_id_width"] != str(row_id_width)
        ):
            raise ValueError(f"orphan shard has invalid dimensions in {path}")
        if self._row_id_width is not None and row_id_width != self._row_id_width:
            raise ValueError(f"orphan shard row-id width changed in {path}")
        expected_header = {
            "whitener_hash": self.whitener_hash,
            "split": self.split,
            "shard_index": str(index),
            "sites": json.dumps(list(self.sites)),
            "d_model": str(self.d_model),
            "dtype": "bfloat16",
            "row_ids_dtype": ROW_IDS_DTYPE_NAME,
            "meta": json.dumps(self.meta, sort_keys=True),
        }
        mismatches = {
            key: {"expected": value, "actual": header.get(key)}
            for key, value in expected_header.items()
            if header.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"orphan shard header mismatch in {path}: "
                + json.dumps(mismatches, sort_keys=True)
            )
        if tuple(acts.shape) != (n_tokens, len(self.sites), self.d_model) or (
            acts.dtype != STORE_DTYPE
        ):
            raise ValueError(
                f"orphan shard payload mismatch in {path}: "
                f"shape={tuple(acts.shape)} dtype={acts.dtype}"
            )
        if tuple(row_ids.shape) != (n_tokens, row_id_width) or (
            row_ids.dtype != ROW_IDS_DTYPE
        ):
            raise ValueError(
                f"orphan shard row identity payload mismatch in {path}: "
                f"shape={tuple(row_ids.shape)} dtype={row_ids.dtype}"
            )
        if not bool(torch.isfinite(acts).all()):
            raise ValueError(f"orphan shard contains non-finite activations in {path}")
        zero_rows = (~(acts != 0).any(dim=(1, 2))).float().mean()
        if float(zero_rows) > self.max_zero_row_frac:
            raise ValueError(f"orphan shard exceeds the zero-row limit in {path}")

        content_bytes = _tensor_byte_view(acts)
        row_bytes = _tensor_byte_view(row_ids)
        content_sha256 = hashlib.sha256(content_bytes).hexdigest()
        row_ids_sha256 = hashlib.sha256(row_bytes).hexdigest()
        for label, observed in (
            ("content_sha256", content_sha256),
            ("row_ids_sha256", row_ids_sha256),
        ):
            claimed = header[label]
            if (
                len(claimed) != 64
                or any(character not in "0123456789abcdef" for character in claimed)
                or claimed != observed
            ):
                raise ValueError(f"orphan shard {label} mismatch in {path}")

        # Recheck the directory immediately before advancing the manifest.  A
        # concurrent or duplicated tail must never be silently hidden by an
        # otherwise valid first orphan.
        actual_files = {candidate.name for candidate in self.dir.glob("*.safetensors")}
        if actual_files != expected_files:
            raise ValueError(
                "orphan recovery became ambiguous during validation: "
                f"expected={sorted(expected_files)}, actual={sorted(actual_files)}"
            )

        self._row_id_width = row_id_width
        self._content_stream_hasher.update(content_bytes)
        self._row_stream_hasher.update(row_bytes)
        self.shards.append(
            {
                "file": path.name,
                "index": index,
                "n_tokens": n_tokens,
                "content_sha256": content_sha256,
                "row_ids_sha256": row_ids_sha256,
                "row_id_width": row_id_width,
                "row_ids_dtype": ROW_IDS_DTYPE_NAME,
            }
        )
        self._next_row_id = self.persisted_tokens
        self._write_manifest(complete=False)

    def add(self, x: torch.Tensor, row_ids: torch.Tensor | None = None) -> None:
        """Append activations and immutable row identities.

        ``row_ids`` is an int64 [n,k] tensor (for example sequence, position,
        token). Capture must supply it. The sequential default keeps small
        synthetic/tests convenient but is not sufficient provenance for a
        real-model capture.
        """
        self._ensure_open()
        if x.dtype in FORBIDDEN_DTYPES:
            raise TypeError("fp16 is forbidden in the harvest/store path")
        if x.ndim != 3 or x.shape[0] <= 0:
            raise ValueError("activation batches must be nonempty rank-3 tensors")
        if x.shape[1] != len(self.sites) or x.shape[2] != self.d_model:
            raise ValueError(f"shape {tuple(x.shape)} does not match store config")
        if row_ids is None:
            row_ids = torch.arange(
                self._next_row_id, self._next_row_id + x.shape[0], dtype=torch.int64
            ).view(-1, 1)
        elif row_ids.dtype != ROW_IDS_DTYPE:
            raise TypeError("row_ids must use int64 exactly")
        row_ids = row_ids.to(device="cpu")
        if row_ids.ndim != 2 or row_ids.shape[0] != x.shape[0] or row_ids.shape[1] <= 0:
            raise ValueError("row_ids must have shape [n, k]")
        if self._row_id_width is None:
            self._row_id_width = int(row_ids.shape[1])
        elif row_ids.shape[1] != self._row_id_width:
            raise ValueError("row_ids width changed within a split")
        values = x.to(device="cpu", dtype=STORE_DTYPE)
        self._next_row_id += x.shape[0]
        offset = 0
        while offset < len(values):
            if self._buffer is None:
                self._buffer = torch.empty(
                    self.tokens_per_shard,
                    len(self.sites),
                    self.d_model,
                    dtype=STORE_DTYPE,
                )
                self._row_id_buffer = torch.empty(
                    self.tokens_per_shard,
                    self._row_id_width,
                    dtype=ROW_IDS_DTYPE,
                )
            assert self._row_id_buffer is not None
            take = min(
                len(values) - offset,
                self.tokens_per_shard - self._buffered,
            )
            self._buffer[self._buffered : self._buffered + take].copy_(
                values[offset : offset + take]
            )
            self._row_id_buffer[self._buffered : self._buffered + take].copy_(
                row_ids[offset : offset + take]
            )
            self._buffered += take
            offset += take
            if self._buffered == self.tokens_per_shard:
                self._flush(self.tokens_per_shard)

    def _flush(self, n_tokens: int) -> None:
        if (
            self._buffer is None
            or self._row_id_buffer is None
            or n_tokens != self._buffered
        ):
            raise RuntimeError("shard flush must consume the complete staging buffer")
        out = self._buffer[:n_tokens]
        out_ids = self._row_id_buffer[:n_tokens]
        self._buffer = None
        self._row_id_buffer = None
        self._buffered = 0
        self._submit(out.detach(), out_ids.detach())

    def _submit(self, acts: torch.Tensor, row_ids: torch.Tensor) -> None:
        """Transfer one complete detached buffer to the one-deep worker."""

        self._ensure_open()
        if self._pending is not None:
            self.synchronize()
        index = len(self.shards)
        content_stream_hasher = self._content_stream_hasher.copy()
        row_stream_hasher = self._row_stream_hasher.copy()
        future = self._worker().submit(
            _persist_shard,
            self._write_contract,
            index,
            acts,
            row_ids,
            content_stream_hasher,
            row_stream_hasher,
        )
        self._pending = _PendingShardWrite(
            future=future,
            content_stream_hasher=content_stream_hasher,
            row_stream_hasher=row_stream_hasher,
        )

    def synchronize(self) -> int:
        """Install the pending durable shard into an fsynced manifest.

        At most one worker-owned shard exists. Until this barrier succeeds,
        ``persisted_tokens`` deliberately excludes it; a process death in that
        interval leaves the one exact orphan accepted by resume validation.
        """

        self._ensure_open()
        pending = self._pending
        if pending is None:
            return self.persisted_tokens
        try:
            result = pending.future.result()
            self._pending = None
            self._content_stream_hasher = pending.content_stream_hasher
            self._row_stream_hasher = pending.row_stream_hasher
            self.shards.append(
                {
                    "file": result.file,
                    "index": result.index,
                    "n_tokens": result.n_tokens,
                    "content_sha256": result.content_sha256,
                    "row_ids_sha256": result.row_ids_sha256,
                    "row_id_width": result.row_id_width,
                    "row_ids_dtype": ROW_IDS_DTYPE_NAME,
                }
            )
            self._write_manifest(complete=False)
            persisted = self.persisted_tokens
            if self._on_durable_shard is not None:
                self._on_durable_shard(persisted)
            return persisted
        except BaseException as exc:  # noqa: BLE001 - poison and re-raise exactly
            self._pending = None
            self._poison = exc
            raise

    def _manifest_payload(self, *, complete: bool) -> dict:
        if self._row_id_width is None:
            raise ValueError("cannot manifest a split without row identities")
        manifest = {
            "format_version": STORE_FORMAT_VERSION,
            "complete": complete,
            "split": self.split,
            "whitener_hash": self.whitener_hash,
            "sites": list(self.sites),
            "d_model": self.d_model,
            "n_tokens": self.persisted_tokens,
            "tokens_per_shard": self.tokens_per_shard,
            "row_ids_dtype": ROW_IDS_DTYPE_NAME,
            "row_id_width": self._row_id_width,
            "shards": self.shards,
            "meta": self.meta,
            "content_stream_sha256": self._content_stream_hasher.hexdigest(),
            "row_stream_sha256": self._row_stream_hasher.hexdigest(),
        }
        manifest["manifest_sha256"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return manifest

    def _write_manifest(self, *, complete: bool) -> dict:
        manifest = self._manifest_payload(complete=complete)
        temporary = self.dir / (MANIFEST_NAME + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.dir / MANIFEST_NAME)
        _fsync_directory(self.dir)
        return manifest

    def close(self) -> dict:
        """Flush, synchronize, and publish the complete split manifest."""

        try:
            # Keep this inside the lifecycle guard: close() is also the final
            # executor join after an earlier synchronize/callback failure.
            self._ensure_open()
            if self._buffered:
                self._flush(self._buffered)
            self.synchronize()
            if not self.shards:
                raise ValueError("cannot close an empty activation-store split")
            manifest = self._manifest_payload(complete=True)
            expected_row_digest = self.meta.get("row_stream_sha256")
            if (
                expected_row_digest is not None
                and expected_row_digest != manifest["row_stream_sha256"]
            ):
                raise ValueError(
                    "derived store row identities do not match the declared raw "
                    "stream digest"
                )
            result = self._write_manifest(complete=True)
            self._closed = True
            return result
        except BaseException as exc:  # noqa: BLE001 - preserve exact cause
            if self._poison is None:
                self._poison = exc
            self._closed = True
            raise
        finally:
            self._shutdown_worker()

    def abort(self) -> int:
        """Drain active persistence but discard the unsubmitted RAM tail.

        The split remains incomplete and resumable. A persistence or durable-
        progress callback failure is re-raised after the executor is joined.
        """

        if self._closed:
            return self.persisted_tokens
        try:
            if self._pending is not None:
                self.synchronize()
            return self.persisted_tokens
        finally:
            self._buffer = None
            self._row_id_buffer = None
            self._buffered = 0
            self._closed = True
            self._shutdown_worker()


def prefetch_batches(
    it: Iterator,
    depth: int = 4,
    *,
    pin_memory: bool | Callable[[torch.Tensor], bool] = False,
) -> Iterator:
    """Drive an I/O-bound batch iterator
    from a daemon thread, holding up to ``depth`` batches ahead of the
    consumer. Order-preserving, so determinism is untouched; worker
    exceptions are re-raised at the consumption point. This overlaps shard
    reads with GPU steps.

    ``pin_memory=True`` pins every tensor leaf. A callable can instead select
    leaves to pin; unselected metadata remains in its original CPU storage.
    Total lookahead is the queue depth plus the producer's current item (and
    any shard held by the source iterator), not exactly ``depth`` resident
    batches. Closing or abandoning the returned generator sets a cancellation
    event, closes the source iterator when supported, and prevents a producer
    from remaining parked on a full queue.
    """
    import queue
    import threading

    if depth <= 0:
        raise ValueError("prefetch depth must be positive")
    if not isinstance(pin_memory, bool) and not callable(pin_memory):
        raise TypeError("pin_memory must be a bool or callable")
    q: queue.Queue = queue.Queue(maxsize=depth)
    end = object()
    stop = threading.Event()

    def prepare(item):
        if torch.is_tensor(item):
            should_pin = pin_memory(item) if callable(pin_memory) else pin_memory
            if not isinstance(should_pin, bool):
                raise TypeError("pin_memory callable must return bool")
            if not should_pin:
                return item
            return item if item.is_pinned() else item.pin_memory()
        if isinstance(item, tuple):
            children = tuple(prepare(value) for value in item)
            if hasattr(item, "_fields"):
                return type(item)(*children)
            return children
        if isinstance(item, list):
            return [prepare(value) for value in item]
        if isinstance(item, Mapping):
            return {key: prepare(value) for key, value in item.items()}
        raise TypeError(
            "prefetched batches must contain only tensors or nested "
            "tuples/lists/mappings of tensors"
        )

    def deliver(item: object) -> bool:
        while not stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def worker() -> None:
        try:
            for x in it:
                if not deliver(prepare(x)):
                    return
            deliver(end)
        except BaseException as e:  # noqa: BLE001 — re-raised consumer-side
            deliver(e)
        finally:
            # Generator close must run in the producer thread. Calling it
            # consumer-side while the producer is inside ``next`` raises
            # ``ValueError: generator already executing``.
            close = getattr(it, "close", None)
            if close is not None:
                close()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        while True:
            item = q.get()
            if item is end:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        thread.join(timeout=1.0)


@dataclass(slots=True)
class _PendingCudaBatch:
    """One host batch retained until its dedicated-stream copy is consumed."""

    host: Any
    device: Any
    ready: torch.cuda.Event


@dataclass(slots=True)
class _PendingCudaFailure:
    """A source/transfer failure queued after every preceding valid batch."""

    error: BaseException


def cuda_prefetch_batches(
    it: Iterator,
    *,
    device: str | torch.device,
    depth: int = 1,
    dtype_policy: (
        torch.dtype | Callable[[torch.Tensor], torch.dtype | None] | None
    ) = None,
    copy_policy: Callable[[torch.Tensor], bool] | None = None,
) -> Iterator:
    """Copy nested CPU tensor batches ahead on one dedicated CUDA stream.

    ``depth`` is the number of device batches held ahead of the batch currently
    yielded to the consumer. Therefore total live lookahead is at most
    ``depth + 1`` batches, matching :func:`prefetch_batches`' queue-plus-current
    accounting. The source iterator is consumed synchronously; composing this
    wrapper outside ``prefetch_batches(..., pin_memory=True)`` overlaps both I/O
    and H2D transfer with CUDA work.

    A static ``torch.dtype`` policy casts floating tensor leaves only. A
    callable receives every copied tensor leaf and returns its target dtype, or
    ``None`` to preserve that leaf's dtype. ``copy_policy`` can select which
    leaves are pinned and copied; unselected leaves remain the same CPU tensor.
    Nested tuples, lists, and mappings are reconstructed with the same order;
    all leaves must initially be CPU tensors.

    Each copy records a CUDA event. Before yielding, the consumer's current
    stream waits for that event and every device tensor records the consumer
    stream, so allocator reuse remains correct without a device-wide
    synchronization. Closing the iterator drains the bounded copy stream and
    closes the source when supported. CPU devices are rejected explicitly: a
    silent fallback here would falsely claim copy/compute overlap.
    """

    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise ValueError("cuda_prefetch_batches requires a CUDA device")
    if depth <= 0:
        raise ValueError("CUDA prefetch depth must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for cuda_prefetch_batches")
    if resolved_device.index is None:
        resolved_device = torch.device("cuda", torch.cuda.current_device())
    if (
        dtype_policy is not None
        and not isinstance(dtype_policy, torch.dtype)
        and not callable(dtype_policy)
    ):
        raise TypeError("dtype_policy must be a torch.dtype, callable, or None")
    if copy_policy is not None and not callable(copy_policy):
        raise TypeError("copy_policy must be callable or None")

    source = iter(it)
    copy_stream = torch.cuda.Stream(device=resolved_device)

    def target_dtype(tensor: torch.Tensor) -> torch.dtype:
        if dtype_policy is None:
            return tensor.dtype
        if isinstance(dtype_policy, torch.dtype):
            return dtype_policy if tensor.is_floating_point() else tensor.dtype
        resolved = dtype_policy(tensor)
        if resolved is None:
            return tensor.dtype
        if not isinstance(resolved, torch.dtype):
            raise TypeError("dtype_policy callable must return a torch.dtype or None")
        return resolved

    def should_copy(tensor: torch.Tensor) -> bool:
        if copy_policy is None:
            return True
        resolved = copy_policy(tensor)
        if not isinstance(resolved, bool):
            raise TypeError("copy_policy callable must return bool")
        return resolved

    def map_batch(value: Any, leaf: Callable[[torch.Tensor], torch.Tensor]) -> Any:
        if torch.is_tensor(value):
            return leaf(value)
        if isinstance(value, tuple):
            children = tuple(map_batch(item, leaf) for item in value)
            if hasattr(value, "_fields"):
                return type(value)(*children)
            return children
        if isinstance(value, list):
            return [map_batch(item, leaf) for item in value]
        if isinstance(value, Mapping):
            return {key: map_batch(item, leaf) for key, item in value.items()}
        raise TypeError(
            "CUDA-prefetched batches must contain only tensors or nested "
            "tuples/lists/mappings of tensors"
        )

    def pin_leaf(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type != "cpu":
            raise ValueError("CUDA-prefetched tensor leaves must be on CPU")
        if not should_copy(tensor):
            return tensor
        return tensor if tensor.is_pinned() else tensor.pin_memory()

    def copy_leaf(tensor: torch.Tensor) -> torch.Tensor:
        if not should_copy(tensor):
            return tensor
        return tensor.to(
            device=resolved_device,
            dtype=target_dtype(tensor),
            non_blocking=True,
        )

    def record_leaf(tensor: torch.Tensor, stream: torch.cuda.Stream) -> torch.Tensor:
        if tensor.device.type == "cuda":
            tensor.record_stream(stream)
        return tensor

    def generate() -> Iterator:
        from collections import deque

        pending: deque[_PendingCudaBatch | _PendingCudaFailure] = deque()
        source_done = False

        def enqueue_one() -> None:
            nonlocal source_done
            if source_done:
                return
            try:
                host_batch = map_batch(next(source), pin_leaf)
                with torch.cuda.stream(copy_stream):
                    device_batch = map_batch(host_batch, copy_leaf)
                    ready = torch.cuda.Event()
                    ready.record(copy_stream)
                pending.append(_PendingCudaBatch(host_batch, device_batch, ready))
            except StopIteration:
                source_done = True
            except BaseException as exc:  # noqa: BLE001 - preserve exact source cause
                source_done = True
                pending.append(_PendingCudaFailure(exc))

        def fill(target: int) -> None:
            while len(pending) < target and not source_done:
                enqueue_one()

        try:
            # One batch is current and ``depth`` more are bounded lookahead.
            fill(depth + 1)
            while pending:
                current = pending.popleft()
                fill(depth)
                if isinstance(current, _PendingCudaFailure):
                    raise current.error
                consumer_stream = torch.cuda.current_stream(resolved_device)
                consumer_stream.wait_event(current.ready)
                map_batch(
                    current.device,
                    lambda tensor: record_leaf(tensor, consumer_stream),
                )
                # Retain ``current.host`` across the yield. This is stricter
                # than relying only on pinned-allocator event tracking.
                yield current.device
        finally:
            try:
                close = getattr(source, "close", None)
                if close is not None:
                    close()
            finally:
                # Pending copies own host buffers. Drain before releasing them
                # on explicit close or consumer-side failure.
                copy_stream.synchronize()

    return generate()


class StoreReader:
    """Sequential buffered-shuffle reads from a written split.

    Per epoch: seeded shard-order permutation; contiguous chunk reads fill
    a ``buffer_tokens`` RAM buffer; the buffer is permuted and emitted as
    [batch, S, d] batches (bf16, CPU); a sub-batch remainder carries into
    the next fill. The seed is the caller's to record — design: BSC and
    baseline share it verbatim.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        expected_whitener_hash: str | None = None,
        sites: Sequence[int] | None = None,
        allow_incomplete: bool = False,
        _allow_unmanifested_tail: bool = False,
    ) -> None:
        if not split or Path(split).name != split:
            raise ValueError("split must be one nonempty path component")
        self.dir = Path(root) / split
        manifest_path = self.dir / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest at {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text())
        claimed_manifest_hash = self.manifest.get("manifest_sha256")
        if claimed_manifest_hash is None:
            raise ValueError(f"legacy unbound store manifest at {manifest_path}")
        unhashed = dict(self.manifest)
        unhashed.pop("manifest_sha256")
        actual_manifest_hash = hashlib.sha256(
            json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if actual_manifest_hash != claimed_manifest_hash:
            raise ValueError(f"manifest hash mismatch at {manifest_path}")
        if self.manifest.get("format_version") != STORE_FORMAT_VERSION:
            raise ValueError(
                f"unsupported activation-store format at {manifest_path}: "
                f"{self.manifest.get('format_version')!r}; expected "
                f"{STORE_FORMAT_VERSION}"
            )
        if not isinstance(self.manifest.get("complete"), bool):
            raise ValueError("manifest complete status must be boolean")
        if self.manifest.get("complete") is not True and not allow_incomplete:
            raise ValueError(f"activation-store split is incomplete at {manifest_path}")
        if self.manifest.get("split") != split:
            raise ValueError(
                f"manifest split {self.manifest.get('split')!r} does not match {split!r}"
            )
        if self.manifest.get("row_ids_dtype") != ROW_IDS_DTYPE_NAME:
            raise ValueError("manifest row_ids_dtype must be int64")
        if (
            not isinstance(self.manifest.get("row_id_width"), int)
            or self.manifest["row_id_width"] <= 0
        ):
            raise ValueError("manifest row_id_width must be positive")
        if (
            not isinstance(self.manifest.get("d_model"), int)
            or self.manifest["d_model"] <= 0
        ):
            raise ValueError("manifest d_model must be positive")
        if (
            not isinstance(self.manifest.get("tokens_per_shard"), int)
            or self.manifest["tokens_per_shard"] <= 0
        ):
            raise ValueError("manifest tokens_per_shard must be positive")
        stored_sites = self.manifest.get("sites")
        if (
            not isinstance(stored_sites, list)
            or not stored_sites
            or any(not isinstance(site, int) for site in stored_sites)
            or len(set(stored_sites)) != len(stored_sites)
        ):
            raise ValueError("manifest sites must be nonempty, integer, and unique")
        records = self.manifest.get("shards")
        if not isinstance(records, list) or not records:
            raise ValueError("activation-store split must contain at least one shard")
        for index, record in enumerate(records):
            expected_file = f"shard_{index:05d}.safetensors"
            if not isinstance(record, dict):
                raise ValueError("manifest shard records must be objects")
            if record.get("index") != index or record.get("file") != expected_file:
                raise ValueError("manifest shard sequence is not canonical")
            if not isinstance(record.get("n_tokens"), int) or record["n_tokens"] <= 0:
                raise ValueError("manifest shard token counts must be positive")
            if (
                not isinstance(record.get("row_id_width"), int)
                or record["row_id_width"] <= 0
            ):
                raise ValueError("manifest shard row_id_width must be positive")
            if record["row_id_width"] != self.manifest["row_id_width"]:
                raise ValueError("manifest row_id_width changed across shards")
            if record.get("row_ids_dtype") != ROW_IDS_DTYPE_NAME:
                raise ValueError("manifest shard row_ids_dtype must be int64")
        if self.manifest.get("n_tokens") != sum(
            record["n_tokens"] for record in records
        ):
            raise ValueError("manifest token count does not equal its shard records")
        expected_files = {record["file"] for record in records}
        actual_files = {path.name for path in self.dir.glob("*.safetensors")}
        next_file = f"shard_{len(records):05d}.safetensors"
        recoverable_tail = (
            _allow_unmanifested_tail
            and self.manifest.get("complete") is False
            and actual_files == expected_files | {next_file}
        )
        if actual_files != expected_files and not recoverable_tail:
            raise ValueError(
                "activation-store shard file set differs from manifest: "
                f"expected={sorted(expected_files)}, actual={sorted(actual_files)}"
            )
        self.whitener_hash = self.manifest["whitener_hash"]
        if (
            expected_whitener_hash is not None
            and self.whitener_hash != expected_whitener_hash
        ):
            raise ValueError(
                f"store {self.dir} was written under whitener "
                f"{self.whitener_hash[:12]}…, expected {expected_whitener_hash[:12]}…"
            )
        self.n_tokens = self.manifest["n_tokens"]
        stored_sites = list(self.manifest["sites"])
        stored_site_dims = list(
            self.manifest.get("meta", {}).get(
                "site_dims", [self.manifest["d_model"]] * len(stored_sites)
            )
        )
        if len(stored_site_dims) != len(stored_sites):
            raise ValueError("manifest site_dims does not match sites")
        if any(
            not isinstance(width, int) or width <= 0 or width > self.manifest["d_model"]
            for width in stored_site_dims
        ):
            raise ValueError("manifest site_dims must be positive and within d_model")
        # The site-subset view selects a
        # subset of the stored site axis by layer number, sliced AFTER shard
        # load so generator consumption (shard order, buffer permutations)
        # is byte-identical to the full-width read at the same seed — the
        # factorial's matched-data guarantee: a single-site cell sees exactly
        # the joint run's token stream, sliced. Stored order is preserved;
        # reordering is refused rather than silently permuting frames.
        self._site_sel: slice | torch.Tensor | None
        if sites is None:
            self.sites = tuple(stored_sites)
            self.site_dims = tuple(int(v) for v in stored_site_dims)
            self._site_sel = None
        else:
            req = [int(s) for s in sites]
            missing = [s for s in req if s not in stored_sites]
            if missing:
                raise ValueError(f"sites {missing} not in store (has {stored_sites})")
            if len(set(req)) != len(req):
                raise ValueError(f"duplicate sites in {req}")
            idx = [stored_sites.index(s) for s in req]
            if idx != sorted(idx):
                raise ValueError(
                    f"sites {req} not in stored order {stored_sites} — "
                    "reordering the site axis is not supported"
                )
            self.sites = tuple(req)
            self.site_dims = tuple(int(stored_site_dims[i]) for i in idx)
            full_axis = idx == list(range(len(stored_sites)))
            contiguous = bool(idx) and idx == list(range(idx[0], idx[-1] + 1))
            if full_axis:
                self._site_sel = None
            elif contiguous:
                self._site_sel = slice(idx[0], idx[-1] + 1)
            else:
                self._site_sel = torch.tensor(idx, dtype=torch.long)
        self.n_sites = len(self.sites)
        self.d_model = self.manifest["d_model"]

    def _subset(self, acts: torch.Tensor) -> torch.Tensor:
        if self._site_sel is None:
            return acts
        if isinstance(self._site_sel, slice):
            return acts[:, self._site_sel]
        return acts.index_select(1, self._site_sel)

    def _shard_payload(
        self, shard: str | dict, *, verify: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from safetensors import safe_open

        record = (
            next(s for s in self.manifest["shards"] if s["file"] == shard)
            if isinstance(shard, str)
            else shard
        )
        name = record["file"]
        path = self.dir / name
        with safe_open(path, framework="pt", device="cpu") as f:
            header = f.metadata()
            expected_header = {
                "whitener_hash": self.whitener_hash,
                "split": self.manifest["split"],
                "shard_index": str(record["index"]),
                "n_tokens": str(record["n_tokens"]),
                "sites": json.dumps(self.manifest["sites"]),
                "d_model": str(self.manifest["d_model"]),
                "dtype": "bfloat16",
                "content_sha256": record["content_sha256"],
                "row_ids_sha256": record["row_ids_sha256"],
                "row_id_width": str(record["row_id_width"]),
                "row_ids_dtype": ROW_IDS_DTYPE_NAME,
                "meta": json.dumps(self.manifest.get("meta", {}), sort_keys=True),
            }
            mismatches = {
                key: {"header": header.get(key), "manifest": value}
                for key, value in expected_header.items()
                if header.get(key) != value
            }
            if mismatches:
                raise ValueError(
                    f"shard header mismatch in {path}: "
                    + json.dumps(mismatches, sort_keys=True)
                )
            keys = set(f.keys())
            if keys != {"acts", "row_ids"}:
                raise ValueError(f"shard tensor set mismatch in {path}: {sorted(keys)}")
            acts = f.get_tensor("acts")
            row_ids = f.get_tensor("row_ids")
        expected_shape = (
            record["n_tokens"],
            len(self.manifest["sites"]),
            self.manifest["d_model"],
        )
        if tuple(acts.shape) != expected_shape or acts.dtype != STORE_DTYPE:
            raise ValueError(
                f"shard payload mismatch in {path}: shape={tuple(acts.shape)} "
                f"dtype={acts.dtype}, expected={expected_shape}/{STORE_DTYPE}"
            )
        if (
            tuple(row_ids.shape) != (record["n_tokens"], record["row_id_width"])
            or row_ids.dtype != ROW_IDS_DTYPE
        ):
            raise ValueError(
                f"row identity payload mismatch in {path}: "
                f"shape={tuple(row_ids.shape)} dtype={row_ids.dtype}; expected "
                f"({record['n_tokens']}, {record['row_id_width']})/{ROW_IDS_DTYPE}"
            )
        if verify:
            checksum = hashlib.sha256(
                acts.contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            if checksum != header["content_sha256"]:
                raise ValueError(f"content checksum mismatch in shard {path}")
            row_checksum = hashlib.sha256(
                row_ids.contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            if row_checksum != header["row_ids_sha256"]:
                raise ValueError(f"row identity checksum mismatch in shard {path}")
        return acts, row_ids

    def _shard_tokens(self, shard: str | dict, *, verify: bool = False) -> torch.Tensor:
        return self._shard_payload(shard, verify=verify)[0]

    def verify(self, *, expected_row_identity: Mapping[str, int] | None = None) -> int:
        """Re-hash every shard and optionally verify its exact row allocation."""
        if self.n_tokens <= 0 or not self.manifest["shards"]:
            raise ValueError("activation-store split is empty")
        if expected_row_identity is not None:
            required = {
                "sequence_start",
                "sequence_stop_exclusive",
                "tokens_per_sequence",
                "position_start",
            }
            if set(expected_row_identity) != required or any(
                not isinstance(expected_row_identity[name], int)
                or isinstance(expected_row_identity[name], bool)
                for name in required
            ):
                raise ValueError("expected row identity contract is malformed")
            sequence_start = expected_row_identity["sequence_start"]
            sequence_stop = expected_row_identity["sequence_stop_exclusive"]
            tokens_per_sequence = expected_row_identity["tokens_per_sequence"]
            position_start = expected_row_identity["position_start"]
            if (
                sequence_start < 0
                or sequence_stop <= sequence_start
                or tokens_per_sequence <= 0
                or position_start < 0
                or (sequence_stop - sequence_start) * tokens_per_sequence
                != self.n_tokens
            ):
                raise ValueError("expected row identity allocation is inconsistent")
        total = 0
        stream = hashlib.sha256()
        row_stream = hashlib.sha256()
        for s in self.manifest["shards"]:
            acts, row_ids = self._shard_payload(s, verify=True)
            if expected_row_identity is not None:
                if row_ids.shape[1] < 2:
                    raise ValueError(
                        "captured row identity lacks sequence and position"
                    )
                offsets = torch.arange(
                    total,
                    total + row_ids.shape[0],
                    dtype=torch.int64,
                )
                expected_sequences = sequence_start + offsets // tokens_per_sequence
                expected_positions = position_start + offsets % tokens_per_sequence
                if not torch.equal(row_ids[:, 0], expected_sequences):
                    mismatch = int(
                        torch.nonzero(
                            row_ids[:, 0] != expected_sequences, as_tuple=False
                        )[0]
                    )
                    raise ValueError(
                        "row identity sequence differs from the canonical split "
                        f"allocation at stored row {total + mismatch}"
                    )
                if not torch.equal(row_ids[:, 1], expected_positions):
                    mismatch = int(
                        torch.nonzero(
                            row_ids[:, 1] != expected_positions, as_tuple=False
                        )[0]
                    )
                    raise ValueError(
                        "row identity position differs from the canonical packed "
                        f"sequence at stored row {total + mismatch}"
                    )
            total += acts.shape[0]
            stream.update(acts.contiguous().view(torch.uint8).numpy().tobytes())
            row_stream.update(row_ids.contiguous().view(torch.uint8).numpy().tobytes())
        if total != self.n_tokens:
            raise ValueError(
                f"manifest claims {self.n_tokens} tokens, shards hold {total}"
            )
        if stream.hexdigest() != self.manifest["content_stream_sha256"]:
            raise ValueError("ordered shard stream digest does not match manifest")
        if row_stream.hexdigest() != self.manifest["row_stream_sha256"]:
            raise ValueError("ordered row identity digest does not match manifest")
        return total

    def sequential_batches(self, batch_size: int) -> Iterator[torch.Tensor]:
        """Stored-order stream, never RAM-resident beyond one shard (eval)."""
        carry: torch.Tensor | None = None
        for s in self.manifest["shards"]:
            acts = self._subset(self._shard_tokens(s))
            if carry is not None:
                needed = batch_size - len(carry)
                if len(acts) < needed:
                    carry = torch.cat((carry, acts), dim=0)
                    continue
                yield torch.cat((carry, acts[:needed]), dim=0)
                acts = acts[needed:]
                carry = None
            n_full = acts.shape[0] // batch_size * batch_size
            for i in range(0, n_full, batch_size):
                yield acts[i : i + batch_size]
            carry = acts[n_full:] if acts.shape[0] > n_full else None
        if carry is not None and carry.shape[0]:
            yield carry

    def sequential_batches_with_ids(
        self, batch_size: int
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Stored-order activations paired with immutable row identities."""
        carry_x: torch.Tensor | None = None
        carry_ids: torch.Tensor | None = None
        for shard in self.manifest["shards"]:
            acts, row_ids = self._shard_payload(shard)
            acts = self._subset(acts)
            if carry_x is not None:
                assert carry_ids is not None
                needed = batch_size - len(carry_x)
                if len(acts) < needed:
                    carry_x = torch.cat((carry_x, acts), dim=0)
                    carry_ids = torch.cat((carry_ids, row_ids), dim=0)
                    continue
                yield (
                    torch.cat((carry_x, acts[:needed]), dim=0),
                    torch.cat((carry_ids, row_ids[:needed]), dim=0),
                )
                acts = acts[needed:]
                row_ids = row_ids[needed:]
                carry_x = None
                carry_ids = None
            n_full = acts.shape[0] // batch_size * batch_size
            for i in range(0, n_full, batch_size):
                yield acts[i : i + batch_size], row_ids[i : i + batch_size]
            carry_x = acts[n_full:] if acts.shape[0] > n_full else None
            carry_ids = row_ids[n_full:] if row_ids.shape[0] > n_full else None
        if carry_x is not None and carry_x.shape[0]:
            assert carry_ids is not None
            yield carry_x, carry_ids

    def shuffled_batches(
        self,
        batch_size: int,
        *,
        seed: int,
        epochs: int | None = None,
        buffer_tokens: int = 131_072,
        prefix_tokens: int | None = None,
    ) -> Iterator[torch.Tensor]:
        """Shuffle an exact immutable prefix with bounded buffering.

        The prefix is resolved in stored order before any shard or token
        permutation. Each epoch emits every prefix row exactly once,
        including a possibly partial final batch.
        """
        if batch_size <= 0 or buffer_tokens < batch_size:
            raise ValueError(
                "batch_size must be positive and buffer_tokens >= batch_size"
            )
        prefix = self.n_tokens if prefix_tokens is None else int(prefix_tokens)
        if prefix <= 0 or prefix > self.n_tokens:
            raise ValueError("prefix_tokens must be in [1, store n_tokens]")
        eligible: list[tuple[dict, int]] = []
        remaining = prefix
        for shard in self.manifest["shards"]:
            if remaining <= 0:
                break
            take = min(remaining, int(shard["n_tokens"]))
            eligible.append((shard, take))
            remaining -= take
        if remaining:
            raise RuntimeError("store manifest ended before prefix_tokens")
        gen = torch.Generator().manual_seed(seed)
        epoch = 0
        while epochs is None or epoch < epochs:
            order = torch.randperm(len(eligible), generator=gen)
            buffer: list[torch.Tensor] = []
            buffered = 0
            for shard_idx in order.tolist():
                shard, take_count = eligible[shard_idx]
                acts = self._subset(self._shard_tokens(shard))[:take_count]
                buffer.append(acts)
                buffered += acts.shape[0]
                while buffered >= buffer_tokens:
                    chunk = buffer[0] if len(buffer) == 1 else torch.cat(buffer, dim=0)
                    take, rest = chunk[:buffer_tokens], chunk[buffer_tokens:]
                    buffer = [rest] if rest.shape[0] else []
                    buffered = rest.shape[0]
                    perm = torch.randperm(take.shape[0], generator=gen)
                    take = take[perm]
                    n_full = take.shape[0] // batch_size * batch_size
                    for i in range(0, n_full, batch_size):
                        yield take[i : i + batch_size]
                    tail = take[n_full:]
                    if tail.shape[0]:
                        buffer.insert(0, tail)
                        buffered += tail.shape[0]
            # End of epoch: emit every remaining row. Dropping a partial
            # batch would silently shrink the declared unique-row pool.
            if buffered:
                chunk = buffer[0] if len(buffer) == 1 else torch.cat(buffer, dim=0)
                perm = torch.randperm(chunk.shape[0], generator=gen)
                chunk = chunk[perm]
                for i in range(0, chunk.shape[0], batch_size):
                    yield chunk[i : i + batch_size]
            epoch += 1
