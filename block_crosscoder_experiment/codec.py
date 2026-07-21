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
3. **Canonical orientation**: per block, rotate the code space to
   diagonalize the calib active-code second moment (descending); sign
   fixed so the active-mean projection is nonnegative. Exploits the
   residual O(b) gauge; frozen thereafter. Without it, an arbitrary
   gauge rotation changes componentwise clipping while the model is
   unchanged (tested: gauge-rotated models produce matching R-D points).
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
import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path

import torch

__all__ = [
    "CodecSpec",
    "Codec",
    "EncodedBatch",
    "fit_codec",
    "evaluate_rd",
    "encode_batch",
    "decode_batch",
]


_CODEC_PAYLOAD_KEYS = {
    "format_version",
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


def _artifact_digest(payload: dict) -> str:
    h = hashlib.sha256()

    def add(value) -> None:
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            h.update(str(tensor.dtype).encode() + str(tuple(tensor.shape)).encode())
            h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                h.update(str(key).encode() + b"\0")
                add(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                add(item)
        else:
            h.update(json.dumps(value, sort_keys=True, default=str).encode())

    add(payload)
    return h.hexdigest()


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

    @property
    def n_included(self) -> int:
        return int(self.rank_to_block.numel())

    def block_to_rank(self, *, device: torch.device | str = "cpu") -> torch.Tensor:
        mapping = torch.full(
            (self.included.numel(),), -1, dtype=torch.long, device=device
        )
        if self.n_included:
            mapping[self.rank_to_block.to(device=device, dtype=torch.long)] = (
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
        lo, hi = self.lo.to(z_can.device), self.hi.to(z_can.device)
        span = (hi - lo).clamp_min(1e-12)
        t = ((z_can - lo) / span).clamp(0.0, 1.0)
        return lo + torch.round(t * levels) / levels * span

    def quantize_indices(self, z_can: torch.Tensor, q: int) -> torch.Tensor:
        """Integer amplitude symbols for an actual round-trip packet."""
        levels = (1 << q) - 1
        lo, hi = self.lo.to(z_can.device), self.hi.to(z_can.device)
        span = (hi - lo).clamp_min(1e-12)
        return torch.round(((z_can - lo) / span).clamp(0, 1) * levels).to(torch.int32)

    def dequantize_indices(self, symbols: torch.Tensor, q: int) -> torch.Tensor:
        levels = (1 << q) - 1
        lo, hi = self.lo.to(symbols.device), self.hi.to(symbols.device)
        return lo + symbols.float() / levels * (hi - lo).clamp_min(1e-12)

    def save(self, path: str | Path) -> None:
        """Atomically serialize every calibration-fit codec parameter."""
        payload = self.to_payload()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    def to_payload(self) -> dict:
        """Return the complete, internally authenticated consumer payload."""

        self._validate_serialized_semantics()
        payload = {
            "format_version": 2,
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
        if payload.get("format_version") != 2:
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


@torch.no_grad()
def _packet_from_output(model, codec: Codec, out, q: int) -> EncodedBatch:
    """Build the one canonical packet representation from a model output."""
    if q not in codec.spec.qs:
        raise ValueError(f"q={q} is not in the frozen codec spec")
    device = next(model.parameters()).device
    included = codec.included.to(device)
    mask = out.mask & included.unsqueeze(0)
    nz = mask.nonzero(as_tuple=False)
    z_can = torch.einsum(
        "gij,ngj->ngi", codec.rotation.to(device), out.z_selected * mask.unsqueeze(-1)
    )
    symbols = codec.quantize_indices(z_can, q)
    compact_ranks = codec.block_to_rank(device=device)[nz[:, 1]]
    if bool((compact_ranks < 0).any()):  # pragma: no cover - mask proves this
        raise ValueError("packet support contains a block outside the rank table")
    return EncodedBatch(
        q=q,
        n_tokens=out.mask.shape[0],
        counts=mask.sum(dim=1).to(torch.int32).cpu(),
        block_ids=compact_ranks.to(torch.int32).cpu(),
        amplitude_symbols=symbols[mask].cpu(),
    )


@torch.no_grad()
def encode_batch(model, codec: Codec, x: torch.Tensor, q: int) -> EncodedBatch:
    """Encode a batch into explicit support and integer amplitude symbols."""
    device = next(model.parameters()).device
    x = x.to(device, torch.float32)
    out = model(x, mode="threshold")
    return _packet_from_output(model, codec, out, q)


@torch.no_grad()
def decode_batch(model, codec: Codec, packet: EncodedBatch) -> torch.Tensor:
    """Decode an explicit packet without access to the source activations."""
    device = next(model.parameters()).device
    G, b = model.cfg.n_blocks, model.cfg.block_dim
    if packet.q not in codec.spec.qs:
        raise ValueError(f"packet q={packet.q} is not in the frozen codec spec")
    if packet.n_tokens < 0 or packet.counts.shape != (packet.n_tokens,):
        raise ValueError("packet counts must have one entry per token")
    if packet.counts.dtype not in {torch.int32, torch.int64}:
        raise TypeError("packet counts must be an integer tensor")
    if bool((packet.counts < 0).any()) or bool(
        (packet.counts > codec.n_included).any()
    ):
        raise ValueError("packet has an impossible support count")
    n_events = int(packet.counts.long().sum())
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
    if bool((packet.block_ids < 0).any()) or bool(
        (packet.block_ids >= codec.n_included).any()
    ):
        raise ValueError("packet block rank is outside the frozen support alphabet")
    levels = (1 << packet.q) - 1
    if bool((packet.amplitude_symbols < 0).any()) or bool(
        (packet.amplitude_symbols > levels).any()
    ):
        raise ValueError("packet amplitude symbol is outside the q-bit alphabet")
    mask = torch.zeros(packet.n_tokens, G, dtype=torch.bool, device=device)
    rows = torch.repeat_interleave(
        torch.arange(packet.n_tokens, device=device), packet.counts.to(device).long()
    )
    ranks = packet.block_ids.to(device).long()
    ids = codec.rank_to_block.to(device=device, dtype=torch.long)[ranks]
    if n_events:
        # Each support is a set. Duplicate IDs within one token would be an
        # ambiguous packet and would silently overwrite an amplitude below.
        starts = torch.repeat_interleave(
            torch.arange(packet.n_tokens), packet.counts.long()
        )
        pairs = torch.stack((starts, packet.block_ids.long()), dim=1)
        if torch.unique(pairs, dim=0).shape[0] != n_events:
            raise ValueError("packet repeats a block id within one token")
    mask[rows, ids] = True
    symbols = torch.zeros(packet.n_tokens, G, b, dtype=torch.int32, device=device)
    symbols[rows, ids] = packet.amplitude_symbols.to(device)
    z_can = codec.dequantize_indices(symbols, packet.q) * mask.unsqueeze(-1)
    z = torch.einsum("gji,ngj->ngi", codec.rotation.to(device), z_can)
    return model.decode(z)


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
    block_events = torch.zeros(G, dtype=torch.long)
    mean_acc = torch.zeros(S, d, dtype=torch.float64)
    n_tokens = 0
    selected_events = 0
    estimated_peak_bytes = 0

    for x in batches:
        x = x.to(device, torch.float32)
        out = model(x, mode="threshold")
        mask = out.mask
        z_sel = out.z_selected
        nz = mask.nonzero()
        selected_events += int(nz.shape[0])
        # Conservative bound for list storage, concatenation overlap,
        # canonical-code workspace, sort indices and per-event IDs.  The
        # ceiling fails closed; calibration never samples or truncates events.
        estimated_peak_bytes = selected_events * (32 + 24 * b)
        if estimated_peak_bytes > spec.max_calibration_event_bytes:
            raise MemoryError(
                "exact codec calibration exceeds its resolved event-memory ceiling: "
                f"estimated {estimated_peak_bytes} > "
                f"{spec.max_calibration_event_bytes} bytes"
            )
        ev_codes.append(z_sel[mask].float().cpu())
        ev_ids.append(nz[:, 1].to(torch.int32).cpu())
        ev_tokens.append((nz[:, 0] + n_tokens).to(torch.int32).cpu())
        block_events += mask.sum(dim=0).cpu()
        mean_acc += x.double().sum(dim=0).cpu()
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

    # Canonical orientation: batched second moments via index_add, eigh
    # descending, sign so the active-mean projection is >= 0.
    M = torch.zeros(G, b, b, dtype=torch.float64)
    M.index_add_(0, ids, torch.einsum("ni,nj->nij", codes.double(), codes.double()))
    mean_code = torch.zeros(G, b, dtype=torch.float64)
    mean_code.index_add_(0, ids, codes.double())
    denom = block_events.clamp_min(1).double()
    M /= denom.view(-1, 1, 1)
    mean_code /= denom.view(-1, 1)
    eye = torch.eye(b, dtype=torch.float64)
    safe_M = torch.where(included.view(-1, 1, 1), M, eye.expand(G, b, b))
    _, evecs = torch.linalg.eigh(safe_M)  # ascending
    evecs = evecs.flip(-1)  # descending eigenvalue order, columns
    R = evecs.transpose(1, 2)  # rows: z_can = R @ z
    sign = torch.sign(torch.einsum("gij,gj->gi", R, mean_code))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    R = R * sign.unsqueeze(-1)

    # Clip quantiles per canonical coordinate.
    codes_can = torch.einsum("nij,nj->ni", R[ids].float(), codes)
    lo = torch.zeros(G, b)
    hi = torch.ones(G, b)
    order = torch.argsort(ids)
    sorted_ids = ids[order]
    sorted_codes = codes_can[order]
    boundaries = torch.searchsorted(sorted_ids, torch.arange(G + 1, dtype=torch.long))
    qs = torch.tensor([spec.clip_lo, spec.clip_hi])
    for g in included.nonzero().flatten().tolist():
        seg = sorted_codes[boundaries[g] : boundaries[g + 1]]
        ql = torch.quantile(seg, qs, dim=0)
        lo[g], hi[g] = ql[0], ql[1]
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
        },
    )


@torch.no_grad()
def evaluate_rd(
    model,
    codec: Codec,
    batches,
    *,
    row_len: int | None = None,
    device: str = "cpu",
) -> dict:
    """Eval pass: per-q distortion through quantized codes + rates, with
    per-sequence accumulators and a sequence bootstrap.

    Real-data callers must yield ``(x, row_ids)`` pairs from the sequential
    store reader; column zero is the immutable sequence ID.  Tensor-only
    batches require ``row_len`` and are labelled as the fixed-length
    synthetic/test fallback.  Mixing the two contracts is rejected.
    """
    spec = codec.spec
    b = model.cfg.block_dim
    S = model.cfg.n_sites
    inc = codec.included.to(device)
    mu = codec.calib_mean.to(device).float()  # [S, d] calib-fit centering
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

    excluded_events = 0
    total_events = 0
    sequence_mode: str | None = None
    current_sequence: int | None = None
    fallback_token_offset = 0
    for item in batches:
        if isinstance(item, tuple):
            if len(item) != 2:
                raise ValueError("R-D batches must be x or (x, row_ids)")
            x, row_ids = item
            if sequence_mode == "fixed_length_fallback":
                raise ValueError("cannot mix stored IDs and fixed-length fallback")
            sequence_mode = "stored_sequence_ids"
            if (
                row_ids.ndim != 2
                or row_ids.shape[0] != x.shape[0]
                or row_ids.shape[1] < 1
            ):
                raise ValueError("row_ids must have shape [tokens, >=1]")
            sequence_ids = row_ids[:, 0].to(device="cpu", dtype=torch.int64)
        else:
            x = item
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
        x = x.to(device, torch.float32)
        out = model(x, mode="threshold")
        raw_mask = out.mask
        mask = raw_mask & inc.unsqueeze(0)
        excluded_events += int((raw_mask & ~inc.unsqueeze(0)).sum())
        total_events += int(raw_mask.sum())
        counts = mask.sum(dim=1)

        # Non-operational support-rate sensitivities, per token.  The exact
        # fixed-width packet rate is assembled below from count and ID widths.
        if codec.n_included:
            sup_bits = -codec.log2_count_prob(counts.cpu()).double() + _log2_binom(
                codec.n_included, counts.cpu()
            )
            act_p = (
                (codec.bernoulli_log2p.to(device) * mask.float()).sum(dim=1).double()
            )
            act_q = (
                (codec.bernoulli_log2q.to(device) * mask.float()).sum(dim=1).double()
            )
            bern_bits = -(act_p.cpu() + (log2_1mq_total - act_q.cpu()))
        else:
            sup_bits = torch.zeros(x.shape[0], dtype=torch.float64)
            bern_bits = torch.zeros(x.shape[0], dtype=torch.float64)

        err_site = {}
        for q in spec.qs:
            # Distortion is measured on the same validated integer packet a
            # saved artifact will decode, never on an algebraically similar
            # floating-point shortcut.
            packet = _packet_from_output(model, codec, out, q)
            xhat = decode_batch(model, codec, packet)
            err_site[q] = (x - xhat).double().pow(2).sum(dim=2).cpu()  # [n, S]
        tot_site = (x - mu).double().pow(2).sum(dim=2).cpu()  # [n, S]

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
            pend["cnt"] += float(counts[sl].sum())
            pend["n"] += run_count
            start += run_count
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
        "packet_roundtrip_validated": True,
        "n_rows": n_rows,
        "n_tokens": int(n_tok.sum()),
        "sequence_grouping": sequence_mode,
        "row_len": row_len if sequence_mode == "fixed_length_fallback" else None,
        "avg_count": float(cnt.sum() / n_tok.sum()),
        "codec_meta": dict(codec.meta),
        "eval_excluded_event_share": excluded_events / max(1, total_events),
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
            "fvu_per_site": [1.0] * S,
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
            "fvu_per_site": (err_site_t.sum(dim=0) / tot_site_t.sum(dim=0)).tolist(),
            "amplitude_bits_per_token": amp_bits,
            "rate_bits_per_token": rate,
            "rate_bits_ci95": list(boot(rate_numerator, n_tok)),
            "ideal_enumerative_rate_bits_per_token": ideal_rate,
            "rate_bits_bernoulli": results["bernoulli_bits_per_token"] + amp_bits,
        }
    return results
