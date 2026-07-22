"""Capture one immutable raw activation stream and derive aligned views."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import torch

from block_crosscoder_experiment.store import (
    NORMALIZATION_MODES,
    STORE_FORMAT_VERSION,
    ShardWriter,
    StoreReader,
    Whitener,
    WhitenerAccumulator,
)

TOKENIZER_CONTRACT_FILES = {
    "gpt2-byte-bpe-files-v1": ("tokenizer.json", "vocab.json", "merges.txt"),
    "gemma3-tokenizer-files-v1": ("tokenizer.json", "tokenizer.model"),
}
TRANSFORMER_LENS_MODEL_NAMES = {
    "openai-community/gpt2": "gpt2",
}
TOKENIZER_PREFLIGHTS = {
    "openai-community/gpt2": {
        "contract": "gpt2-byte-bpe-files-v1",
        "class": "GPT2Tokenizer",
        "bos_token_id": 50_256,
        "vocab_sha256": (
            "sha256:e35d8b86ebd35ebd260d040aa455e09759f7e675f4dbb7f3d727516f27eca190"
        ),
    },
    "google/gemma-3-4b-pt": {
        "contract": "gemma3-tokenizer-files-v1",
        "class": "GemmaTokenizer",
        "bos_token_id": 2,
        "vocab_sha256": (
            "sha256:4ab2b66fed16d7e79cfb30bd2168ee3da6d848a6ff9b0753cd62a5841c9328ad"
        ),
    },
}
CAPTURE_STATE_NAME = "capture.state.json"
CAPTURE_MANIFEST_NAME = "capture.json"
DEFAULT_MAX_WRITER_RESIDENCY_BYTES = 8 * 1024**3
CAPTURE_PROFILE_SPLITS = {
    "phase2": (
        "normalization_fit",
        "calibration",
        "development",
        "confirmation",
        "train",
    ),
    "phase3": (
        "normalization_fit",
        "calibration",
        "stability",
        "final",
        "train",
    ),
}


def transformer_lens_model_name(hf_model_name: str) -> str:
    """Return the TransformerLens registry name for a pinned HF repository."""
    return TRANSFORMER_LENS_MODEL_NAMES.get(hf_model_name, hf_model_name)


def pack_token_rows(
    token_iter: Iterator[list[int]], *, ctx: int, bos_id: int, n_rows: int
) -> Iterator[torch.Tensor]:
    """Pack concatenated documents into BOS-prefixed fixed-context rows."""
    buffer: list[int] = []
    produced = 0
    for document in token_iter:
        buffer.extend(document)
        while len(buffer) >= ctx - 1 and produced < n_rows:
            row = [bos_id] + buffer[: ctx - 1]
            buffer = buffer[ctx - 1 :]
            produced += 1
            yield torch.tensor(row, dtype=torch.long)
        if produced >= n_rows:
            return


@dataclass(frozen=True)
class SourceSpec:
    model: str
    revision: str | None
    hook: str

    @classmethod
    def parse(cls, text: str) -> "SourceSpec":
        # MODEL|REVISION|HOOK; an empty revision resolves the current HF ref to
        # an immutable commit before capture.
        parts = text.split("|", 2)
        if len(parts) != 3 or not parts[0] or not parts[2]:
            raise ValueError("--source must be MODEL|REVISION|HOOK")
        return cls(parts[0], parts[1] or None, parts[2])


@dataclass(slots=True)
class _PendingCaptureCopy:
    """One activation batch retained until its asynchronous D2H copy lands."""

    source: torch.Tensor
    host: torch.Tensor
    row_ids: torch.Tensor
    ready: torch.cuda.Event

    def resolve(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.ready.synchronize()
        return self.host, self.row_ids


def _overlap_cuda_capture_copies(
    batches: Iterator[tuple[torch.Tensor, torch.Tensor]],
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Overlap one bounded pinned activation D2H copy with the next forward.

    The source iterator performs the model forward synchronously on CUDA. We
    pull one item ahead before resolving the preceding transfer, which lets a
    dedicated copy stream drain the previous activations while the default
    stream executes the next model batch. Both CUDA source storage and pinned
    host storage stay live until the copy event completes.
    """

    source = iter(batches)
    transfer_stream: torch.cuda.Stream | None = None

    def enqueue(
        item: tuple[torch.Tensor, torch.Tensor],
    ) -> _PendingCaptureCopy:
        nonlocal transfer_stream
        activations, row_ids = item
        if not activations.is_cuda:
            raise ValueError("overlapped capture requires CUDA activations")
        if row_ids.device.type != "cpu":
            raise ValueError("capture row identities must remain on CPU")
        if transfer_stream is None:
            transfer_stream = torch.cuda.Stream(device=activations.device)
        elif transfer_stream.device != activations.device:
            raise ValueError("capture activation device changed between batches")
        host = torch.empty_like(
            activations,
            device="cpu",
            pin_memory=True,
        )
        produced = torch.cuda.Event()
        produced.record(torch.cuda.current_stream(activations.device))
        with torch.cuda.stream(transfer_stream):
            transfer_stream.wait_event(produced)
            host.copy_(activations, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(transfer_stream)
        return _PendingCaptureCopy(activations, host, row_ids, ready)

    pending: _PendingCaptureCopy | None = None
    try:
        try:
            pending = enqueue(next(source))
        except StopIteration:
            return
        for item in source:
            following = enqueue(item)
            host, row_ids = pending.resolve()
            pending = following
            yield host, row_ids
        host, row_ids = pending.resolve()
        pending = None
        yield host, row_ids
    finally:
        # A consumer-side close or a source exception may leave the lookahead
        # transfer in flight. Drain it before either pinned host storage or its
        # retained CUDA source can leave scope.
        if transfer_stream is not None:
            transfer_stream.synchronize()
        close = getattr(source, "close", None)
        if close is not None:
            close()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _immutable_revision(value: object, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(f"{label} did not resolve to an immutable 40-hex commit")
    return value


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path.parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _dependency_versions() -> dict[str, str]:
    names = (
        "block-crosscoder-experiment",
        "datasets",
        "huggingface-hub",
        "sae-lens",
        "safetensors",
        "torch",
        "transformer-lens",
        "transformers",
    )
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "uninstalled"
    return versions


def capture_implementation_contract() -> dict[str, object]:
    """Bind executable capture code and dependency versions into provenance."""

    from block_crosscoder_experiment import store

    return {
        "schema": "bsc-capture-implementation-v1",
        "python": platform.python_version(),
        "dependencies": _dependency_versions(),
        "data_module_sha256": _file_sha256(Path(__file__)),
        "store_module_sha256": _file_sha256(Path(store.__file__)),
    }


def load_pinned_tokenizer(model: str, revision: str, contract: str):
    """Load and validate the exact slow tokenizer used by capture and TL."""

    from transformers import AutoTokenizer

    try:
        expected = TOKENIZER_PREFLIGHTS[model]
    except KeyError as exc:
        raise ValueError(
            f"no reviewed tokenizer preflight is declared for {model!r}"
        ) from exc
    if contract != expected["contract"]:
        raise ValueError(
            f"tokenizer contract {contract!r} is incompatible with {model!r}; "
            f"expected {expected['contract']!r}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        revision=revision,
        use_fast=False,
    )
    observed = {
        "class": type(tokenizer).__name__,
        "bos_token_id": tokenizer.bos_token_id,
        "vocab_sha256": "sha256:" + _canonical_hash(tokenizer.get_vocab()),
    }
    mismatches = {
        key: {"expected": expected[key], "actual": observed[key]}
        for key in observed
        if expected[key] != observed[key]
    }
    if mismatches:
        raise ValueError(
            "tokenizer preflight failed: " + json.dumps(mismatches, sort_keys=True)
        )
    return tokenizer


def tokenizer_contract_hash(
    model: str,
    revision: str,
    contract: str,
) -> str:
    """Hash the ordered immutable tokenizer files named by a plan contract."""
    from huggingface_hub import snapshot_download

    try:
        filenames = TOKENIZER_CONTRACT_FILES[contract]
    except KeyError as exc:
        raise ValueError(f"unsupported tokenizer contract {contract!r}") from exc
    snapshot = Path(
        snapshot_download(
            model,
            revision=revision,
            allow_patterns=list(filenames),
        )
    )
    digest = hashlib.sha256()
    missing: list[str] = []
    for filename in filenames:
        path = snapshot / filename
        if not path.is_file():
            missing.append(filename)
            continue
        digest.update(filename.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    if missing:
        raise ValueError(
            f"tokenizer contract {contract!r} is missing files {missing} "
            f"at {model}@{revision}"
        )
    return "sha256:" + digest.hexdigest()


def parse_split_sizes(values: list[str] | None) -> dict[str, int]:
    if not values:
        raise ValueError("split sizes must be declared explicitly")
    result: dict[str, int] = {}
    for value in values:
        name, sep, raw = value.partition("=")
        if not sep or not name or Path(name).name != name:
            raise ValueError("split sizes must be NAME=POSITIVE_TOKENS")
        try:
            size = int(raw)
        except ValueError as exc:
            raise ValueError("split sizes must be NAME=POSITIVE_TOKENS") from exc
        if size <= 0:
            raise ValueError("split sizes must be NAME=POSITIVE_TOKENS")
        if name in result:
            raise ValueError(f"duplicate split {name}")
        result[name] = size
    required = {"normalization_fit", "calibration", "train"}
    if missing := required - result.keys():
        raise ValueError(f"missing required splits: {sorted(missing)}")
    return result


def parse_capture_split_sizes(
    values: list[str] | None, *, profile: str | None
) -> dict[str, int]:
    """Parse and enforce one complete phase-specific capture role set."""

    if profile not in CAPTURE_PROFILE_SPLITS:
        choices = ", ".join(CAPTURE_PROFILE_SPLITS)
        raise ValueError(
            f"capture profile must be explicitly declared as one of: {choices}"
        )
    result = parse_split_sizes(values)
    required = CAPTURE_PROFILE_SPLITS[profile]
    required_set = set(required)
    observed_set = set(result)
    missing = [name for name in required if name not in observed_set]
    unexpected = [name for name in result if name not in required_set]
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise ValueError(
            f"{profile} capture requires exactly the split roles {list(required)}; "
            + "; ".join(details)
        )
    # Corpus allocation is semantically meaningful: roles receive consecutive
    # packed-sequence ranges.  Canonicalize it from the profile instead of
    # allowing CLI argument order to change which examples become train,
    # development, confirmation, stability, or final evidence.
    return {name: result[name] for name in required}


def estimate_store_bytes(
    split_sizes: dict[str, int], site_dims: Iterable[int], *, n_views: int = 1
) -> int:
    # Payload only. Safetensors headers and int64 row IDs are included as a
    # conservative fixed per-token allowance.
    dimensions = tuple(int(width) for width in site_dims)
    if not dimensions or any(width <= 0 for width in dimensions):
        raise ValueError("site dimensions must be nonempty and positive")
    if n_views <= 0:
        raise ValueError("n_views must be positive")
    per_token = 2 * sum(dimensions) + 3 * 8
    return sum(split_sizes.values()) * per_token * n_views


def estimate_writer_residency_bytes(
    site_dims: Iterable[int],
    *,
    tokens_per_shard: int = 150_000,
    row_id_width: int = 3,
) -> dict[str, int]:
    """Return the exact one-deep writer payload and peak residency bounds.

    Stores are physically padded to ``max(site_dims)``.  The async contract
    owns at most one detached shard while the producer fills one staging
    shard, so the refusal bound is exactly two full physical payloads.
    """

    dimensions = tuple(int(width) for width in site_dims)
    if not dimensions or any(width <= 0 for width in dimensions):
        raise ValueError("site dimensions must be nonempty and positive")
    if tokens_per_shard <= 0:
        raise ValueError("tokens_per_shard must be positive")
    if row_id_width <= 0:
        raise ValueError("row_id_width must be positive")
    per_token = 2 * len(dimensions) * max(dimensions) + 8 * row_id_width
    shard_payload = tokens_per_shard * per_token
    return {
        "bytes_per_token": per_token,
        "shard_payload_bytes": shard_payload,
        "pending_shard_bytes": shard_payload,
        "staging_shard_bytes": shard_payload,
        "writer_residency_bytes": 2 * shard_payload,
    }


def estimate_capture_pipeline_residency_bytes(
    writer: dict[str, int],
    site_dims: Iterable[int],
    *,
    batch_rows: int,
    context: int,
    drop_positions: int,
    cuda_overlap: bool,
) -> dict[str, int | str]:
    """Price the bounded activation-copy lookahead beside the shard writer."""

    dimensions = tuple(int(width) for width in site_dims)
    if not dimensions or any(width <= 0 for width in dimensions):
        raise ValueError("site dimensions must be nonempty and positive")
    if batch_rows <= 0 or context <= 0 or not 0 <= drop_positions < context:
        raise ValueError("capture batch/context geometry is invalid")
    batch_tokens = batch_rows * (context - drop_positions)
    activation_bytes = batch_tokens * len(dimensions) * max(dimensions) * 2
    row_identity_bytes = batch_tokens * 3 * 8
    # Pulling one item ahead owns the current consumer batch and one pending
    # batch. The previous CUDA source is deliberately retained until its D2H
    # event completes; row identities remain CPU-resident throughout.
    lookahead = 2 if cuda_overlap else 0
    pinned_host_bytes = lookahead * activation_bytes
    retained_row_identity_bytes = lookahead * row_identity_bytes
    retained_cuda_bytes = lookahead * activation_bytes
    writer_bytes = int(writer["writer_residency_bytes"])
    return {
        "contract": (
            "two_pinned_activation_d2h_lookahead_v1"
            if cuda_overlap
            else "synchronous_cpu_capture_v1"
        ),
        "activation_batch_bytes": activation_bytes,
        "row_identity_batch_bytes": row_identity_bytes,
        "pinned_activation_buffer_count": lookahead,
        "pinned_activation_host_bytes": pinned_host_bytes,
        "retained_row_identity_host_bytes": retained_row_identity_bytes,
        "retained_cuda_source_bytes": retained_cuda_bytes,
        "peak_host_pipeline_bytes": (
            writer_bytes + pinned_host_bytes + retained_row_identity_bytes
        ),
        "peak_cuda_capture_lookahead_bytes": retained_cuda_bytes,
    }


def _enforce_writer_residency(
    estimate: dict[str, int], *, max_writer_residency_bytes: int
) -> None:
    if max_writer_residency_bytes <= 0:
        raise ValueError("max_writer_residency_bytes must be positive")
    required = estimate["writer_residency_bytes"]
    if required > max_writer_residency_bytes:
        raise ValueError(
            "one-deep shard writer residency exceeds the configured refusal "
            f"limit: required={required} bytes, "
            f"limit={max_writer_residency_bytes} bytes"
        )


def _enforce_capture_pipeline_residency(
    estimate: dict[str, int | str], *, max_host_residency_bytes: int
) -> None:
    if max_host_residency_bytes <= 0:
        raise ValueError("max_host_residency_bytes must be positive")
    required = int(estimate["peak_host_pipeline_bytes"])
    if required > max_host_residency_bytes:
        raise ValueError(
            "capture pipeline host residency exceeds the configured refusal "
            f"limit: required={required} bytes, "
            f"limit={max_host_residency_bytes} bytes"
        )


def whole_sequence_split_plan(
    split_sizes: dict[str, int], tokens_per_sequence: int
) -> dict[str, dict[str, int]]:
    """Allocate requested token minima without sharing a sequence across splits."""
    if tokens_per_sequence <= 0:
        raise ValueError("tokens_per_sequence must be positive")
    next_sequence = 0
    result: dict[str, dict[str, int]] = {}
    for name, requested in split_sizes.items():
        n_sequences = math.ceil(requested / tokens_per_sequence)
        stop = next_sequence + n_sequences
        result[name] = {
            "requested_tokens": requested,
            "actual_tokens": n_sequences * tokens_per_sequence,
            "sequence_start": next_sequence,
            "sequence_stop_exclusive": stop,
            "tokens_per_sequence": tokens_per_sequence,
        }
        next_sequence = stop
    return result


def _ensure_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"refusing nonempty output directory {path}")
    path.mkdir(parents=True, exist_ok=True)


def _normalization_fit_requested_tokens(capture: dict, fit_reader: StoreReader) -> int:
    """Resolve the exact declared prefix, excluding sequence-rounding surplus."""

    split_plan = capture.get("split_plan")
    if not isinstance(split_plan, dict):
        raise ValueError("capture.json lacks its split allocation")
    spec = split_plan.get("normalization_fit")
    if not isinstance(spec, dict):
        raise ValueError("capture.json lacks normalization_fit allocation")
    requested = spec.get("requested_tokens")
    actual = spec.get("actual_tokens")
    if (
        not isinstance(requested, int)
        or isinstance(requested, bool)
        or requested <= 0
        or not isinstance(actual, int)
        or isinstance(actual, bool)
        or actual != fit_reader.n_tokens
        or requested > actual
    ):
        raise ValueError("normalization_fit allocation is inconsistent with its store")
    return requested


def _sequential_prefix(
    reader: StoreReader, *, batch_size: int, n_tokens: int
) -> Iterator[torch.Tensor]:
    remaining = n_tokens
    for batch in reader.sequential_batches(batch_size):
        if remaining <= 0:
            break
        take = min(remaining, batch.shape[0])
        yield batch[:take]
        remaining -= take
    if remaining:
        raise ValueError("activation store ended before the declared fit prefix")


def derive_views(
    raw_root: Path,
    out_root: Path,
    modes: Iterable[str],
    *,
    batch_size: int = 4096,
    tokens_per_shard: int = 150_000,
    max_writer_residency_bytes: int = DEFAULT_MAX_WRITER_RESIDENCY_BYTES,
) -> dict[str, dict]:
    """Fit transforms once and derive byte-aligned views from a raw store."""
    modes = tuple(modes)
    if not modes or len(set(modes)) != len(modes):
        raise ValueError("derive modes must be nonempty and unique")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if raw_root.resolve() == out_root.resolve():
        raise ValueError("raw and derived roots must differ")
    unknown = set(modes) - set(NORMALIZATION_MODES)
    if unknown:
        raise ValueError(f"unknown normalization modes: {sorted(unknown)}")
    capture_path = raw_root / "capture.json"
    if not capture_path.is_file():
        raise ValueError(f"raw store lacks {capture_path}")
    capture = json.loads(capture_path.read_text())
    source = capture.get("source")
    source_hash = capture.get("source_hash")
    if not isinstance(source, dict) or source_hash != _canonical_hash(source):
        raise ValueError("capture.json source contract hash mismatch")
    capture_sha256 = _file_sha256(capture_path)
    fit_reader = StoreReader(raw_root, "normalization_fit")
    fit_reader.verify()
    if fit_reader.whitener_hash != f"raw:{source_hash}":
        raise ValueError("normalization-fit split is not bound to capture source")
    writer_residency = estimate_writer_residency_bytes(
        fit_reader.site_dims,
        tokens_per_shard=tokens_per_shard,
        row_id_width=int(fit_reader.manifest["row_id_width"]),
    )
    _enforce_writer_residency(
        writer_residency,
        max_writer_residency_bytes=max_writer_residency_bytes,
    )
    fit_tokens = _normalization_fit_requested_tokens(capture, fit_reader)
    accumulator = WhitenerAccumulator(
        fit_reader.n_sites,
        fit_reader.d_model,
        track_covariance="whiten" in modes,
    )
    for x in _sequential_prefix(fit_reader, batch_size=batch_size, n_tokens=fit_tokens):
        accumulator.update(x.float())
    centered_norm = None
    if "sqrt_d" in modes:
        fitted_mean = (accumulator.sum / accumulator.n).float()
        totals = torch.zeros(fit_reader.n_sites, dtype=torch.float64)
        for x in _sequential_prefix(
            fit_reader, batch_size=batch_size, n_tokens=fit_tokens
        ):
            for site, width in enumerate(fit_reader.site_dims):
                totals[site] += (
                    (x[:, site, :width].float() - fitted_mean[site, :width])
                    .norm(dim=-1)
                    .double()
                    .sum()
                )
        centered_norm = totals / accumulator.n
    split_names = [
        path.name
        for path in sorted(raw_root.iterdir())
        if path.is_dir() and (path / "split.json").exists()
    ]
    # Verification authenticates immutable source bytes and does not depend on
    # the requested normalization mode.  Reusing these stateless readers avoids
    # re-hashing every raw shard once per derived view.
    source_readers = {split: StoreReader(raw_root, split) for split in split_names}
    for reader in source_readers.values():
        reader.verify()
    results: dict[str, dict] = {}
    for mode in modes:
        view_root = out_root / mode
        _ensure_empty(view_root)
        # Only immutable content identities enter the transform hash.  The
        # local raw-store path is a locator kept in derived shard manifests,
        # not a scientific identity (moving a store must not change W).
        transform_source_meta = {
            "source_capture_sha256": capture_sha256,
            "source_hash": source_hash,
            "source_fit_manifest_sha256": fit_reader.manifest["manifest_sha256"],
            "source_fit_row_stream_sha256": fit_reader.manifest["row_stream_sha256"],
            "source_fit_content_stream_sha256": fit_reader.manifest[
                "content_stream_sha256"
            ],
            "source_fit_requested_tokens": fit_tokens,
            "transform_contract": "content_addressed_materialized_view-v1",
            "site_dims": list(fit_reader.site_dims),
        }
        transform = accumulator.finalize(
            sites=fit_reader.sites,
            meta=transform_source_meta,
            mode=mode,
            mean_centered_norm=centered_norm if mode == "sqrt_d" else None,
        )
        transform.save(view_root / "whitener.pt")
        view_splits = {}
        for split in split_names:
            reader = source_readers[split]
            meta = {
                **reader.manifest.get("meta", {}),
                **transform_source_meta,
                "source_raw_root": str(raw_root.resolve()),
                "derived_view": True,
                "source_split_manifest_sha256": reader.manifest["manifest_sha256"],
                "row_stream_sha256": reader.manifest["row_stream_sha256"],
                "normalization": mode,
                "writer_pipeline": {
                    "contract": "one_pending_shard_v1",
                    **writer_residency,
                    "max_writer_residency_bytes": max_writer_residency_bytes,
                },
            }
            writer = ShardWriter(
                view_root,
                split,
                whitener_hash=transform.hash,
                sites=reader.sites,
                d_model=reader.d_model,
                meta=meta,
                tokens_per_shard=tokens_per_shard,
            )
            try:
                for x, row_ids in reader.sequential_batches_with_ids(batch_size):
                    writer.add(transform.apply(x), row_ids)
                manifest = writer.close()
            except BaseException as producer_error:  # noqa: BLE001
                try:
                    writer.abort()
                except BaseException as drain_error:  # noqa: BLE001
                    raise BaseExceptionGroup(
                        "derived-view production and shard persistence both failed",
                        [producer_error, drain_error],
                    ) from None
                raise
            if manifest["row_stream_sha256"] != reader.manifest["row_stream_sha256"]:
                raise RuntimeError("derived view changed row identity")
            StoreReader(
                view_root, split, expected_whitener_hash=transform.hash
            ).verify()
            view_splits[split] = manifest
        results[mode] = {
            "whitener_hash": transform.hash,
            "writer_pipeline": {
                "contract": "one_pending_shard_v1",
                **writer_residency,
                "max_writer_residency_bytes": max_writer_residency_bytes,
            },
            "splits": view_splits,
        }
    return results


def fit_transform_artifacts(
    raw_root: Path,
    out_root: Path,
    modes: Iterable[str],
    *,
    batch_size: int = 4096,
) -> dict[str, dict]:
    """Fit content-addressed transforms without duplicating activation shards.

    This is the Phase-3 single-view contract: the immutable bf16 raw store is
    retained once and a loader applies one frozen, invertible transform in
    fp32.  Every transform binds both ``capture.json`` and the exact
    normalization-fit row/content manifests.
    """
    modes = tuple(modes)
    if not modes or len(set(modes)) != len(modes):
        raise ValueError("fit-transform modes must be nonempty and unique")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if raw_root.resolve() == out_root.resolve():
        raise ValueError("raw and transform roots must differ")
    unknown = set(modes) - set(NORMALIZATION_MODES)
    if unknown:
        raise ValueError(f"unknown normalization modes: {sorted(unknown)}")
    capture_path = raw_root / "capture.json"
    if not capture_path.is_file():
        raise ValueError(f"raw store lacks {capture_path}")
    capture = json.loads(capture_path.read_text())
    source = capture.get("source")
    source_hash = capture.get("source_hash")
    if not isinstance(source, dict) or source_hash != _canonical_hash(source):
        raise ValueError("capture.json source contract hash mismatch")

    fit_reader = StoreReader(raw_root, "normalization_fit")
    fit_reader.verify()
    if fit_reader.whitener_hash != f"raw:{source_hash}":
        raise ValueError("normalization-fit split is not bound to capture source")
    fit_tokens = _normalization_fit_requested_tokens(capture, fit_reader)
    accumulator = WhitenerAccumulator(
        fit_reader.n_sites,
        fit_reader.d_model,
        track_covariance="whiten" in modes,
    )
    for x in _sequential_prefix(fit_reader, batch_size=batch_size, n_tokens=fit_tokens):
        accumulator.update(x.float())
    centered_norm = None
    if "sqrt_d" in modes:
        fitted_mean = (accumulator.sum / accumulator.n).float()
        totals = torch.zeros(fit_reader.n_sites, dtype=torch.float64)
        for x in _sequential_prefix(
            fit_reader, batch_size=batch_size, n_tokens=fit_tokens
        ):
            for site, width in enumerate(fit_reader.site_dims):
                totals[site] += (
                    (x[:, site, :width].float() - fitted_mean[site, :width])
                    .norm(dim=-1)
                    .double()
                    .sum()
                )
        centered_norm = totals / accumulator.n

    capture_sha256 = _file_sha256(capture_path)
    common_meta = {
        "source_capture_sha256": capture_sha256,
        "source_hash": source_hash,
        "source_fit_manifest_sha256": fit_reader.manifest["manifest_sha256"],
        "source_fit_row_stream_sha256": fit_reader.manifest["row_stream_sha256"],
        "source_fit_content_stream_sha256": fit_reader.manifest[
            "content_stream_sha256"
        ],
        "source_fit_requested_tokens": fit_tokens,
        "transform_contract": "content_addressed_transform_only-v1",
        "transform_only": True,
        "site_dims": list(fit_reader.site_dims),
    }
    results: dict[str, dict] = {}
    for mode in modes:
        transform = accumulator.finalize(
            sites=fit_reader.sites,
            meta=common_meta,
            mode=mode,
            mean_centered_norm=centered_norm if mode == "sqrt_d" else None,
        )
        artifact_root = out_root / mode / transform.hash
        artifact_root.mkdir(parents=True, exist_ok=True)
        transform_path = artifact_root / "whitener.pt"
        if transform_path.exists():
            existing = Whitener.load(transform_path)
            if existing.hash != transform.hash:
                raise ValueError(f"content-address collision at {artifact_root}")
        else:
            transform.save(transform_path)
        manifest = {
            "schema": "bsc-transform-artifact-v1",
            "mode": mode,
            "transform_hash": transform.hash,
            "whitener_sha256": _file_sha256(transform_path),
            "source_capture_sha256": capture_sha256,
            "source_hash": source_hash,
            "source_fit_manifest_sha256": fit_reader.manifest["manifest_sha256"],
            "source_fit_row_stream_sha256": fit_reader.manifest["row_stream_sha256"],
            "source_fit_content_stream_sha256": fit_reader.manifest[
                "content_stream_sha256"
            ],
            "source_fit_requested_tokens": fit_tokens,
            "source_raw_root": str(raw_root.resolve()),
            "transform_contract": "content_addressed_transform_only-v1",
        }
        manifest_path = artifact_root / "transform.json"
        encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        if manifest_path.exists() and manifest_path.read_text() != encoded:
            raise ValueError(f"existing transform manifest differs at {manifest_path}")
        manifest_path.write_text(encoded)
        results[mode] = {
            **manifest,
            "path": str(transform_path),
            "manifest": str(manifest_path),
        }
    return results


def verify_alignment(roots: Iterable[Path]) -> dict:
    roots = tuple(roots)
    if len(roots) < 2:
        raise ValueError("alignment verification needs at least two stores")
    if len({root.resolve() for root in roots}) != len(roots):
        raise ValueError("alignment stores must be unique")
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"activation-store root does not exist: {root}")
    split_sets = [
        {p.name for p in root.iterdir() if p.is_dir() and (p / "split.json").exists()}
        for root in roots
    ]
    if any(splits != split_sets[0] for splits in split_sets[1:]):
        raise ValueError("stores expose different split sets")
    if not split_sets[0]:
        raise ValueError("activation stores expose no nonempty splits")
    result = {}
    for split in sorted(split_sets[0]):
        readers = [StoreReader(root, split) for root in roots]
        for reader in readers:
            reader.verify()
        identities = {
            (
                reader.n_tokens,
                reader.manifest["row_stream_sha256"],
                tuple(reader.manifest["sites"]),
                reader.manifest["d_model"],
            )
            for reader in readers
        }
        if len(identities) != 1:
            raise ValueError(f"row alignment mismatch for split {split}")
        result[split] = {
            "n_tokens": readers[0].n_tokens,
            "row_stream_sha256": readers[0].manifest["row_stream_sha256"],
        }
    return result


def capture(
    args: argparse.Namespace,
    *,
    failure_injector: Callable[[str, int], None] | None = None,
) -> dict[str, object]:
    """Capture one pinned model's hooks into a resumable immutable row stream."""

    if args.context <= 1 or args.drop_positions < 0:
        raise ValueError(
            "context must exceed one and drop_positions cannot be negative"
        )
    if (
        args.batch_rows <= 0
        or args.write_batch_tokens <= 0
        or args.tokens_per_shard <= 0
    ):
        raise ValueError("capture batch and shard sizes must be positive")
    if args.resume:
        if not args.out.is_dir() or not (args.out / CAPTURE_STATE_NAME).is_file():
            raise ValueError("--resume requires an existing capture.state.json")
    elif args.out.exists():
        if not args.out.is_dir() or any(args.out.iterdir()):
            raise ValueError(f"refusing nonempty output directory {args.out}")
    raw_sources = [SourceSpec.parse(value) for value in args.source]
    if not raw_sources:
        raise ValueError("capture requires at least one source hook")
    hooks = [source.hook for source in raw_sources]
    if len(set(hooks)) != len(hooks):
        raise ValueError("capture hooks must be unique")
    profile = getattr(args, "profile", None)
    split_sizes = parse_capture_split_sizes(args.split, profile=profile)
    tokens_per_row = args.context - args.drop_positions
    if tokens_per_row <= 0:
        raise ValueError("drop_positions must be smaller than context")

    from datasets import load_dataset
    from huggingface_hub import HfApi
    from sae_lens import HookedSAETransformer

    # Resolve every mutable ref before touching the output.  Cross-model
    # capture is deliberately refused: this project studies cross-layer
    # features in one model, and has no token-alignment recipe for model diffing.
    hf = HfApi()
    resolved_sources: list[tuple[tuple[str, str], str]] = []
    for source in raw_sources:
        info = hf.model_info(source.model, revision=source.revision)
        revision = _immutable_revision(info.sha, label=f"model {source.model!r}")
        resolved_sources.append(((source.model, revision), source.hook))
    model_revisions = {key for key, _ in resolved_sources}
    if len(model_revisions) != 1:
        raise ValueError(
            "capture is single-model-only; all hooks must resolve to one model revision"
        )
    model_key = next(iter(model_revisions))
    model_name, model_revision = model_key
    loader_name = transformer_lens_model_name(model_name)
    tokenizer = load_pinned_tokenizer(
        model_name, model_revision, args.tokenizer_contract
    )
    tokenizer_hash = tokenizer_contract_hash(
        model_name, model_revision, args.tokenizer_contract
    )
    model = HookedSAETransformer.from_pretrained_no_processing(
        loader_name,
        revision=model_revision,
        dtype=torch.bfloat16,
        tokenizer=tokenizer,
    ).to(args.device)
    model.eval()
    if getattr(model, "tokenizer", None) is not tokenizer:
        raise ValueError("TransformerLens did not retain the explicit pinned tokenizer")
    hook_dict = getattr(model, "hook_dict", {})
    if hook_dict:
        missing_hooks = [hook for hook in hooks if hook not in hook_dict]
        if missing_hooks:
            raise ValueError(f"model does not expose requested hooks: {missing_hooks}")
    # TransformerLens' stop_at_layer is exclusive.  Residual hooks are bound
    # at the start of their named block, so max(block)+1 captures every
    # requested activation without executing unrelated later transformer
    # blocks or the output head.  Unknown hook namespaces conservatively keep
    # the full forward.
    hook_layers = []
    for hook in hooks:
        match = re.fullmatch(r"blocks\.(\d+)\..+", hook)
        if match is None:
            hook_layers = []
            break
        hook_layers.append(int(match.group(1)))
    stop_at_layer = max(hook_layers) + 1 if hook_layers else None
    d_model = int(model.cfg.d_model)
    if d_model <= 0:
        raise ValueError("model d_model must be positive")
    site_dims = [d_model] * len(hooks)
    max_writer_residency_bytes = int(
        getattr(
            args,
            "max_writer_residency_bytes",
            DEFAULT_MAX_WRITER_RESIDENCY_BYTES,
        )
    )
    writer_residency = estimate_writer_residency_bytes(
        site_dims,
        tokens_per_shard=args.tokens_per_shard,
        row_id_width=3,
    )
    cuda_capture_overlap = str(args.device).startswith("cuda")
    capture_pipeline_residency = estimate_capture_pipeline_residency_bytes(
        writer_residency,
        site_dims,
        batch_rows=args.batch_rows,
        context=args.context,
        drop_positions=args.drop_positions,
        cuda_overlap=cuda_capture_overlap,
    )
    _enforce_capture_pipeline_residency(
        capture_pipeline_residency,
        max_host_residency_bytes=max_writer_residency_bytes,
    )

    corpus_info = hf.dataset_info(args.corpus, revision=args.corpus_revision)
    corpus_revision = _immutable_revision(
        corpus_info.sha,
        label=f"corpus {args.corpus!r}",
    )
    source_meta = {
        "format_version": 2,
        "sources": [
            {"model": key[0], "revision": key[1], "hook": hook}
            for key, hook in resolved_sources
        ],
        "corpus": args.corpus,
        "corpus_config": args.corpus_config,
        "corpus_revision": corpus_revision,
        "corpus_split": args.corpus_split,
        "text_field": args.text_field,
        "context": args.context,
        "drop_positions": args.drop_positions,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_sha256": "sha256:" + _canonical_hash(tokenizer.get_vocab()),
        "add_special_tokens": False,
        "bos_token_id": int(tokenizer.bos_token_id),
        "packing_algorithm": "bos_prefixed_greedy_document_stream_v1",
        "sequence_allocation": "whole_packed_contexts_v1",
        "tokenizer_hashes": [tokenizer_hash],
        "tokenizer_contract": args.tokenizer_contract,
        "store_contract_version": args.store_contract_version,
        "alignment_version": args.alignment_version,
        "alignment_audit": args.alignment_audit,
        "row_identity_columns": ["sequence", "position", "token_id"],
        "capture_mode": "raw_once",
        "model_loader": "transformer_lens_from_pretrained_no_processing_v1",
        "transformer_lens_model_names": [loader_name],
        "model_forward_dtype": "bfloat16",
        "store_dtype": "bfloat16",
    }
    source_hash = _canonical_hash(source_meta)
    split_plan = whole_sequence_split_plan(split_sizes, tokens_per_row)
    split_order = list(split_sizes)
    implementation = dict(capture_implementation_contract())
    implementation["runtime"] = {
        "requested_device": str(args.device),
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_name": (
            torch.cuda.get_device_name(torch.device(args.device))
            if str(args.device).startswith("cuda") and torch.cuda.is_available()
            else None
        ),
    }
    binding = {
        "schema": "bsc-capture-binding-v1",
        "campaign_profile": profile,
        "source_hash": source_hash,
        "split_order": split_order,
        "split_plan": split_plan,
        "capture_implementation": implementation,
        "sites": list(range(len(hooks))),
        "site_dims": site_dims,
        "d_model": d_model,
        "physical_store_format_version": STORE_FORMAT_VERSION,
        "batch_rows": args.batch_rows,
        "write_batch_tokens": args.write_batch_tokens,
        "tokens_per_shard": args.tokens_per_shard,
        "writer_pipeline": {
            "contract": "one_pending_shard_v1",
            **writer_residency,
            "max_writer_residency_bytes": max_writer_residency_bytes,
        },
        "capture_transfer_pipeline": capture_pipeline_residency,
    }
    binding_sha256 = _canonical_hash(binding)
    state_path = args.out / CAPTURE_STATE_NAME
    capture_path = args.out / CAPTURE_MANIFEST_NAME

    if args.resume:
        state = json.loads(state_path.read_text())
        if (
            state.get("schema") != "bsc-capture-state-v1"
            or state.get("binding") != binding
            or state.get("binding_sha256") != binding_sha256
        ):
            raise ValueError("resume capture binding differs from the existing state")
        # These exact temporary paths are never committed evidence.  They may
        # survive process death between write and atomic rename, so resume
        # discards them only after the immutable binding has been revalidated.
        for temporary in (
            args.out / (CAPTURE_STATE_NAME + ".tmp"),
            args.out / (CAPTURE_MANIFEST_NAME + ".tmp"),
        ):
            temporary.unlink(missing_ok=True)
        for split in split_order:
            split_dir = args.out / split
            if split_dir.is_dir():
                (split_dir / "split.json.tmp").unlink(missing_ok=True)
                for temporary in split_dir.glob("shard_*.tmp"):
                    temporary.unlink()
    else:
        _ensure_empty(args.out)
        state = {
            "schema": "bsc-capture-state-v1",
            "status": "in_progress",
            "binding": binding,
            "binding_sha256": binding_sha256,
            "progress": {},
        }
        _atomic_json(state_path, state)

    allowed_entries = set(split_order) | {CAPTURE_STATE_NAME, CAPTURE_MANIFEST_NAME}
    foreign = sorted(
        path.name for path in args.out.iterdir() if path.name not in allowed_entries
    )
    if foreign:
        raise ValueError(f"capture output contains unbound entries: {foreign}")

    def split_meta(split: str) -> dict[str, object]:
        spec = split_plan[split]
        return {
            **source_meta,
            "site_dims": site_dims,
            "split_requested_tokens": spec["requested_tokens"],
            "split_actual_tokens": spec["actual_tokens"],
            "sequence_start": spec["sequence_start"],
            "sequence_stop_exclusive": spec["sequence_stop_exclusive"],
            "tokens_per_sequence": tokens_per_row,
            "sequence_allocation": "whole_packed_contexts_v1",
            "capture_binding_sha256": binding_sha256,
            "ordered_split_allocation": split_order,
        }

    def resume_split_writer(
        split: str,
        *,
        on_durable_shard: Callable[[int], None] | None = None,
    ) -> ShardWriter:
        """Recover only the exact durable tail permitted by ShardWriter."""

        return ShardWriter(
            args.out,
            split,
            whitener_hash=f"raw:{source_hash}",
            sites=range(len(hooks)),
            d_model=d_model,
            meta=split_meta(split),
            tokens_per_shard=args.tokens_per_shard,
            resume=True,
            on_durable_shard=on_durable_shard,
        )

    def verified_reader(split: str, *, allow_incomplete: bool = False) -> StoreReader:
        reader = StoreReader(args.out, split, allow_incomplete=allow_incomplete)
        reader.verify()
        expected = {
            "whitener_hash": f"raw:{source_hash}",
            "sites": list(range(len(hooks))),
            "d_model": d_model,
            "tokens_per_shard": args.tokens_per_shard,
            "meta": split_meta(split),
        }
        mismatches = {
            key: {"expected": value, "actual": reader.manifest.get(key)}
            for key, value in expected.items()
            if reader.manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"capture split {split!r} changed contract: "
                + json.dumps(mismatches, sort_keys=True)
            )
        if reader.n_tokens > split_plan[split]["actual_tokens"]:
            raise ValueError(f"capture split {split!r} exceeds its allocation")
        return reader

    def verify_sequence_identity(split: str, reader: StoreReader) -> None:
        spec = split_plan[split]
        offset = 0
        for _, row_ids in reader.sequential_batches_with_ids(args.write_batch_tokens):
            linear = torch.arange(offset, offset + row_ids.shape[0], dtype=torch.int64)
            expected_sequence = spec["sequence_start"] + linear // tokens_per_row
            expected_position = args.drop_positions + linear % tokens_per_row
            if not torch.equal(row_ids[:, 0], expected_sequence) or not torch.equal(
                row_ids[:, 1], expected_position
            ):
                raise RuntimeError(f"split {split!r} bisected or reused a sequence")
            offset += row_ids.shape[0]
        if offset != spec["actual_tokens"]:
            raise RuntimeError(f"split {split!r} does not cover its exact allocation")

    # Reconstruct progress solely from verified shard manifests.  Mutable state
    # is informational; it is never trusted over content-bound store evidence.
    persisted_by_split: dict[str, int] = {}
    complete_splits: dict[str, dict[str, object]] = {}
    saw_gap_or_partial = False
    for split in split_order:
        split_dir = args.out / split
        manifest_path = split_dir / "split.json"
        if not split_dir.exists():
            saw_gap_or_partial = True
            persisted_by_split[split] = 0
            continue
        if not manifest_path.is_file():
            if not any(split_dir.iterdir()):
                saw_gap_or_partial = True
                persisted_by_split[split] = 0
                continue
            # The first shard can be directory-fsynced before the first
            # per-split manifest exists.  ShardWriter accepts only one exact,
            # content-valid shard_00000 tail and atomically manifests it.
            resume_split_writer(split)
        else:
            try:
                manifest_probe = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError):
                # Preserve StoreReader's normal, evidence-bearing refusal path.
                manifest_probe = None
            if isinstance(manifest_probe, dict) and (
                manifest_probe.get("complete") is False
            ):
                # This is also the only path allowed to adopt a single exact
                # shard rename which outran its manifest update.
                resume_split_writer(split)
        reader = verified_reader(split, allow_incomplete=True)
        if saw_gap_or_partial and reader.n_tokens:
            raise ValueError("capture split manifests are not an ordered prefix")
        persisted_by_split[split] = reader.n_tokens
        expected_tokens = split_plan[split]["actual_tokens"]
        if reader.manifest.get("complete") is True:
            if reader.n_tokens != expected_tokens:
                raise ValueError(f"complete split {split!r} has the wrong token count")
            verify_sequence_identity(split, reader)
            complete_splits[split] = dict(split_plan[split])
        else:
            if reader.n_tokens > expected_tokens:
                raise ValueError(f"incomplete split {split!r} exceeds its allocation")
            if reader.n_tokens == expected_tokens:
                # A crash after the final shard's incomplete manifest but
                # before the complete=true replacement is recoverable without
                # replay.  Verify the exact allocated identity stream first,
                # then atomically finalize and re-read through the strict path.
                verify_sequence_identity(split, reader)
                finalizer = resume_split_writer(split)
                if finalizer.persisted_tokens != expected_tokens:
                    raise ValueError(
                        f"split {split!r} resume cursor changed before finalization"
                    )
                finalizer.close()
                reader = verified_reader(split)
                verify_sequence_identity(split, reader)
                complete_splits[split] = dict(split_plan[split])
            else:
                saw_gap_or_partial = True

    state["progress"] = persisted_by_split
    _atomic_json(state_path, state)
    total_tokens = sum(spec["actual_tokens"] for spec in split_plan.values())
    committed_tokens = sum(persisted_by_split.values())
    if committed_tokens > total_tokens:
        raise ValueError("capture progress exceeds the ordered split allocation")

    if committed_tokens == total_tokens:
        expected_capture: dict[str, object] = {
            "schema": "bsc-capture-manifest-v1",
            "source": source_meta,
            "source_hash": source_hash,
            "split_order": split_order,
            "split_plan": split_plan,
            "splits": complete_splits,
            "capture_implementation": implementation,
            "capture_binding_sha256": binding_sha256,
        }
        if capture_path.is_file():
            existing_capture = json.loads(capture_path.read_text())
            if existing_capture != expected_capture:
                raise ValueError("capture.json differs from the resume binding")
        else:
            existing_capture = expected_capture
            _atomic_json(capture_path, existing_capture)
        state["status"] = "complete"
        state["capture_manifest_sha256"] = _file_sha256(capture_path)
        _atomic_json(state_path, state)
        return existing_capture
    if capture_path.exists():
        raise ValueError("incomplete capture must not have capture.json")

    dataset = load_dataset(
        args.corpus,
        name=args.corpus_config,
        split=args.corpus_split,
        streaming=True,
        revision=corpus_revision,
    )

    def documents() -> Iterator[list[int]]:
        for document in dataset:
            text = document.get(args.text_field)
            if not isinstance(text, str):
                raise ValueError(f"corpus row lacks text field {args.text_field!r}")
            yield tokenizer.encode(text, add_special_tokens=False)

    total_rows = sum(
        spec["sequence_stop_exclusive"] - spec["sequence_start"]
        for spec in split_plan.values()
    )
    start_sequence, first_offset = divmod(committed_tokens, tokens_per_row)
    all_rows = pack_token_rows(
        documents(),
        ctx=args.context,
        bos_id=tokenizer.bos_token_id,
        n_rows=total_rows,
    )
    remaining_rows = itertools.islice(all_rows, start_sequence, None)

    capture_positions = torch.arange(
        args.drop_positions,
        args.context,
        dtype=torch.int64,
    )

    @torch.inference_mode()
    def process_rows(
        batch_rows: list[tuple[int, torch.Tensor]], skip: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_ids = torch.tensor([item[0] for item in batch_rows], dtype=torch.int64)
        toks_cpu = torch.stack([item[1] for item in batch_rows])
        toks = toks_cpu.to(args.device)
        forward_kwargs = {}
        if stop_at_layer is not None and callable(getattr(model, "forward", None)):
            forward_kwargs["stop_at_layer"] = stop_at_layer
        _, cache = model.run_with_cache(
            toks,
            names_filter=lambda name, selected=set(hooks): name in selected,
            return_type=None,
            **forward_kwargs,
        )
        per_source: list[torch.Tensor] = []
        for hook in hooks:
            acts = cache[hook]
            if acts.ndim != 3 or acts.shape[-1] != d_model:
                raise ValueError(
                    f"hook {hook!r} emitted shape {tuple(acts.shape)}, expected "
                    f"[batch, context, {d_model}]"
                )
            per_source.append(acts[:, args.drop_positions :])
        stacked = torch.stack(per_source, dim=2).reshape(-1, len(per_source), d_model)
        position = capture_positions.view(1, -1).expand(toks_cpu.shape[0], -1)
        identity = torch.stack(
            (
                seq_ids.view(-1, 1).expand_as(position),
                position,
                toks_cpu[:, args.drop_positions :].to(torch.int64),
            ),
            dim=-1,
        ).reshape(-1, 3)
        return stacked[skip:], identity[skip:]

    def device_activation_batches() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        batch_rows: list[tuple[int, torch.Tensor]] = []
        skip = first_offset
        for sequence_id, row in enumerate(remaining_rows, start=start_sequence):
            batch_rows.append((sequence_id, row))
            if len(batch_rows) == args.batch_rows:
                yield process_rows(batch_rows, skip)
                batch_rows = []
                skip = 0
        if batch_rows:
            yield process_rows(batch_rows, skip)

    def activation_batches() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        batches = device_activation_batches()
        if cuda_capture_overlap:
            yield from _overlap_cuda_capture_copies(batches)
            return
        for activations, row_ids in batches:
            yield activations.cpu(), row_ids

    stream = iter(activation_batches())
    pending_x: torch.Tensor | None = None
    pending_ids: torch.Tensor | None = None

    def take_slices(n_tokens: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield exact stream slices without concatenating a transient batch."""

        nonlocal pending_x, pending_ids
        remaining = n_tokens
        while remaining:
            if pending_x is None or not pending_x.shape[0]:
                try:
                    pending_x, pending_ids = next(stream)
                except StopIteration as exc:
                    raise RuntimeError(
                        "corpus exhausted before the declared capture allocation"
                    ) from exc
            assert pending_ids is not None
            count = min(remaining, pending_x.shape[0])
            yield pending_x[:count], pending_ids[:count]
            pending_x, pending_ids = pending_x[count:], pending_ids[count:]
            remaining -= count

    capture_splits: dict[str, dict[str, object]] = dict(complete_splits)
    for split in split_order:
        n_tokens = split_plan[split]["actual_tokens"]
        persisted = persisted_by_split[split]
        if persisted == n_tokens:
            continue
        split_dir = args.out / split
        resume_split = (split_dir / "split.json").is_file()

        def durable_progress(after: int, *, active_split: str = split) -> None:
            # ShardWriter invokes this only after the incomplete split
            # manifest and its directory entry are durable.  This callback is
            # therefore the single deterministic capture-progress edge.
            state["progress"][active_split] = after
            _atomic_json(state_path, state)
            if failure_injector is not None:
                failure_injector(active_split, after)

        writer = (
            resume_split_writer(split, on_durable_shard=durable_progress)
            if resume_split
            else ShardWriter(
                args.out,
                split,
                whitener_hash=f"raw:{source_hash}",
                sites=range(len(hooks)),
                d_model=d_model,
                meta=split_meta(split),
                tokens_per_shard=args.tokens_per_shard,
                on_durable_shard=durable_progress,
            )
        )
        if writer.persisted_tokens != persisted:
            raise ValueError(
                f"split {split!r} resume cursor changed after verification"
            )
        remaining = n_tokens - persisted
        try:
            while remaining:
                count = min(remaining, args.write_batch_tokens)
                for x, row_ids in take_slices(count):
                    writer.add(x, row_ids)
                remaining -= count
            manifest = writer.close()
        except BaseException as producer_error:  # noqa: BLE001
            try:
                writer.abort()
            except BaseException as drain_error:  # noqa: BLE001
                raise BaseExceptionGroup(
                    "capture production and shard persistence both failed",
                    [producer_error, drain_error],
                ) from None
            raise
        reader = verified_reader(split)
        verify_sequence_identity(split, reader)
        state["progress"][split] = manifest["n_tokens"]
        _atomic_json(state_path, state)
        capture_splits[split] = dict(split_plan[split])
        print(
            json.dumps(
                {
                    "split": split,
                    "tokens": manifest["n_tokens"],
                    "requested_tokens": split_plan[split]["requested_tokens"],
                    "sequences": (
                        split_plan[split]["sequence_stop_exclusive"]
                        - split_plan[split]["sequence_start"]
                    ),
                    "row_stream_sha256": manifest["row_stream_sha256"],
                }
            )
        )

    capture_manifest: dict[str, object] = {
        "schema": "bsc-capture-manifest-v1",
        "source": source_meta,
        "source_hash": source_hash,
        "split_order": split_order,
        "split_plan": split_plan,
        "splits": capture_splits,
        "capture_implementation": implementation,
        "capture_binding_sha256": binding_sha256,
    }
    _atomic_json(capture_path, capture_manifest)
    state["status"] = "complete"
    state["progress"] = {
        split: split_plan[split]["actual_tokens"] for split in split_order
    }
    state["capture_manifest_sha256"] = _file_sha256(capture_path)
    _atomic_json(state_path, state)
    return capture_manifest


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture")
    cap.add_argument(
        "--source",
        action="append",
        required=True,
        help="MODEL|REVISION|HOOK; repeat for aligned sites",
    )
    cap.add_argument("--corpus", default="HuggingFaceFW/fineweb-edu")
    cap.add_argument("--corpus-config", default="sample-10BT")
    cap.add_argument("--corpus-revision", default=None)
    cap.add_argument("--corpus-split", default="train")
    cap.add_argument("--text-field", default="text")
    cap.add_argument(
        "--tokenizer-contract",
        choices=tuple(TOKENIZER_CONTRACT_FILES),
        required=True,
        help="ordered tokenizer-file hash contract bound by the study plan",
    )
    cap.add_argument(
        "--store-contract-version",
        choices=(
            "activation-store-v3-derived-views",
            "activation-store-v3-single-view",
        ),
        default="activation-store-v3-derived-views",
    )
    cap.add_argument(
        "--alignment-version",
        default="identical-tokenizer-row-identity-v1",
    )
    cap.add_argument(
        "--alignment-audit",
        default="not_applicable:single-model-identical-tokenizer",
    )
    cap.add_argument("--context", type=int, default=128)
    cap.add_argument("--drop-positions", type=int, default=1)
    cap.add_argument("--batch-rows", type=int, default=8)
    cap.add_argument("--write-batch-tokens", type=int, default=65_536)
    cap.add_argument("--tokens-per-shard", type=int, default=150_000)
    cap.add_argument(
        "--max-writer-residency-bytes",
        type=int,
        default=DEFAULT_MAX_WRITER_RESIDENCY_BYTES,
        help=(
            "refuse capture if the shard writer plus bounded pinned activation "
            "lookahead exceed this host-memory bound"
        ),
    )
    cap.add_argument(
        "--profile",
        choices=tuple(CAPTURE_PROFILE_SPLITS),
        required=True,
        help="campaign capture contract: phase2 pilot or phase3 publication",
    )
    cap.add_argument("--split", action="append", required=True, help="NAME=TOKENS")
    cap.add_argument("--device", default="cuda")
    cap.add_argument("--out", type=Path, required=True)
    cap.add_argument(
        "--resume",
        action="store_true",
        help="verify the content-bound prefix and continue at the next durable shard",
    )

    derive = sub.add_parser("derive")
    derive.add_argument("--raw", type=Path, required=True)
    derive.add_argument("--out", type=Path, required=True)
    derive.add_argument(
        "--mode", action="append", choices=NORMALIZATION_MODES, required=True
    )
    derive.add_argument("--batch-size", type=int, default=4096)
    derive.add_argument("--tokens-per-shard", type=int, default=150_000)
    derive.add_argument(
        "--max-writer-residency-bytes",
        type=int,
        default=DEFAULT_MAX_WRITER_RESIDENCY_BYTES,
        help="refuse derivation if staging plus one pending shard exceeds this bound",
    )

    fit_transform = sub.add_parser("fit-transform")
    fit_transform.add_argument("--raw", type=Path, required=True)
    fit_transform.add_argument("--out", type=Path, required=True)
    fit_transform.add_argument(
        "--mode", action="append", choices=NORMALIZATION_MODES, required=True
    )
    fit_transform.add_argument("--batch-size", type=int, default=4096)

    verify = sub.add_parser("verify")
    verify.add_argument("--store", type=Path, action="append", required=True)

    estimate = sub.add_parser("estimate")
    estimate.add_argument("--split", action="append", required=True)
    estimate.add_argument("--site-dim", action="append", type=int, required=True)
    estimate.add_argument("--views", type=int, default=1)
    estimate.add_argument("--tokens-per-shard", type=int, default=150_000)
    estimate.add_argument("--row-id-width", type=int, default=3)

    args = parser.parse_args(argv)
    if args.command == "capture":
        capture(args)
    elif args.command == "derive":
        payload = derive_views(
            args.raw,
            args.out,
            args.mode,
            batch_size=args.batch_size,
            tokens_per_shard=args.tokens_per_shard,
            max_writer_residency_bytes=args.max_writer_residency_bytes,
        )
        print(json.dumps(payload, indent=2))
    elif args.command == "fit-transform":
        payload = fit_transform_artifacts(
            args.raw, args.out, args.mode, batch_size=args.batch_size
        )
        print(json.dumps(payload, indent=2))
    elif args.command == "verify":
        if len(args.store) == 1:
            root = args.store[0]
            payload = {
                split.name: StoreReader(root, split.name).verify()
                for split in root.iterdir()
                if split.is_dir() and (split / "split.json").exists()
            }
        else:
            payload = verify_alignment(args.store)
        print(json.dumps(payload, indent=2))
    else:
        split_sizes = parse_split_sizes(args.split)
        nbytes = estimate_store_bytes(split_sizes, args.site_dim, n_views=args.views)
        writer_residency = estimate_writer_residency_bytes(
            args.site_dim,
            tokens_per_shard=args.tokens_per_shard,
            row_id_width=args.row_id_width,
        )
        print(
            json.dumps(
                {
                    "bytes": nbytes,
                    "gib": nbytes / 2**30,
                    "writer": writer_residency,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
