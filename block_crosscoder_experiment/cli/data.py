"""Capture one immutable raw activation stream and derive aligned views."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import re
import shutil
import socket
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

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
VIEW_MANIFEST_NAME = "view.json"
DEFAULT_MAX_WRITER_RESIDENCY_BYTES = 8 * 1024**3
DEFAULT_PREWRITE_METADATA_RESERVE_BYTES = 1024**2
DEFAULT_FREE_SPACE_FLOOR_FRAC = 0.15
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


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _enforce_prewrite_storage(
    destination: Path,
    required_bytes: int,
    *,
    operation: str,
) -> dict[str, int | str]:
    """Refuse a producer before writing to its actual destination device."""

    if required_bytes < 0:
        raise ValueError("prewrite storage requirement cannot be negative")
    parent = _nearest_existing_parent(destination)
    status = parent.stat()
    usage = shutil.disk_usage(parent)
    floor_reserve = int(usage.total * DEFAULT_FREE_SPACE_FLOOR_FRAC)
    available = max(0, int(usage.free) - floor_reserve)
    result: dict[str, int | str] = {
        "operation": operation,
        "destination": str(destination.expanduser().resolve()),
        "filesystem_path": str(parent),
        "device": int(status.st_dev),
        "required_bytes": int(required_bytes),
        "raw_free_bytes": int(usage.free),
        "free_space_floor_bytes": floor_reserve,
        "available_above_floor_bytes": available,
    }
    if required_bytes > available:
        raise ValueError(
            f"{operation} storage preflight failed on destination device "
            f"{status.st_dev}: required={required_bytes} bytes, "
            f"available_above_15pct_floor={available} bytes "
            f"(raw_free={usage.free}, floor={floor_reserve}) at {parent}"
        )
    return result


def _transform_storage_bytes(transform: Whitener) -> int:
    tensor_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            transform.mean,
            transform.W,
            transform.ridge,
            transform.eigenvalues,
        )
    )
    return tensor_bytes + DEFAULT_PREWRITE_METADATA_RESERVE_BYTES


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
            # The consumer has requested another batch, so it has finished
            # copying this transient pinned view into the writer staging
            # buffer. Release the yielded aliases before allocating the next
            # lookahead destination; otherwise three pinned batches coexist.
            del host, row_ids
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
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
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _producer_lock_path(output_root: Path) -> Path:
    resolved = output_root.expanduser().resolve()
    return resolved.parent / f".{resolved.name}.bsc-producer.lock"


@contextmanager
def _producer_lock(output_root: Path, *, operation: str) -> Iterator[None]:
    """Hold one nonblocking producer lease outside an immutable output tree."""

    lock_path = _producer_lock_path(output_root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "unknown owner"
            raise ValueError(
                f"{operation} output is locked by another producer at {lock_path}: "
                f"{owner}"
            ) from exc
        owner = {
            "schema": "bsc-data-producer-lock-v1",
            "operation": operation,
            "output_root": str(output_root.expanduser().resolve()),
            "host": socket.gethostname(),
            "pid": os.getpid(),
        }
        handle.seek(0)
        handle.truncate()
        json.dump(owner, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _save_whitener_atomic(transform: Whitener, path: Path) -> None:
    """Publish deterministic transform bytes through a unique temporary name."""

    path.parent.mkdir(parents=True, exist_ok=True)
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
            # A file object gives torch.save the stable ``archive/`` member
            # prefix; a random path would leak its basename into the ZIP bytes.
            torch.save(
                {
                    "mean": transform.mean,
                    "W": transform.W,
                    "ridge": transform.ridge,
                    "eigenvalues": transform.eigenvalues,
                    "sites": list(transform.sites),
                    "n_fit_tokens": transform.n_fit_tokens,
                    "meta": transform.meta,
                    "hash": transform.hash,
                },
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory = os.open(path.parent, flags)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _dependency_versions() -> dict[str, str]:
    names = (
        "block-crosscoder-experiment",
        "datasets",
        "huggingface-hub",
        "numpy",
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


def expected_capture_source_contract(values: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize the capture source contract declared by a resolved cell.

    This intentionally lives beside the capture producer so planning, capture,
    preflight, and cell execution can share one canonical map from immutable
    study decisions to the JSON capture contract.
    """

    def sequence(name: str) -> tuple[Any, ...]:
        value = values[name]
        if not isinstance(value, (tuple, list)):
            raise ValueError(f"{name} must be a sequence")
        return tuple(value)

    hooks = tuple(str(item) for item in sequence("data.store_sites"))
    if not hooks:
        raise ValueError("data.store_sites cannot be empty")

    def per_site(name: str) -> tuple[str, ...]:
        items = tuple(str(item) for item in sequence(name))
        if len(items) == 1:
            return items * len(hooks)
        if len(items) != len(hooks):
            raise ValueError(f"{name} must contain one value or one per store site")
        return items

    def singleton(name: str) -> str:
        items = sequence(name)
        if len(items) != 1:
            raise ValueError(f"{name} must contain exactly one value")
        return str(items[0])

    drop_policy = str(values["data.context_drop_policy"])
    if drop_policy == "none":
        drop_positions = 0
    elif drop_policy == "drop_bos_position_0":
        drop_positions = 1
    else:
        raise ValueError(f"unsupported data.context_drop_policy {drop_policy!r}")
    models = per_site("data.source_models")
    revisions = per_site("data.source_model_revisions")
    base: dict[str, Any] = {
        "sources": [
            {"model": model, "revision": revision, "hook": hook}
            for model, revision, hook in zip(models, revisions, hooks)
        ],
        "corpus": singleton("data.corpus"),
        "corpus_config": singleton("data.corpus_config"),
        "corpus_revision": singleton("data.corpus_revision"),
        "corpus_split": singleton("data.corpus_split"),
        "context": int(values["data.context_length"]),
        "drop_positions": drop_positions,
        "tokenizer_hashes": [str(item) for item in sequence("data.tokenizer_hashes")],
        "tokenizer_contract": str(values["data.tokenizer_contract"]),
        "store_contract_version": str(values["data.store_contract_version"]),
        "alignment_version": str(values["data.alignment_version"]),
        "alignment_audit": str(values["data.alignment_audit"]),
    }
    capture_pairs = sequence("data.capture_contract")
    capture: dict[str, Any] = {}
    for item in capture_pairs:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("data.capture_contract entries must be key/value pairs")
        key, value = item
        if not isinstance(key, str) or not key or key in capture:
            raise ValueError("data.capture_contract has an invalid or duplicate key")
        capture[key] = list(value) if isinstance(value, (tuple, list)) else value
    overlap = set(base).intersection(capture)
    if overlap:
        raise ValueError(
            "data.capture_contract duplicates resolved fields: "
            + ", ".join(sorted(overlap))
        )
    return {**base, **capture}


def expected_capture_allocation(
    values: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, dict[str, int]]]:
    """Rebuild the canonical whole-sequence allocation for a resolved cell."""

    declared = values["data.split_sizes"]
    if not isinstance(declared, (tuple, list)) or not declared:
        raise ValueError("data.split_sizes must be a nonempty sequence")
    drop_policy = str(values["data.context_drop_policy"])
    if drop_policy == "none":
        drop_positions = 0
    elif drop_policy == "drop_bos_position_0":
        drop_positions = 1
    else:
        raise ValueError(f"unsupported data.context_drop_policy {drop_policy!r}")
    tokens_per_sequence = int(values["data.context_length"]) - drop_positions
    if tokens_per_sequence <= 0:
        raise ValueError("capture tokens per sequence must be positive")
    split_sizes: dict[str, int] = {}
    for item in declared:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("data.split_sizes entries must be name/count pairs")
        name, count = str(item[0]), int(item[1])
        if not name or name in split_sizes or count <= 0:
            raise ValueError("data.split_sizes has an invalid or duplicate entry")
        split_sizes[name] = count
    return tuple(split_sizes), whole_sequence_split_plan(
        split_sizes, tokens_per_sequence
    )


def validate_capture_manifest(capture: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate an embedded capture binding and its duplicated fields."""

    if capture.get("schema") != "bsc-capture-manifest-v1":
        raise ValueError("capture manifest has an unknown schema")
    source = capture.get("source")
    if not isinstance(source, dict) or capture.get("source_hash") != _canonical_hash(
        source
    ):
        raise ValueError("capture manifest source hash mismatch")
    binding = capture.get("capture_binding")
    if (
        not isinstance(binding, dict)
        or binding.get("schema") != "bsc-capture-binding-v1"
    ):
        raise ValueError("capture manifest lacks its canonical embedded binding")
    required_binding_keys = {
        "schema",
        "campaign_profile",
        "source_hash",
        "split_order",
        "split_plan",
        "capture_implementation",
        "sites",
        "site_dims",
        "d_model",
        "physical_store_format_version",
        "batch_rows",
        "write_batch_tokens",
        "tokens_per_shard",
        "writer_pipeline",
        "capture_transfer_pipeline",
    }
    if set(binding) != required_binding_keys:
        raise ValueError("capture binding has missing or unexpected fields")
    implementation = binding.get("capture_implementation")
    if (
        not isinstance(implementation, dict)
        or implementation.get("schema") != "bsc-capture-implementation-v1"
    ):
        raise ValueError("capture implementation contract is malformed")
    binding_sha256 = _canonical_hash(binding)
    if capture.get("capture_binding_sha256") != binding_sha256:
        raise ValueError("capture binding digest mismatch")
    sites = binding.get("sites")
    site_dims = binding.get("site_dims")
    d_model = binding.get("d_model")
    positive_integer_fields = (
        binding.get("batch_rows"),
        binding.get("write_batch_tokens"),
        binding.get("tokens_per_shard"),
    )
    if (
        not isinstance(sites, list)
        or sites != list(range(len(sites)))
        or not isinstance(site_dims, list)
        or len(site_dims) != len(sites)
        or not site_dims
        or any(
            not isinstance(width, int) or isinstance(width, bool) or width <= 0
            for width in site_dims
        )
        or not isinstance(d_model, int)
        or isinstance(d_model, bool)
        or d_model != max(site_dims)
        or binding.get("physical_store_format_version") != STORE_FORMAT_VERSION
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in positive_integer_fields
        )
        or not isinstance(binding.get("writer_pipeline"), dict)
        or not isinstance(binding.get("capture_transfer_pipeline"), dict)
    ):
        raise ValueError("capture binding geometry or execution fields are malformed")
    duplicated = {
        "source_hash": capture.get("source_hash"),
        "split_order": capture.get("split_order"),
        "split_plan": capture.get("split_plan"),
        "capture_implementation": capture.get("capture_implementation"),
    }
    mismatches = {
        key: {"manifest": value, "binding": binding.get(key)}
        for key, value in duplicated.items()
        if value != binding.get(key)
    }
    if capture.get("splits") != capture.get("split_plan"):
        mismatches["splits"] = {
            "manifest": capture.get("splits"),
            "binding": binding.get("split_plan"),
        }
    profile = binding.get("campaign_profile")
    required_roles = CAPTURE_PROFILE_SPLITS.get(str(profile))
    if (
        required_roles is None
        or capture.get("split_order") != list(required_roles)
        or not isinstance(capture.get("split_plan"), dict)
        or set(capture.get("split_plan", {})) != set(required_roles or ())
    ):
        mismatches["campaign_profile"] = {
            "manifest": capture.get("split_order"),
            "binding": profile,
        }
    if mismatches:
        raise ValueError(
            "capture manifest differs from its embedded binding: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return dict(binding)


def validate_derived_view_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate a derived-view root envelope without reading shard bytes."""

    required_keys = {
        "schema",
        "mode",
        "transform_hash",
        "whitener_sha256",
        "source_capture_sha256",
        "source_capture_manifest_sha256",
        "source_capture",
        "source_hash",
        "capture_binding_sha256",
        "split_order",
        "splits",
        "view_manifest_sha256",
    }
    if set(manifest) != required_keys:
        raise ValueError("derived-view root manifest has missing or unexpected fields")
    claimed_digest = manifest.get("view_manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("view_manifest_sha256", None)
    if (
        manifest.get("schema") != "bsc-derived-view-manifest-v1"
        or not isinstance(claimed_digest, str)
        or claimed_digest != _canonical_hash(unsigned)
    ):
        raise ValueError("derived-view root manifest digest mismatch")
    source_capture = manifest.get("source_capture")
    if not isinstance(source_capture, dict):
        raise ValueError("derived-view root manifest lacks embedded capture evidence")
    capture_binding = validate_capture_manifest(source_capture)
    if (
        manifest.get("source_capture_manifest_sha256")
        != _canonical_hash(source_capture)
        or manifest.get("source_hash") != source_capture.get("source_hash")
        or manifest.get("capture_binding_sha256") != _canonical_hash(capture_binding)
    ):
        raise ValueError("derived-view embedded capture binding mismatch")
    mode = manifest.get("mode")
    split_order = manifest.get("split_order")
    split_records = manifest.get("splits")
    if (
        mode not in NORMALIZATION_MODES
        or not isinstance(split_order, list)
        or not split_order
        or any(not isinstance(split, str) or not split for split in split_order)
        or len(set(split_order)) != len(split_order)
        or not isinstance(split_records, dict)
        or set(split_records) != set(split_order)
    ):
        raise ValueError("derived-view root manifest has malformed roles")
    return dict(manifest)


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
    split_sizes: dict[str, int],
    site_dims: Iterable[int],
    *,
    n_views: int = 1,
    row_id_width: int = 3,
) -> int:
    # Payload only. Safetensors headers and int64 row IDs are included as a
    # conservative fixed per-token allowance.
    dimensions = tuple(int(width) for width in site_dims)
    if not dimensions or any(width <= 0 for width in dimensions):
        raise ValueError("site dimensions must be nonempty and positive")
    if n_views <= 0:
        raise ValueError("n_views must be positive")
    if row_id_width <= 0:
        raise ValueError("row_id_width must be positive")
    per_token = 2 * len(dimensions) * max(dimensions) + row_id_width * 8
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


def _derive_views_unlocked(
    raw_root: Path,
    out_root: Path,
    modes: Iterable[str],
    *,
    batch_size: int = 4096,
    tokens_per_shard: int = 150_000,
    max_writer_residency_bytes: int = DEFAULT_MAX_WRITER_RESIDENCY_BYTES,
    resume: bool = False,
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
    if not isinstance(capture, dict):
        raise ValueError("capture.json must contain an object")
    capture_binding = validate_capture_manifest(capture)
    source_hash = str(capture["source_hash"])
    # Authenticate the complete raw role set and canonical row allocation
    # before fitting any transform or touching the output tree.
    verify_store_root(raw_root)
    capture_sha256 = _file_sha256(capture_path)
    capture_manifest_sha256 = _canonical_hash(capture)
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
    available_splits = {
        path.name
        for path in raw_root.iterdir()
        if path.is_dir() and (path / "split.json").exists()
    }
    declared_order = capture.get("split_order")
    if (
        not isinstance(declared_order, list)
        or not declared_order
        or any(not isinstance(item, str) or not item for item in declared_order)
        or len(set(declared_order)) != len(declared_order)
        or set(declared_order) != available_splits
    ):
        raise ValueError("capture split order differs from its materialized raw splits")
    split_names = tuple(declared_order)
    # Verification authenticates immutable source bytes and does not depend on
    # the requested normalization mode.  Reusing these stateless readers avoids
    # re-hashing every raw shard once per derived view.
    source_readers = {split: StoreReader(raw_root, split) for split in split_names}
    results: dict[str, dict] = {}
    for mode in modes:
        view_root = out_root / mode
        if resume:
            view_root.mkdir(parents=True, exist_ok=True)
            allowed_entries = set(split_names) | {"whitener.pt", VIEW_MANIFEST_NAME}
            foreign = sorted(
                path.name
                for path in view_root.iterdir()
                if path.name not in allowed_entries
            )
            if foreign:
                raise ValueError(
                    f"cannot resume derived view {mode!r}; unbound entries: {foreign}"
                )
        else:
            _ensure_empty(view_root)
        # Only immutable content identities enter the transform hash.  The
        # local raw-store path is a locator kept in derived shard manifests,
        # not a scientific identity (moving a store must not change W).
        transform_source_meta = {
            "source_capture_sha256": capture_sha256,
            "source_capture_manifest_sha256": capture_manifest_sha256,
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
        transform_path = view_root / "whitener.pt"
        _enforce_prewrite_storage(
            transform_path,
            0 if transform_path.is_file() else _transform_storage_bytes(transform),
            operation=f"derive {mode!r} transform",
        )
        if transform_path.is_file():
            try:
                existing_transform = Whitener.load(transform_path)
            except (OSError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"cannot resume derived view {mode!r}; whitener is invalid: {exc}"
                ) from exc
            if existing_transform.hash != transform.hash:
                raise ValueError(
                    f"cannot resume derived view {mode!r}; whitener binding differs"
                )
        elif resume and any(view_root.iterdir()):
            raise ValueError(
                f"cannot resume derived view {mode!r}; output exists without "
                "its bound whitener.pt"
            )
        else:
            _save_whitener_atomic(transform, transform_path)
        prior_view_manifest: dict[str, Any] | None = None
        prior_view_manifest_path = view_root / VIEW_MANIFEST_NAME
        if resume and prior_view_manifest_path.is_file():
            try:
                prior_payload = json.loads(prior_view_manifest_path.read_text())
                if not isinstance(prior_payload, dict):
                    raise ValueError("manifest is not an object")
                prior_view_manifest = validate_derived_view_manifest(prior_payload)
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"cannot resume derived view {mode!r}; root manifest is invalid: "
                    f"{exc}"
                ) from exc
            invariant_binding = {
                "mode": mode,
                "transform_hash": transform.hash,
                "whitener_sha256": _file_sha256(transform_path),
                "source_capture_sha256": capture_sha256,
                "source_capture_manifest_sha256": capture_manifest_sha256,
                "source_capture": capture,
                "source_hash": source_hash,
                "capture_binding_sha256": _canonical_hash(capture_binding),
                "split_order": list(split_names),
            }
            if any(
                prior_view_manifest.get(key) != value
                for key, value in invariant_binding.items()
            ):
                raise ValueError(
                    f"cannot resume derived view {mode!r}; root binding differs"
                )
        view_splits = {}
        missing_split_seen = False
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
            split_dir = view_root / split
            if resume and split_dir.exists():
                if missing_split_seen:
                    raise ValueError(
                        f"cannot resume derived view {mode!r}; complete splits are "
                        f"not an ordered prefix (unexpected {split!r})"
                    )
                manifest_path = split_dir / "split.json"
                if not split_dir.is_dir() or not manifest_path.is_file():
                    raise ValueError(
                        f"cannot resume derived view {mode!r}/{split}; partial split "
                        f"has no complete manifest; remove exactly {split_dir} after "
                        "review, then rerun with --resume"
                    )
                try:
                    existing_reader = StoreReader(
                        view_root,
                        split,
                        expected_whitener_hash=transform.hash,
                    )
                    existing_reader.verify()
                except (OSError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"cannot resume derived view {mode!r}/{split}; completed "
                        f"split is invalid: {exc}"
                    ) from exc
                mismatches = {
                    "sites": list(existing_reader.sites) != list(reader.sites),
                    "site_dims": list(existing_reader.site_dims)
                    != list(reader.site_dims),
                    "d_model": existing_reader.d_model != reader.d_model,
                    "n_tokens": existing_reader.n_tokens != reader.n_tokens,
                    "tokens_per_shard": (
                        existing_reader.manifest.get("tokens_per_shard")
                        != tokens_per_shard
                    ),
                    "row_stream_sha256": (
                        existing_reader.manifest.get("row_stream_sha256")
                        != reader.manifest.get("row_stream_sha256")
                    ),
                    "meta": existing_reader.manifest.get("meta") != meta,
                }
                failed = sorted(name for name, differs in mismatches.items() if differs)
                if failed:
                    raise ValueError(
                        f"cannot resume derived view {mode!r}/{split}; completed "
                        f"split binding differs in {failed}"
                    )
                if prior_view_manifest is not None:
                    expected_root_record = {
                        "manifest_sha256": existing_reader.manifest["manifest_sha256"],
                        "content_stream_sha256": existing_reader.manifest[
                            "content_stream_sha256"
                        ],
                        "row_stream_sha256": existing_reader.manifest[
                            "row_stream_sha256"
                        ],
                        "n_tokens": existing_reader.n_tokens,
                    }
                    if prior_view_manifest["splits"].get(split) != expected_root_record:
                        raise ValueError(
                            f"cannot resume derived view {mode!r}/{split}; split "
                            "differs from the authenticated root manifest"
                        )
                view_splits[split] = existing_reader.manifest
                continue
            missing_split_seen = True
            _enforce_prewrite_storage(
                split_dir,
                estimate_store_bytes(
                    {split: reader.n_tokens},
                    reader.site_dims,
                    n_views=1,
                    row_id_width=int(reader.manifest["row_id_width"]),
                ),
                operation=f"derive {mode!r}/{split}",
            )
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
        view_manifest = {
            "schema": "bsc-derived-view-manifest-v1",
            "mode": mode,
            "transform_hash": transform.hash,
            "whitener_sha256": _file_sha256(transform_path),
            "source_capture_sha256": capture_sha256,
            "source_capture_manifest_sha256": capture_manifest_sha256,
            "source_capture": capture,
            "source_hash": source_hash,
            "capture_binding_sha256": _canonical_hash(capture_binding),
            "split_order": list(split_names),
            "splits": {
                split: {
                    "manifest_sha256": manifest["manifest_sha256"],
                    "content_stream_sha256": manifest["content_stream_sha256"],
                    "row_stream_sha256": manifest["row_stream_sha256"],
                    "n_tokens": manifest["n_tokens"],
                }
                for split, manifest in view_splits.items()
            },
        }
        view_manifest["view_manifest_sha256"] = _canonical_hash(view_manifest)
        view_manifest_path = view_root / VIEW_MANIFEST_NAME
        if view_manifest_path.is_file():
            try:
                existing_view_manifest = json.loads(view_manifest_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"cannot resume derived view {mode!r}; view manifest is invalid"
                ) from exc
            if existing_view_manifest != view_manifest:
                raise ValueError(
                    f"cannot resume derived view {mode!r}; view manifest binding differs"
                )
        else:
            _enforce_prewrite_storage(
                view_manifest_path,
                DEFAULT_PREWRITE_METADATA_RESERVE_BYTES,
                operation=f"derive {mode!r} root manifest",
            )
            _atomic_json(view_manifest_path, view_manifest)
        results[mode] = {
            "whitener_hash": transform.hash,
            "view_manifest": view_manifest,
            "writer_pipeline": {
                "contract": "one_pending_shard_v1",
                **writer_residency,
                "max_writer_residency_bytes": max_writer_residency_bytes,
            },
            "splits": view_splits,
        }
    return results


def derive_views(
    raw_root: Path,
    out_root: Path,
    modes: Iterable[str],
    *,
    batch_size: int = 4096,
    tokens_per_shard: int = 150_000,
    max_writer_residency_bytes: int = DEFAULT_MAX_WRITER_RESIDENCY_BYTES,
    resume: bool = False,
) -> dict[str, dict]:
    with _producer_lock(out_root, operation="derive"):
        return _derive_views_unlocked(
            raw_root,
            out_root,
            modes,
            batch_size=batch_size,
            tokens_per_shard=tokens_per_shard,
            max_writer_residency_bytes=max_writer_residency_bytes,
            resume=resume,
        )


def _fit_transform_artifacts_unlocked(
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
    if not isinstance(capture, dict):
        raise ValueError("capture.json must contain an object")
    validate_capture_manifest(capture)
    source_hash = str(capture["source_hash"])
    verify_store_root(raw_root)

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
    capture_manifest_sha256 = _canonical_hash(capture)
    common_meta = {
        "source_capture_sha256": capture_sha256,
        "source_capture_manifest_sha256": capture_manifest_sha256,
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
        transform_path = artifact_root / "whitener.pt"
        manifest_path = artifact_root / "transform.json"
        _enforce_prewrite_storage(
            artifact_root,
            (
                0
                if transform_path.is_file() and manifest_path.is_file()
                else _transform_storage_bytes(transform)
            ),
            operation=f"fit-transform {mode!r}",
        )
        artifact_root.mkdir(parents=True, exist_ok=True)
        if transform_path.exists():
            existing = Whitener.load(transform_path)
            if existing.hash != transform.hash:
                raise ValueError(f"content-address collision at {artifact_root}")
        else:
            _save_whitener_atomic(transform, transform_path)
        manifest = {
            "schema": "bsc-transform-artifact-v1",
            "mode": mode,
            "transform_hash": transform.hash,
            "whitener_sha256": _file_sha256(transform_path),
            "source_capture_sha256": capture_sha256,
            "source_capture_manifest_sha256": capture_manifest_sha256,
            "source_capture": capture,
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
        encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        if manifest_path.exists():
            if manifest_path.read_text() != encoded:
                raise ValueError(
                    f"existing transform manifest differs at {manifest_path}"
                )
        else:
            _atomic_json(manifest_path, manifest)
        results[mode] = {
            **manifest,
            "path": str(transform_path),
            "manifest": str(manifest_path),
        }
    return results


def fit_transform_artifacts(
    raw_root: Path,
    out_root: Path,
    modes: Iterable[str],
    *,
    batch_size: int = 4096,
) -> dict[str, dict]:
    with _producer_lock(out_root, operation="fit-transform"):
        return _fit_transform_artifacts_unlocked(
            raw_root,
            out_root,
            modes,
            batch_size=batch_size,
        )


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
                tuple(reader.site_dims),
                reader.manifest["d_model"],
            )
            for reader in readers
        }
        if len(identities) != 1:
            raise ValueError(f"row alignment mismatch for split {split}")
        result[split] = {
            "n_tokens": readers[0].n_tokens,
            "row_stream_sha256": readers[0].manifest["row_stream_sha256"],
            "site_dims": list(readers[0].site_dims),
        }
    return result


def _verify_raw_store_root(root: Path) -> dict[str, object]:
    """Verify one raw capture root, including its complete declared role set."""

    if not root.is_dir():
        raise ValueError(f"activation-store root does not exist: {root}")
    split_names = tuple(
        path.name
        for path in sorted(root.iterdir())
        if path.is_dir() and (path / "split.json").is_file()
    )
    if not split_names:
        raise ValueError("activation store exposes no splits")
    capture_path = root / CAPTURE_MANIFEST_NAME
    if not capture_path.is_file():
        raise ValueError("single-store verification requires capture.json")
    try:
        capture_payload = json.loads(capture_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read capture manifest: {exc}") from exc
    if not isinstance(capture_payload, dict):
        raise ValueError("capture manifest must be a JSON object")
    binding = validate_capture_manifest(capture_payload)
    declared_order = tuple(str(item) for item in capture_payload["split_order"])
    if split_names != tuple(sorted(declared_order)):
        raise ValueError(
            "activation store split set differs from the declared capture roles"
        )
    result: dict[str, object] = {}
    for split in declared_order:
        reader = StoreReader(root, split)
        allocation = capture_payload["split_plan"][split]
        source = capture_payload["source"]
        drop_positions = source.get("drop_positions")
        if not isinstance(drop_positions, int) or isinstance(drop_positions, bool):
            raise ValueError("capture source drop_positions is malformed")
        verification = reader.verify(
            expected_row_identity={
                "sequence_start": allocation["sequence_start"],
                "sequence_stop_exclusive": allocation["sequence_stop_exclusive"],
                "tokens_per_sequence": allocation["tokens_per_sequence"],
                "position_start": drop_positions,
            }
        )
        meta = reader.manifest.get("meta", {})
        expected_meta = {
            "split_requested_tokens": allocation["requested_tokens"],
            "split_actual_tokens": allocation["actual_tokens"],
            "sequence_start": allocation["sequence_start"],
            "sequence_stop_exclusive": allocation["sequence_stop_exclusive"],
            "tokens_per_sequence": allocation["tokens_per_sequence"],
            "ordered_split_allocation": list(declared_order),
            "capture_binding_sha256": _canonical_hash(binding),
        }
        if any(meta.get(key) != value for key, value in expected_meta.items()):
            raise ValueError(f"split {split!r} differs from its capture allocation")
        if reader.n_tokens != int(allocation["actual_tokens"]):
            raise ValueError(
                f"split {split!r} row count differs from capture allocation"
            )
        if reader.whitener_hash != f"raw:{capture_payload['source_hash']}":
            raise ValueError(f"split {split!r} is not bound to its capture source")
        if (
            list(reader.sites) != binding["sites"]
            or list(reader.site_dims) != binding["site_dims"]
            or reader.d_model != binding["d_model"]
        ):
            raise ValueError(f"split {split!r} geometry differs from capture binding")
        result[split] = verification
    return result


def _verify_derived_store_root(root: Path) -> dict[str, object]:
    """Verify one materialized normalization view from its root manifest."""

    manifest_path = root / VIEW_MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read derived-view manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("derived-view manifest must be an object")
    manifest = validate_derived_view_manifest(manifest)
    mode = manifest.get("mode")
    split_order = manifest.get("split_order")
    split_records = manifest.get("splits")
    assert isinstance(mode, str)
    assert isinstance(split_order, list)
    assert isinstance(split_records, dict)
    allowed = set(split_order) | {"whitener.pt", VIEW_MANIFEST_NAME}
    actual = {path.name for path in root.iterdir()}
    if actual != allowed:
        raise ValueError(
            "derived-view root entries differ from its manifest: "
            + json.dumps(
                {"expected": sorted(allowed), "actual": sorted(actual)},
                sort_keys=True,
            )
        )
    transform_path = root / "whitener.pt"
    if _file_sha256(transform_path) != manifest.get("whitener_sha256"):
        raise ValueError("derived-view whitener file digest mismatch")
    transform = Whitener.load(transform_path)
    if transform.hash != manifest.get("transform_hash") or transform.mode != mode:
        raise ValueError("derived-view whitener binding mismatch")
    if (
        transform.meta.get("source_capture_sha256")
        != manifest.get("source_capture_sha256")
        or transform.meta.get("source_capture_manifest_sha256")
        != manifest.get("source_capture_manifest_sha256")
        or transform.meta.get("source_hash") != manifest.get("source_hash")
    ):
        raise ValueError("derived-view source binding mismatch")

    result: dict[str, object] = {}
    for split in split_order:
        record = split_records[split]
        if not isinstance(record, dict):
            raise ValueError(f"derived-view split record {split!r} is malformed")
        if set(record) != {
            "manifest_sha256",
            "content_stream_sha256",
            "row_stream_sha256",
            "n_tokens",
        }:
            raise ValueError(f"derived-view split record {split!r} is malformed")
        reader = StoreReader(root, split, expected_whitener_hash=transform.hash)
        meta = reader.manifest.get("meta", {})
        identity_fields = (
            "sequence_start",
            "sequence_stop_exclusive",
            "tokens_per_sequence",
            "drop_positions",
        )
        if any(
            not isinstance(meta.get(name), int) or isinstance(meta.get(name), bool)
            for name in identity_fields
        ):
            raise ValueError(
                f"derived-view split {split!r} lacks canonical row allocation"
            )
        verification = reader.verify(
            expected_row_identity={
                "sequence_start": meta["sequence_start"],
                "sequence_stop_exclusive": meta["sequence_stop_exclusive"],
                "tokens_per_sequence": meta["tokens_per_sequence"],
                "position_start": meta["drop_positions"],
            }
        )
        expected_record = {
            "manifest_sha256": reader.manifest["manifest_sha256"],
            "content_stream_sha256": reader.manifest["content_stream_sha256"],
            "row_stream_sha256": reader.manifest["row_stream_sha256"],
            "n_tokens": reader.n_tokens,
        }
        if record != expected_record:
            raise ValueError(
                f"derived-view split {split!r} differs from its root manifest"
            )
        if (
            meta.get("derived_view") is not True
            or meta.get("normalization") != mode
            or meta.get("source_capture_sha256")
            != manifest.get("source_capture_sha256")
            or meta.get("source_hash") != manifest.get("source_hash")
            or tuple(reader.sites) != transform.sites
            or tuple(reader.site_dims) != transform.site_dims
        ):
            raise ValueError(
                f"derived-view split {split!r} has a divergent source/geometry binding"
            )
        result[split] = verification
    return result


def verify_store_root(root: Path) -> dict[str, object]:
    """Verify one complete raw capture or one complete materialized view."""

    if not root.is_dir():
        raise ValueError(f"activation-store root does not exist: {root}")
    if (root / CAPTURE_MANIFEST_NAME).is_file():
        return _verify_raw_store_root(root)
    if (root / VIEW_MANIFEST_NAME).is_file():
        return _verify_derived_store_root(root)
    raise ValueError(
        f"single-store verification requires {CAPTURE_MANIFEST_NAME} or "
        f"{VIEW_MANIFEST_NAME}"
    )


def _capture_unlocked(
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
    full_capture_bytes = estimate_store_bytes(
        {split: int(spec["actual_tokens"]) for split, spec in split_plan.items()},
        site_dims,
        n_views=1,
        row_id_width=3,
    )
    _enforce_prewrite_storage(
        args.out,
        (
            DEFAULT_PREWRITE_METADATA_RESERVE_BYTES
            if args.resume
            else full_capture_bytes
        ),
        operation="capture",
    )

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
        for pattern in (
            f".{CAPTURE_STATE_NAME}.*.tmp",
            f".{CAPTURE_MANIFEST_NAME}.*.tmp",
        ):
            for temporary in args.out.glob(pattern):
                temporary.unlink()
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

    remaining_capture_bytes = estimate_store_bytes(
        {
            split: int(split_plan[split]["actual_tokens"])
            - int(persisted_by_split[split])
            for split in split_order
        },
        site_dims,
        n_views=1,
        row_id_width=3,
    )
    _enforce_prewrite_storage(
        args.out,
        remaining_capture_bytes,
        operation="capture resume remainder" if args.resume else "capture remainder",
    )
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
            "capture_binding": binding,
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
                # Drop the exhausted views before advancing the capture
                # generator. RHS evaluation of ``next`` otherwise retains the
                # prior pinned allocation while the lookahead allocates its
                # following destination, violating the two-buffer bound.
                pending_x = None
                pending_ids = None
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
        "capture_binding": binding,
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


def capture(
    args: argparse.Namespace,
    *,
    failure_injector: Callable[[str, int], None] | None = None,
) -> dict[str, object]:
    with _producer_lock(args.out, operation="capture"):
        return _capture_unlocked(args, failure_injector=failure_injector)


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
        "--resume",
        action="store_true",
        help=(
            "verify and reuse a complete ordered split prefix, then continue "
            "missing derived splits"
        ),
    )
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
            resume=args.resume,
        )
        print(json.dumps(payload, indent=2))
    elif args.command == "fit-transform":
        payload = fit_transform_artifacts(
            args.raw, args.out, args.mode, batch_size=args.batch_size
        )
        print(json.dumps(payload, indent=2))
    elif args.command == "verify":
        try:
            if len(args.store) == 1:
                payload = verify_store_root(args.store[0])
            else:
                for root in args.store:
                    verify_store_root(root)
                payload = verify_alignment(args.store)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            parser.exit(2, f"error: {exc}\n")
        print(json.dumps(payload, indent=2))
    else:
        split_sizes = parse_split_sizes(args.split)
        nbytes = estimate_store_bytes(
            split_sizes,
            args.site_dim,
            n_views=args.views,
            row_id_width=args.row_id_width,
        )
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
