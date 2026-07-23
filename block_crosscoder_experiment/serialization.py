"""Canonical tensor-payload hashing shared by durable artifact writers."""

from __future__ import annotations

import hashlib
import math
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping

import torch


MODEL_STATE_DIGEST_CONTRACT = "sha256_merkle_16m_v1"
TYPED_PAYLOAD_DIGEST_CONTRACT = "sha256_typed_frames_v1"
_MODEL_STATE_DIGEST_CHUNK_BYTES = 16 << 20
_MODEL_STATE_DIGEST_MAX_WORKERS = 16


def _update_frame(
    digest: Any,
    type_name: str,
    payload: bytes | bytearray | memoryview,
) -> None:
    """Append one unambiguous typed, length-prefixed byte frame."""

    type_bytes = type_name.encode("ascii")
    digest.update(struct.pack(">I", len(type_bytes)))
    digest.update(type_bytes)
    view = memoryview(payload)
    if not view.c_contiguous:
        raise TypeError("typed digest frames require contiguous bytes")
    byte_view = view.cast("B")
    digest.update(struct.pack(">Q", byte_view.nbytes))
    digest.update(byte_view)


def typed_payload_digest(value: Any) -> str:
    """Hash JSON-like values and dense tensors with injective typed framing.

    Container kinds and lengths are authenticated, mappings require string
    keys and are ordered by their UTF-8 bytes, and tensor storage is streamed
    through a ``memoryview``.  Unsupported Python objects fail closed instead
    of being collapsed through ``str()``.
    """

    digest = hashlib.sha256()
    _update_frame(
        digest,
        "contract",
        TYPED_PAYLOAD_DIGEST_CONTRACT.encode("ascii"),
    )

    def add(item: Any) -> None:
        if torch.is_tensor(item):
            if item.layout != torch.strided:
                raise TypeError("typed payload digest requires dense strided tensors")
            tensor = item.detach().cpu().resolve_conj().resolve_neg().contiguous()
            _update_frame(digest, "value-type", b"tensor")
            _update_frame(digest, "tensor-dtype", str(tensor.dtype).encode("ascii"))
            _update_frame(digest, "tensor-rank", struct.pack(">Q", tensor.ndim))
            for dimension in tensor.shape:
                _update_frame(
                    digest,
                    "tensor-dimension",
                    struct.pack(">Q", int(dimension)),
                )
            byte_array = tensor.reshape(-1).view(torch.uint8).numpy()
            _update_frame(digest, "tensor-bytes", memoryview(byte_array))
        elif isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise TypeError("typed payload digest mappings require string keys")
            _update_frame(digest, "value-type", b"mapping")
            _update_frame(digest, "container-length", struct.pack(">Q", len(item)))
            for key in sorted(item, key=lambda candidate: candidate.encode("utf-8")):
                _update_frame(digest, "mapping-key", key.encode("utf-8"))
                add(item[key])
        elif isinstance(item, list):
            _update_frame(digest, "value-type", b"list")
            _update_frame(digest, "container-length", struct.pack(">Q", len(item)))
            for child in item:
                add(child)
        elif isinstance(item, tuple):
            _update_frame(digest, "value-type", b"tuple")
            _update_frame(digest, "container-length", struct.pack(">Q", len(item)))
            for child in item:
                add(child)
        elif item is None:
            _update_frame(digest, "none", b"")
        elif isinstance(item, bool):
            _update_frame(digest, "bool", b"\x01" if item else b"\x00")
        elif isinstance(item, int):
            _update_frame(digest, "int", str(item).encode("ascii"))
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("typed payload digest refuses non-finite floats")
            _update_frame(digest, "float64", struct.pack(">d", item))
        elif isinstance(item, str):
            _update_frame(digest, "string", item.encode("utf-8"))
        elif isinstance(item, (bytes, bytearray, memoryview)):
            _update_frame(digest, "bytes", memoryview(item))
        else:
            raise TypeError(
                "typed payload digest does not support "
                f"{type(item).__module__}.{type(item).__qualname__}"
            )

    add(value)
    return digest.hexdigest()


def tensor_payload_digest(value: Any) -> str:
    """Hash a nested tensor payload under the current typed-frame contract."""

    return typed_payload_digest(value)


def model_state_digest(
    state: Mapping[str, torch.Tensor],
    *,
    max_workers: int | None = None,
) -> str:
    """Hash model tensors with a deterministic parallel SHA-256 Merkle tree.

    ``max_workers`` affects execution only.  The fixed chunking and ordered root
    make the result independent of worker count and completion order.
    """

    if not isinstance(state, Mapping) or not state:
        raise ValueError("model state digest requires a nonempty mapping")
    if max_workers is not None and (type(max_workers) is not int or max_workers < 1):
        raise ValueError("model state digest max_workers must be a positive integer")
    if any(not isinstance(name, str) or not name for name in state):
        raise ValueError("model state digest requires nonempty string field names")
    arrays: list[Any] = []
    metadata: list[tuple[int, bytes, bytes, tuple[int, ...], int, int]] = []
    jobs: list[tuple[int, bytes, bytes, tuple[int, ...], int, int, memoryview]] = []
    for field_index, (name, tensor) in enumerate(state.items()):
        if not torch.is_tensor(tensor):
            raise ValueError("model state digest requires named tensors")
        if tensor.layout != torch.strided:
            raise ValueError("model state digest requires dense strided tensors")
        contiguous = tensor.detach().cpu().contiguous()
        byte_array = contiguous.reshape(-1).view(torch.uint8).numpy()
        arrays.append(byte_array)
        raw = memoryview(byte_array)
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(contiguous.dtype).encode("ascii")
        chunk_count = (
            len(raw) + _MODEL_STATE_DIGEST_CHUNK_BYTES - 1
        ) // _MODEL_STATE_DIGEST_CHUNK_BYTES
        metadata.append(
            (
                field_index,
                name_bytes,
                dtype_bytes,
                tuple(contiguous.shape),
                len(raw),
                chunk_count,
            )
        )
        for chunk_index, start in enumerate(
            range(0, len(raw), _MODEL_STATE_DIGEST_CHUNK_BYTES)
        ):
            jobs.append(
                (
                    field_index,
                    name_bytes,
                    dtype_bytes,
                    tuple(contiguous.shape),
                    len(raw),
                    chunk_index,
                    raw[start : start + _MODEL_STATE_DIGEST_CHUNK_BYTES],
                )
            )

    def digest_leaf(
        job: tuple[int, bytes, bytes, tuple[int, ...], int, int, memoryview],
    ) -> bytes:
        (
            field_index,
            name_bytes,
            dtype_bytes,
            shape,
            tensor_nbytes,
            chunk_index,
            chunk,
        ) = job
        leaf = hashlib.sha256()
        contract = MODEL_STATE_DIGEST_CONTRACT.encode("ascii")
        leaf.update(b"bsc-model-state-leaf\0")
        leaf.update(struct.pack(">I", len(contract)))
        leaf.update(contract)
        leaf.update(struct.pack(">I", field_index))
        leaf.update(struct.pack(">I", len(name_bytes)))
        leaf.update(name_bytes)
        leaf.update(struct.pack(">I", len(dtype_bytes)))
        leaf.update(dtype_bytes)
        leaf.update(struct.pack(">I", len(shape)))
        for dimension in shape:
            leaf.update(struct.pack(">Q", dimension))
        leaf.update(struct.pack(">Q", tensor_nbytes))
        leaf.update(struct.pack(">QQ", chunk_index, len(chunk)))
        leaf.update(chunk)
        return leaf.digest()

    if jobs:
        workers = min(
            max_workers or _MODEL_STATE_DIGEST_MAX_WORKERS,
            len(jobs),
            os.cpu_count() or 1,
        )
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="bsc-model-digest",
        ) as executor:
            leaf_digests = list(executor.map(digest_leaf, jobs))
    else:
        leaf_digests = []

    root = hashlib.sha256()
    contract = MODEL_STATE_DIGEST_CONTRACT.encode("ascii")
    root.update(b"bsc-model-state-digest\0")
    root.update(struct.pack(">I", len(contract)))
    root.update(contract)
    root.update(struct.pack(">Q", _MODEL_STATE_DIGEST_CHUNK_BYTES))
    root.update(struct.pack(">I", len(metadata)))
    leaf_offset = 0
    for field_index, name_bytes, dtype_bytes, shape, nbytes, chunk_count in metadata:
        root.update(struct.pack(">I", field_index))
        root.update(struct.pack(">I", len(name_bytes)))
        root.update(name_bytes)
        root.update(struct.pack(">I", len(dtype_bytes)))
        root.update(dtype_bytes)
        root.update(struct.pack(">I", len(shape)))
        for dimension in shape:
            root.update(struct.pack(">Q", dimension))
        root.update(struct.pack(">QQ", nbytes, chunk_count))
        for chunk_index, digest in enumerate(
            leaf_digests[leaf_offset : leaf_offset + chunk_count]
        ):
            root.update(struct.pack(">Q", chunk_index))
            root.update(digest)
        leaf_offset += chunk_count
    if leaf_offset != len(leaf_digests):
        raise RuntimeError("model state digest leaf accounting mismatch")
    return root.hexdigest()


__all__ = [
    "MODEL_STATE_DIGEST_CONTRACT",
    "TYPED_PAYLOAD_DIGEST_CONTRACT",
    "model_state_digest",
    "tensor_payload_digest",
    "typed_payload_digest",
]
