"""Canonical tensor-payload hashing shared by durable artifact writers."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping

import torch


MODEL_STATE_DIGEST_CONTRACT = "sha256_merkle_16m_v1"
_MODEL_STATE_DIGEST_CHUNK_BYTES = 16 << 20
_MODEL_STATE_DIGEST_MAX_WORKERS = 16


def tensor_payload_digest(value: Any) -> str:
    """Hash nested JSON scalars and dense tensors without copying host bytes."""

    digest = hashlib.sha256()

    def canonical_json(payload: Any) -> str:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    def add(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(canonical_json(list(tensor.shape)).encode("ascii"))
            byte_array = tensor.reshape(-1).view(torch.uint8).numpy()
            digest.update(memoryview(byte_array))
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item):
                digest.update(str(key).encode("utf-8") + b"\0")
                add(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"seq\0")
            for child in item:
                add(child)
        else:
            digest.update(
                json.dumps(
                    item,
                    sort_keys=True,
                    allow_nan=False,
                    default=str,
                ).encode("utf-8")
            )
            digest.update(b"\0")

    add(value)
    return digest.hexdigest()


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
    jobs: list[
        tuple[int, bytes, bytes, tuple[int, ...], int, int, memoryview]
    ] = []
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
    "model_state_digest",
    "tensor_payload_digest",
]
