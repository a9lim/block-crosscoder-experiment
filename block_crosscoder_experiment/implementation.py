"""Small runtime helpers shared by the cell and data commands."""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch


CANONICAL_EXECUTOR_SCHEMA = "bsc-cell-executor-simple-v1"
CANONICAL_EXECUTOR_PROCESS_MODEL = "ordinary-stage-files-v1"


def _cuda_index(device: torch.device | str) -> int:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("CUDA lock requires a CUDA device")
    return torch.cuda.current_device() if resolved.index is None else resolved.index


def cuda_execution_lock_path(device: torch.device | str) -> Path:
    """Return the ordinary per-user lock used to avoid concurrent GPU runs."""

    return Path(f"/var/tmp/bsc-gpu-{os.getuid()}-{_cuda_index(device)}.lock")


@contextmanager
def host_cuda_execution_lock(
    device: torch.device | str,
    *,
    operation: str,
    owner_id: str,
) -> Iterator[None]:
    """Serialize project work on one visible CUDA device."""

    del operation, owner_id
    resolved = torch.device(device)
    if resolved.type != "cuda":
        yield
        return
    path = cuda_execution_lock_path(resolved)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
