"""Ordinary filesystem helpers retained for call-site compatibility."""

from __future__ import annotations

import os
from pathlib import Path


def durable_mkdir(
    path: str | Path,
    *,
    mode: int = 0o777,
    parents: bool = False,
    exist_ok: bool = False,
) -> Path:
    target = Path(path)
    target.mkdir(mode=mode, parents=parents, exist_ok=exist_ok)
    return target


def fsync_file(path: str | Path) -> None:
    del path


def fsync_directory(path: str | Path) -> None:
    del path


def durable_replace(
    temporary: str | Path,
    destination: str | Path,
    *,
    file_already_synced: bool = False,
) -> None:
    del file_already_synced
    os.replace(temporary, destination)


def durable_create(
    temporary: str | Path,
    destination: str | Path,
    *,
    file_already_synced: bool = False,
) -> None:
    del file_already_synced
    destination_path = Path(destination)
    if destination_path.exists():
        raise FileExistsError(destination_path)
    os.replace(temporary, destination_path)
