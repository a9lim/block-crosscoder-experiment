"""Small POSIX durability primitives for immutable experiment artifacts.

An atomic rename prevents readers from observing partial bytes, but it does not
by itself make either the file contents or the new directory entry durable
across a sudden host/power loss.  Scientific stage artifacts use this module so
the append-only campaign journal cannot outlive the files it commits.
"""

from __future__ import annotations

import os
from pathlib import Path


def fsync_file(path: str | Path) -> None:
    """Flush one already-written regular file to stable storage."""

    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory(path: str | Path) -> None:
    """Flush directory-entry updates for ``path`` to stable storage."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(Path(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_replace(
    temporary: str | Path,
    destination: str | Path,
    *,
    file_already_synced: bool = False,
) -> None:
    """Durably publish ``temporary`` at ``destination``.

    The caller must place both paths on the same filesystem.  When the writer
    already flushed the temporary file descriptor, ``file_already_synced``
    avoids reopening it solely for a duplicate fsync.
    """

    temporary_path = Path(temporary)
    destination_path = Path(destination)
    if temporary_path.parent.resolve() != destination_path.parent.resolve():
        raise ValueError("durable replacement requires one parent directory")
    if not file_already_synced:
        fsync_file(temporary_path)
    os.replace(temporary_path, destination_path)
    fsync_directory(destination_path.parent)
