"""Small POSIX durability primitives for immutable experiment artifacts.

An atomic rename prevents readers from observing partial bytes, but it does not
by itself make either the file contents or the new directory entry durable
across a sudden host/power loss.  Scientific stage artifacts use this module so
the append-only campaign journal cannot outlive the files it commits.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path


def durable_mkdir(
    path: str | Path,
    *,
    mode: int = 0o777,
    parents: bool = False,
    exist_ok: bool = False,
) -> Path:
    """Create a directory and durably publish every newly created link.

    ``Path.mkdir(parents=True)`` can leave an otherwise complete artifact tree
    unreachable after sudden power loss because none of the parent directory
    entries are flushed.  This primitive creates missing components from the
    nearest existing ancestor outward and fsyncs that ancestor after each
    link.  A concurrent creator is accepted only under ``exist_ok``/``parents``
    semantics and its parent link is still flushed.
    """

    target = Path(path)
    try:
        target_stat = target.stat()
    except FileNotFoundError:
        target_stat = None
    if target_stat is not None:
        if not stat.S_ISDIR(target_stat.st_mode):
            raise FileExistsError(
                f"directory path exists but is not a directory: {target}"
            )
        if not exist_ok:
            raise FileExistsError(f"directory already exists: {target}")
        return target

    missing: list[Path] = [target]
    ancestor = target.parent
    if parents:
        while not ancestor.exists():
            missing.append(ancestor)
            if ancestor.parent == ancestor:
                break
            ancestor = ancestor.parent
    elif not ancestor.is_dir():
        raise FileNotFoundError(f"parent directory does not exist: {ancestor}")

    for directory in reversed(missing):
        parent = directory.parent
        try:
            os.mkdir(directory, mode)
        except FileExistsError:
            if not directory.is_dir() or (directory == target and not exist_ok):
                raise
        fsync_directory(parent)
    return target


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


def durable_create(
    temporary: str | Path,
    destination: str | Path,
    *,
    file_already_synced: bool = False,
) -> None:
    """Durably create ``destination`` without ever replacing an existing file."""

    temporary_path = Path(temporary)
    destination_path = Path(destination)
    if temporary_path.parent.resolve() != destination_path.parent.resolve():
        raise ValueError("durable creation requires one parent directory")
    if not file_already_synced:
        fsync_file(temporary_path)
    os.link(temporary_path, destination_path)
    fsync_directory(destination_path.parent)
    temporary_path.unlink()
