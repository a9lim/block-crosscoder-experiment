from pathlib import Path

import pytest

import block_crosscoder_experiment.durability as durability


def test_durable_mkdir_flushes_each_new_parent_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "campaign" / "cells" / "cell-1"
    flushed: list[Path] = []
    monkeypatch.setattr(
        durability,
        "fsync_directory",
        lambda path: flushed.append(Path(path)),
    )

    assert durability.durable_mkdir(target, parents=True) == target

    assert target.is_dir()
    assert flushed == [
        tmp_path,
        tmp_path / "campaign",
        tmp_path / "campaign" / "cells",
    ]


def test_durable_mkdir_existing_and_collision_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    flushed: list[Path] = []
    monkeypatch.setattr(
        durability,
        "fsync_directory",
        lambda path: flushed.append(Path(path)),
    )

    assert durability.durable_mkdir(existing, exist_ok=True) == existing
    assert flushed == []
    with pytest.raises(FileExistsError, match="already exists"):
        durability.durable_mkdir(existing)

    obstruction = tmp_path / "file"
    obstruction.write_bytes(b"not a directory")
    with pytest.raises(FileExistsError, match="not a directory"):
        durability.durable_mkdir(obstruction, exist_ok=True)
    with pytest.raises(FileNotFoundError, match="parent directory"):
        durability.durable_mkdir(tmp_path / "missing" / "child")


def test_durable_mkdir_accepts_concurrent_creator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "shared"
    original_stat = Path.stat
    injected_creation = False

    def stat_with_concurrent_creation(path, *args, **kwargs):
        nonlocal injected_creation
        try:
            return original_stat(path, *args, **kwargs)
        except FileNotFoundError:
            if path == target and not injected_creation:
                injected_creation = True
                durability.os.mkdir(target)
            raise

    monkeypatch.setattr(Path, "stat", stat_with_concurrent_creation)

    assert durability.durable_mkdir(target, exist_ok=True) == target
    assert target.is_dir()


def test_durable_replace_flushes_bytes_before_name_and_parent(
    tmp_path: Path, monkeypatch
) -> None:
    temporary = tmp_path / "artifact.tmp"
    destination = tmp_path / "artifact.bin"
    temporary.write_bytes(b"complete")
    events: list[tuple[str, Path, Path | None]] = []

    monkeypatch.setattr(
        durability,
        "fsync_file",
        lambda path: events.append(("file", Path(path), None)),
    )
    original_replace = durability.os.replace

    def replace(source, target):
        events.append(("replace", Path(source), Path(target)))
        original_replace(source, target)

    monkeypatch.setattr(durability.os, "replace", replace)
    monkeypatch.setattr(
        durability,
        "fsync_directory",
        lambda path: events.append(("directory", Path(path), None)),
    )

    durability.durable_replace(temporary, destination)

    assert events == [
        ("file", temporary, None),
        ("replace", temporary, destination),
        ("directory", tmp_path, None),
    ]
    assert destination.read_bytes() == b"complete"


def test_durable_replace_can_reuse_an_already_flushed_file(
    tmp_path: Path, monkeypatch
) -> None:
    temporary = tmp_path / "artifact.tmp"
    destination = tmp_path / "artifact.bin"
    temporary.write_bytes(b"complete")
    file_flushes = 0
    directory_flushes = 0

    def file_flush(_path):
        nonlocal file_flushes
        file_flushes += 1

    def directory_flush(_path):
        nonlocal directory_flushes
        directory_flushes += 1

    monkeypatch.setattr(durability, "fsync_file", file_flush)
    monkeypatch.setattr(durability, "fsync_directory", directory_flush)
    durability.durable_replace(
        temporary,
        destination,
        file_already_synced=True,
    )

    assert file_flushes == 0
    assert directory_flushes == 1
    assert destination.read_bytes() == b"complete"
