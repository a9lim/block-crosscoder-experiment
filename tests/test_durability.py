from pathlib import Path

import block_crosscoder_experiment.durability as durability


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
