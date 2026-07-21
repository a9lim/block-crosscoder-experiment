"""Adversarial checks for the activation-store integrity boundary.

Each test corrupts exactly one binding layer.  A valid manifest hash is
recomputed only when the test needs to reach a deeper shard or stream check.
"""

import copy
import hashlib
import json
from pathlib import Path
import shutil

import pytest
import torch

import block_crosscoder_experiment.store as store_module
from block_crosscoder_experiment.store import ShardWriter, StoreReader


SITES = [2, 5]
D_MODEL = 3


def _canonical_hash(manifest: dict) -> str:
    unhashed = copy.deepcopy(manifest)
    unhashed.pop("manifest_sha256", None)
    return hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_manifest(path: Path, manifest: dict, *, rehash: bool) -> None:
    manifest = copy.deepcopy(manifest)
    if rehash:
        manifest["manifest_sha256"] = _canonical_hash(manifest)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def _write_store(
    root: Path,
    *,
    split: str = "train",
    seed: int = 0,
    sites: list[int] = SITES,
    d_model: int = D_MODEL,
    n_tokens: int = 9,
    tokens_per_shard: int = 4,
    meta: dict | None = None,
) -> tuple[dict, torch.Tensor]:
    writer = ShardWriter(
        root,
        split,
        whitener_hash="whitener-fixture",
        sites=sites,
        d_model=d_model,
        meta={"panel": "integrity-fixture"} if meta is None else meta,
        tokens_per_shard=tokens_per_shard,
        free_space_floor_frac=0.0,
    )
    # Nonzero, deterministic, and distinct across stores and rows.
    values = torch.arange(1, n_tokens * len(sites) * d_model + 1)
    values = values.reshape(n_tokens, len(sites), d_model).float()
    values = values + seed * 1000
    writer.add(values)
    return writer.close(), values


def _rewrite_shard(
    path: Path,
    *,
    metadata_updates: dict[str, str] | None = None,
    mutate_payload=None,
) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata())
        tensors = {key: handle.get_tensor(key).clone() for key in handle.keys()}
    metadata.update(metadata_updates or {})
    if mutate_payload is not None:
        mutate_payload(tensors["acts"])
    temporary = path.with_suffix(path.suffix + ".rewrite")
    save_file(
        {key: tensor.contiguous() for key, tensor in tensors.items()},
        temporary,
        metadata=metadata,
    )
    temporary.replace(path)


def _leave_durable_tail_orphan(root: Path, monkeypatch) -> tuple[dict, torch.Tensor]:
    values = (
        torch.arange(1, 13 * len(SITES) * D_MODEL + 1)
        .reshape(13, len(SITES), D_MODEL)
        .float()
    )
    writer = ShardWriter(
        root,
        "train",
        whitener_hash="whitener-fixture",
        sites=SITES,
        d_model=D_MODEL,
        meta={"panel": "integrity-fixture"},
        tokens_per_shard=4,
        free_space_floor_frac=0.0,
    )
    writer.add(values[:4])
    prior_manifest = json.loads((root / "train" / "split.json").read_text())

    def crash_before_manifest(*, complete: bool) -> dict:
        assert complete is False
        raise RuntimeError("injected after durable shard rename")

    monkeypatch.setattr(writer, "_write_manifest", crash_before_manifest)
    with pytest.raises(RuntimeError, match="durable shard rename"):
        writer.add(values[4:8])
    return prior_manifest, values


def test_resume_adopts_one_verified_durable_tail_and_fsyncs_manifest(
    tmp_path, monkeypatch
):
    prior_manifest, values = _leave_durable_tail_orphan(tmp_path, monkeypatch)
    manifest_path = tmp_path / "train" / "split.json"
    assert json.loads(manifest_path.read_text()) == prior_manifest
    assert (tmp_path / "train" / "shard_00001.safetensors").is_file()

    fsynced: list[Path] = []
    monkeypatch.setattr(
        store_module,
        "_fsync_directory",
        lambda path: fsynced.append(Path(path)),
    )
    resumed = ShardWriter(
        tmp_path,
        "train",
        whitener_hash="whitener-fixture",
        sites=SITES,
        d_model=D_MODEL,
        meta={"panel": "integrity-fixture"},
        tokens_per_shard=4,
        free_space_floor_frac=0.0,
        resume=True,
    )
    assert resumed.persisted_tokens == 8
    adopted = json.loads(manifest_path.read_text())
    assert adopted["complete"] is False
    assert [item["file"] for item in adopted["shards"]] == [
        "shard_00000.safetensors",
        "shard_00001.safetensors",
    ]
    assert fsynced == [tmp_path / "train"]

    resumed.add(values[8:])
    recovered_manifest = resumed.close()
    uninterrupted_manifest, _ = _write_store(
        tmp_path / "uninterrupted",
        n_tokens=13,
    )
    assert recovered_manifest == uninterrupted_manifest
    assert StoreReader(tmp_path, "train").verify() == 13


@pytest.mark.parametrize("mutation", ["corrupt", "ambiguous"])
def test_resume_refuses_corrupt_or_ambiguous_durable_tail(
    tmp_path, monkeypatch, mutation
):
    _leave_durable_tail_orphan(tmp_path, monkeypatch)
    orphan = tmp_path / "train" / "shard_00001.safetensors"
    if mutation == "corrupt":
        _rewrite_shard(orphan, mutate_payload=lambda acts: acts.add_(1))
        message = "orphan shard content_sha256 mismatch"
    else:
        shutil.copyfile(orphan, tmp_path / "train" / "shard_00002.safetensors")
        message = "shard file set differs from manifest"

    with pytest.raises(ValueError, match=message):
        ShardWriter(
            tmp_path,
            "train",
            whitener_hash="whitener-fixture",
            sites=SITES,
            d_model=D_MODEL,
            meta={"panel": "integrity-fixture"},
            tokens_per_shard=4,
            free_space_floor_frac=0.0,
            resume=True,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "legacy unbound store manifest"),
        ("mismatch", "manifest hash mismatch"),
    ],
)
def test_manifest_self_hash_is_mandatory_and_refuses_mutation(
    tmp_path, mutation, message
):
    manifest, _ = _write_store(tmp_path)
    manifest_path = tmp_path / "train" / "split.json"
    if mutation == "missing":
        manifest.pop("manifest_sha256")
    else:
        manifest["n_tokens"] += 1
    _write_manifest(manifest_path, manifest, rehash=False)

    with pytest.raises(ValueError, match=message):
        StoreReader(tmp_path, "train")


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("split", "eval"),
        ("shard_index", "99"),
        ("n_tokens", "999"),
        ("sites", json.dumps(list(reversed(SITES)))),
        ("d_model", "99"),
        ("dtype", "float32"),
        ("content_sha256", "0" * 64),
        ("row_ids_sha256", "0" * 64),
        ("row_id_width", "99"),
        ("row_ids_dtype", "int32"),
        ("meta", "{}"),
    ],
)
def test_every_shard_header_binding_is_checked(tmp_path, field, bad_value):
    manifest, _ = _write_store(tmp_path)
    shard_path = tmp_path / "train" / manifest["shards"][0]["file"]
    _rewrite_shard(shard_path, metadata_updates={field: bad_value})

    reader = StoreReader(tmp_path, "train")
    with pytest.raises(ValueError, match=rf"shard header mismatch.*{field}"):
        reader.verify()


def test_valid_safetensors_with_mutated_payload_fails_content_checksum(tmp_path):
    manifest, _ = _write_store(tmp_path)
    shard_path = tmp_path / "train" / manifest["shards"][0]["file"]

    def corrupt(acts: torch.Tensor) -> None:
        acts[0, 0, 0] += 7

    # Preserve the complete original header while replacing one payload value.
    _rewrite_shard(shard_path, mutate_payload=corrupt)
    reader = StoreReader(tmp_path, "train")
    with pytest.raises(ValueError, match="content checksum mismatch"):
        reader.verify()


def test_ordered_stream_digest_is_sensitive_to_shard_order(tmp_path):
    manifest, _ = _write_store(tmp_path)
    manifest["shards"] = list(reversed(manifest["shards"]))
    _write_manifest(tmp_path / "train" / "split.json", manifest, rehash=True)

    # Schema v3 refuses the non-canonical record sequence before reading any
    # payload; the ordered stream digest remains a second independent guard.
    with pytest.raises(ValueError, match="shard sequence is not canonical"):
        StoreReader(tmp_path, "train")


def test_row_identity_dtype_is_bound_in_manifest_header_and_payload(tmp_path):
    manifest, _ = _write_store(tmp_path)
    manifest_path = tmp_path / "train" / "split.json"
    manifest["row_ids_dtype"] = "int32"
    _write_manifest(manifest_path, manifest, rehash=True)
    with pytest.raises(ValueError, match="row_ids_dtype must be int64"):
        StoreReader(tmp_path, "train")

    manifest, _ = _write_store(tmp_path / "payload")
    shard_path = tmp_path / "payload" / "train" / manifest["shards"][0]["file"]
    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata())
        acts = handle.get_tensor("acts").clone()
        row_ids = handle.get_tensor("row_ids").to(torch.int32)
    replacement = shard_path.with_suffix(".rewrite")
    save_file(
        {"acts": acts, "row_ids": row_ids},
        replacement,
        metadata=metadata,
    )
    replacement.replace(shard_path)
    reader = StoreReader(tmp_path / "payload", "train")
    with pytest.raises(ValueError, match="row identity payload mismatch"):
        reader.verify()


def test_format_version_and_complete_status_are_refusal_boundaries(tmp_path):
    manifest, _ = _write_store(tmp_path)
    manifest_path = tmp_path / "train" / "split.json"
    manifest["format_version"] = 2
    _write_manifest(manifest_path, manifest, rehash=True)
    with pytest.raises(ValueError, match="unsupported activation-store format"):
        StoreReader(tmp_path, "train")

    manifest["format_version"] = 3
    manifest["complete"] = False
    _write_manifest(manifest_path, manifest, rehash=True)
    with pytest.raises(ValueError, match="split is incomplete"):
        StoreReader(tmp_path, "train")
    assert StoreReader(tmp_path, "train", allow_incomplete=True).verify() == 9


def test_derived_view_propagates_raw_row_stream_identity(tmp_path):
    raw_manifest, _ = _write_store(tmp_path / "raw", split="train")
    raw = StoreReader(tmp_path / "raw", "train")
    assert raw.verify() == raw_manifest["n_tokens"]

    derived_writer = ShardWriter(
        tmp_path / "derived",
        "train",
        whitener_hash="whitener-fixture",
        sites=[SITES[0]],
        d_model=D_MODEL,
        meta={
            "panel": "integrity-fixture",
            "derived_from": "raw/train",
            "row_stream_sha256": raw_manifest["row_stream_sha256"],
        },
        tokens_per_shard=5,
        free_space_floor_frac=0.0,
    )
    for batch, row_ids in raw.sequential_batches_with_ids(3):
        # Site slicing and a value transform change content but not row identity.
        derived_writer.add(batch[:, :1].float() * 2, row_ids=row_ids)
    derived_manifest = derived_writer.close()

    assert derived_manifest["row_stream_sha256"] == raw_manifest["row_stream_sha256"]
    assert (
        derived_manifest["content_stream_sha256"]
        != raw_manifest["content_stream_sha256"]
    )
    derived = StoreReader(tmp_path / "derived", "train")
    assert derived.manifest["row_stream_sha256"] == raw.manifest["row_stream_sha256"]
    assert derived.verify() == raw_manifest["n_tokens"]


def test_coherently_relabelled_foreign_shard_is_caught_by_stream_digest(tmp_path):
    manifest_a, _ = _write_store(tmp_path / "a", seed=1)
    manifest_b, _ = _write_store(tmp_path / "b", seed=2)
    record_a = manifest_a["shards"][1]
    record_b = manifest_b["shards"][1]
    assert record_a["n_tokens"] == record_b["n_tokens"]

    shard_a = tmp_path / "a" / "train" / record_a["file"]
    shard_b = tmp_path / "b" / "train" / record_b["file"]
    shutil.copyfile(shard_b, shard_a)

    # Model the strongest accidental mix: the shard record travels with the
    # foreign payload, so its header and per-shard checksum both validate.
    manifest_a["shards"][1]["content_sha256"] = record_b["content_sha256"]
    _write_manifest(tmp_path / "a" / "train" / "split.json", manifest_a, rehash=True)
    reader = StoreReader(tmp_path / "a", "train")
    reader._shard_tokens(reader.manifest["shards"][1], verify=True)

    with pytest.raises(ValueError, match="ordered shard stream digest"):
        reader.verify()
