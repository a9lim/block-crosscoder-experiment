from __future__ import annotations

import gc
import hashlib
import json
import shutil
import sys
import types
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from block_crosscoder_experiment.cli.data import (
    _overlap_cuda_capture_copies,
    _canonical_hash,
    _producer_lock,
    _producer_lock_path,
    capture,
    derive_views,
    estimate_capture_pipeline_residency_bytes,
    estimate_store_bytes,
    estimate_writer_residency_bytes,
    fit_transform_artifacts,
    load_pinned_tokenizer,
    parse_capture_split_sizes,
    parse_split_sizes,
    tokenizer_contract_hash,
    transformer_lens_model_name,
    validate_capture_manifest,
    verify_store_root,
    verify_alignment,
    whole_sequence_split_plan,
)
from block_crosscoder_experiment.cli import data as data_module
from block_crosscoder_experiment.cli import matrix as matrix_module
from block_crosscoder_experiment.cli.matrix import (
    _resolve_phase2_view_dispatch,
    _run_with_optional_view_dispatch,
    _storage_preflight,
    _verified_existing_input_storage,
)
from block_crosscoder_experiment.cli.matrix import main as matrix_main
from block_crosscoder_experiment.cli.run_cell import _expected_real_source_contract
from block_crosscoder_experiment.store import ShardWriter, StoreReader, Whitener
from block_crosscoder_experiment.studies import (
    FrozenSelection,
    GPT2_VOCAB_HASH,
    Phase,
    PHASE3_VOCAB_HASH,
    StudyError,
    build_phase2_blueprint,
    build_phase2_plan,
    build_phase1_plan,
    resolved_candidate_execution_signature,
)


def test_transformer_lens_loader_name_preserves_pinned_repo_identity():
    assert transformer_lens_model_name("openai-community/gpt2") == "gpt2"
    assert transformer_lens_model_name("google/gemma-3-1b-pt") == "google/gemma-3-1b-pt"


def test_capture_cli_rejects_retired_store_contract_before_loading(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        data_module.main(
            [
                "capture",
                "--source",
                "openai-community/gpt2|" + "a" * 40 + "|blocks.3.hook_resid_pre",
                "--tokenizer-contract",
                "gpt2-byte-bpe-files-v1",
                "--profile",
                "phase2",
                "--store-contract-version",
                "activation-store-v2",
                "--split",
                "train=1",
                "--out",
                str(tmp_path / "store"),
            ]
        )
    assert exc_info.value.code == 2


def test_capture_cli_requires_profile_and_complete_profile_roles(tmp_path):
    common = [
        "capture",
        "--source",
        "openai-community/gpt2|" + "a" * 40 + "|blocks.3.hook_resid_pre",
        "--tokenizer-contract",
        "gpt2-byte-bpe-files-v1",
        "--split",
        "normalization_fit=1",
        "--split",
        "calibration=1",
        "--split",
        "train=1",
        "--out",
        str(tmp_path / "store"),
    ]
    with pytest.raises(SystemExit) as exc_info:
        data_module.main(common)
    assert exc_info.value.code == 2

    with pytest.raises(ValueError, match="missing.*development.*confirmation"):
        data_module.main([*common, "--profile", "phase2"])


def _raw_store(root, *, offset=0.0, authenticated_profile="phase2"):
    root.mkdir(parents=True, exist_ok=True)
    source = {
        "format_version": 2,
        "sources": [
            {
                "model": "test/model",
                "revision": "a" * 40,
                "hook": "blocks.0.hook_resid_pre",
            },
            {
                "model": "test/model",
                "revision": "a" * 40,
                "hook": "blocks.1.hook_resid_pre",
            },
        ],
        "corpus": "test/corpus",
        "corpus_config": "default",
        "corpus_revision": "b" * 40,
        "corpus_split": "train",
        "text_field": "text",
        "context": 2,
        "drop_positions": 1,
        "tokenizer_class": "TestTokenizer",
        "tokenizer_vocab_sha256": "sha256:" + "c" * 64,
        "add_special_tokens": False,
        "bos_token_id": 0,
        "packing_algorithm": "bos_prefixed_greedy_document_stream_v1",
        "sequence_allocation": "whole_packed_contexts_v1",
        "tokenizer_hashes": ["sha256:" + "d" * 64],
        "tokenizer_contract": "gpt2-byte-bpe-files-v1",
        "store_contract_version": "activation-store-v3-single-view",
        "alignment_version": "identical-tokenizer-row-identity-v1",
        "alignment_audit": "not_applicable:single-model-identical-tokenizer",
        "row_identity_columns": ["sequence", "position", "token_id"],
        "capture_mode": "raw_once",
        "model_loader": "transformer_lens_from_pretrained_no_processing_v1",
        "transformer_lens_model_names": ["test/model"],
        "model_forward_dtype": "bfloat16",
        "store_dtype": "bfloat16",
    }
    source_hash = hashlib.sha256(
        json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    split_sizes = {
        "normalization_fit": 64,
        "calibration": 48,
        "development": 32,
        "confirmation": 32,
        "train": 80,
    }
    split_plan = whole_sequence_split_plan(split_sizes, 1)
    capture_payload = None
    binding_sha256 = None
    if authenticated_profile is not None:
        implementation = {
            "schema": "bsc-capture-implementation-v1",
            "python": "3.12.0",
            "dependencies": {"torch": "test"},
            "data_module_sha256": "e" * 64,
            "store_module_sha256": "f" * 64,
            "runtime": {
                "requested_device": "cpu",
                "torch_cuda_version": None,
                "cuda_device_name": None,
            },
        }
        binding = {
            "schema": data_module.CAPTURE_BINDING_SCHEMA,
            "campaign_profile": authenticated_profile,
            "source_hash": source_hash,
            "split_order": list(split_sizes),
            "split_plan": split_plan,
            "capture_implementation": implementation,
            "sites": [0, 1],
            "site_dims": [5, 5],
            "d_model": 5,
            "physical_store_format_version": data_module.STORE_FORMAT_VERSION,
            "batch_rows": 1,
            "write_batch_tokens": 1,
            "tokens_per_shard": 17,
            "writer_pipeline": {
                "contract": "one_pending_shard_v1",
                "bytes_per_token": 44,
                "shard_payload_bytes": 748,
                "pending_shard_bytes": 748,
                "staging_shard_bytes": 748,
                "writer_residency_bytes": 1496,
                "max_writer_residency_bytes": 4096,
            },
            "capture_transfer_pipeline": {
                "contract": "synchronous_cpu_capture_v1",
                "activation_batch_bytes": 20,
                "row_identity_batch_bytes": 24,
                "pinned_activation_buffer_count": 0,
                "pinned_activation_host_bytes": 0,
                "retained_row_identity_host_bytes": 0,
                "retained_cuda_source_bytes": 0,
                "peak_host_pipeline_bytes": 1496,
                "peak_cuda_capture_lookahead_bytes": 0,
            },
        }
        binding_sha256 = hashlib.sha256(
            json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    gen = torch.Generator().manual_seed(4)
    for split, n in split_sizes.items():
        allocation = split_plan[split]
        meta = {
            "site_dims": [5, 5],
            "split_requested_tokens": n,
            "split_actual_tokens": n,
        }
        if binding_sha256 is not None:
            meta.update(
                {
                    **source,
                    "sequence_start": allocation["sequence_start"],
                    "sequence_stop_exclusive": allocation["sequence_stop_exclusive"],
                    "tokens_per_sequence": allocation["tokens_per_sequence"],
                    "sequence_allocation": "whole_packed_contexts_v1",
                    "capture_binding_sha256": binding_sha256,
                    "ordered_split_allocation": list(split_sizes),
                }
            )
        writer = ShardWriter(
            root,
            split,
            whitener_hash=f"raw:{source_hash}",
            sites=(0, 1),
            d_model=5,
            meta=meta,
            tokens_per_shard=17,
            free_space_floor_frac=0,
        )
        x = torch.randn(n, 2, 5, generator=gen) + offset
        ids = torch.stack(
            (
                torch.arange(n) + allocation["sequence_start"],
                torch.ones(n, dtype=torch.int64),
                torch.arange(n, dtype=torch.int64),
            ),
            dim=1,
        )
        writer.add(x, ids)
        writer.close()
    if authenticated_profile is not None:
        capture_payload = {
            "schema": data_module.CAPTURE_MANIFEST_SCHEMA,
            "source": source,
            "source_hash": source_hash,
            "split_order": list(split_sizes),
            "split_plan": split_plan,
            "splits": {
                split: data_module._capture_split_record(
                    root,
                    split,
                    StoreReader(root, split),
                    split_plan[split],
                )
                for split in split_sizes
            },
            "capture_implementation": implementation,
            "capture_binding": binding,
            "capture_binding_sha256": binding_sha256,
        }
        capture_payload["capture_content_sha256"] = data_module._canonical_hash(
            capture_payload
        )
    else:
        capture_payload = {
            "source": source,
            "source_hash": source_hash,
            "split_order": list(split_sizes),
            "split_plan": split_plan,
            "splits": split_plan,
        }
    (root / "capture.json").write_text(json.dumps(capture_payload) + "\n")


def test_derive_views_preserves_row_identity(tmp_path):
    raw = tmp_path / "raw"
    _raw_store(raw)
    out = tmp_path / "views"
    derive_views(
        raw,
        out,
        ("none", "scalar_rms", "sqrt_d"),
        batch_size=13,
    )
    aligned = verify_alignment((out / "none", out / "scalar_rms", out / "sqrt_d"))
    assert aligned["development"]["n_tokens"] == 32
    assert (
        StoreReader(raw, "train").manifest["row_stream_sha256"]
        == StoreReader(out / "scalar_rms", "train").manifest["row_stream_sha256"]
    )
    verified = verify_store_root(out / "scalar_rms")
    assert set(verified) == {
        "normalization_fit",
        "calibration",
        "development",
        "confirmation",
        "train",
    }


@pytest.mark.parametrize("producer", [derive_views, fit_transform_artifacts])
def test_transform_producers_require_authenticated_capture_manifest(tmp_path, producer):
    raw = tmp_path / "raw"
    _raw_store(raw)
    capture_path = raw / "capture.json"
    capture_payload = json.loads(capture_path.read_text())
    capture_payload["capture_binding"]["d_model"] = 4
    capture_payload["capture_content_sha256"] = data_module._canonical_hash(
        {
            key: value
            for key, value in capture_payload.items()
            if key != "capture_content_sha256"
        }
    )
    capture_path.write_text(json.dumps(capture_payload) + "\n")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="capture binding digest mismatch"):
        producer(raw, output, ("none",))
    assert not output.exists()


def test_derived_view_root_manifest_detects_missing_or_divergent_members(tmp_path):
    raw = tmp_path / "raw"
    views = tmp_path / "views"
    _raw_store(raw)
    derive_views(raw, views, ("none",))
    view = views / "none"
    manifest_path = view / data_module.VIEW_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["splits"]["train"]["n_tokens"] -= 1
    unsigned = dict(manifest)
    unsigned.pop("view_manifest_sha256")
    manifest["view_manifest_sha256"] = data_module._canonical_hash(unsigned)
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ValueError, match="split record 'train'.*source capture"):
        verify_store_root(view)

    derive_views(raw, tmp_path / "fresh", ("none",))
    fresh_view = tmp_path / "fresh" / "none"
    shutil.rmtree(fresh_view / "confirmation")
    with pytest.raises(ValueError, match="entries differ"):
        verify_store_root(fresh_view)

    evidence_views = tmp_path / "evidence"
    derive_views(raw, evidence_views, ("none",))
    evidence_view = evidence_views / "none"
    evidence_manifest_path = evidence_view / data_module.VIEW_MANIFEST_NAME
    evidence_manifest = json.loads(evidence_manifest_path.read_text())
    evidence_manifest["source_capture"]["capture_binding"]["d_model"] = 4
    evidence_manifest["source_capture"]["capture_content_sha256"] = (
        data_module._canonical_hash(
            {
                key: value
                for key, value in evidence_manifest["source_capture"].items()
                if key != "capture_content_sha256"
            }
        )
    )
    evidence_unsigned = dict(evidence_manifest)
    evidence_unsigned.pop("view_manifest_sha256")
    evidence_manifest["view_manifest_sha256"] = data_module._canonical_hash(
        evidence_unsigned
    )
    evidence_manifest_path.write_text(json.dumps(evidence_manifest) + "\n")
    with pytest.raises(ValueError, match="capture binding digest mismatch"):
        verify_store_root(evidence_view)


def test_current_data_manifests_reject_extra_fields_even_when_rehashed(tmp_path):
    raw = tmp_path / "raw"
    _raw_store(raw)
    capture = json.loads((raw / "capture.json").read_text())
    capture["ignored_future_field"] = True
    capture["capture_content_sha256"] = data_module._canonical_hash(
        {
            key: value
            for key, value in capture.items()
            if key != "capture_content_sha256"
        }
    )
    with pytest.raises(ValueError, match="capture manifest keys mismatch"):
        validate_capture_manifest(capture)

    views = tmp_path / "views"
    derive_views(raw, views, ("none",))
    view_manifest = json.loads((views / "none" / "view.json").read_text())
    view_manifest["splits"]["train"]["ignored_future_field"] = True
    unsigned_view = dict(view_manifest)
    unsigned_view.pop("view_manifest_sha256")
    view_manifest["view_manifest_sha256"] = data_module._canonical_hash(unsigned_view)
    with pytest.raises(ValueError, match="derived-view split record.*keys mismatch"):
        data_module.validate_derived_view_manifest(view_manifest)

    transforms = tmp_path / "transforms"
    record = fit_transform_artifacts(raw, transforms, ("none",))["none"]
    transform_manifest = json.loads(Path(record["manifest"]).read_text())
    transform_manifest["ignored_future_field"] = True
    unsigned_transform = dict(transform_manifest)
    unsigned_transform.pop("transform_manifest_sha256")
    transform_manifest["transform_manifest_sha256"] = data_module._canonical_hash(
        unsigned_transform
    )
    with pytest.raises(ValueError, match="transform manifest keys mismatch"):
        data_module.validate_transform_artifact_manifest(transform_manifest)


def test_standalone_view_rejects_self_rehashed_source_manifest_forgery(tmp_path):
    raw = tmp_path / "raw"
    views = tmp_path / "views"
    _raw_store(raw)
    derive_views(raw, views, ("none",))
    view = views / "none"

    split_path = view / "train" / "split.json"
    split_manifest = json.loads(split_path.read_text())
    split_manifest["meta"]["source_split_manifest_sha256"] = "0" * 64
    unsigned_split = dict(split_manifest)
    unsigned_split.pop("manifest_sha256")
    split_manifest["manifest_sha256"] = data_module._canonical_hash(unsigned_split)
    split_path.write_text(json.dumps(split_manifest) + "\n")

    view_path = view / "view.json"
    view_manifest = json.loads(view_path.read_text())
    view_manifest["splits"]["train"]["manifest_sha256"] = split_manifest[
        "manifest_sha256"
    ]
    unsigned_view = dict(view_manifest)
    unsigned_view.pop("view_manifest_sha256")
    view_manifest["view_manifest_sha256"] = data_module._canonical_hash(unsigned_view)
    view_path.write_text(json.dumps(view_manifest) + "\n")

    with pytest.raises(
        ValueError,
        match="shard header mismatch|divergent source/geometry binding",
    ):
        verify_store_root(view)


def test_derive_resume_reuses_complete_prefix_and_continues_missing_splits(tmp_path):
    raw = tmp_path / "raw"
    _raw_store(raw)
    out = tmp_path / "views"
    original = derive_views(raw, out, ("scalar_rms",), batch_size=13)
    view = out / "scalar_rms"
    preserved = {
        split: (view / split / "split.json").read_bytes()
        for split in ("normalization_fit", "calibration")
    }
    shutil.rmtree(view / "development")
    shutil.rmtree(view / "confirmation")
    shutil.rmtree(view / "train")

    resumed = derive_views(
        raw,
        out,
        ("scalar_rms",),
        batch_size=11,
        resume=True,
    )
    assert (
        resumed["scalar_rms"]["whitener_hash"]
        == original["scalar_rms"]["whitener_hash"]
    )
    assert {
        split: (view / split / "split.json").read_bytes() for split in preserved
    } == preserved
    for split in (
        "normalization_fit",
        "calibration",
        "development",
        "confirmation",
        "train",
    ):
        StoreReader(view, split).verify()

    # Fully complete resume is an idempotent verification pass.
    assert (
        derive_views(
            raw,
            out,
            ("scalar_rms",),
            batch_size=17,
            resume=True,
        )
        == resumed
    )


def test_derive_resume_refuses_partial_or_divergent_prefix(tmp_path):
    raw = tmp_path / "raw"
    _raw_store(raw)
    partial_out = tmp_path / "partial"
    derive_views(raw, partial_out, ("none",), batch_size=13)
    partial_view = partial_out / "none"
    (partial_view / "development" / "split.json").unlink()
    shutil.rmtree(partial_view / "train")
    with pytest.raises(ValueError, match="partial split.*remove exactly"):
        derive_views(raw, partial_out, ("none",), resume=True)

    complete_out = tmp_path / "complete"
    derive_views(raw, complete_out, ("none",), batch_size=13)
    with pytest.raises(ValueError, match="tokens_per_shard"):
        derive_views(
            raw,
            complete_out,
            ("none",),
            tokens_per_shard=17,
            resume=True,
        )


def test_data_producer_lock_refuses_a_concurrent_writer(tmp_path):
    output = tmp_path / "capture"
    with _producer_lock(output, operation="outer"):
        with pytest.raises(ValueError, match="locked by another producer"):
            with _producer_lock(output, operation="inner"):
                raise AssertionError("concurrent producer unexpectedly acquired lock")
    assert not output.exists()
    lock_payload = json.loads(_producer_lock_path(output).read_text())
    assert lock_payload["schema"] == "bsc-data-producer-lock-v1"
    assert lock_payload["pid"] > 0


def test_data_producer_lock_refuses_a_symlink_file(tmp_path):
    output = tmp_path / "symlinked-lock-output"
    lock_path = _producer_lock_path(output)
    assert not lock_path.exists()
    target = tmp_path / "target"
    target.write_text("must remain unchanged\n")
    lock_path.symlink_to(target)
    try:
        with pytest.raises(ValueError, match="safe data-producer lock file"):
            with _producer_lock(output, operation="capture"):
                raise AssertionError("symlinked lock unexpectedly opened")
        assert target.read_text() == "must remain unchanged\n"
    finally:
        lock_path.unlink()


def test_immutable_json_publication_never_clobbers_a_racing_name(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"publisher":"other"}\n')
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        data_module._atomic_json(
            path,
            {"publisher": "ours"},
            overwrite=False,
        )
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".manifest.json.*.tmp"))


def test_fit_transform_artifact_binds_capture_and_fit_stream_without_shards(
    tmp_path, monkeypatch
):
    raw = tmp_path / "raw"
    _raw_store(raw)
    out = tmp_path / "transforms"
    fsync_calls = []
    real_fsync = data_module.os.fsync

    def tracked_fsync(descriptor):
        fsync_calls.append(descriptor)
        return real_fsync(descriptor)

    monkeypatch.setattr(data_module.os, "fsync", tracked_fsync)
    result = fit_transform_artifacts(raw, out, ("scalar_rms",), batch_size=13)
    record = result["scalar_rms"]
    artifact_root = out / "scalar_rms" / record["transform_hash"]
    assert (artifact_root / "whitener.pt").is_file()
    assert (artifact_root / "transform.json").is_file()
    assert len(fsync_calls) >= 2  # manifest file and containing directory
    assert not (artifact_root / "transform.json.tmp").exists()
    assert not any(path.name == "split.json" for path in artifact_root.rglob("*"))
    assert (
        record["source_fit_row_stream_sha256"]
        == StoreReader(raw, "normalization_fit").manifest["row_stream_sha256"]
    )
    # Idempotent content-addressed reruns verify rather than overwrite.
    again = fit_transform_artifacts(raw, out, ("scalar_rms",), batch_size=17)
    assert again["scalar_rms"]["transform_hash"] == record["transform_hash"]


def test_transform_identity_is_content_addressed_not_store_path(tmp_path):
    original = tmp_path / "original" / "raw"
    relocated = tmp_path / "relocated" / "raw"
    _raw_store(original)
    shutil.copytree(original, relocated)
    first = fit_transform_artifacts(
        original, tmp_path / "first", ("scalar_rms",), batch_size=11
    )
    second = fit_transform_artifacts(
        relocated, tmp_path / "second", ("scalar_rms",), batch_size=11
    )
    assert (
        first["scalar_rms"]["transform_hash"] == second["scalar_rms"]["transform_hash"]
    )
    assert hashlib.sha256(Path(first["scalar_rms"]["path"]).read_bytes()).digest() == (
        hashlib.sha256(Path(second["scalar_rms"]["path"]).read_bytes()).digest()
    )


def test_alignment_refuses_different_rows(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _raw_store(a)
    _raw_store(b)
    # Rebuild one split with different explicit identities.
    split = b / "development"
    for child in split.iterdir():
        child.unlink()
    writer = ShardWriter(
        b,
        "development",
        whitener_hash="raw:test",
        sites=(0, 1),
        d_model=5,
        free_space_floor_frac=0,
    )
    writer.add(torch.randn(32, 2, 5), torch.arange(100, 132).view(-1, 1))
    writer.close()
    with pytest.raises(ValueError, match="alignment"):
        verify_alignment((a, b))


def test_alignment_refuses_different_structural_site_dimensions(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _raw_store(a)
    _raw_store(b)
    original = StoreReader(b, "development")
    batches = list(original.sequential_batches_with_ids(64))
    meta = {**original.manifest["meta"], "site_dims": [4, 5]}
    shutil.rmtree(b / "development")
    writer = ShardWriter(
        b,
        "development",
        whitener_hash=original.whitener_hash,
        sites=original.sites,
        d_model=original.d_model,
        meta=meta,
        tokens_per_shard=17,
        free_space_floor_frac=0,
    )
    for acts, row_ids in batches:
        writer.add(acts, row_ids)
    writer.close()
    with pytest.raises(ValueError, match="alignment"):
        verify_alignment((a, b))


def test_split_parser_and_estimate():
    splits = parse_split_sizes(
        ["normalization_fit=2", "calibration=3", "eval=5", "train=7"]
    )
    metadata = (
        data_module.DEFAULT_PREWRITE_METADATA_RESERVE_BYTES
        + len(splits) * data_module.DEFAULT_SPLIT_MANIFEST_RESERVE_BYTES
        + len(splits)
        * (
            data_module.DEFAULT_SHARD_HEADER_RESERVE_BYTES
            + data_module.DEFAULT_SHARD_MANIFEST_RECORD_RESERVE_BYTES
        )
    )
    assert estimate_store_bytes(splits, (4, 6), n_views=2) == (17 * 48 + metadata) * 2
    assert (
        estimate_store_bytes(splits, (4, 6), n_views=2, row_id_width=5)
        == (17 * 64 + metadata) * 2
    )
    writer = estimate_writer_residency_bytes(
        (4, 6), tokens_per_shard=10, row_id_width=3
    )
    assert writer == {
        "bytes_per_token": 48,
        "shard_payload_bytes": 480,
        "pending_shard_bytes": 480,
        "staging_shard_bytes": 480,
        "writer_residency_bytes": 960,
    }
    assert estimate_capture_pipeline_residency_bytes(
        writer,
        (4, 6),
        batch_rows=2,
        context=8,
        drop_positions=1,
        cuda_overlap=True,
    ) == {
        "contract": "two_pinned_activation_d2h_lookahead_v1",
        "activation_batch_bytes": 336,
        "row_identity_batch_bytes": 336,
        "pinned_activation_buffer_count": 2,
        "pinned_activation_host_bytes": 672,
        "retained_row_identity_host_bytes": 672,
        "retained_cuda_source_bytes": 672,
        "peak_host_pipeline_bytes": 2304,
        "peak_cuda_capture_lookahead_bytes": 672,
    }
    with pytest.raises(ValueError):
        parse_split_sizes(["train=2"])
    with pytest.raises(ValueError, match="explicitly"):
        parse_split_sizes(None)
    phase3 = parse_split_sizes(
        [
            "normalization_fit=2",
            "calibration=3",
            "final=5",
            "train=7",
        ]
    )
    assert phase3["final"] == 5


def test_store_prewrite_estimate_covers_actual_small_sharded_store(tmp_path):
    root = tmp_path / "raw"
    _raw_store(root)
    split_sizes = {
        split: StoreReader(root, split).n_tokens
        for split in data_module.CAPTURE_PROFILE_SPLITS["phase2"]
    }
    estimate = estimate_store_bytes(
        split_sizes,
        (5, 5),
        row_id_width=3,
        tokens_per_shard=17,
    )
    actual = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    assert estimate >= actual


def test_capture_split_profiles_require_exact_complete_role_sets():
    phase2 = parse_capture_split_sizes(
        [
            "normalization_fit=2",
            "calibration=3",
            "train=7",
            "development=5",
            "confirmation=11",
        ],
        profile="phase2",
    )
    assert tuple(phase2) == (
        "normalization_fit",
        "calibration",
        "development",
        "confirmation",
        "train",
    )
    phase3 = parse_capture_split_sizes(
        [
            "normalization_fit=2",
            "calibration=3",
            "train=7",
            "stability=5",
            "final=11",
        ],
        profile="phase3",
    )
    assert tuple(phase3) == (
        "normalization_fit",
        "calibration",
        "stability",
        "final",
        "train",
    )

    with pytest.raises(ValueError, match="missing.*confirmation"):
        parse_capture_split_sizes(
            [
                "normalization_fit=2",
                "calibration=3",
                "train=7",
                "development=5",
            ],
            profile="phase2",
        )
    with pytest.raises(ValueError, match="unexpected.*development"):
        parse_capture_split_sizes(
            [
                "normalization_fit=2",
                "calibration=3",
                "train=7",
                "stability=5",
                "final=11",
                "development=13",
            ],
            profile="phase3",
        )
    with pytest.raises(ValueError, match="explicitly declared"):
        parse_capture_split_sizes(
            ["normalization_fit=2", "calibration=3", "train=7"],
            profile=None,
        )


def test_whole_sequence_split_plan_rounds_each_split_without_overlap():
    plan = whole_sequence_split_plan(
        {"normalization_fit": 5, "calibration": 7, "eval": 1, "train": 12},
        4,
    )
    assert [spec["actual_tokens"] for spec in plan.values()] == [8, 8, 4, 12]
    intervals = [
        (spec["sequence_start"], spec["sequence_stop_exclusive"])
        for spec in plan.values()
    ]
    assert intervals == [(0, 2), (2, 4), (4, 5), (5, 8)]
    assert all(
        spec["actual_tokens"]
        == (spec["sequence_stop_exclusive"] - spec["sequence_start"]) * 4
        for spec in plan.values()
    )


def test_tokenizer_contract_hash_binds_ordered_file_names_and_bytes(
    tmp_path, monkeypatch
):
    (tmp_path / "tokenizer.json").write_bytes(b"tokenizer")
    (tmp_path / "vocab.json").write_bytes(b"vocab")
    (tmp_path / "merges.txt").write_bytes(b"merges")
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *args, **kwargs: str(tmp_path),
    )
    import hashlib

    expected = hashlib.sha256()
    for name in ("tokenizer.json", "vocab.json", "merges.txt"):
        expected.update(name.encode() + b"\0")
        expected.update((tmp_path / name).read_bytes())
    assert tokenizer_contract_hash("model", "revision", "gpt2-byte-bpe-files-v1") == (
        "sha256:" + expected.hexdigest()
    )


def test_unicode_vocab_hash_is_canonical_utf8_not_ascii_escaped():
    value = {"é": 1, "漢": 2}
    expected = hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert _canonical_hash(value) == expected


def test_reviewed_vocab_preflights_match_utf8_plan_contracts():
    assert data_module.TOKENIZER_PREFLIGHTS["openai-community/gpt2"][
        "vocab_sha256"
    ] == GPT2_VOCAB_HASH == (
        "sha256:179cad62d906b7217f1c9431ece06e7a78a7721f9580960147a6c1ea0a53fc65"
    )
    assert data_module.TOKENIZER_PREFLIGHTS["google/gemma-3-4b-pt"][
        "vocab_sha256"
    ] == PHASE3_VOCAB_HASH == (
        "sha256:4ab2b66fed16d7e79cfb30bd2168ee3da6d848a6ff9b0753cd62a5841c9328ad"
    )


def test_pinned_tokenizer_preflight_binds_revision_class_bos_and_vocab(
    monkeypatch,
):
    class ReviewedTokenizer:
        bos_token_id = 7

        def get_vocab(self):
            return {"é": 1, "token": 2}

    calls = []

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append((args, kwargs))
            return ReviewedTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=AutoTokenizer),
    )
    monkeypatch.setitem(
        data_module.TOKENIZER_PREFLIGHTS,
        "reviewed/model",
        {
            "contract": "gpt2-byte-bpe-files-v1",
            "class": "ReviewedTokenizer",
            "bos_token_id": 7,
            "vocab_sha256": "sha256:"
            + _canonical_hash(ReviewedTokenizer().get_vocab()),
        },
    )
    tokenizer = load_pinned_tokenizer(
        "reviewed/model", "immutable-sha", "gpt2-byte-bpe-files-v1"
    )
    assert isinstance(tokenizer, ReviewedTokenizer)
    assert calls == [
        (("reviewed/model",), {"revision": "immutable-sha", "use_fast": False})
    ]
    with pytest.raises(ValueError, match="incompatible"):
        load_pinned_tokenizer(
            "reviewed/model", "immutable-sha", "gemma3-tokenizer-files-v1"
        )


def _mock_capture_runtime(monkeypatch):
    plan = build_phase2_plan(seeds=(0,), smoke=True)
    values = plan.stages[0].cells[0].decision_map
    expected_source = _expected_real_source_contract(values)
    fake_vocab = {"unicode-é": 1, "token": 2}
    expected_vocab_digest = expected_source["tokenizer_vocab_sha256"].removeprefix(
        "sha256:"
    )
    real_canonical_hash = data_module._canonical_hash

    class GPT2Tokenizer:
        bos_token_id = 50_256
        eos_token_id = 50_256
        unk_token_id = 50_256

        def get_vocab(self):
            return fake_vocab

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return list(range(10, 2_010))

    tokenizer = GPT2Tokenizer()

    def canonical_hash(value):
        if value is fake_vocab:
            return expected_vocab_digest
        return real_canonical_hash(value)

    class HfApi:
        def model_info(self, model, revision):
            assert model == expected_source["sources"][0]["model"]
            assert revision == expected_source["sources"][0]["revision"]
            return SimpleNamespace(sha=revision)

        def dataset_info(self, corpus, revision):
            assert corpus == expected_source["corpus"]
            assert revision == expected_source["corpus_revision"]
            return SimpleNamespace(sha=revision)

    loader_calls = []

    class FakeModel:
        def __init__(self, explicit_tokenizer):
            assert explicit_tokenizer is tokenizer
            self.tokenizer = GPT2Tokenizer()
            self.cfg = SimpleNamespace(d_model=2)
            self.hook_dict = {
                item["hook"]: object() for item in expected_source["sources"]
            }

        def to(self, device):
            return self

        def eval(self):
            return self

        def forward(self, toks, *, stop_at_layer=None):  # pragma: no cover
            raise AssertionError("run_with_cache should own the forward")

        def run_with_cache(
            self,
            toks,
            *,
            names_filter,
            return_type,
            stop_at_layer=None,
        ):
            assert return_type is None
            expected_layers = [
                int(item["hook"].split(".")[1]) for item in expected_source["sources"]
            ]
            assert stop_at_layer == max(expected_layers) + 1
            cache = {}
            for index, item in enumerate(expected_source["sources"]):
                hook = item["hook"]
                assert names_filter(hook)
                cache[hook] = toks.float().unsqueeze(-1).repeat(1, 1, 2) + index
            return None, cache

    class HookedSAETransformer:
        @classmethod
        def from_pretrained_no_processing(cls, name, **kwargs):
            loader_calls.append((name, kwargs))
            return FakeModel(kwargs["tokenizer"])

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=HfApi)
    )
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(
            load_dataset=lambda *args, **kwargs: [{"text": "enough tokens"}]
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "sae_lens",
        types.SimpleNamespace(HookedSAETransformer=HookedSAETransformer),
    )
    monkeypatch.setattr(data_module, "_canonical_hash", canonical_hash)
    monkeypatch.setattr(
        data_module,
        "load_pinned_tokenizer",
        lambda model, revision, contract: tokenizer,
    )
    monkeypatch.setattr(
        data_module,
        "tokenizer_contract_hash",
        lambda model, revision, contract: expected_source["tokenizer_hashes"][0],
    )
    monkeypatch.setattr(
        data_module,
        "capture_implementation_contract",
        lambda: {
            "schema": "bsc-capture-implementation-v1",
            "python": "3.12.0",
            "dependencies": {"test-runtime": "exact"},
            "data_module_sha256": "a" * 64,
            "store_module_sha256": "b" * 64,
        },
    )

    def args(out, *, resume=False, sources=None):
        source_values = sources or [
            "|".join((item["model"], item["revision"], item["hook"]))
            for item in expected_source["sources"]
        ]
        return SimpleNamespace(
            source=source_values,
            corpus=expected_source["corpus"],
            corpus_config=expected_source["corpus_config"],
            corpus_revision=expected_source["corpus_revision"],
            corpus_split=expected_source["corpus_split"],
            text_field=expected_source["text_field"],
            tokenizer_contract=expected_source["tokenizer_contract"],
            store_contract_version=expected_source["store_contract_version"],
            alignment_version=expected_source["alignment_version"],
            alignment_audit=expected_source["alignment_audit"],
            context=expected_source["context"],
            drop_positions=expected_source["drop_positions"],
            batch_rows=2,
            write_batch_tokens=64,
            tokens_per_shard=64,
            profile="phase2",
            split=[
                "normalization_fit=2",
                "calibration=2",
                "development=2",
                "confirmation=2",
                "train=2",
            ],
            device="cpu",
            out=out,
            resume=resume,
        )

    return expected_source, loader_calls, args


def test_transformer_lens_tokenizer_copy_must_preserve_integer_contract():
    class PinnedTokenizer:
        bos_token_id = 7
        eos_token_id = 8
        unk_token_id = 9

        def __init__(self, vocab):
            self._vocab = vocab

        def get_vocab(self):
            return self._vocab

    pinned = PinnedTokenizer({"unicode-é": 1, "token": 2})
    compatible_copy = PinnedTokenizer(dict(pinned.get_vocab()))
    data_module._validate_transformer_lens_tokenizer(pinned, compatible_copy)

    incompatible_copy = PinnedTokenizer({"unicode-é": 1, "drift": 2})
    with pytest.raises(ValueError, match="differs from the explicit pinned tokenizer"):
        data_module._validate_transformer_lens_tokenizer(pinned, incompatible_copy)


def test_capture_exact_source_contract_and_failure_resume_stream_identity(
    tmp_path, monkeypatch
):
    expected_source, loader_calls, make_args = _mock_capture_runtime(monkeypatch)
    uninterrupted_root = tmp_path / "uninterrupted"
    resumed_root = tmp_path / "resumed"
    uninterrupted = capture(make_args(uninterrupted_root))
    assert uninterrupted["source"] == expected_source
    assert uninterrupted["split_order"] == [
        "normalization_fit",
        "calibration",
        "development",
        "confirmation",
        "train",
    ]
    assert uninterrupted["capture_implementation"]["dependencies"] == {
        "test-runtime": "exact"
    }
    assert (
        uninterrupted["capture_binding"]["capture_implementation"]
        == uninterrupted["capture_implementation"]
    )
    validate_capture_manifest(uninterrupted)
    tampered_capture = json.loads(json.dumps(uninterrupted))
    tampered_capture["capture_binding"]["capture_implementation"]["dependencies"] = {
        "test-runtime": "forged"
    }
    tampered_capture["capture_content_sha256"] = data_module._canonical_hash(
        {
            key: value
            for key, value in tampered_capture.items()
            if key != "capture_content_sha256"
        }
    )
    with pytest.raises(ValueError, match="implementation differs"):
        validate_capture_manifest(tampered_capture)
    zero_digest_capture = json.loads(json.dumps(uninterrupted))
    zero_digest_capture["capture_binding_sha256"] = "0" * 64
    zero_digest_capture["capture_content_sha256"] = data_module._canonical_hash(
        {
            key: value
            for key, value in zero_digest_capture.items()
            if key != "capture_content_sha256"
        }
    )
    with pytest.raises(ValueError, match="binding digest mismatch"):
        validate_capture_manifest(zero_digest_capture)
    arbitrary_implementation = json.loads(json.dumps(uninterrupted))
    arbitrary_implementation["capture_implementation"] = {
        "schema": "evil-arbitrary-code-v1"
    }
    arbitrary_implementation["capture_binding"]["capture_implementation"] = {
        "schema": "evil-arbitrary-code-v1"
    }
    arbitrary_implementation["capture_binding_sha256"] = data_module._canonical_hash(
        arbitrary_implementation["capture_binding"]
    )
    arbitrary_implementation["capture_content_sha256"] = data_module._canonical_hash(
        {
            key: value
            for key, value in arbitrary_implementation.items()
            if key != "capture_content_sha256"
        }
    )
    with pytest.raises(ValueError, match="capture implementation keys mismatch"):
        validate_capture_manifest(arbitrary_implementation)
    assert uninterrupted["source"]["transformer_lens_model_names"] == ["gpt2"]
    assert loader_calls[0][0] == "gpt2"
    assert loader_calls[0][1]["revision"] == expected_source["sources"][0]["revision"]
    assert loader_calls[0][1]["tokenizer"] is not None

    derived_root = tmp_path / "derived"
    derive_views(
        uninterrupted_root,
        derived_root,
        ("scalar_rms",),
        batch_size=32,
    )
    derived_transform = Whitener.load(derived_root / "scalar_rms" / "whitener.pt")
    assert uninterrupted["split_plan"]["normalization_fit"]["actual_tokens"] == 127
    assert derived_transform.n_fit_tokens == 2
    assert derived_transform.meta["source_fit_requested_tokens"] == 2

    transform_root = tmp_path / "transforms"
    fitted = fit_transform_artifacts(
        uninterrupted_root,
        transform_root,
        ("scalar_rms",),
        batch_size=32,
    )["scalar_rms"]
    fitted_transform = Whitener.load(fitted["path"])
    assert fitted_transform.n_fit_tokens == 2
    assert fitted_transform.meta["source_fit_requested_tokens"] == 2

    fired = False

    def fail_once(split, persisted):
        nonlocal fired
        if not fired:
            fired = True
            raise RuntimeError(f"injected after {split}:{persisted}")

    with pytest.raises(RuntimeError, match="injected"):
        capture(make_args(resumed_root), failure_injector=fail_once)
    partial = StoreReader(
        resumed_root,
        "normalization_fit",
        allow_incomplete=True,
    )
    assert partial.verify() == 64
    resumed = capture(make_args(resumed_root, resume=True))
    assert resumed == uninterrupted
    validate_capture_manifest(resumed)
    for split in uninterrupted["split_order"]:
        left = StoreReader(uninterrupted_root, split)
        right = StoreReader(resumed_root, split)
        assert left.manifest == right.manifest
        for shard in left.manifest["shards"]:
            left_acts, left_ids = left._shard_payload(shard, verify=True)
            right_acts, right_ids = right._shard_payload(shard, verify=True)
            assert torch.equal(left_acts, right_acts)
            assert torch.equal(left_ids, right_ids)


def test_completed_capture_resume_is_idempotent_and_rebuilds_final_manifest(
    tmp_path, monkeypatch
):
    _, _, make_args = _mock_capture_runtime(monkeypatch)
    root = tmp_path / "capture"
    completed = capture(make_args(root))
    assert capture(make_args(root, resume=True)) == completed

    (root / "capture.json").unlink()
    rebuilt = capture(make_args(root, resume=True))
    assert rebuilt == completed
    assert rebuilt["capture_binding"] == completed["capture_binding"]
    validate_capture_manifest(rebuilt)


def test_verify_store_root_rejects_wrong_source_binding_and_row_identity(
    tmp_path, monkeypatch
):
    _, _, make_args = _mock_capture_runtime(monkeypatch)

    def rewrite_train(root, *, whitener_hash=None, corrupt_identity=False):
        reader = StoreReader(root, "train")
        batches = list(reader.sequential_batches_with_ids(512))
        meta = reader.manifest["meta"]
        shutil.rmtree(root / "train")
        writer = ShardWriter(
            root,
            "train",
            whitener_hash=(
                reader.whitener_hash if whitener_hash is None else whitener_hash
            ),
            sites=reader.sites,
            d_model=reader.d_model,
            meta=meta,
            tokens_per_shard=reader.manifest["tokens_per_shard"],
            free_space_floor_frac=0,
        )
        for acts, row_ids in batches:
            if corrupt_identity:
                row_ids = row_ids.clone()
                row_ids[0, 0] += 1
            writer.add(acts, row_ids)
        writer.close()

    wrong_source = tmp_path / "wrong-source"
    capture(make_args(wrong_source))
    rewrite_train(wrong_source, whitener_hash="raw:" + "f" * 64)
    with pytest.raises(ValueError, match="not bound to its capture source"):
        verify_store_root(wrong_source)

    wrong_identity = tmp_path / "wrong-identity"
    capture(make_args(wrong_identity))
    rewrite_train(wrong_identity, corrupt_identity=True)
    with pytest.raises(ValueError, match="sequence differs"):
        verify_store_root(wrong_identity)


def test_data_verify_rejects_empty_and_incomplete_capture_role_sets(
    tmp_path, monkeypatch
):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit) as exc_info:
        data_module.main(["verify", "--store", str(empty)])
    assert exc_info.value.code == 2

    _, _, make_args = _mock_capture_runtime(monkeypatch)
    root = tmp_path / "capture"
    capture(make_args(root))
    assert set(verify_store_root(root)) == {
        "normalization_fit",
        "calibration",
        "development",
        "confirmation",
        "train",
    }
    extra = root / "undeclared"
    extra.mkdir()
    (extra / "split.json").write_text("{}\n")
    with pytest.raises(ValueError, match="split set differs"):
        verify_store_root(root)


def test_capture_refuses_pipeline_residency_before_creating_output(
    tmp_path, monkeypatch
):
    _, _, make_args = _mock_capture_runtime(monkeypatch)
    out = tmp_path / "refused"
    args = make_args(out)
    args.max_writer_residency_bytes = 1
    with pytest.raises(
        ValueError,
        match="pipeline host residency.*required=.*limit=1",
    ):
        capture(args)
    assert not out.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_capture_copy_overlap_is_byte_exact_ordered_and_close_safe():
    identities = [torch.full((5, 3), index, dtype=torch.int64) for index in range(4)]
    expected = [
        (torch.arange(60, dtype=torch.bfloat16).reshape(5, 3, 4) + index).cuda()
        for index in range(4)
    ]

    def source():
        for activation, row_ids in zip(expected, identities, strict=True):
            yield activation.clone(), row_ids

    observed = list(_overlap_cuda_capture_copies(source()))
    assert all(
        torch.equal(row_ids, reference)
        for (_, row_ids), reference in zip(observed, identities, strict=True)
    )
    assert all(host.is_pinned() for host, _ in observed)
    for (host, _), reference in zip(observed, expected, strict=True):
        assert torch.equal(host, reference.cpu())

    closed = False

    def closing_source():
        nonlocal closed
        try:
            yield from source()
        finally:
            closed = True

    stream = _overlap_cuda_capture_copies(closing_source())
    next(stream)
    stream.close()
    assert closed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_capture_copy_overlap_holds_only_two_pinned_destinations(monkeypatch):
    real_empty_like = torch.empty_like
    destinations: list[weakref.ReferenceType[torch.Tensor]] = []
    peak_live = 0

    def tracked_empty_like(*args, **kwargs):
        nonlocal peak_live
        live_before = sum(reference() is not None for reference in destinations)
        result = real_empty_like(*args, **kwargs)
        destinations.append(weakref.ref(result))
        peak_live = max(peak_live, live_before + 1)
        return result

    monkeypatch.setattr(data_module.torch, "empty_like", tracked_empty_like)
    identities = torch.zeros(4, 3, dtype=torch.int64)

    def source():
        for index in range(6):
            yield (
                torch.full(
                    (4, 2, 8),
                    index,
                    dtype=torch.bfloat16,
                    device="cuda",
                ),
                identities,
            )

    stream = _overlap_cuda_capture_copies(source())
    expected_value = 0
    while True:
        try:
            item = next(stream)
        except StopIteration:
            break
        host, row_ids = item
        assert bool((host == expected_value).all())
        expected_value += 1
        del item, host, row_ids
        gc.collect()
    assert expected_value == 6
    assert peak_live == 2


def test_capture_streams_slices_without_torch_cat(tmp_path, monkeypatch):
    _, _, make_args = _mock_capture_runtime(monkeypatch)

    def forbidden_cat(*args, **kwargs):
        raise AssertionError("capture assembled transient concatenation")

    monkeypatch.setattr(torch, "cat", forbidden_cat)
    manifest = capture(make_args(tmp_path / "direct-slices"))
    assert manifest["split_order"][-1] == "train"


def test_derive_refuses_writer_residency_before_creating_output(tmp_path):
    raw = tmp_path / "raw"
    _raw_store(raw)
    out = tmp_path / "refused-views"
    with pytest.raises(ValueError, match="writer residency.*required=.*limit=1"):
        derive_views(
            raw,
            out,
            ("none",),
            batch_size=13,
            max_writer_residency_bytes=1,
        )
    assert not out.exists()


def test_capture_resume_adopts_first_shard_rename_before_first_manifest(
    tmp_path, monkeypatch
):
    _, _, make_args = _mock_capture_runtime(monkeypatch)
    uninterrupted_root = tmp_path / "uninterrupted"
    resumed_root = tmp_path / "resumed"
    uninterrupted = capture(make_args(uninterrupted_root))

    original_write_manifest = ShardWriter._write_manifest
    fired = False

    def crash_once(self, *, complete):
        nonlocal fired
        if self.split == "normalization_fit" and not complete and not fired:
            fired = True
            raise RuntimeError("injected after first shard rename")
        return original_write_manifest(self, complete=complete)

    monkeypatch.setattr(ShardWriter, "_write_manifest", crash_once)
    with pytest.raises(RuntimeError, match="first shard rename"):
        capture(make_args(resumed_root))
    assert (resumed_root / "normalization_fit" / "shard_00000.safetensors").is_file()
    assert not (resumed_root / "normalization_fit" / "split.json").exists()

    monkeypatch.setattr(ShardWriter, "_write_manifest", original_write_manifest)
    resumed = capture(make_args(resumed_root, resume=True))
    assert resumed == uninterrupted
    for split in uninterrupted["split_order"]:
        assert (
            StoreReader(resumed_root, split).manifest
            == StoreReader(uninterrupted_root, split).manifest
        )


def test_capture_resume_finalizes_full_incomplete_split_without_replay(
    tmp_path, monkeypatch
):
    _, _, make_args = _mock_capture_runtime(monkeypatch)
    uninterrupted_root = tmp_path / "uninterrupted"
    resumed_root = tmp_path / "resumed"
    uninterrupted = capture(make_args(uninterrupted_root))

    original_write_manifest = ShardWriter._write_manifest
    fired = False

    def crash_once(self, *, complete):
        nonlocal fired
        if self.split == "normalization_fit" and complete and not fired:
            fired = True
            raise RuntimeError("injected before complete manifest")
        return original_write_manifest(self, complete=complete)

    monkeypatch.setattr(ShardWriter, "_write_manifest", crash_once)
    with pytest.raises(RuntimeError, match="complete manifest"):
        capture(make_args(resumed_root))
    partial = StoreReader(
        resumed_root,
        "normalization_fit",
        allow_incomplete=True,
    )
    assert partial.manifest["complete"] is False
    assert (
        partial.verify()
        == uninterrupted["split_plan"]["normalization_fit"]["actual_tokens"]
    )

    monkeypatch.setattr(ShardWriter, "_write_manifest", original_write_manifest)
    resumed = capture(make_args(resumed_root, resume=True))
    assert resumed == uninterrupted
    for split in uninterrupted["split_order"]:
        assert (
            StoreReader(resumed_root, split).manifest
            == StoreReader(uninterrupted_root, split).manifest
        )


def test_capture_refuses_duplicate_hooks_and_multiple_models_before_load(
    tmp_path, monkeypatch
):
    expected_source, loader_calls, make_args = _mock_capture_runtime(monkeypatch)
    first = expected_source["sources"][0]
    duplicate = "|".join((first["model"], first["revision"], first["hook"]))
    with pytest.raises(ValueError, match="hooks must be unique"):
        capture(make_args(tmp_path / "duplicate", sources=[duplicate, duplicate]))
    assert not loader_calls

    second_model = f"other/model|{first['revision']}|blocks.1.hook_resid_pre"

    # Permit model-info resolution for the alternate only; capture must still
    # refuse the cross-model contract before tokenizer or model loading.
    class MultiHfApi:
        def model_info(self, model, revision):
            return SimpleNamespace(sha=revision)

        def dataset_info(self, corpus, revision):
            return SimpleNamespace(sha=revision)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(HfApi=MultiHfApi),
    )
    with pytest.raises(ValueError, match="single-model-only"):
        capture(make_args(tmp_path / "multi", sources=[duplicate, second_model]))
    assert not loader_calls


def test_all_data_producers_preflight_their_actual_destinations(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    _raw_store(raw)
    calls = []

    def checked(destination, required_bytes, *, operation):
        calls.append((Path(destination).resolve(), required_bytes, operation))
        return {
            "destination": str(Path(destination).resolve()),
            "required_bytes": required_bytes,
        }

    monkeypatch.setattr(data_module, "_enforce_prewrite_storage", checked)
    derive_views(raw, tmp_path / "views", ("none",))
    fit_transform_artifacts(raw, tmp_path / "transforms", ("scalar_rms",))

    _, _, make_args = _mock_capture_runtime(monkeypatch)
    capture_out = tmp_path / "capture"
    capture(make_args(capture_out))

    assert any(
        destination == (tmp_path / "views" / "none" / "whitener.pt").resolve()
        and operation == "derive 'none' transform"
        for destination, _, operation in calls
    )
    assert any(
        "fit-transform 'scalar_rms'" == operation
        and str(destination).startswith(str((tmp_path / "transforms").resolve()))
        for destination, _, operation in calls
    )
    assert any(
        destination == capture_out.resolve() and operation == "capture"
        for destination, _, operation in calls
    )


def test_incremental_storage_preflight_credits_only_verified_inputs(
    tmp_path, monkeypatch
):
    raw = tmp_path / "raw"
    _raw_store(raw)
    for name in (
        "BSC_ACTIVATION_STORE",
        "BSC_STORE_ROOT",
        "BSC_RAW_STORE_ROOT",
        "BSC_RAW_STORE",
        "BSC_TRANSFORM_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BSC_RAW_STORE_ROOT", str(raw))
    verified = _verified_existing_input_storage()
    assert verified["verified_existing_input_bytes"] > 0
    assert len(verified["inputs"][0]["splits"]) == 5
    free = 50
    monkeypatch.setattr(
        matrix_module.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=1_000, used=950, free=free),
    )
    estimate = verified["verified_existing_input_bytes"] + 100
    preflight = _storage_preflight(tmp_path / "campaign", estimate)
    assert (
        preflight["credited_existing_input_bytes"]
        == verified["verified_existing_input_bytes"]
    )
    assert preflight["additional_storage_bytes_required"] == 100
    assert preflight["sufficient"] is False
    original_verify = StoreReader.verify

    def unexpected_rehash(self):
        raise AssertionError("unchanged stat-bound stores should reuse receipts")

    monkeypatch.setattr(StoreReader, "verify", unexpected_rehash)
    cached = _storage_preflight(tmp_path / "campaign", estimate)
    assert (
        cached["credited_existing_input_bytes"]
        == verified["verified_existing_input_bytes"]
    )
    monkeypatch.setattr(StoreReader, "verify", original_verify)

    shard = next(raw.rglob("*.safetensors"))
    corrupted = bytearray(shard.read_bytes())
    corrupted[-1] ^= 0xFF
    shard.write_bytes(corrupted)
    with pytest.raises(StudyError, match="checksum"):
        _verified_existing_input_storage()


def test_matrix_store_receipt_rechecks_content_when_stat_receipt_is_refreshed(
    tmp_path, monkeypatch
):
    raw = tmp_path / "raw"
    campaign_root = tmp_path / "campaign"
    _raw_store(raw)
    monkeypatch.setenv("BSC_RAW_STORE_ROOT", str(raw))
    verified = _verified_existing_input_storage()
    _storage_preflight(
        campaign_root,
        verified["verified_existing_input_bytes"] + 1,
    )
    receipts = [
        (path, json.loads(path.read_text()))
        for path in (campaign_root / ".store-verification").glob("*.json")
    ]
    receipt_path, receipt = next(
        (path, payload) for path, payload in receipts if payload["split"] == "train"
    )
    assert receipt["content_probes"]
    shard = raw / "train" / "shard_00000.safetensors"
    body = bytearray(shard.read_bytes())
    body[-1] ^= 1
    shard.write_bytes(body)

    # A stat-only external refresh is insufficient: the verifier-issued probe
    # stays old, invalidates the receipt, and triggers the full store checksum.
    status = shard.stat()
    receipt["stat_fingerprint"]["shards"][0] = {
        "path": str(shard.resolve()),
        "size_bytes": status.st_size,
        "mtime_ns": status.st_mtime_ns,
        "ctime_ns": status.st_ctime_ns,
        "device": status.st_dev,
        "inode": status.st_ino,
    }
    receipt_path.write_text(json.dumps(receipt) + "\n")
    with pytest.raises(StudyError, match="checksum"):
        _storage_preflight(
            campaign_root,
            verified["verified_existing_input_bytes"] + 1,
        )


def test_prewrite_storage_gate_preserves_fifteen_percent_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(
        data_module.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=1_000, used=800, free=200),
    )
    with pytest.raises(ValueError, match="available_above_15pct_floor=50"):
        data_module._enforce_prewrite_storage(
            tmp_path / "destination",
            51,
            operation="test producer",
        )
    accepted = data_module._enforce_prewrite_storage(
        tmp_path / "destination",
        50,
        operation="test producer",
    )
    assert accepted["raw_free_bytes"] == 200
    assert accepted["free_space_floor_bytes"] == 150
    assert accepted["available_above_floor_bytes"] == 50
    assert accepted["device"] == tmp_path.stat().st_dev


def test_storage_preflight_does_not_aggregate_distinct_destination_devices(
    tmp_path, monkeypatch
):
    campaign_root = tmp_path / "campaign"
    raw_root = tmp_path / "raw"
    view_root = tmp_path / "views"

    class FakeParent:
        def __init__(self, name, device):
            self.name = name
            self.device = device

        def stat(self):
            return SimpleNamespace(st_dev=self.device)

        def __str__(self):
            return self.name

    parents = {
        "campaign": FakeParent("campaign-fs", 1),
        "raw": FakeParent("raw-fs", 2),
        "views": FakeParent("view-fs", 3),
    }
    monkeypatch.setattr(
        matrix_module,
        "_nearest_existing_parent",
        lambda path: parents[
            "views"
            if "views" in str(path)
            else "raw"
            if "raw" in str(path)
            else "campaign"
        ],
    )
    usage = {
        "campaign-fs": SimpleNamespace(total=1_000, used=750, free=250),
        "raw-fs": SimpleNamespace(total=1_000, used=600, free=400),
        "view-fs": SimpleNamespace(total=1_000, used=700, free=300),
    }
    monkeypatch.setattr(
        matrix_module.shutil,
        "disk_usage",
        lambda parent: usage[str(parent)],
    )
    monkeypatch.setattr(
        matrix_module,
        "_verified_existing_input_storage",
        lambda **kwargs: {
            "verified_existing_input_bytes": 800,
            "inputs": [{"root": str(raw_root), "verified_bytes": 800}],
        },
    )
    monkeypatch.setattr(
        matrix_module,
        "_estimated_plan_input_storage_bytes",
        lambda plan: 1_000,
    )
    monkeypatch.setattr(
        matrix_module,
        "_configured_input_roots",
        lambda explicit=(): (raw_root, view_root),
    )
    preflight = _storage_preflight(
        campaign_root,
        1_100,
        plan=object(),
        input_roots=(view_root,),
    )
    by_device = {item["device"]: item for item in preflight["filesystem_preflights"]}
    assert by_device[1]["required_bytes"] == 100
    # The 200-byte unmaterialized remainder is conservatively required on
    # every declared input destination, even though the raw root supplied all
    # existing credit. Planning may override this; scientific launch expects
    # the verified remainder to be zero.
    assert by_device[2]["required_bytes"] == 200
    assert by_device[3]["required_bytes"] == 200
    assert by_device[2]["sufficient"] is True
    assert by_device[3]["sufficient"] is False
    assert preflight["sufficient"] is False


def test_storage_preflight_sums_same_device_roles(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    view_root = tmp_path / "views"

    class SameParent:
        def stat(self):
            return SimpleNamespace(st_dev=7)

        def __str__(self):
            return "shared-fs"

    parent = SameParent()
    monkeypatch.setattr(matrix_module, "_nearest_existing_parent", lambda path: parent)
    monkeypatch.setattr(
        matrix_module.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=1_000, used=600, free=400),
    )
    monkeypatch.setattr(
        matrix_module,
        "_verified_existing_input_storage",
        lambda **kwargs: {"verified_existing_input_bytes": 800, "inputs": []},
    )
    monkeypatch.setattr(
        matrix_module,
        "_estimated_plan_input_storage_bytes",
        lambda plan: 1_000,
    )
    monkeypatch.setattr(
        matrix_module,
        "_configured_input_roots",
        lambda explicit=(): (view_root,),
    )
    preflight = _storage_preflight(
        root,
        1_100,
        plan=object(),
        input_roots=(view_root,),
    )
    [shared] = preflight["filesystem_preflights"]
    assert shared["required_bytes"] == 300
    assert shared["available_above_floor_bytes"] == 250
    assert shared["sufficient"] is False


def test_storage_credit_is_plan_bound_and_phase1_is_always_zero(tmp_path, monkeypatch):
    raw = tmp_path / "unrelated"
    _raw_store(raw, authenticated_profile="phase2")
    monkeypatch.setenv("BSC_RAW_STORE_ROOT", str(raw))
    phase1 = build_phase1_plan((0,), smoke=True)
    phase1_credit = _verified_existing_input_storage(plan=phase1)
    assert phase1_credit["verified_existing_input_bytes"] == 0
    assert phase1_credit["inputs"] == []

    # The capture is internally authenticated but belongs to a different
    # source and split allocation, so it cannot buy credit for Phase 2.
    phase2 = build_phase2_plan((0,), smoke=True)
    phase2_credit = _verified_existing_input_storage(plan=phase2)
    assert phase2_credit["verified_existing_input_bytes"] == 0
    assert phase2_credit["inputs"][0]["eligible_for_plan"] is False

    _, _, make_args = _mock_capture_runtime(monkeypatch)
    matching = tmp_path / "matching"
    matching_args = make_args(matching)
    matching_args.split = [
        f"{role}=64" for role in data_module.CAPTURE_PROFILE_SPLITS["phase2"]
    ]
    capture(matching_args)
    monkeypatch.setenv("BSC_RAW_STORE_ROOT", str(matching))
    matching_credit = _verified_existing_input_storage(plan=phase2)
    assert matching_credit["verified_existing_input_bytes"] > 0
    assert matching_credit["inputs"][0]["eligible_for_plan"] is True


def test_matrix_run_rechecks_storage_and_exits_nonzero_after_cell_failure(
    tmp_path, monkeypatch, capsys
):
    plan = build_phase1_plan((0,), smoke=True)

    class FakeCampaign:
        def __init__(self):
            self.plan = plan

        def status(self):
            return {"plan_id": plan.plan_id}

    campaign = FakeCampaign()
    monkeypatch.setattr(matrix_module, "Campaign", lambda root: campaign)
    preflight_calls = []

    def checked(root, estimate, *, allow_insufficient, plan, input_roots=()):
        preflight_calls.append((root, allow_insufficient, plan, input_roots))
        return {"sufficient": True}

    monkeypatch.setattr(matrix_module, "_checked_storage_extension", checked)
    monkeypatch.setattr(
        matrix_module,
        "_run_with_optional_view_dispatch",
        lambda campaign, args: SimpleNamespace(
            failed_cells=1,
            to_dict=lambda: {
                "selected_cells": 1,
                "completed_cells": 0,
                "failed_cells": 1,
                "skipped_cells": 0,
            },
        ),
    )
    with pytest.raises(SystemExit) as exc_info:
        matrix_main(["run", "--root", str(tmp_path)])
    assert exc_info.value.code == 1
    assert preflight_calls == [(tmp_path, False, plan, ())]
    output = json.loads(capsys.readouterr().out)
    assert output["run"]["failed_cells"] == 1


def test_matrix_run_replays_current_declared_resource_budget_before_dispatch(
    tmp_path, monkeypatch
):
    plan = build_phase1_plan((0,), smoke=True)
    campaign = SimpleNamespace(plan=plan)
    monkeypatch.setattr(matrix_module, "Campaign", lambda root: campaign)
    monkeypatch.setattr(
        matrix_module,
        "enforce_plan_resources",
        lambda loaded: (_ for _ in ()).throw(
            matrix_module.BudgetExceeded("current plan exceeds declared budget")
        ),
    )
    dispatched = []
    monkeypatch.setattr(
        matrix_module,
        "_run_with_optional_view_dispatch",
        lambda campaign, args: dispatched.append(True),
    )
    with pytest.raises(SystemExit) as exc_info:
        matrix_main(("run", "--root", str(tmp_path)))
    assert exc_info.value.code == 2
    assert dispatched == []


def test_matrix_sigterm_unwinds_dispatch_and_restores_prior_handler(
    tmp_path, monkeypatch
):
    plan = build_phase1_plan((0,), smoke=True)
    campaign = SimpleNamespace(plan=plan)
    monkeypatch.setattr(matrix_module, "Campaign", lambda root: campaign)
    monkeypatch.setattr(matrix_module, "enforce_plan_resources", lambda plan: plan)
    monkeypatch.setattr(
        matrix_module,
        "_checked_storage_extension",
        lambda *args, **kwargs: {"sufficient": True},
    )
    previous = object()
    current = {"handler": previous}
    installs = []
    monkeypatch.setattr(
        matrix_module.signal,
        "getsignal",
        lambda signum: current["handler"],
    )

    def install(signum, handler):
        installs.append((signum, handler))
        prior = current["handler"]
        current["handler"] = handler
        return prior

    monkeypatch.setattr(matrix_module.signal, "signal", install)

    def terminated(campaign, args):
        current["handler"](matrix_module.signal.SIGTERM, None)
        raise AssertionError("SIGTERM handler returned")

    monkeypatch.setattr(matrix_module, "_run_with_optional_view_dispatch", terminated)
    with pytest.raises(SystemExit) as exc_info:
        matrix_main(("run", "--root", str(tmp_path)))
    assert exc_info.value.code == 128 + matrix_module.signal.SIGTERM
    assert current["handler"] is previous
    assert installs[-1] == (matrix_module.signal.SIGTERM, previous)


def test_phase2_view_dispatch_is_per_cell_and_fails_closed_on_manifests(
    tmp_path,
):
    raw = tmp_path / "raw"
    views = tmp_path / "views"
    _raw_store(raw)
    derive_views(raw, views, ("none", "scalar_rms"), batch_size=13)
    split_sizes = (
        ("normalization_fit", 64),
        ("calibration", 48),
        ("development", 32),
        ("confirmation", 32),
        ("train", 80),
    )
    source = json.loads((raw / "capture.json").read_text())["source"]
    base_source_keys = {
        "sources",
        "corpus",
        "corpus_config",
        "corpus_revision",
        "corpus_split",
        "context",
        "drop_positions",
        "tokenizer_hashes",
        "tokenizer_contract",
        "store_contract_version",
        "alignment_version",
        "alignment_audit",
    }
    common_values = {
        "data.split_sizes": split_sizes,
        "data.context_drop_policy": "drop_bos_position_0",
        "data.context_length": source["context"],
        "data.store_sites": tuple(item["hook"] for item in source["sources"]),
        "data.source_models": tuple(item["model"] for item in source["sources"]),
        "data.source_model_revisions": tuple(
            item["revision"] for item in source["sources"]
        ),
        "data.corpus": (source["corpus"],),
        "data.corpus_config": (source["corpus_config"],),
        "data.corpus_revision": (source["corpus_revision"],),
        "data.corpus_split": (source["corpus_split"],),
        "data.tokenizer_hashes": tuple(source["tokenizer_hashes"]),
        "data.tokenizer_contract": source["tokenizer_contract"],
        "data.store_contract_version": source["store_contract_version"],
        "data.alignment_version": source["alignment_version"],
        "data.alignment_audit": source["alignment_audit"],
        "data.capture_contract": tuple(
            (key, value) for key, value in source.items() if key not in base_source_keys
        ),
    }
    cells = {
        "none-cell": SimpleNamespace(
            decision_map={
                **common_values,
                "data.normalization": "none",
            }
        ),
        "scalar-cell": SimpleNamespace(
            decision_map={
                **common_values,
                "data.normalization": "scalar_rms",
            }
        ),
    }
    dispatched = _resolve_phase2_view_dispatch(views, cells)
    assert dispatched == {
        "none-cell": (views / "none").resolve(),
        "scalar-cell": (views / "scalar_rms").resolve(),
    }

    missing = {
        "missing": SimpleNamespace(
            decision_map={
                "data.normalization": "whiten",
                "data.split_sizes": split_sizes,
            }
        )
    }
    with pytest.raises(StudyError, match="does not exist"):
        _resolve_phase2_view_dispatch(views, missing)

    foreign = views / "none" / "foreign.bin"
    foreign.write_bytes(b"not part of the view")
    with pytest.raises(StudyError, match="root entries differ"):
        _resolve_phase2_view_dispatch(views, {"none-cell": cells["none-cell"]})
    foreign.unlink()

    manifest_path = views / "none" / "train" / "split.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["n_tokens"] += 1
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(StudyError, match="manifest hash mismatch"):
        _resolve_phase2_view_dispatch(views, {"none-cell": cells["none-cell"]})


def test_phase2_view_dispatch_resume_includes_running_cells(tmp_path):
    calls = []

    class FakeCampaign:
        plan = SimpleNamespace(phase=Phase.PHASE2)

        def runnable_cell_ids(self, **kwargs):
            calls.append(kwargs)
            return ()

    args = SimpleNamespace(
        resume=True,
        stop_after=None,
        view_root=tmp_path,
        python=sys.executable,
        module="block_crosscoder_experiment.cli.run_cell",
        cells=None,
        limit=None,
    )
    summary = _run_with_optional_view_dispatch(FakeCampaign(), args)
    assert summary.selected_cells == 0
    assert calls == [{"include_failed": True, "include_resume_required": True}]


@pytest.mark.parametrize(
    "extra",
    [
        ("--limit", "0"),
        ("--limit", "1", "--cell", "phase1.some-cell.s0"),
    ],
)
def test_matrix_run_rejects_zero_or_ambiguous_selection(extra, tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        matrix_main(("run", "--root", str(tmp_path), *extra))
    assert exc_info.value.code == 2


def test_matrix_cli_dispatches_complete_smoke_family_branch_lifecycle(
    tmp_path, monkeypatch, capsys
):
    blueprint = build_phase2_blueprint((0,), smoke=True)
    plan = build_phase2_plan((0,), smoke=True)
    family = next(
        item
        for item in blueprint.comparator_families
        if item.name == "bsf_grassmannian"
    )
    blueprint_path = tmp_path / "blueprint.json"
    blueprint_path.write_text(json.dumps(blueprint.to_manifest()) + "\n")

    class FakeCampaign:
        def __init__(self):
            self.plan = plan
            self.blueprint_path = blueprint_path
            self.family_extensions = []
            self.revisits = []

        def status(self):
            return {"plan_id": self.plan.plan_id}

        def select_family_root(self, family_name, *, out=None):
            assert family_name == family.name
            return {"schema": "selection", "family": family_name}

        def select_family_revisit_inputs(self, family_name, *, out=None):
            assert family_name == family.name
            return {"schema": "nomination", "family": family_name}

        def extend_family(
            self,
            extended,
            *,
            family_name,
            selection,
            selection_path,
        ):
            assert family_name == family.name
            self.plan = extended
            self.family_extensions.append(extended.stages[-1].name)

        def extend_family_revisit(
            self,
            extended,
            *,
            family_name,
            selection_path,
        ):
            assert family_name == family.name
            self.plan = extended
            self.revisits.append(extended.stages[-1].name)

    campaign = FakeCampaign()
    monkeypatch.setattr(matrix_module, "Campaign", lambda root: campaign)
    monkeypatch.setattr(
        matrix_module,
        "_checked_storage_extension",
        lambda *args, **kwargs: {"sufficient": True},
    )

    matrix_main(
        [
            "select-family-root",
            "--root",
            str(tmp_path / "campaign"),
            "--family",
            family.name,
        ]
    )

    def candidate_groups(stage):
        groups = {}
        for cell in stage.cells:
            groups.setdefault(cell.candidate_id, []).append(cell)
        return [tuple(items) for _, items in sorted(groups.items())]

    def frozen(policy, cells, universe):
        return FrozenSelection.from_cells(
            policy,
            cells,
            [0.5 + index for index in range(len(cells))],
            [
                "sha256:" + hashlib.sha256(cell.cell_id.encode()).hexdigest()
                for cell in cells
            ],
            "sha256:" + hashlib.sha256(universe.encode()).hexdigest(),
        )

    root_cells = next(
        group
        for group in candidate_groups(campaign.plan.stages[0])
        if group[0].recipe_name == family.root_recipe_name
    )
    selection = frozen(family.root_selection_policy, root_cells, "family-root")
    selection_path = tmp_path / "family-selection.json"

    family_stages = []
    for round_index in range(len(family.rounds)):
        selection_path.write_text(
            json.dumps({"selected": [selection.to_dict()]}) + "\n"
        )
        matrix_main(
            [
                "advance-family",
                "--root",
                str(tmp_path / "campaign"),
                "--family",
                family.name,
                "--selection",
                str(selection_path),
                "--allow-insufficient-local-storage",
            ]
        )
        stage = campaign.plan.stages[-1]
        family_stages.append(stage)
        if round_index + 1 < len(family.rounds):
            selection = frozen(
                stage.selection_policy,
                candidate_groups(stage)[0],
                f"round-{round_index}",
            )

    nomination_universe = "all-family-rounds"
    nomination_groups = []
    seen_signatures = set()
    for stage in family_stages:
        for group in candidate_groups(stage):
            signature = resolved_candidate_execution_signature(group)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            nomination_groups.append(group)
            if len(nomination_groups) == family.revisit.top_k:
                break
        if len(nomination_groups) == family.revisit.top_k:
            break
    assert len(nomination_groups) == family.revisit.top_k
    nominations = [
        frozen(family.revisit.nomination_policy, group, nomination_universe)
        for group in nomination_groups
    ]
    nomination_path = tmp_path / "family-top2.json"
    nomination_path.write_text(
        json.dumps({"selected": [item.to_dict() for item in nominations]}) + "\n"
    )
    matrix_main(
        [
            "nominate-family-revisit",
            "--root",
            str(tmp_path / "campaign"),
            "--family",
            family.name,
        ]
    )
    matrix_main(
        [
            "revisit-family",
            "--root",
            str(tmp_path / "campaign"),
            "--family",
            family.name,
            "--selection",
            str(nomination_path),
            "--allow-insufficient-local-storage",
        ]
    )
    assert campaign.family_extensions == [item.name for item in family.rounds]
    assert campaign.revisits == [family.revisit.name]
    assert campaign.plan.stages[-1].selection_policy == family.revisit.selection_policy
    assert len(candidate_groups(campaign.plan.stages[-1])) == 2
    assert capsys.readouterr().out
