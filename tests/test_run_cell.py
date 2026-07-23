"""End-to-end and fail-closed tests for the generic cell executor."""

from __future__ import annotations

import copy
import fcntl
import json
import hashlib
import io
import inspect
import math
import os
import subprocess
import sys
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

from block_crosscoder_experiment.campaign import (
    Campaign,
    CampaignRunner,
    QUALIFICATION_SCHEMA,
    RunState,
)
from block_crosscoder_experiment.cli.run_cell import (
    CellExecutionError,
    _Context,
    _consume_chunked_rd_evaluation_batch,
    _FileFingerprint,
    _RawEndpointErrorCache,
    _RetainedArtifactKey,
    _StageExecutionCache,
    _VERIFIED_STORE_BINDINGS,
    _accumulate_chunked_recovery_association,
    _assert_deployment_snapshot_digest,
    _assert_durable_snapshot_schema,
    _assert_model_snapshot_lineage_current,
    _assert_serialized_snapshot_current,
    _apply_encoder_scale_calibration,
    _encoder_scale_fit_batches,
    _execution_rng_snapshot,
    _finalize_development_time_sharing,
    _evaluate_cached_time_sharing,
    _balanced_schedule_uses_upper,
    _expected_capture_allocation,
    _expected_real_source_contract,
    _fixed_rate_raw_score,
    _gather_event_factor_blocks,
    _gpu_lock_path,
    _host_gpu_execution_lock,
    _lower_convex_rate_envelope,
    _load_deployable_codec,
    _load_deployment_schedule_bundle,
    _load_capture_contract,
    _matching_pathologies,
    _mapped_support_confusion_counts,
    _model_config,
    _normalize_model_only_consumer_state,
    _normalization_record,
    _persisted_view_validation,
    _phase1_identification_evidence,
    _phase1_identification_outcome,
    _production_precision_preflight,
    _resolve_real_store,
    _resolved_runtime_device,
    _selection_validation_metrics,
    _selected_time_sharing_plans,
    _synthetic_batches,
    _synthetic_dataset,
    _synthetic_source_contract,
    _synchronous_model_snapshot,
    _support_confusion,
    _support_matched_subspace_overlap,
    _tensor_payload_digest,
    _time_sharing_plan_key,
    _transform_on_cuda,
    _training_batches,
    _validate_final_checkpoint,
    _validate_preparation_contract,
    _verify_real_source_contract,
    _verify_store_reader_once,
    _write_deployment_schedule_bundle,
    validate_cell_config,
)
from block_crosscoder_experiment.codec import (
    Codec,
    CodecSpec,
    _RDEvaluationInput,
    _RDEvaluationSelection,
    _RDEvaluationSession,
    fit_codec,
)
from block_crosscoder_experiment.serialization import (
    MODEL_STATE_DIGEST_CONTRACT,
    model_state_digest,
)
from block_crosscoder_experiment.cli.data import (
    CAPTURE_BINDING_SCHEMA,
    CAPTURE_MANIFEST_SCHEMA,
    capture_implementation_contract,
    estimate_capture_pipeline_residency_bytes,
    estimate_writer_residency_bytes,
    fit_transform_artifacts,
    validate_capture_manifest,
)
import block_crosscoder_experiment.implementation as implementation_module
from block_crosscoder_experiment.model import BSCConfig, BlockCrosscoder
from block_crosscoder_experiment.runtime_limits import (
    CODE_NORM_CUDA_IMPLEMENTATION,
    DECODER_RETRACTION_CHOLESKY_QR_IMPLEMENTATION,
    DECODER_RETRACTION_HOUSEHOLDER_QR_IMPLEMENTATION,
    DECODER_RETRACTION_NOT_APPLICABLE,
    DECODER_RETRACTION_SYMMETRIC_POLAR_IMPLEMENTATION,
    FACTORIZED_EXECUTION_DIRECT_RANK_SPACE_IMPLEMENTATION,
    FACTORIZED_EXECUTION_FACTOR_REGULARIZERS_IMPLEMENTATION,
    FACTORIZED_EXECUTION_MATERIALIZED_REFERENCE_IMPLEMENTATION,
    FACTORIZED_EXECUTION_NOT_APPLICABLE,
    DECODED_ENERGY_EXACT_IMPLEMENTATION,
    DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION,
    ISOLATED_LOSS_EXACT_IMPLEMENTATION,
    ISOLATED_LOSS_MAPPED_IMPLEMENTATION,
    MAP_NUCLEAR_GUARDED_MATMUL_IMPLEMENTATION,
    RD_EVALUATION_TOKEN_CHUNK,
    SPARSE_DECODE_CUDA_IMPLEMENTATION,
)
from block_crosscoder_experiment.store import (
    STORE_FORMAT_VERSION,
    ShardWriter,
    StoreReader,
)
from block_crosscoder_experiment.studies import (
    BSC_FACTOR_CONTESTS,
    CellSpec,
    FrozenPanelDecision,
    FrozenPanelEntry,
    FrozenSelection,
    Phase,
    Phase1Blueprint,
    PHASE2_INELIGIBLE_SELECTION_SCORE,
    PHASE2_SELECTION_METRIC_KEY,
    PHASE2_SELECTION_METRIC_PATH,
    RELEASE_DIAGNOSTIC_RECIPES,
    StageSpec,
    StudyError,
    StudyPlan,
    build_phase1_plan,
    build_phase2_plan,
    build_phase2_blueprint,
    build_phase3_blueprint,
    build_phase3_plan,
    canonical_json,
    engineering,
    materialize_child_plan,
    merge_decisions,
)

import pytest
import torch

import block_crosscoder_experiment.cli.run_cell as run_cell_module


REPO = Path(__file__).resolve().parents[1]


class _WeakrefableCacheOwner:
    pass


def _retained_cache_key() -> _RetainedArtifactKey:
    return _RetainedArtifactKey(
        cell_id="cell-a",
        producer_stage="train",
        consumer_stage="calibrate",
        artifact_kind="checkpoint",
        canonical_path="/immutable/cell-a/checkpoint.pt",
        sha256="a" * 64,
        size_bytes=101,
        fingerprint=_FileFingerprint(
            device=11,
            inode=22,
            size_bytes=101,
            mtime_ns=33,
            ctime_ns=44,
        ),
        model_config_sha256="b" * 64,
    )


def test_resolved_runtime_device_canonicalizes_implicit_cuda_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    assert torch.device("cuda:2") != torch.device("cuda")
    assert _resolved_runtime_device("cuda") == torch.device("cuda:2")
    assert _resolved_runtime_device("cuda:1") == torch.device("cuda:1")
    assert _resolved_runtime_device("cpu") == torch.device("cpu")


@pytest.mark.parametrize(
    "changed_field",
    (
        "cell_id",
        "producer_stage",
        "consumer_stage",
        "artifact_kind",
        "canonical_path",
        "sha256",
        "size_bytes",
        "fingerprint.device",
        "fingerprint.inode",
        "fingerprint.size_bytes",
        "fingerprint.mtime_ns",
        "fingerprint.ctime_ns",
        "model_config_sha256",
    ),
)
def test_retained_checkpoint_refuses_every_cache_key_mismatch(
    changed_field: str,
) -> None:
    expected = _retained_cache_key()
    if changed_field.startswith("fingerprint."):
        field = changed_field.removeprefix("fingerprint.")
        actual = replace(
            expected,
            fingerprint=replace(
                expected.fingerprint,
                **{field: getattr(expected.fingerprint, field) + 1},
            ),
        )
    else:
        value = getattr(expected, changed_field)
        actual = replace(
            expected,
            **{
                changed_field: (
                    value + 1 if isinstance(value, int) else value + "-changed"
                )
            },
        )
    cache = _StageExecutionCache()
    cache.remember_checkpoint(
        actual,
        _WeakrefableCacheOwner(),
        {},
        object(),
        released_owner_refs={},
    )
    with pytest.raises(
        CellExecutionError,
        match="binding differs from journaled artifact",
    ):
        cache.take_checkpoint(expected)
    assert cache.checkpoint is None


def test_retained_checkpoint_refuses_a_live_released_training_owner() -> None:
    cache = _StageExecutionCache()
    owner = _WeakrefableCacheOwner()
    cache.remember_checkpoint(
        _retained_cache_key(),
        _WeakrefableCacheOwner(),
        {},
        object(),
        released_owner_refs={"optimizer": weakref.ref(owner)},
    )
    with pytest.raises(
        CellExecutionError,
        match="released training owners remain live",
    ):
        cache.take_checkpoint(_retained_cache_key())
    assert cache.checkpoint is None


def test_retained_deployment_refuses_a_live_durable_validation_model() -> None:
    cache = _StageExecutionCache()
    key = _retained_cache_key()
    validation_model = _WeakrefableCacheOwner()
    cache.remember_deployment(
        key,
        {"model_cfg": {}},
        _WeakrefableCacheOwner(),
        object(),
        {},
        object(),
        discarded_validation_model_ref=weakref.ref(validation_model),
    )
    with pytest.raises(
        CellExecutionError,
        match="durable validation model remains live",
    ):
        cache.take_deployment(key)
    assert cache.deployment is None


@pytest.mark.parametrize("heavy_field", ("model_state", "codec_payload"))
def test_retained_deployment_refuses_heavy_durable_fields(
    heavy_field: str,
) -> None:
    cache = _StageExecutionCache()
    discarded = _WeakrefableCacheOwner()
    discarded_ref = weakref.ref(discarded)
    del discarded
    with pytest.raises(
        CellExecutionError,
        match="contains heavy durable fields",
    ):
        cache.remember_deployment(
            _retained_cache_key(),
            {"model_cfg": {}, heavy_field: {}},
            _WeakrefableCacheOwner(),
            object(),
            {},
            object(),
            discarded_validation_model_ref=discarded_ref,
        )


def _model_only_handoff_fixture(*, decoder_bias: bool = True) -> BlockCrosscoder:
    return BlockCrosscoder(
        BSCConfig(
            n_blocks=2,
            block_dim=2,
            n_sites=2,
            d_model=3,
            k=1,
            decoder_constraint="free",
            decoder_bias=decoder_bias,
            decoder_init_preconditioning="none",
            decoder_init_operation_order=(
                "gaussian_mask_rescale_then_declared_constraint"
            ),
        )
    )


def _normalized_model_snapshot():
    model = _model_only_handoff_fixture()
    _normalize_model_only_consumer_state(model)
    model.eval()
    snapshot, lineage = _synchronous_model_snapshot(model)
    return model, snapshot, lineage


def test_model_snapshot_lineage_does_not_consume_process_rng() -> None:
    model = _model_only_handoff_fixture()
    _normalize_model_only_consumer_state(model)
    model.eval()
    rng_before = _execution_rng_snapshot()
    snapshot, lineage = _synchronous_model_snapshot(model)
    assert _execution_rng_snapshot() == rng_before
    _assert_model_snapshot_lineage_current(
        model,
        lineage,
        label="test model",
    )
    _assert_serialized_snapshot_current(
        snapshot,
        lineage,
        label="test snapshot",
    )
    _assert_durable_snapshot_schema(
        snapshot,
        lineage,
        label="test durable artifact",
    )
    assert _execution_rng_snapshot() == rng_before


def test_model_snapshot_lineage_can_defer_value_binding_to_parent_payload(
    monkeypatch,
) -> None:
    model = _model_only_handoff_fixture()
    _normalize_model_only_consumer_state(model)
    model.eval()
    monkeypatch.setattr(
        run_cell_module,
        "model_state_digest",
        lambda *_args, **_kwargs: pytest.fail("unexpected model-only digest"),
    )
    snapshot, lineage = _synchronous_model_snapshot(
        model,
        include_model_digest=False,
    )
    assert lineage.snapshot_digest_contract is None
    assert lineage.snapshot_sha256 is None
    _assert_serialized_snapshot_current(snapshot, lineage, label="deployment snapshot")


def test_model_snapshot_lineage_refuses_version_drift() -> None:
    model, _, lineage = _normalized_model_snapshot()
    assert model.D is not None
    with torch.no_grad():
        model.D.add_(1.0)
    with pytest.raises(CellExecutionError, match="tensor identity/storage/version"):
        _assert_model_snapshot_lineage_current(
            model,
            lineage,
            label="test model",
        )


def test_model_snapshot_lineage_refuses_parameter_replacement() -> None:
    model, _, lineage = _normalized_model_snapshot()
    assert model.D is not None
    model.D = torch.nn.Parameter(model.D.detach().clone())
    with pytest.raises(CellExecutionError, match="tensor identity/storage/version"):
        _assert_model_snapshot_lineage_current(
            model,
            lineage,
            label="test model",
        )


def test_model_snapshot_lineage_refuses_config_drift() -> None:
    model, _, lineage = _normalized_model_snapshot()
    model.cfg.k += 1
    with pytest.raises(CellExecutionError, match="model config"):
        _assert_model_snapshot_lineage_current(
            model,
            lineage,
            label="test model",
        )


def test_model_snapshot_lineage_refuses_serialized_mapping_replacement() -> None:
    _, snapshot, lineage = _normalized_model_snapshot()
    with pytest.raises(
        CellExecutionError, match="replaced the serialized state mapping"
    ):
        _assert_serialized_snapshot_current(
            dict(snapshot),
            lineage,
            label="test snapshot",
        )


def test_model_snapshot_lineage_refuses_serialized_tensor_replacement() -> None:
    _, snapshot, lineage = _normalized_model_snapshot()
    name = next(iter(snapshot))
    snapshot[name] = snapshot[name].clone()
    with pytest.raises(CellExecutionError, match="mutated or replaced"):
        _assert_serialized_snapshot_current(
            snapshot,
            lineage,
            label="test snapshot",
        )


def test_model_snapshot_lineage_refuses_state_dict_callbacks() -> None:
    model = _model_only_handoff_fixture()
    model.register_state_dict_pre_hook(lambda *_: None)
    with pytest.raises(CellExecutionError, match="callback-free state_dict"):
        _synchronous_model_snapshot(model)


def test_model_snapshot_lineage_refuses_model_subclasses() -> None:
    class DerivedBlockCrosscoder(BlockCrosscoder):
        pass

    base = _model_only_handoff_fixture()
    model = DerivedBlockCrosscoder(base.cfg)
    with pytest.raises(CellExecutionError, match="canonical BlockCrosscoder type"):
        _synchronous_model_snapshot(model)


def test_model_snapshot_lineage_refuses_state_dict_override() -> None:
    model = _model_only_handoff_fixture()
    model.state_dict = lambda *args, **kwargs: {}  # type: ignore[method-assign]
    with pytest.raises(CellExecutionError, match="callback-free state_dict"):
        _synchronous_model_snapshot(model)


def test_model_snapshot_lineage_refuses_save_to_state_dict_override() -> None:
    model = _model_only_handoff_fixture()
    model._save_to_state_dict = (  # type: ignore[method-assign]
        lambda *args, **kwargs: None
    )
    with pytest.raises(CellExecutionError, match="callback-free state_dict"):
        _synchronous_model_snapshot(model)


def test_retained_checkpoint_recertifies_snapshot_lineage() -> None:
    model, _, lineage = _normalized_model_snapshot()
    cache = _StageExecutionCache()
    cache.remember_checkpoint(
        _retained_cache_key(),
        model,
        {},
        lineage,
        released_owner_refs={},
    )
    assert model.D is not None
    with torch.no_grad():
        model.D.add_(1.0)
    with pytest.raises(CellExecutionError, match="drifted after"):
        cache.take_checkpoint(_retained_cache_key())
    assert cache.checkpoint is None


def test_durable_snapshot_schema_refuses_tensor_contract_drift() -> None:
    _, snapshot, lineage = _normalized_model_snapshot()
    name = next(iter(snapshot))
    forged = dict(snapshot)
    forged[name] = forged[name].reshape(-1)
    with pytest.raises(CellExecutionError, match="schema differs"):
        _assert_durable_snapshot_schema(
            forged,
            lineage,
            label="forged artifact",
        )


def test_snapshot_bound_save_refuses_replaced_torch_save(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, snapshot, lineage = _normalized_model_snapshot()
    monkeypatch.setattr(run_cell_module.torch, "save", lambda *_args, **_kwargs: None)
    with pytest.raises(CellExecutionError, match="native blocking torch.save"):
        run_cell_module._save_immutable_torch(
            tmp_path / "artifact.pt",
            {"model_state": snapshot},
            model_lineage=lineage,
            model_state_field="model_state",
        )


def test_model_only_handoff_clears_every_parameter_gradient() -> None:
    model = _model_only_handoff_fixture()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    _normalize_model_only_consumer_state(model)
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("decoder_bias", (False, True))
def test_model_only_handoff_matches_fresh_requires_grad_schema(
    decoder_bias: bool,
) -> None:
    model = _model_only_handoff_fixture(decoder_bias=decoder_bias)
    fresh = _model_only_handoff_fixture(decoder_bias=decoder_bias)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _normalize_model_only_consumer_state(model)
    assert {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    } == {name: parameter.requires_grad for name, parameter in fresh.named_parameters()}
    assert model.c.requires_grad is decoder_bias


def test_model_only_handoff_invalidates_threshold_validation_cache() -> None:
    model = _model_only_handoff_fixture()
    with torch.no_grad():
        model.theta.fill_(0.25)
    model._require_calibrated_threshold("test")
    assert model._validated_theta_key is not None
    _normalize_model_only_consumer_state(model)
    assert model._validated_theta_key is None


def test_immutable_torch_save_is_byte_exact_across_fresh_processes(
    tmp_path: Path,
) -> None:
    script = """
import sys
from pathlib import Path
import torch
from block_crosscoder_experiment.cli.run_cell import _save_immutable_torch

payload = {
    "schema": "deterministic-save-test-v1",
    "model_state": {
        "weight": torch.arange(24, dtype=torch.float32).reshape(4, 6),
        "index": torch.tensor([7, 2, 9], dtype=torch.int64),
    },
    "meta": {"alpha": 0.5, "names": ["a", "b"]},
}
_save_immutable_torch(Path(sys.argv[1]), payload)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    outputs = (tmp_path / "first.pt", tmp_path / "second.pt")
    for output in outputs:
        subprocess.run(
            [sys.executable, "-c", script, str(output)],
            cwd=tmp_path,
            env=environment,
            check=True,
        )
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    assert (
        hashlib.sha256(outputs[0].read_bytes()).digest()
        == hashlib.sha256(outputs[1].read_bytes()).digest()
    )
    with ZipFile(outputs[0]) as archive:
        assert all(name.startswith("archive/") for name in archive.namelist())


def test_stage_digest_cache_hashes_an_unchanged_output_exactly_once(
    tmp_path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "checkpoint.pt"
    artifact.write_bytes(b"checkpoint-payload")
    manifest = tmp_path / "stage.json"
    calls = 0
    real_sha256 = run_cell_module._sha256

    def counted_sha256(path: Path) -> str:
        nonlocal calls
        calls += 1
        return real_sha256(path)

    monkeypatch.setattr(run_cell_module, "_sha256", counted_sha256)
    cache = run_cell_module._ArtifactDigestCache()
    expected = cache.digest(artifact)
    run_cell_module._emit_stage_manifest(
        manifest,
        cell_id="cell",
        stage="train",
        root=tmp_path,
        artifacts=(("checkpoint", artifact),),
        digest=cache.digest,
    )
    assert calls == 1
    assert json.loads(manifest.read_text())["artifacts"][0]["sha256"] == expected


def test_worker_digest_cache_reuses_only_a_matching_journal_receipt(
    tmp_path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "checkpoint.pt"
    artifact.write_bytes(b"checkpoint-payload")
    calls = 0
    real_sha256 = run_cell_module._sha256

    def counted_sha256(path: Path) -> str:
        nonlocal calls
        calls += 1
        return real_sha256(path)

    monkeypatch.setattr(run_cell_module, "_sha256", counted_sha256)
    cache = run_cell_module._ArtifactDigestCache()
    expected = cache.digest(artifact)
    assert cache.verify(
        artifact,
        sha256=expected,
        size_bytes=artifact.stat().st_size,
    ) == run_cell_module._FileFingerprint.from_path(artifact)
    assert calls == 1

    with pytest.raises(CellExecutionError, match="size mismatch"):
        cache.verify(
            artifact,
            sha256=expected,
            size_bytes=artifact.stat().st_size + 1,
        )
    with pytest.raises(CellExecutionError, match="hash mismatch"):
        cache.verify(
            artifact,
            sha256="0" * 64,
            size_bytes=artifact.stat().st_size,
        )
    assert cache.verify(
        artifact,
        sha256=expected,
        size_bytes=artifact.stat().st_size,
    ) == run_cell_module._FileFingerprint.from_path(artifact)
    assert calls == 1


def test_worker_digest_cache_rehashes_and_refuses_same_size_in_place_mutation(
    tmp_path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "checkpoint.pt"
    artifact.write_bytes(b"checkpoint-payload")
    calls = 0
    real_sha256 = run_cell_module._sha256

    def counted_sha256(path: Path) -> str:
        nonlocal calls
        calls += 1
        return real_sha256(path)

    monkeypatch.setattr(run_cell_module, "_sha256", counted_sha256)
    cache = run_cell_module._ArtifactDigestCache()
    expected = cache.digest(artifact)
    before = run_cell_module._FileFingerprint.from_path(artifact)
    with artifact.open("r+b") as handle:
        handle.write(b"tampered----------")
        handle.flush()
        os.fsync(handle.fileno())
    stat = artifact.stat()
    os.utime(
        artifact,
        ns=(stat.st_atime_ns, before.mtime_ns),
    )
    after = run_cell_module._FileFingerprint.from_path(artifact)
    assert after.device == before.device
    assert after.inode == before.inode
    assert after.size_bytes == before.size_bytes
    assert after.mtime_ns == before.mtime_ns
    assert after.ctime_ns != before.ctime_ns
    assert after != before
    with pytest.raises(CellExecutionError, match="hash mismatch"):
        cache.verify(
            artifact,
            sha256=expected,
            size_bytes=before.size_bytes,
        )
    assert calls == 2


def test_persistent_worker_injects_one_digest_cache_across_stage_requests(
    monkeypatch,
) -> None:
    requests = "\n".join(
        (
            json.dumps(
                {
                    "stage": stage,
                    "artifacts_out": f"/{stage}.json",
                    "resume": False,
                }
            )
            for stage in ("prepare", "train")
        )
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(requests + "\n"))
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    observed = []

    def capture_request(*args, **kwargs) -> None:
        observed.append(kwargs["artifact_digests"])

    monkeypatch.setattr(run_cell_module, "_execute_stage_request", capture_request)
    run_cell_module._worker_main(Path("/unused-cell.json"))
    assert len(observed) == 2
    assert observed[0] is observed[1]


def test_persistent_worker_exits_after_digest_bound_stage_failure(
    monkeypatch,
) -> None:
    requests = "\n".join(
        json.dumps(
            {
                "stage": stage,
                "artifacts_out": f"/{stage}.json",
                "resume": False,
            }
        )
        for stage in ("prepare", "train")
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(requests + "\n"))
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    calls = 0

    def fail_request(*args, **kwargs) -> None:
        nonlocal calls
        calls += 1
        raise CellExecutionError("injected stage failure")

    monkeypatch.setattr(run_cell_module, "_execute_stage_request", fail_request)
    run_cell_module._worker_main(Path("/unused-cell.json"))
    assert calls == 1
    assert json.loads(output.getvalue())["ok"] is False


def test_evaluate_uses_one_common_selector_and_shared_stream() -> None:
    source = inspect.getsource(run_cell_module._evaluate)
    joint_source = inspect.getsource(
        run_cell_module._evaluate_rate_distortion_and_raw_space
    )
    assert source.count("_prefetched_evaluation_batches(") == 0
    assert "evaluate_selector_and_shared_code_modes(" not in source
    assert joint_source.count("evaluate_selector_and_shared_code_modes(") == 1
    assert joint_source.count("_RDEvaluationSession(") == 1
    assert joint_source.count("for rd_input in joint_inputs()") == 1
    assert "_threshold_batch_consumer=consume_threshold_batch" in joint_source
    assert "_evaluate_native_selector(" not in source
    assert source.count("_evaluate_rate_distortion_and_raw_space(") == 1
    assert "evaluate_rd(" not in source
    assert "encode_batch(" not in source


def test_rd_consumer_chunk_preserves_complete_ordered_outer_payload() -> None:
    batch_tokens = RD_EVALUATION_TOKEN_CHUNK + 17
    transformed = torch.arange(batch_tokens * 6, dtype=torch.float32).reshape(
        batch_tokens, 2, 3
    )
    row_ids = torch.arange(batch_tokens * 3).reshape(batch_tokens, 3)
    context = transformed + 0.5
    z = torch.arange(batch_tokens * 8, dtype=torch.float32).reshape(batch_tokens, 4, 2)
    scores = torch.arange(batch_tokens * 4, dtype=torch.float32).reshape(
        batch_tokens, 4
    )
    mask = scores.remainder(3) == 0

    class RecordingSession:
        def __init__(self) -> None:
            self.calls = []

        def consume(self, item, *, threshold_selection) -> None:
            self.calls.append((item, threshold_selection))

    session = RecordingSession()
    _consume_chunked_rd_evaluation_batch(
        session,
        _RDEvaluationInput(transformed, row_ids, context),
        _RDEvaluationSelection(z, scores, mask),
    )

    assert [len(item.transformed) for item, _ in session.calls] == [
        RD_EVALUATION_TOKEN_CHUNK,
        17,
    ]
    assert torch.equal(
        torch.cat([item.transformed for item, _ in session.calls]), transformed
    )
    assert torch.equal(torch.cat([item.row_ids for item, _ in session.calls]), row_ids)
    assert torch.equal(torch.cat([item.context for item, _ in session.calls]), context)
    for field, expected in (("z", z), ("scores", scores), ("mask", mask)):
        assert torch.equal(
            torch.cat([getattr(selection, field) for _, selection in session.calls]),
            expected,
        )


def test_rd_consumer_chunk_is_bit_exact_to_one_full_session_batch() -> None:
    generator = torch.Generator().manual_seed(3901)
    model = BlockCrosscoder(
        BSCConfig(
            n_blocks=4,
            block_dim=2,
            n_sites=2,
            d_model=3,
            k=1,
            selection="batch_topk",
        )
    )
    calibration = torch.randn(64, 2, 3, generator=generator)
    model.fit_threshold_([calibration], target_avg_blocks=1)
    codec = fit_codec(
        model,
        [calibration],
        CodecSpec(qs=(2, 4), floor=1, n_bootstrap=8),
    )
    batch_tokens = RD_EVALUATION_TOKEN_CHUNK + 17
    transformed = torch.randn(batch_tokens, 2, 3, generator=generator)
    row_ids = torch.stack(
        (torch.arange(batch_tokens), torch.arange(batch_tokens)),
        dim=1,
    )
    selected, _, _ = model.select_with_materialized(transformed, mode="threshold")
    selection = _RDEvaluationSelection(
        selected.z,
        selected.scores,
        selected.mask,
    )
    rd_input = _RDEvaluationInput(transformed, row_ids, transformed.clone())

    full_session = _RDEvaluationSession(model, codec)
    full_session.consume(rd_input, threshold_selection=selection)
    expected = full_session.finalize()

    chunked_session = _RDEvaluationSession(model, codec)
    _consume_chunked_rd_evaluation_batch(chunked_session, rd_input, selection)
    actual = chunked_session.finalize()

    assert actual == expected
    assert _tensor_payload_digest(actual) == _tensor_payload_digest(expected)


def test_calibrate_prefetches_and_closes_all_three_cuda_traversals() -> None:
    source = inspect.getsource(run_cell_module._calibrate)
    assert source.count("_prefetched_evaluation_batches(") == 3
    assert "_evaluation_batches(" not in source.replace(
        "_prefetched_evaluation_batches(",
        "",
    )
    assert source.count("with closing(") == 3
    assert "achieved_events_device.add_(" in source
    assert "achieved_events += int(selected.sum())" not in source


def test_raw_observer_packs_grouped_metric_readback() -> None:
    source = inspect.getsource(run_cell_module._evaluate_rate_distortion_and_raw_space)
    assert "grouped_metrics = torch.stack(" in source
    assert "grouped_metrics[:, index + 1].tolist()" in source
    assert "grouped_errors[q].sum(dim=1).cpu().tolist()" not in source


@pytest.mark.parametrize("target", ("cpu", "cuda"))
def test_event_factor_blocks_are_gathered_before_host_transfer(target) -> None:
    if target == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device(target)
    selected_code = torch.arange(4 * 5 * 3, device=device).reshape(4, 5, 3)
    selected_mask = torch.tensor(
        [
            [True, False, True, False, True],
            [False, True, False, True, False],
            [True, True, False, False, True],
            [False, False, True, True, False],
        ],
        device=device,
    )
    event_factor = torch.tensor([2, 0, 1, 2])
    factor_to_group = torch.tensor([4, 1, 3])

    code, mask = _gather_event_factor_blocks(
        selected_code,
        selected_mask,
        event_factor,
        factor_to_group,
    )
    rows = torch.arange(len(event_factor), device=device)
    groups = factor_to_group[event_factor].to(device)
    assert torch.equal(code, selected_code[rows, groups])
    assert torch.equal(mask, selected_mask[rows, groups])
    assert code.shape == (4, 3)
    assert mask.shape == (4,)
    code_only, absent_mask = _gather_event_factor_blocks(
        selected_code,
        None,
        event_factor,
        factor_to_group,
    )
    assert torch.equal(code_only, code)
    assert absent_mask is None


def test_persisted_view_validation_matches_allclose_contract() -> None:
    expected = torch.tensor([1.0, -2.0, 0.0], dtype=torch.bfloat16)
    actual = expected.float() + torch.tensor([0.01, -0.02, 0.011])
    maximum, agrees = _persisted_view_validation(actual, expected)
    assert maximum == pytest.approx(0.02)
    assert agrees is True

    refused = actual.clone()
    refused[2] = 0.013
    maximum, agrees = _persisted_view_validation(refused, expected)
    assert maximum == pytest.approx(0.02)
    assert agrees is False


def test_subspace_overlap_is_bound_to_the_support_selected_group() -> None:
    # Each factor's geometrically best group is deliberately the *other*
    # group.  Recovery must report the overlap at the support assignment,
    # rather than splicing together evidence from unrelated learned blocks.
    overlaps = torch.tensor([[0.1, 0.95], [0.9, 0.2]])
    support_assignment = torch.tensor([0, 1], dtype=torch.long)
    matched = _support_matched_subspace_overlap(overlaps, support_assignment)
    assert matched.tolist() == pytest.approx([0.1, 0.2])
    assert matched.tolist() != pytest.approx(overlaps.max(dim=1).values.tolist())


def _cell(
    *,
    phase: Phase = Phase.PHASE1,
    recipe_index: int = 0,
    seed: int = 0,
) -> CellSpec:
    if phase is Phase.PHASE1:
        recipe_name = BSC_FACTOR_CONTESTS[recipe_index].name
        base = next(
            cell
            for cell in build_phase1_plan(seeds=(seed,), smoke=True).cells
            if cell.recipe_name == recipe_name
        )
    else:
        base = build_phase2_plan(seeds=(seed,), smoke=True).cells[0]
    return replace(
        base,
        name=f"{phase.value}.test.executor{recipe_index}.s{seed}",
        stage="test",
    )


def _stiefel_decoded_energy_cell(*, seed: int = 0) -> CellSpec:
    base = _cell(recipe_index=0, seed=seed)
    overrides = {
        "model.selection_score": "decoded_energy",
        "implementation.decoded_energy_implementation": (
            DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION
        ),
        "optimizer.retract_every_steps": 1,
    }
    return replace(
        base,
        name=f"phase1.test.stiefel_decoded_energy.s{seed}",
        decisions=tuple(
            replace(decision, value=overrides[decision.name])
            if decision.name in overrides
            else decision
            for decision in base.decisions
        ),
    )


def _mapped_isolated_loss_cell(*, seed: int = 0) -> CellSpec:
    base = _cell(recipe_index=0, seed=seed)
    overrides = {
        "model.selection_score": "isolated_loss_decrease",
        "model.decoder": "free_scale_controlled",
        "model.decoder_bias": False,
        "objective.reconstruction": "squared_l2",
        "implementation.decoder_retraction_implementation": (
            DECODER_RETRACTION_NOT_APPLICABLE
        ),
        "implementation.isolated_loss_decrease_implementation": (
            ISOLATED_LOSS_MAPPED_IMPLEMENTATION
        ),
    }
    return replace(
        base,
        name=f"phase1.test.mapped_isolated_loss.s{seed}",
        decisions=tuple(
            replace(decision, value=overrides[decision.name])
            if decision.name in overrides
            else decision
            for decision in base.decisions
        ),
    )


def _campaign(tmp_path: Path, cell: CellSpec) -> Campaign:
    source_stage = build_phase1_plan(
        seeds=(cell.seed,), smoke=bool(cell.decision_map["runtime.smoke"])
    ).stages[-1]
    assert source_stage.selection_policy is not None
    selection_policy = replace(
        source_stage.selection_policy,
        eligible_recipe_names=(cell.recipe_name,),
    )
    plan = StudyPlan(
        f"test_executor_{cell.phase.value}_{cell.seed}_{cell.recipe_name}",
        cell.phase,
        (StageSpec("test", (cell,), selection_policy=selection_policy),),
    )
    blueprint = Phase1Blueprint(
        name="run_cell_test_blueprint",
        seeds=(cell.seed,),
        initial_stages=plan.stages,
        rounds=(),
    )
    campaign = Campaign(tmp_path / "campaign")
    # Executor integration tests intentionally use one tiny smoke cell. Patch
    # registration's canonical builders only for this non-scientific fixture;
    # _Context still authenticates the persisted plan, blueprint, history,
    # journal, and exact cell path in the subprocess.
    with (
        patch(
            "block_crosscoder_experiment.campaign.build_phase1_blueprint",
            return_value=blueprint,
        ),
        patch(
            "block_crosscoder_experiment.campaign.build_phase1_plan",
            return_value=plan,
        ),
    ):
        campaign.register(plan, blueprint_manifest=blueprint.to_manifest())
    return campaign


def _runner(campaign: Campaign, **extra_env: str) -> CampaignRunner:
    pythonpath = str(REPO)
    if os.environ.get("PYTHONPATH"):
        pythonpath += os.pathsep + os.environ["PYTHONPATH"]
    return CampaignRunner(
        campaign,
        python=sys.executable,
        env={"PYTHONPATH": pythonpath, **extra_env},
    )


@pytest.mark.parametrize("cell_index", (0, 7))
def test_executor_rejects_unbound_phase2_preview_cell(
    tmp_path, monkeypatch, cell_index
):
    blueprint = build_phase2_blueprint(seeds=(0,), smoke=True)
    plan = build_phase2_plan(seeds=(0,), smoke=True)
    cell = plan.cells[cell_index]
    campaign = Campaign(tmp_path)
    campaign.plan_path.write_text(
        json.dumps(plan.to_manifest(), indent=2, sort_keys=True) + "\n"
    )
    campaign.blueprint_path.write_text(
        json.dumps(blueprint.to_manifest(), indent=2, sort_keys=True) + "\n"
    )
    campaign.plans_dir.mkdir(parents=True)
    (campaign.plans_dir / "preview.json").write_text(
        json.dumps(plan.to_manifest(), indent=2, sort_keys=True) + "\n"
    )
    cell_path = campaign.cell_manifest_path(cell.cell_id)
    cell_dir = cell_path.parent
    cell_dir.mkdir(parents=True)
    cell_path.write_text(
        json.dumps(cell.to_manifest(), indent=2, sort_keys=True) + "\n"
    )
    monkeypatch.setenv("BSC_CAMPAIGN_ROOT", str(tmp_path))

    with pytest.raises(
        CellExecutionError,
        match="planning estimates only|authenticated Phase-1 decision",
    ):
        _Context(cell_path, cell_dir / "artifacts.json", "prepare")


def _phase3_cell() -> CellSpec:
    blueprint = build_phase3_blueprint(seeds=(0,), smoke=True)
    phase2_blueprint = build_phase2_blueprint(seeds=(0,), smoke=True)
    phase2 = build_phase2_plan(seeds=(0,), smoke=True)
    while phase2.stages[-1].selection_policy is not None:
        stage = phase2.stages[-1]
        groups: dict[str, list[CellSpec]] = {}
        for cell in stage.cells:
            groups.setdefault(cell.candidate_id, []).append(cell)
        eligible = [
            group
            for group in groups.values()
            if not stage.selection_policy.eligible_recipe_names
            or group[0].recipe_name in stage.selection_policy.eligible_recipe_names
        ]
        eligible = [
            group
            for group in eligible
            if all(
                cell.decision_map["qualification.promotable"] is True for cell in group
            )
        ]
        selected = tuple(
            sorted(
                sorted(eligible, key=lambda group: group[0].candidate_id)[0],
                key=lambda item: item.seed,
            )
        )
        frozen = FrozenSelection.from_cells(
            stage.selection_policy,
            selected,
            tuple(0.5 for _ in selected),
            tuple(
                "sha256:" + hashlib.sha256(cell.cell_id.encode()).hexdigest()
                for cell in selected
            ),
            "sha256:" + hashlib.sha256(stage.name.encode()).hexdigest(),
        )
        phase2 = materialize_child_plan(phase2, phase2_blueprint, frozen)
    anchors: dict[str, tuple[CellSpec, ...]] = {}
    for cell in phase2.stages[0].cells:
        anchors.setdefault(cell.recipe_name, tuple())
        anchors[cell.recipe_name] += (cell,)
    finalists: dict[str, tuple[CellSpec, ...]] = {}
    for cell in phase2.stages[-1].cells:
        finalists.setdefault(cell.candidate_id, tuple())
        finalists[cell.candidate_id] += (cell,)
    finalist = next(
        cells
        for cells in finalists.values()
        if cells[0].decision_map["data.normalization"] == "scalar_rms"
    )
    entries = []
    for slot in blueprint.panel_slots:
        if slot.role == "selected_finalist":
            entries.append(
                FrozenPanelEntry.from_cells(
                    panel_slot=slot.name,
                    role=slot.role,
                    source_cells=finalist,
                    selection_ids=finalist[0].decision_map[
                        "selection.upstream_selection_ids"
                    ],
                    qualification_sha256s=(
                        "sha256:" + hashlib.sha256(b"final-qualification").hexdigest(),
                    ),
                    confirmation_sha256s=(
                        "sha256:" + hashlib.sha256(b"final-confirmation").hexdigest(),
                    ),
                )
            )
        else:
            root_cells = anchors[str(slot.comparator_recipe_name)]
            derived_recipe_id = (
                "derived-recipe:"
                + hashlib.sha256(f"family:{slot.name}".encode()).hexdigest()
            )
            source_cells = tuple(
                replace(
                    cell,
                    stage=f"family_{slot.name}_top2_revisit_16m",
                    recipe_name=f"derived_family_{slot.name}_revisit_winner",
                    recipe_id=derived_recipe_id,
                    decisions=merge_decisions(
                        cell.decisions,
                        (
                            engineering(
                                "selection.comparator_family_name",
                                slot.comparator_family_name,
                                rationale="test fixture binds the calibrated family",
                            ),
                            engineering(
                                "selection.comparator_family_blueprint_id",
                                slot.comparator_family_id,
                                rationale="test fixture binds the family blueprint",
                            ),
                            engineering(
                                "selection.family_root_recipe_id",
                                slot.comparator_recipe_id,
                                rationale="test fixture binds the family root lineage",
                            ),
                        ),
                    ),
                )
                for cell in root_cells
            )
            entries.append(
                FrozenPanelEntry.from_cells(
                    panel_slot=slot.name,
                    role=slot.role,
                    source_cells=source_cells,
                    selection_ids=(
                        "selection:"
                        + hashlib.sha256(
                            f"family-selection:{slot.name}".encode()
                        ).hexdigest(),
                    ),
                    qualification_sha256s=(
                        "sha256:" + hashlib.sha256(slot.name.encode()).hexdigest(),
                    ),
                )
            )
    panel = FrozenPanelDecision(
        source_phase2_plan_id=phase2.plan_id,
        source_phase2_blueprint_id=phase2_blueprint.blueprint_id,
        phase2_campaign_manifest_sha256=(
            "sha256:" + hashlib.sha256(b"phase2-campaign").hexdigest()
        ),
        selection_universe_sha256=(
            "sha256:" + hashlib.sha256(b"phase2-universe").hexdigest()
        ),
        entries=tuple(entries),
    )
    return build_phase3_plan(seeds=(0,), smoke=True, panel_decision=panel).cells[0]


@pytest.mark.parametrize(
    ("label", "cell_factory"),
    (
        ("Phase-1", lambda: build_phase1_plan(seeds=(0,), smoke=True).cells[0]),
        ("Phase-3", _phase3_cell),
    ),
)
def test_executor_rejects_orphan_phase1_and_phase3_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    cell_factory,
) -> None:
    cell = cell_factory()
    cell_dir = tmp_path / "cells" / label.lower().replace("-", "")
    cell_dir.mkdir(parents=True)
    cell_path = cell_dir / "cell.json"
    cell_path.write_text(
        json.dumps(cell.to_manifest(), indent=2, sort_keys=True) + "\n"
    )
    monkeypatch.setenv("BSC_CAMPAIGN_ROOT", str(tmp_path))

    with pytest.raises(
        CellExecutionError,
        match="exact active campaign binding",
    ):
        _Context(cell_path, cell_dir / "artifacts.json", "prepare")


def test_tiny_phase1_cell_runs_all_five_stages_and_binds_inputs(tmp_path: Path) -> None:
    cell = _cell()
    campaign = _campaign(tmp_path, cell)
    summary = _runner(campaign).run(limit=1)
    assert summary.to_dict() == {
        "selected_cells": 1,
        "completed_cells": 1,
        "failed_cells": 0,
        "skipped_cells": 0,
    }
    record = campaign.record(cell.cell_id)
    assert record.state is RunState.QUALIFIED
    refs = record.artifact_map
    for kind in (
        "preparation",
        "checkpoint",
        "training_report",
        "calibration",
        "deployment_codec",
        "deployment_schedules",
        "calibration_record",
        "evaluation",
        "qualification",
    ):
        refs[kind].verify(campaign.root)

    qualification = json.loads(refs["qualification"].resolve(campaign.root).read_text())
    assert qualification["schema"] == QUALIFICATION_SCHEMA
    assert qualification["qualified"] is True
    assert all(qualification["checks"].values())
    assert set(qualification["checks"]) == {
        "deployment_schedule_integrity",
        "encoder_scale_calibration_integrity",
        "finite",
        "method_endpoints",
        "precision_preflight_integrity",
        "provenance",
        "regularizer_calibration_integrity",
        "resource_compliance",
        "selection_score_diagnostics_integrity",
        "scientific_endpoint_complete",
        "split_integrity",
    }
    assert isinstance(qualification["scientific_outcome"]["passed"], bool)
    assert qualification["promotion_eligible"] is False
    assert "runtime_smoke" in qualification["promotion_ineligible_reasons"]
    assert qualification["selection_eligible_for_protocol_test"] is True
    assert qualification["selection_eligibility_mode"] == "smoke_protocol_only"
    assert qualification["inputs"] == {
        kind: refs[kind].sha256
        for kind in (
            "preparation",
            "checkpoint",
            "calibration",
            "deployment_codec",
            "deployment_schedules",
            "evaluation",
        )
    }
    preparation = json.loads(refs["preparation"].resolve(campaign.root).read_text())
    assert qualification["implementation_identity"] == preparation["implementation"]
    assert (
        qualification["implementation_identity_sha256"]
        == preparation["implementation_sha256"]
    )
    assert set(qualification["implementation_identity"]["dependencies"]) == {
        "datasets",
        "block-crosscoder-experiment",
        "huggingface-hub",
        "numpy",
        "sae-lens",
        "safetensors",
        "torch",
        "transformer-lens",
        "transformers",
        "triton",
    }
    evaluation = json.loads(refs["evaluation"].resolve(campaign.root).read_text())
    assert evaluation["synthetic_recovery"]["native"]["n_truth_factors"] > 0
    assert evaluation["synthetic_recovery"]["deployed"]["n_truth_factors"] > 0
    for endpoint in ("native", "deployed"):
        assert (
            evaluation["synthetic_recovery"][endpoint]["n_factor_calibration_examples"]
            == cell.decision_map["data.synthetic_factor_calibration_examples"]
        )
        assert (
            evaluation["synthetic_recovery"][endpoint]["n_examples"]
            == (cell.decision_map["data.synthetic_development_examples"])
        )
    assert evaluation["native_selector"]["mode"] == "topk"
    assert evaluation["deployed_selector"]["mode"] == "threshold"
    assert evaluation["codec_roundtrip"]["source_free_decode"] is True
    assert evaluation["rate_distortion"]["points"]["8"]["rate_bits_per_token"] >= 0


def test_preparation_contract_rejects_cell_divergent_synthetic_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = _cell(seed=83)
    campaign = _campaign(tmp_path, cell)
    assert _runner(campaign).run(stop_after="prepare").completed_cells == 1
    campaign.transition(cell.cell_id, RunState.RUNNING)
    monkeypatch.setenv("BSC_CAMPAIGN_ROOT", str(campaign.root))
    ctx = _Context(
        campaign.cell_manifest_path(cell.cell_id),
        campaign.cell_dir(cell.cell_id) / "contract-test-artifacts.json",
        "train",
    )
    preparation = json.loads(ctx.preparation.read_text())
    cell_manifest_sha256 = hashlib.sha256(ctx.cell_path.read_bytes()).hexdigest()
    _validate_preparation_contract(
        ctx.cell,
        preparation,
        cell_manifest_sha256=cell_manifest_sha256,
    )

    mutations = []
    wrong_range = copy.deepcopy(preparation)
    wrong_range["data"]["ranges"]["evaluation"][0] += 1
    mutations.append(wrong_range)
    wrong_normalization = copy.deepcopy(preparation)
    wrong_normalization["data"]["normalization"]["scale"][0] *= 2.0
    mutations.append(wrong_normalization)
    wrong_stream = copy.deepcopy(preparation)
    wrong_stream["data"]["evaluation_stream"] = "confirmation"
    mutations.append(wrong_stream)
    wrong_seed = copy.deepcopy(preparation)
    wrong_seed["random"]["eval_data_seed"] += 1
    mutations.append(wrong_seed)
    for forged in mutations:
        with pytest.raises(CellExecutionError, match="differs|stale"):
            _validate_preparation_contract(
                ctx.cell,
                forged,
                cell_manifest_sha256=cell_manifest_sha256,
            )


def test_executor_ignores_forged_state_snapshot_and_replays_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = _cell(seed=89)
    campaign = _campaign(tmp_path, cell)
    assert _runner(campaign).run(stop_after="prepare").completed_cells == 1
    campaign.transition(cell.cell_id, RunState.RUNNING)
    snapshot = json.loads(campaign.state_path(cell.cell_id).read_text())
    snapshot["state"] = "trained"
    snapshot["artifacts"] = [
        {
            "kind": "checkpoint",
            "path": "/tmp/forged-checkpoint.pt",
            "sha256": "f" * 64,
            "size_bytes": 1,
        }
    ]
    campaign.state_path(cell.cell_id).write_text(json.dumps(snapshot) + "\n")
    monkeypatch.setenv("BSC_CAMPAIGN_ROOT", str(campaign.root))
    ctx = _Context(
        campaign.cell_manifest_path(cell.cell_id),
        campaign.cell_dir(cell.cell_id) / "journal-state-artifacts.json",
        "train",
    )
    state, artifacts = ctx.state()
    assert state == "running"
    assert set(artifacts) == {"preparation", "prepare_manifest"}


def test_persistent_worker_is_byte_exact_with_one_shot_stage_processes(
    tmp_path: Path,
) -> None:
    class OneShotCampaignRunner(CampaignRunner):
        @property
        def _supports_persistent_worker(self) -> bool:
            return False

    cell = _cell(seed=43)
    persistent = _campaign(tmp_path / "persistent", cell)
    one_shot = _campaign(tmp_path / "one-shot", cell)
    persistent_runner = _runner(persistent)
    one_shot_runner = OneShotCampaignRunner(
        one_shot,
        python=persistent_runner.python,
        env=persistent_runner.env,
    )

    assert persistent_runner.run(limit=1).failed_cells == 0
    assert one_shot_runner.run(limit=1).failed_cells == 0
    persistent_refs = persistent.record(cell.cell_id).artifact_map
    one_shot_refs = one_shot.record(cell.cell_id).artifact_map
    assert persistent_refs.keys() == one_shot_refs.keys()
    assert {
        kind: (ref.sha256, ref.size_bytes) for kind, ref in persistent_refs.items()
    } == {kind: (ref.sha256, ref.size_bytes) for kind, ref in one_shot_refs.items()}
    preparation = json.loads(
        persistent_refs["preparation"].resolve(persistent.root).read_text()
    )
    durable_deployment = torch.load(
        persistent_refs["deployment_codec"].resolve(persistent.root),
        map_location="cpu",
        weights_only=True,
    )
    assert "model_state" in durable_deployment
    assert "codec_payload" in durable_deployment
    assert preparation["implementation"]["executor_schema"] == ("bsc-cell-executor-v13")
    assert preparation["implementation"]["executor_process_model"] == (
        "persistent_exact_snapshot_lineage_v5"
    )


def test_persistent_worker_restarts_are_byte_exact_from_every_stage(
    tmp_path: Path,
) -> None:
    cell = _cell(seed=47)
    exact_kinds = (
        "checkpoint",
        "calibration",
        "deployment_codec",
        "deployment_schedules",
        "evaluation",
        "qualification",
    )

    def completed_fingerprints(campaign: Campaign) -> dict[str, tuple[str, int]]:
        refs = campaign.record(cell.cell_id).artifact_map
        assert campaign.record(cell.cell_id).state is RunState.QUALIFIED
        return {
            kind: (refs[kind].sha256, refs[kind].size_bytes) for kind in exact_kinds
        }

    uninterrupted = _campaign(tmp_path / "uninterrupted", cell)
    assert _runner(uninterrupted).run(limit=1).failed_cells == 0
    expected = completed_fingerprints(uninterrupted)

    repeated = _campaign(tmp_path / "repeated", cell)
    assert _runner(repeated).run(limit=1).failed_cells == 0
    assert completed_fingerprints(repeated) == expected

    for stage in ("prepare", "train", "calibrate", "evaluate"):
        campaign = _campaign(tmp_path / f"restart-{stage}", cell)
        first = _runner(campaign).run(limit=1, stop_after=stage)
        assert first.completed_cells == 1
        restarted = Campaign(campaign.root)
        resumed = _runner(restarted).run(limit=1, resume=True)
        assert resumed.failed_cells == 0
        assert completed_fingerprints(restarted) == expected


def test_factorized_masked_decoded_energy_cell_runs_through_saved_codec(
    tmp_path: Path,
) -> None:
    base = _cell(recipe_index=2, seed=41)
    overrides = {
        "model.decoder": "free_scale_controlled",
        "model.site_rank": 2,
        "model.selection_score": "decoded_energy",
        "objective.encoder_site_mask_probability": 0.10,
        "implementation.decoder_retraction_implementation": "not_applicable_v1",
        "implementation.factorized_execution_implementation": (
            FACTORIZED_EXECUTION_DIRECT_RANK_SPACE_IMPLEMENTATION
        ),
    }
    cell = replace(
        base,
        name="phase1.test.factorized_masked_decoded_energy.s41",
        decisions=tuple(
            replace(decision, value=overrides[decision.name])
            if decision.name in overrides
            else decision
            for decision in base.decisions
        ),
    )
    model_cfg, train_cfg = validate_cell_config(cell)
    assert model_cfg.site_rank == 2
    assert model_cfg.factorized_execution_implementation == (
        FACTORIZED_EXECUTION_DIRECT_RANK_SPACE_IMPLEMENTATION
    )
    assert model_cfg.selection_score == "decoded_energy"
    assert train_cfg.encoder_site_mask_probability == 0.10

    def with_factorized_implementation(value: str) -> CellSpec:
        return replace(
            cell,
            decisions=tuple(
                replace(decision, value=value)
                if decision.name == "implementation.factorized_execution_implementation"
                else decision
                for decision in cell.decisions
            ),
        )

    assert _model_config(
        with_factorized_implementation(
            FACTORIZED_EXECUTION_MATERIALIZED_REFERENCE_IMPLEMENTATION
        )
    ).factorized_execution_implementation == (
        FACTORIZED_EXECUTION_MATERIALIZED_REFERENCE_IMPLEMENTATION
    )
    for value, message in (
        (FACTORIZED_EXECUTION_NOT_APPLICABLE, "violates its carrier predicate"),
        ("ambient_cuda_default", "unknown factorized-execution"),
        ("direct_rank_space_bmm_bounded_v1", "unknown factorized-execution"),
        ("materialized_site_tensor_reference_v1", "unknown factorized-execution"),
        ("direct_rank_space_prepacked_core_bmm_v2", "unknown factorized-execution"),
    ):
        with pytest.raises(CellExecutionError, match=message):
            _model_config(with_factorized_implementation(value))

    campaign = _campaign(tmp_path, cell)
    summary = _runner(campaign).run(limit=1)
    assert summary.failed_cells == 0
    record = campaign.record(cell.cell_id)
    assert record.state is RunState.QUALIFIED
    evaluation = json.loads(
        record.artifact_map["evaluation"].resolve(campaign.root).read_text()
    )
    for endpoint in ("native", "deployed"):
        dependence = evaluation["shared_code"][endpoint]["functional_dependence"]
        assert {"pre_selection", "post_selection"}.issubset(dependence)
        assert dependence["pre_selection"]["delta_by_site_block"]


def test_deployable_codec_is_the_complete_validated_consumer_artifact(
    tmp_path: Path,
) -> None:
    cell = _stiefel_decoded_energy_cell(seed=37)
    campaign = _campaign(tmp_path, cell)
    assert _runner(campaign).run(limit=1).failed_cells == 0
    refs = campaign.record(cell.cell_id).artifact_map
    deployment_path = refs["deployment_codec"].resolve(campaign.root)
    payload = torch.load(deployment_path, map_location="cpu", weights_only=True)
    preparation_hash = refs["preparation"].sha256

    loaded, model, codec, summary, verified_deployment_digest = _load_deployable_codec(
        deployment_path,
        cell_id=cell.cell_id,
        checkpoint_hash=refs["checkpoint"].sha256,
        calibration_hash=refs["calibration"].sha256,
        preparation_hash=preparation_hash,
        device=torch.device("cpu"),
    )
    assert verified_deployment_digest == payload["artifact_sha256"]
    assert loaded["schema"] == "bsc-deployable-codec-v2"
    assert model.cfg.n_blocks == codec.included.numel()
    assert (
        model.cfg.decoded_energy_implementation
        == DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION
    )
    assert (
        model.cfg.decoder_retraction_implementation
        == DECODER_RETRACTION_CHOLESKY_QR_IMPLEMENTATION
    )
    assert summary["accepted_tokens"] > 0

    self_consistent_substitution = copy.deepcopy(payload)
    substitution_name = "E"
    assert substitution_name in self_consistent_substitution["model_state"]
    self_consistent_substitution["model_state"][substitution_name].view(-1)[0].add_(
        1e-3
    )
    unsigned_substitution = {
        key: value
        for key, value in self_consistent_substitution.items()
        if key != "artifact_sha256"
    }
    self_consistent_substitution["artifact_sha256"] = _tensor_payload_digest(
        unsigned_substitution
    )
    substitution_path = tmp_path / "self-consistent-model-substitution.pt"
    torch.save(self_consistent_substitution, substitution_path)
    (
        _substituted_payload,
        _substituted_model,
        _substituted_codec,
        _substituted_summary,
        verified_substitution_digest,
    ) = _load_deployable_codec(
        substitution_path,
        cell_id=cell.cell_id,
        checkpoint_hash=refs["checkpoint"].sha256,
        calibration_hash=refs["calibration"].sha256,
        preparation_hash=preparation_hash,
        device=torch.device("cpu"),
    )
    with pytest.raises(CellExecutionError, match="exact pre-save snapshot"):
        _assert_deployment_snapshot_digest(
            expected=verified_deployment_digest,
            verified=verified_substitution_digest,
        )

    checkpoint_payload = torch.load(
        refs["checkpoint"].resolve(campaign.root),
        map_location="cpu",
        weights_only=True,
    )
    assert (
        checkpoint_payload["model_state_digest_contract"] == MODEL_STATE_DIGEST_CONTRACT
    )
    assert checkpoint_payload["model_state_sha256"] == model_state_digest(
        checkpoint_payload["model"]
    )
    wrong_digest_contract = copy.deepcopy(checkpoint_payload)
    wrong_digest_contract["model_state_digest_contract"] = "sha256_merkle_16m_v0"
    wrong_digest_contract_path = tmp_path / "wrong-model-digest-contract.pt"
    torch.save(wrong_digest_contract, wrong_digest_contract_path)
    with pytest.raises(CellExecutionError, match="model-state digest mismatch"):
        _validate_final_checkpoint(
            wrong_digest_contract_path,
            checkpoint_payload["run_binding"],
        )
    bitflipped_checkpoint = copy.deepcopy(checkpoint_payload)
    bitflipped_name = next(
        name
        for name, tensor in bitflipped_checkpoint["model"].items()
        if tensor.is_floating_point() and tensor.numel()
    )
    bitflipped_checkpoint["model"][bitflipped_name].view(-1)[0].add_(1.0)
    bitflipped_path = tmp_path / "same-schema-bitflipped-checkpoint.pt"
    torch.save(bitflipped_checkpoint, bitflipped_path)
    with pytest.raises(CellExecutionError, match="model-state digest mismatch"):
        _validate_final_checkpoint(
            bitflipped_path,
            checkpoint_payload["run_binding"],
        )
    assert (
        checkpoint_payload["model_cfg"]["decoder_retraction_implementation"]
        == DECODER_RETRACTION_CHOLESKY_QR_IMPLEMENTATION
    )
    assert (
        checkpoint_payload["run_binding"]["model_cfg"][
            "decoder_retraction_implementation"
        ]
        == DECODER_RETRACTION_CHOLESKY_QR_IMPLEMENTATION
    )
    assert (
        checkpoint_payload["model_cfg"]["factorized_execution_implementation"]
        == FACTORIZED_EXECUTION_NOT_APPLICABLE
    )
    assert (
        checkpoint_payload["model_cfg"]["code_norm_implementation"]
        == CODE_NORM_CUDA_IMPLEMENTATION
    )
    assert (
        checkpoint_payload["model_cfg"]["sparse_decode_implementation"]
        == SPARSE_DECODE_CUDA_IMPLEMENTATION
    )
    assert (
        checkpoint_payload["model_cfg"]["map_nuclear_implementation"]
        == MAP_NUCLEAR_GUARDED_MATMUL_IMPLEMENTATION
    )
    mismatched_checkpoint = copy.deepcopy(checkpoint_payload)
    mismatched_checkpoint["model_cfg"]["decoded_energy_implementation"] = (
        DECODED_ENERGY_EXACT_IMPLEMENTATION
    )
    checkpoint_path = tmp_path / "mismatched-checkpoint-config.pt"
    torch.save(mismatched_checkpoint, checkpoint_path)
    with pytest.raises(CellExecutionError, match="top-level configuration"):
        _validate_final_checkpoint(
            checkpoint_path,
            checkpoint_payload["run_binding"],
        )

    missing_retraction_identity = copy.deepcopy(checkpoint_payload)
    missing_retraction_identity["model_cfg"].pop("decoder_retraction_implementation")
    missing_retraction_identity["run_binding"]["model_cfg"].pop(
        "decoder_retraction_implementation"
    )
    missing_retraction_path = tmp_path / "missing-retraction-identity.pt"
    torch.save(missing_retraction_identity, missing_retraction_path)
    with pytest.raises(
        CellExecutionError,
        match="lacks decoder_retraction_implementation",
    ):
        _validate_final_checkpoint(
            missing_retraction_path,
            missing_retraction_identity["run_binding"],
        )

    missing_factorized_identity = copy.deepcopy(checkpoint_payload)
    missing_factorized_identity["model_cfg"].pop("factorized_execution_implementation")
    missing_factorized_identity["run_binding"]["model_cfg"].pop(
        "factorized_execution_implementation"
    )
    missing_factorized_path = tmp_path / "missing-factorized-identity.pt"
    torch.save(missing_factorized_identity, missing_factorized_path)
    with pytest.raises(
        CellExecutionError,
        match="lacks factorized_execution_implementation",
    ):
        _validate_final_checkpoint(
            missing_factorized_path,
            missing_factorized_identity["run_binding"],
        )

    missing_sparse_identity = copy.deepcopy(checkpoint_payload)
    missing_sparse_identity["model_cfg"].pop("sparse_decode_implementation")
    missing_sparse_identity["run_binding"]["model_cfg"].pop(
        "sparse_decode_implementation"
    )
    missing_sparse_path = tmp_path / "missing-sparse-identity.pt"
    torch.save(missing_sparse_identity, missing_sparse_path)
    with pytest.raises(
        CellExecutionError,
        match="lacks sparse_decode_implementation",
    ):
        _validate_final_checkpoint(
            missing_sparse_path,
            missing_sparse_identity["run_binding"],
        )

    missing_code_norm_identity = copy.deepcopy(checkpoint_payload)
    missing_code_norm_identity["model_cfg"].pop("code_norm_implementation")
    missing_code_norm_identity["run_binding"]["model_cfg"].pop(
        "code_norm_implementation"
    )
    missing_code_norm_path = tmp_path / "missing-code-norm-identity.pt"
    torch.save(missing_code_norm_identity, missing_code_norm_path)
    with pytest.raises(
        CellExecutionError,
        match="lacks code_norm_implementation",
    ):
        _validate_final_checkpoint(
            missing_code_norm_path,
            missing_code_norm_identity["run_binding"],
        )

    missing_map_nuclear_identity = copy.deepcopy(checkpoint_payload)
    missing_map_nuclear_identity["model_cfg"].pop("map_nuclear_implementation")
    missing_map_nuclear_identity["run_binding"]["model_cfg"].pop(
        "map_nuclear_implementation"
    )
    missing_map_nuclear_path = tmp_path / "missing-map-nuclear-identity.pt"
    torch.save(missing_map_nuclear_identity, missing_map_nuclear_path)
    with pytest.raises(
        CellExecutionError,
        match="lacks map_nuclear_implementation",
    ):
        _validate_final_checkpoint(
            missing_map_nuclear_path,
            missing_map_nuclear_identity["run_binding"],
        )

    forged_optimizer = copy.deepcopy(checkpoint_payload)
    forged_optimizer["optimizer"]["param_groups"][0]["fused"] = True
    forged_optimizer_path = tmp_path / "forged-optimizer-kernel.pt"
    torch.save(forged_optimizer, forged_optimizer_path)
    with pytest.raises(CellExecutionError, match="optimizer contract"):
        _validate_final_checkpoint(
            forged_optimizer_path,
            checkpoint_payload["run_binding"],
        )

    nested_identity_mismatch = copy.deepcopy(payload)
    nested_codec = Codec.from_payload(
        nested_identity_mismatch["codec_payload"],
        source="test nested codec",
    )
    nested_codec.meta["model_cfg"]["decoded_energy_implementation"] = (
        DECODED_ENERGY_EXACT_IMPLEMENTATION
    )
    nested_identity_mismatch["codec_payload"] = nested_codec.to_payload()
    unsigned = {
        key: value
        for key, value in nested_identity_mismatch.items()
        if key != "artifact_sha256"
    }
    nested_identity_mismatch["artifact_sha256"] = _tensor_payload_digest(unsigned)
    nested_identity_path = tmp_path / "nested-model-identity-mismatch.pt"
    torch.save(nested_identity_mismatch, nested_identity_path)
    with pytest.raises(CellExecutionError, match="model config|model-config"):
        _load_deployable_codec(
            nested_identity_path,
            cell_id=cell.cell_id,
            checkpoint_hash=refs["checkpoint"].sha256,
            calibration_hash=refs["calibration"].sha256,
            preparation_hash=preparation_hash,
            device=torch.device("cpu"),
        )

    finite_off_manifold = copy.deepcopy(payload)
    finite_off_manifold["model_state"]["D"].mul_(1.01)
    assert torch.isfinite(finite_off_manifold["model_state"]["D"]).all()
    unsigned = {
        key: value
        for key, value in finite_off_manifold.items()
        if key != "artifact_sha256"
    }
    finite_off_manifold["artifact_sha256"] = _tensor_payload_digest(unsigned)
    finite_gram_path = tmp_path / "finite-off-manifold-model.pt"
    torch.save(finite_off_manifold, finite_gram_path)
    with pytest.raises(CellExecutionError, match="decoded-energy invariant"):
        _load_deployable_codec(
            finite_gram_path,
            cell_id=cell.cell_id,
            checkpoint_hash=refs["checkpoint"].sha256,
            calibration_hash=refs["calibration"].sha256,
            preparation_hash=preparation_hash,
            device=torch.device("cpu"),
        )

    corrupt_tensor = copy.deepcopy(payload)
    state_name = next(iter(corrupt_tensor["model_state"]))
    corrupt_tensor["model_state"][state_name].view(-1)[0] += 1
    tensor_path = tmp_path / "corrupt-model-tensor.pt"
    torch.save(corrupt_tensor, tensor_path)
    with pytest.raises(CellExecutionError, match="internal content hash"):
        _load_deployable_codec(
            tensor_path,
            cell_id=cell.cell_id,
            checkpoint_hash=refs["checkpoint"].sha256,
            calibration_hash=refs["calibration"].sha256,
            preparation_hash=preparation_hash,
            device=torch.device("cpu"),
        )

    missing_state = copy.deepcopy(payload)
    missing_state["model_state"].pop(next(iter(missing_state["model_state"])))
    unsigned = {
        key: value for key, value in missing_state.items() if key != "artifact_sha256"
    }
    missing_state["artifact_sha256"] = _tensor_payload_digest(unsigned)
    state_path = tmp_path / "missing-model-state.pt"
    torch.save(missing_state, state_path)
    with pytest.raises(CellExecutionError, match="reconstruct consumer"):
        _load_deployable_codec(
            state_path,
            cell_id=cell.cell_id,
            checkpoint_hash=refs["checkpoint"].sha256,
            calibration_hash=refs["calibration"].sha256,
            preparation_hash=preparation_hash,
            device=torch.device("cpu"),
        )

    corrupt_codec = copy.deepcopy(payload)
    corrupt_codec["codec_payload"]["lo"].view(-1)[0] += 1
    unsigned = {
        key: value for key, value in corrupt_codec.items() if key != "artifact_sha256"
    }
    corrupt_codec["artifact_sha256"] = _tensor_payload_digest(unsigned)
    codec_path = tmp_path / "corrupt-nested-codec.pt"
    torch.save(corrupt_codec, codec_path)
    with pytest.raises(CellExecutionError, match="codec artifact hash mismatch"):
        _load_deployable_codec(
            codec_path,
            cell_id=cell.cell_id,
            checkpoint_hash=refs["checkpoint"].sha256,
            calibration_hash=refs["calibration"].sha256,
            preparation_hash=preparation_hash,
            device=torch.device("cpu"),
        )

    missing_normalization = copy.deepcopy(payload)
    missing_normalization.pop("normalization")
    unsigned = {
        key: value
        for key, value in missing_normalization.items()
        if key != "artifact_sha256"
    }
    missing_normalization["artifact_sha256"] = _tensor_payload_digest(unsigned)
    normalization_path = tmp_path / "missing-normalization.pt"
    torch.save(missing_normalization, normalization_path)
    with pytest.raises(CellExecutionError, match="field set mismatch"):
        _load_deployable_codec(
            normalization_path,
            cell_id=cell.cell_id,
            checkpoint_hash=refs["checkpoint"].sha256,
            calibration_hash=refs["calibration"].sha256,
            preparation_hash=preparation_hash,
            device=torch.device("cpu"),
        )

    malformed_normalization = copy.deepcopy(payload)
    malformed_normalization["normalization"]["record"]["scale"] = []
    unsigned = {
        key: value
        for key, value in malformed_normalization.items()
        if key != "artifact_sha256"
    }
    malformed_normalization["artifact_sha256"] = _tensor_payload_digest(unsigned)
    malformed_path = tmp_path / "malformed-normalization.pt"
    torch.save(malformed_normalization, malformed_path)
    with pytest.raises(CellExecutionError, match="normalization scale shape"):
        _load_deployable_codec(
            malformed_path,
            cell_id=cell.cell_id,
            checkpoint_hash=refs["checkpoint"].sha256,
            calibration_hash=refs["calibration"].sha256,
            preparation_hash=preparation_hash,
            device=torch.device("cpu"),
        )


def test_mapped_isolated_loss_identity_round_trips_and_refuses_forgery(
    tmp_path: Path,
) -> None:
    cell = _mapped_isolated_loss_cell(seed=41)
    campaign = _campaign(tmp_path, cell)
    assert _runner(campaign).run(limit=1).failed_cells == 0
    refs = campaign.record(cell.cell_id).artifact_map
    deployment_path = refs["deployment_codec"].resolve(campaign.root)
    payload, model, codec, _summary, _verified_deployment_digest = (
        _load_deployable_codec(
            deployment_path,
            cell_id=cell.cell_id,
            checkpoint_hash=refs["checkpoint"].sha256,
            calibration_hash=refs["calibration"].sha256,
            preparation_hash=refs["preparation"].sha256,
            device=torch.device("cpu"),
        )
    )
    assert model.cfg.isolated_loss_decrease_implementation == (
        ISOLATED_LOSS_MAPPED_IMPLEMENTATION
    )
    assert payload["model_cfg"]["isolated_loss_decrease_implementation"] == (
        ISOLATED_LOSS_MAPPED_IMPLEMENTATION
    )
    assert (
        codec.meta["model_cfg"]["isolated_loss_decrease_implementation"]
        == ISOLATED_LOSS_MAPPED_IMPLEMENTATION
    )

    forged = copy.deepcopy(payload)
    forged["model_cfg"]["isolated_loss_decrease_implementation"] = (
        ISOLATED_LOSS_EXACT_IMPLEMENTATION
    )
    unsigned = {key: value for key, value in forged.items() if key != "artifact_sha256"}
    forged["artifact_sha256"] = _tensor_payload_digest(unsigned)
    forged_path = tmp_path / "forged-mapped-identity.pt"
    torch.save(forged, forged_path)
    with pytest.raises(CellExecutionError, match="model config|model-config"):
        _load_deployable_codec(
            forged_path,
            cell_id=cell.cell_id,
            checkpoint_hash=refs["checkpoint"].sha256,
            calibration_hash=refs["calibration"].sha256,
            preparation_hash=refs["preparation"].sha256,
            device=torch.device("cpu"),
        )

    checkpoint_payload = torch.load(
        refs["checkpoint"].resolve(campaign.root),
        map_location="cpu",
        weights_only=True,
    )
    checkpoint_payload["model_cfg"].pop("isolated_loss_decrease_implementation")
    checkpoint_payload["run_binding"]["model_cfg"].pop(
        "isolated_loss_decrease_implementation"
    )
    missing_path = tmp_path / "missing-mapped-identity.pt"
    torch.save(checkpoint_payload, missing_path)
    with pytest.raises(
        CellExecutionError,
        match="lacks isolated_loss_decrease_implementation",
    ):
        _validate_final_checkpoint(
            missing_path,
            checkpoint_payload["run_binding"],
        )


@pytest.mark.parametrize(
    ("decision_name", "decision_value", "message"),
    (
        (
            "implementation.decoded_energy_implementation",
            "ambient_cuda_default",
            "unknown decoded-energy implementation identity",
        ),
        (
            "implementation.code_norm_implementation",
            "ambient_cuda_default",
            "unknown code-norm implementation identity",
        ),
        (
            "optimizer.retract_every_steps",
            20,
            "violates its carrier predicate",
        ),
    ),
)
def test_runner_refuses_unknown_or_ineligible_decoded_energy_implementation(
    decision_name: str,
    decision_value,
    message: str,
) -> None:
    cell = _stiefel_decoded_energy_cell()
    changed = replace(
        cell,
        decisions=tuple(
            replace(decision, value=decision_value)
            if decision.name == decision_name
            else decision
            for decision in cell.decisions
        ),
    )
    with pytest.raises(CellExecutionError, match=message):
        _model_config(changed)


def test_runner_resolves_and_refuses_mapped_isolated_loss_implementation() -> None:
    cell = _mapped_isolated_loss_cell()
    cfg = _model_config(cell)
    assert cfg.isolated_loss_decrease_implementation == (
        ISOLATED_LOSS_MAPPED_IMPLEMENTATION
    )

    for decision_name, decision_value, message in (
        (
            "implementation.isolated_loss_decrease_implementation",
            "ambient_cuda_default",
            "unknown isolated-loss implementation identity",
        ),
        (
            "model.decoder",
            "concatenated_stiefel",
            "violates its carrier predicate",
        ),
    ):
        changed = replace(
            cell,
            decisions=tuple(
                replace(decision, value=decision_value)
                if decision.name == decision_name
                else decision
                for decision in cell.decisions
            ),
        )
        with pytest.raises(CellExecutionError, match=message):
            _model_config(changed)

    mean_squared = replace(
        cell,
        decisions=tuple(
            replace(decision, value="mean_squared")
            if decision.name == "objective.reconstruction"
            else decision
            for decision in cell.decisions
        ),
    )
    assert _model_config(mean_squared).reconstruction_loss == "mean_squared"


def test_runner_resolves_and_refuses_decoder_retraction_implementations() -> None:
    qr = _cell()
    assert (
        _model_config(qr).decoder_retraction_implementation
        == DECODER_RETRACTION_CHOLESKY_QR_IMPLEMENTATION
    )

    def changed(cell: CellSpec, **overrides) -> CellSpec:
        return replace(
            cell,
            decisions=tuple(
                replace(decision, value=overrides[decision.name])
                if decision.name in overrides
                else decision
                for decision in cell.decisions
            ),
        )

    householder = changed(
        qr,
        **{
            "implementation.decoder_retraction_implementation": (
                DECODER_RETRACTION_HOUSEHOLDER_QR_IMPLEMENTATION
            )
        },
    )
    assert (
        _model_config(householder).decoder_retraction_implementation
        == DECODER_RETRACTION_HOUSEHOLDER_QR_IMPLEMENTATION
    )

    polar = changed(
        qr,
        **{
            "model.decoder": "concatenated_stiefel_polar",
            "implementation.decoder_retraction_implementation": (
                DECODER_RETRACTION_SYMMETRIC_POLAR_IMPLEMENTATION
            ),
        },
    )
    assert (
        _model_config(polar).decoder_retraction_implementation
        == DECODER_RETRACTION_SYMMETRIC_POLAR_IMPLEMENTATION
    )

    free = changed(
        qr,
        **{
            "model.decoder": "free_scale_controlled",
            "implementation.decoder_retraction_implementation": (
                DECODER_RETRACTION_NOT_APPLICABLE
            ),
            "implementation.decoded_energy_implementation": (
                DECODED_ENERGY_EXACT_IMPLEMENTATION
            ),
        },
    )
    assert (
        _model_config(free).decoder_retraction_implementation
        == DECODER_RETRACTION_NOT_APPLICABLE
    )

    for identity, message in (
        ("ambient_cuda_default", "unknown decoder-retraction"),
        (
            DECODER_RETRACTION_SYMMETRIC_POLAR_IMPLEMENTATION,
            "violates its carrier predicate",
        ),
        (DECODER_RETRACTION_NOT_APPLICABLE, "violates its carrier predicate"),
    ):
        with pytest.raises(CellExecutionError, match=message):
            _model_config(
                changed(
                    qr,
                    **{"implementation.decoder_retraction_implementation": identity},
                )
            )


def test_decoded_energy_preflight_precedes_every_score_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = _stiefel_decoded_energy_cell(seed=39)
    campaign = _campaign(tmp_path, cell)
    runner = _runner(campaign)
    assert runner.run(stop_after="prepare").completed_cells == 1
    campaign.transition(cell.cell_id, RunState.RUNNING)
    monkeypatch.setenv("BSC_CAMPAIGN_ROOT", str(campaign.root))
    ctx = _Context(
        campaign.cell_manifest_path(cell.cell_id),
        campaign.cell_dir(cell.cell_id) / "preflight-order-artifacts.json",
        "train",
    )
    events: list[str] = []
    original_validate = BlockCrosscoder.validate_decoded_energy_implementation

    def track_validation(model):
        record = original_validate(model)
        if record["applicable"]:
            events.append("validated")
        return record

    class FirstScoreSeen(RuntimeError):
        pass

    def stop_at_first_score(model, *args, **kwargs):
        events.append("score")
        raise FirstScoreSeen

    with monkeypatch.context() as order:
        order.setattr(
            BlockCrosscoder,
            "validate_decoded_energy_implementation",
            track_validation,
        )
        order.setattr(BlockCrosscoder, "scores", stop_at_first_score)
        with pytest.raises(FirstScoreSeen):
            run_cell_module._train(ctx, ctx.prerequisites(), resume=False)
    assert events
    assert events[0] == "validated"
    assert events[-1] == "score"
    assert "validated" in events[: events.index("score")]


def test_failed_scientific_outcome_still_yields_admissible_terminal_report(
    tmp_path: Path,
) -> None:
    base = _cell(seed=31)
    thresholds = tuple(
        (name, 1.01 if name == "aggregate.recovered_factor_fraction_min" else value)
        for name, value in base.decision_map[
            "qualification.phase1_identification_thresholds"
        ]
    )
    cell = replace(
        base,
        name="phase1.test.expected_negative.s31",
        decisions=tuple(
            replace(decision, value=thresholds)
            if decision.name == "qualification.phase1_identification_thresholds"
            else decision
            for decision in base.decisions
        ),
    )
    campaign = _campaign(tmp_path, cell)
    summary = _runner(campaign).run(limit=1)
    assert summary.completed_cells == 1
    assert summary.failed_cells == 0
    record = campaign.record(cell.cell_id)
    assert record.state is RunState.QUALIFIED
    qualification = json.loads(
        record.artifact_map["qualification"].resolve(campaign.root).read_text()
    )
    assert all(qualification["checks"].values())
    assert qualification["scientific_outcome"]["passed"] is False
    assert (
        qualification["scientific_outcome"]["checks"]["phase1_identification"] is False
    )
    assert qualification["promotion_eligible"] is False
    assert "scientific_outcome_failed" in qualification["promotion_ineligible_reasons"]


def test_token_layer_norm_identification_is_explicitly_inapplicable() -> None:
    thresholds = _cell(seed=0).decision_map[
        "qualification.phase1_identification_thresholds"
    ]
    recovery = {
        "identification_metrics_eligible": False,
        "identification_metrics_ineligible_reason": (
            "token_layer_normalization_is_not_a_fixed_linear_factor_map"
        ),
        "support_precision": 1.0,
        "support_recall": 1.0,
    }
    native = _phase1_identification_evidence(
        recovery,
        thresholds,
        margin_normalization_contract="piecewise_available_headroom_signed_margin_v2",
    )
    deployed = _phase1_identification_evidence(
        recovery,
        thresholds,
        margin_normalization_contract="piecewise_available_headroom_signed_margin_v2",
    )
    assert native["applicable"] is False
    assert native["passed"] is None
    assert native["margin"] is None
    assert native["ineligible_reason"] == (
        "token_layer_normalization_is_not_a_fixed_linear_factor_map"
    )
    assert native["aggregate"] == {
        "support_precision_diagnostic": 1.0,
        "support_recall_diagnostic": 1.0,
    }
    validation = _selection_validation_metrics(
        Phase.PHASE1,
        identification={"native": native, "deployed": deployed},
        fixed_rate={},
    )
    assert validation == {
        "phase1_identification_applicable": False,
        "phase1_identification_conjunction": False,
        "phase1_identification_margin": None,
    }
    passed, inapplicable = _phase1_identification_outcome(
        Phase.PHASE1,
        {"native": native, "deployed": deployed},
        validation,
    )
    assert passed is True
    assert inapplicable == {
        "phase1_identification": (
            "token_layer_normalization_is_not_a_fixed_linear_factor_map"
        )
    }


def test_scientific_prepare_requires_clean_committed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _cell()
    cell = replace(
        base,
        decisions=tuple(
            replace(decision, value=False)
            if decision.name == "runtime.smoke"
            else decision
            for decision in base.decisions
        ),
    )
    ctx = SimpleNamespace(
        cell=cell,
        values=cell.decision_map,
        cell_path=tmp_path / "cell.json",
        preparation=tmp_path / "preparation.json",
    )
    identity = run_cell_module._implementation_identity()
    identity["provenance"]["git"]["source_dirty"] = True
    monkeypatch.setattr(run_cell_module, "_implementation_identity", lambda: identity)
    with pytest.raises(CellExecutionError, match="scientific execution requires"):
        run_cell_module._prepare(ctx)


def test_final_training_artifacts_are_reused_not_mutated_on_resume(
    tmp_path: Path,
) -> None:
    cell = _cell(seed=1)
    campaign = _campaign(tmp_path, cell)
    runner = _runner(campaign)
    assert runner.run(stop_after="prepare").completed_cells == 1
    campaign.transition(cell.cell_id, RunState.RUNNING)

    first = runner._invoke(cell.cell_id, "train", resume=False)
    first_refs = {item.kind: item for item in first}
    second = runner._invoke(cell.cell_id, "train", resume=True)
    second_refs = {item.kind: item for item in second}
    assert first_refs["checkpoint"].sha256 == second_refs["checkpoint"].sha256
    assert first_refs["training_report"].sha256 == second_refs["training_report"].sha256
    first_refs["checkpoint"].verify(campaign.root)
    second_refs["checkpoint"].verify(campaign.root)
    assert not (
        campaign.cell_dir(cell.cell_id) / "outputs" / "training-progress.pt"
    ).exists()


def test_isolated_loss_decrease_manifest_lifecycle_and_resume_are_exact(
    tmp_path: Path,
) -> None:
    base = _cell(recipe_index=2, seed=44)
    cell = replace(
        base,
        name="phase1.test.isolated_loss_decrease.s44",
        decisions=merge_decisions(
            base.decisions,
            (
                engineering(
                    "model.selection_score",
                    "isolated_loss_decrease",
                    rationale="exercise the algebraic observed-loss score end to end",
                ),
            ),
        ),
    )
    assert CellSpec.from_manifest(json.loads(json.dumps(cell.to_manifest()))) == cell
    resume_campaign = _campaign(tmp_path / "resume", cell)
    resume_runner = _runner(resume_campaign)
    assert resume_runner.run(stop_after="prepare").completed_cells == 1
    resume_campaign.transition(cell.cell_id, RunState.RUNNING)
    first = {
        item.kind: item
        for item in resume_runner._invoke(cell.cell_id, "train", resume=False)
    }
    resumed = {
        item.kind: item
        for item in resume_runner._invoke(cell.cell_id, "train", resume=True)
    }
    assert resumed["checkpoint"].sha256 == first["checkpoint"].sha256
    assert resumed["training_report"].sha256 == first["training_report"].sha256

    campaign = _campaign(tmp_path / "lifecycle", cell)
    runner = _runner(campaign)
    assert runner.run(limit=1).completed_cells == 1
    record = campaign.record(cell.cell_id)
    assert record.state is RunState.QUALIFIED
    checkpoint_ref = record.artifact_map["checkpoint"]
    report_ref = record.artifact_map["training_report"]
    checkpoint = torch.load(
        checkpoint_ref.resolve(campaign.root),
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["model_cfg"]["selection_score"] == ("isolated_loss_decrease")
    assert report_ref.sha256
    evaluation = json.loads(
        record.artifact_map["evaluation"].resolve(campaign.root).read_text()
    )
    for endpoint_name in ("native_selector", "deployed_selector"):
        endpoint = evaluation[endpoint_name]
        diagnostics = endpoint["isolated_loss_gain_diagnostics"]
        assert diagnostics["schema"] == "bsc-isolated-loss-gain-diagnostics-v1"
        assert diagnostics["applicable"] is True
        assert diagnostics["candidate_event_count"] == (
            endpoint["n_tokens"] * cell.decision_map["model.groups"]
        )
        assert (
            sum(
                diagnostics[f"candidate_{sign}_gain_count"]
                for sign in ("negative", "zero", "positive")
            )
            == diagnostics["candidate_event_count"]
        )
        assert diagnostics["candidate_negative_gain_fraction"] >= 0.0
    qualification = json.loads(
        record.artifact_map["qualification"].resolve(campaign.root).read_text()
    )
    assert qualification["checks"]["selection_score_diagnostics_integrity"] is True


@pytest.mark.parametrize("target_ratio,site_rank", ((0.0, None), (0.03, 2)))
def test_initial_loss_ratio_regularizer_is_resolved_once_and_resume_exact(
    tmp_path: Path,
    target_ratio: float,
    site_rank: int | None,
) -> None:
    base = _cell(recipe_index=2, seed=43)
    factorized_decisions = (
        ()
        if site_rank is None
        else (
            engineering(
                "model.decoder",
                "free_scale_controlled",
                rationale="exercise factor-space regularization",
            ),
            engineering(
                "model.site_rank",
                site_rank,
                rationale="exercise rank-two factor-space regularization",
            ),
            engineering(
                "implementation.decoder_retraction_implementation",
                DECODER_RETRACTION_NOT_APPLICABLE,
                rationale="free factorized decoders do not retract",
            ),
            engineering(
                "implementation.factorized_execution_implementation",
                FACTORIZED_EXECUTION_FACTOR_REGULARIZERS_IMPLEMENTATION,
                rationale="bind the factor-space nuclear objective",
            ),
        )
    )
    cell = replace(
        base,
        name=(
            "phase1.test.regularizer_ratio_"
            + str(target_ratio).replace(".", "p")
            + ("_full" if site_rank is None else f"_rank{site_rank}")
            + ".s43"
        ),
        decisions=merge_decisions(
            base.decisions,
            (
                engineering(
                    "objective.regularizer",
                    "end_to_end_map_nuclear",
                    rationale="exercise ratio-calibration provenance",
                ),
                engineering(
                    "objective.regularizer_reduction",
                    "sum_blocks",
                    rationale="exercise the SASA map penalty path",
                ),
                engineering(
                    "objective.regularizer_coefficient",
                    0.0,
                    rationale="reserve the coefficient for calibration",
                ),
                engineering(
                    "objective.regularizer_coefficient_mode",
                    "initial_loss_ratio",
                    rationale="exercise the dimensionless fit contract",
                ),
                engineering(
                    "objective.regularizer_target_initial_ratio",
                    target_ratio,
                    rationale="exercise a declared ratio-ladder point",
                ),
                engineering(
                    "objective.regularizer_calibration_contract",
                    "post_init_train_prefix_true_observation_fp32_v1",
                    rationale="bind the first training batch and fp32 fit",
                ),
                *factorized_decisions,
            ),
        ),
    )
    if site_rank is not None:
        stale = replace(
            cell,
            decisions=tuple(
                replace(
                    decision,
                    value=FACTORIZED_EXECUTION_DIRECT_RANK_SPACE_IMPLEMENTATION,
                )
                if decision.name == "implementation.factorized_execution_implementation"
                else decision
                for decision in cell.decisions
            ),
        )
        with pytest.raises(CellExecutionError, match="objective predicate"):
            _model_config(stale)
    campaign = _campaign(tmp_path / f"ratio-{target_ratio}", cell)
    runner = _runner(campaign)
    assert runner.run(stop_after="prepare").completed_cells == 1
    campaign.transition(cell.cell_id, RunState.RUNNING)
    first = {
        item.kind: item for item in runner._invoke(cell.cell_id, "train", resume=False)
    }
    checkpoint = torch.load(
        first["checkpoint"].resolve(campaign.root),
        map_location="cpu",
        weights_only=True,
    )
    report = json.loads(first["training_report"].resolve(campaign.root).read_text())
    calibration = checkpoint["run_binding"]["initialization"]["regularizer_calibration"]
    assert report["regularizer_calibration"] == calibration
    assert calibration["mode"] == "initial_loss_ratio"
    assert calibration["target_initial_ratio"] == target_ratio
    assert calibration["achieved_initial_ratio"] == pytest.approx(target_ratio)
    assert calibration["initial_reconstruction_loss"] > 0
    assert calibration["initial_regularizer_unweighted"] > 0
    assert (
        calibration["resolved_coefficient"] == report["model_cfg"]["lambda_regularizer"]
    )
    assert report["model_cfg"]["factorized_execution_implementation"] == (
        FACTORIZED_EXECUTION_NOT_APPLICABLE
        if site_rank is None
        else FACTORIZED_EXECUTION_FACTOR_REGULARIZERS_IMPLEMENTATION
    )
    assert (calibration["resolved_coefficient"] == 0.0) is (target_ratio == 0.0)
    second = {
        item.kind: item for item in runner._invoke(cell.cell_id, "train", resume=True)
    }
    assert second["checkpoint"].sha256 == first["checkpoint"].sha256
    assert second["training_report"].sha256 == first["training_report"].sha256


def test_nonzero_ratio_regularizer_resumes_exactly_from_recovery_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_ratio = 0.03
    base = _cell(recipe_index=2, seed=45)
    cell = replace(
        base,
        name="phase1.test.regularizer_ratio_interrupted.s45",
        decisions=merge_decisions(
            base.decisions,
            (
                engineering(
                    "objective.regularizer",
                    "end_to_end_map_nuclear",
                    rationale="exercise ratio-calibration recovery",
                ),
                engineering(
                    "objective.regularizer_reduction",
                    "sum_blocks",
                    rationale="exercise the SASA map penalty path",
                ),
                engineering(
                    "objective.regularizer_coefficient",
                    0.0,
                    rationale="reserve the coefficient for calibration",
                ),
                engineering(
                    "objective.regularizer_coefficient_mode",
                    "initial_loss_ratio",
                    rationale="exercise recovery with a resolved coefficient",
                ),
                engineering(
                    "objective.regularizer_target_initial_ratio",
                    target_ratio,
                    rationale="require a nonzero calibrated coefficient",
                ),
                engineering(
                    "objective.regularizer_calibration_contract",
                    "post_init_train_prefix_true_observation_fp32_v1",
                    rationale="bind the first training batch and fp32 fit",
                ),
            ),
        ),
    )

    baseline_campaign = _campaign(tmp_path / "baseline", cell)
    baseline_runner = _runner(baseline_campaign)
    assert baseline_runner.run(stop_after="prepare").completed_cells == 1
    baseline_campaign.transition(cell.cell_id, RunState.RUNNING)
    baseline = {
        item.kind: item
        for item in baseline_runner._invoke(cell.cell_id, "train", resume=False)
    }
    baseline_checkpoint = torch.load(
        baseline["checkpoint"].resolve(baseline_campaign.root),
        map_location="cpu",
        weights_only=True,
    )

    resumed_campaign = _campaign(tmp_path / "resumed", cell)
    resumed_runner = _runner(resumed_campaign)
    assert resumed_runner.run(stop_after="prepare").completed_cells == 1
    resumed_campaign.transition(cell.cell_id, RunState.RUNNING)
    monkeypatch.setenv("BSC_CAMPAIGN_ROOT", str(resumed_campaign.root))
    ctx = _Context(
        resumed_campaign.cell_manifest_path(cell.cell_id),
        resumed_campaign.cell_dir(cell.cell_id) / "train-test-artifacts.json",
        "train",
    )
    prerequisites = ctx.prerequisites()
    original_training_batches = run_cell_module._training_batches

    def interrupt_after_recovery_checkpoint(*args, **kwargs):
        for index, batch in enumerate(original_training_batches(*args, **kwargs)):
            if index == 2:
                raise RuntimeError("intentional interruption after recovery checkpoint")
            yield batch

    with monkeypatch.context() as interruption:
        interruption.setattr(
            run_cell_module,
            "_training_batches",
            interrupt_after_recovery_checkpoint,
        )
        with pytest.raises(
            RuntimeError,
            match="intentional interruption after recovery checkpoint",
        ):
            run_cell_module._train(ctx, prerequisites, resume=False)

    assert ctx.progress.is_file()
    assert not ctx.checkpoint.exists()
    recovery = torch.load(ctx.progress, map_location="cpu", weights_only=True)
    assert recovery["step_idx"] == 2
    assert recovery["accepted_tokens"] == 32
    assert recovery["data_cursor"] == {"next_token": 32, "stream": "train"}
    resolved = recovery["model_cfg"]["lambda_regularizer"]
    assert resolved > 0.0
    assert recovery["run_binding"]["model_cfg"]["lambda_regularizer"] == resolved

    original_load_checkpoint = run_cell_module.Trainer.load_checkpoint
    load_calls: list[Path] = []

    def tracking_load_checkpoint(
        cls,
        path,
        *,
        device="cpu",
        expected_binding=None,
    ):
        load_calls.append(Path(path))
        return original_load_checkpoint(
            path,
            device=device,
            expected_binding=expected_binding,
        )

    with monkeypatch.context() as resume_patch:
        resume_patch.setattr(
            run_cell_module.Trainer,
            "load_checkpoint",
            classmethod(tracking_load_checkpoint),
        )
        resumed = dict(run_cell_module._train(ctx, prerequisites, resume=True))

    assert load_calls == [ctx.progress]
    assert not ctx.progress.exists()
    resumed_checkpoint = torch.load(
        resumed["checkpoint"], map_location="cpu", weights_only=True
    )

    def assert_nested_exact(actual, expected) -> None:
        if torch.is_tensor(expected):
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            assert torch.equal(
                actual.contiguous().reshape(-1).view(torch.uint8),
                expected.contiguous().reshape(-1).view(torch.uint8),
            )
        elif isinstance(expected, dict):
            assert actual.keys() == expected.keys()
            for key in expected:
                assert_nested_exact(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            assert type(actual) is type(expected)
            assert len(actual) == len(expected)
            for actual_item, expected_item in zip(actual, expected):
                assert_nested_exact(actual_item, expected_item)
        else:
            assert actual == expected

    assert_nested_exact(resumed_checkpoint, baseline_checkpoint)


def test_orphan_final_checkpoint_rebuilds_an_exact_cursor_report(
    tmp_path: Path,
) -> None:
    cell = _cell(seed=29)
    campaign = _campaign(tmp_path, cell)
    runner = _runner(campaign)
    assert runner.run(stop_after="prepare").completed_cells == 1
    campaign.transition(cell.cell_id, RunState.RUNNING)
    first = {
        item.kind: item for item in runner._invoke(cell.cell_id, "train", resume=False)
    }
    first["training_report"].resolve(campaign.root).unlink()

    resumed = {
        item.kind: item for item in runner._invoke(cell.cell_id, "train", resume=True)
    }
    assert resumed["checkpoint"].sha256 == first["checkpoint"].sha256
    report = json.loads(resumed["training_report"].resolve(campaign.root).read_text())
    assert report["attempted_tokens"] == cell.decision_map["data.train_tokens"]
    assert report["data_cursor"] == {
        "next_token": cell.decision_map["data.train_tokens"],
        "stream": "train",
    }
    assert (
        report["step_idx"]
        == (
            cell.decision_map["data.train_tokens"]
            + cell.decision_map["optimizer.batch_tokens"]
            - 1
        )
        // cell.decision_map["optimizer.batch_tokens"]
    )
    checkpoint = torch.load(
        resumed["checkpoint"].resolve(campaign.root),
        map_location="cpu",
        weights_only=True,
    )
    assert report["terminal_log"] == checkpoint["history"][-1]


def test_synthetic_truth_is_independent_of_learner_width_and_support() -> None:
    base = _cell(seed=17)
    learner_changes = {
        "model.block_width": 1,
        "model.active_blocks": 1,
    }
    changed = replace(
        base,
        name="phase1.test.learner_only_delta.s17",
        decisions=tuple(
            replace(decision, value=learner_changes[decision.name])
            if decision.name in learner_changes
            else decision
            for decision in base.decisions
        ),
    )
    left = _synthetic_dataset(base, "train")
    right = _synthetic_dataset(changed, "train")
    assert (
        left.protocol_dict()["stream_digest"] == right.protocol_dict()["stream_digest"]
    )
    assert torch.equal(left.sample(32, start=0).x, right.sample(32, start=0).x)
    assert _synthetic_source_contract(base.decision_map) == _synthetic_source_contract(
        changed.decision_map
    )


def test_every_materialized_prefix_cell_resolves_without_hidden_defaults() -> None:
    phase1 = build_phase1_plan(seeds=(0,), smoke=True)
    phase2 = build_phase2_plan(seeds=(0,), smoke=True)
    for cell in (*phase1.cells, *phase2.cells):
        model, train = validate_cell_config(cell)
        values = cell.decision_map
        assert (
            model.decoder_init_distribution == values["model.decoder_init_distribution"]
        )
        assert (
            model.decoder_init_preconditioning
            == values["model.decoder_init_preconditioning"]
        )
        assert (
            model.decoder_init_operation_order
            == values["model.decoder_init_operation_order"]
        )
        assert train.total_steps == (
            int(values["data.train_tokens"]) + int(values["optimizer.batch_tokens"]) - 1
        ) // int(values["optimizer.batch_tokens"])
    for cell in phase1.cells:
        dataset = _synthetic_dataset(cell, "train")
        _normalization_record(dataset, cell.decision_map)
        _synthetic_source_contract(cell.decision_map)


def test_bsf_encoder_scale_fit_has_an_independent_declared_prefix() -> None:
    cell = next(
        cell
        for cell in build_phase1_plan(seeds=(0,), smoke=True).cells
        if cell.recipe_name == "bsf_vanilla_primary"
    )
    values = cell.decision_map
    assert values["data.normalization"] == "none"
    assert values["data.normalization_fit_count"] == 0
    assert values["model.encoder_scale_fit_split"] == "train_unique_prefix"
    assert values["model.encoder_scale_fit_count"] == 64
    train = _synthetic_dataset(cell, "train")
    preparation = {
        "data": {
            "kind": "synthetic",
            "normalization": _normalization_record(train, values),
        }
    }
    ctx = SimpleNamespace(cell=cell, values=values)
    batches = list(_encoder_scale_fit_batches(ctx, preparation))
    assert sum(len(batch) for batch in batches) == 64
    assert all(torch.isfinite(batch).all() for batch in batches)


def test_group_lasso_encoder_scale_fit_remeasures_postactivation_norm() -> None:
    cell = next(
        cell
        for cell in build_phase1_plan(seeds=(0,), smoke=True).cells
        if cell.recipe_name == "bsf_group_lasso_primary"
    )
    values = cell.decision_map
    train = _synthetic_dataset(cell, "train")
    preparation = {
        "data": {
            "kind": "synthetic",
            "normalization": _normalization_record(train, values),
            "source_contract": _synthetic_source_contract(values),
        }
    }
    result = _apply_encoder_scale_calibration(
        SimpleNamespace(cell=cell, values=values),
        preparation,
        BlockCrosscoder(_model_config(cell)),
    )
    assert result["statistic"] == "global_fp64_mean_postactivation_block_norm"
    assert result["solver"] == "positive_bracketed_bisection_remeasure_v1"
    assert result["remeasured_post_fit"] is True
    assert (
        abs(result["mean_block_norm_after"] - result["target"]) <= result["tolerance"]
    )
    assert result["mean_block_norm_after"] != pytest.approx(
        result["mean_block_norm_before"] * result["scale_multiplier"],
        abs=1e-6,
    )


def test_phase1_confirmation_uses_a_distinct_bound_stream() -> None:
    cell = _cell(seed=19)
    development = _synthetic_dataset(cell, "eval")
    confirmation = _synthetic_dataset(cell, "confirmation")
    assert (
        development.protocol_dict()["stream_digest"]
        != confirmation.protocol_dict()["stream_digest"]
    )
    assert (
        development.protocol_dict()["split_seed"]
        == cell.decision_map["random.eval_data_seed"]
    )
    assert (
        confirmation.protocol_dict()["split_seed"]
        == cell.decision_map["random.confirmation_data_seed"]
    )
    assert not torch.equal(
        development.sample(16, start=0).x,
        confirmation.sample(16, start=0).x,
    )
    train = _synthetic_dataset(cell, "train")
    preparation = {
        "data": {
            "kind": "synthetic",
            "evaluation_stream": "confirmation",
            "normalization": _normalization_record(train, cell.decision_map),
            "ranges": {
                "factor_calibration": [0, 16],
                "calibration": [16, 32],
                "evaluation": [64, 128],
            },
        }
    }
    factor_calibration_batches = list(
        _synthetic_batches(cell, preparation, "factor_calibration", 8)
    )
    calibration_batches = list(_synthetic_batches(cell, preparation, "calibration", 8))
    assert torch.equal(
        torch.cat(factor_calibration_batches), development.sample(16, start=0).x
    )
    assert torch.equal(
        torch.cat(calibration_batches), development.sample(16, start=16).x
    )
    batches = list(_synthetic_batches(cell, preparation, "evaluation", 16))
    expected = confirmation.sample(64, start=64).x
    assert torch.equal(torch.cat(batches), expected)


def test_store_verification_receipt_skips_rehash_until_file_stat_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "store"
    writer = ShardWriter(
        root,
        "train",
        whitener_hash="raw:test",
        sites=(0,),
        d_model=3,
        meta={"site_dims": [3]},
        tokens_per_shard=8,
    )
    writer.add(
        torch.randn(8, 1, 3),
        torch.stack((torch.arange(8), torch.arange(8)), dim=1),
    )
    writer.close()
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()
    monkeypatch.delenv("BSC_VERIFICATION_CACHE_ROOT", raising=False)
    monkeypatch.setenv("BSC_CAMPAIGN_ROOT", str(campaign_root))
    calls = 0
    original_verify = StoreReader.verify

    def counted(reader, **kwargs):
        nonlocal calls
        calls += 1
        return original_verify(reader, **kwargs)

    monkeypatch.setattr(StoreReader, "verify", counted)
    _VERIFIED_STORE_BINDINGS.clear()
    _verify_store_reader_once(StoreReader(root, "train"), root, "train")
    assert calls == 1
    _VERIFIED_STORE_BINDINGS.clear()  # simulate a fresh stage subprocess
    _verify_store_reader_once(StoreReader(root, "train"), root, "train")
    assert calls == 1
    [receipt_path] = (campaign_root / ".store-verification").glob("*.json")
    receipt = json.loads(receipt_path.read_text())
    assert receipt["content_probes"]
    assert receipt["content_probes"][0]["length"] > 0

    shard = root / "train" / "shard_00000.safetensors"
    os.utime(shard, None)
    _VERIFIED_STORE_BINDINGS.clear()
    _verify_store_reader_once(StoreReader(root, "train"), root, "train")
    assert calls == 2


def test_store_verification_cache_override_is_rejected(tmp_path, monkeypatch):
    root = tmp_path / "store"
    writer = ShardWriter(
        root,
        "train",
        whitener_hash="raw:test",
        sites=(0,),
        d_model=3,
        meta={"site_dims": [3]},
        tokens_per_shard=8,
    )
    writer.add(
        torch.randn(8, 1, 3),
        torch.stack((torch.arange(8), torch.arange(8)), dim=1),
    )
    writer.close()
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()
    monkeypatch.setenv("BSC_CAMPAIGN_ROOT", str(campaign_root))
    monkeypatch.setenv("BSC_VERIFICATION_CACHE_ROOT", str(tmp_path / "forged"))
    _VERIFIED_STORE_BINDINGS.clear()
    with pytest.raises(CellExecutionError, match="unsupported.*BSC_CAMPAIGN_ROOT"):
        _verify_store_reader_once(StoreReader(root, "train"), root, "train")


def test_store_receipt_reuse_rechecks_complete_stat_fingerprint(
    tmp_path, monkeypatch
):
    root = tmp_path / "store"
    writer = ShardWriter(
        root,
        "train",
        whitener_hash="raw:test",
        sites=(0,),
        d_model=3,
        meta={"site_dims": [3]},
        tokens_per_shard=8,
    )
    writer.add(
        torch.randn(8, 1, 3),
        torch.stack((torch.arange(8), torch.arange(8)), dim=1),
    )
    writer.close()
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()
    monkeypatch.delenv("BSC_VERIFICATION_CACHE_ROOT", raising=False)
    monkeypatch.setenv("BSC_CAMPAIGN_ROOT", str(campaign_root))
    _VERIFIED_STORE_BINDINGS.clear()
    _verify_store_reader_once(StoreReader(root, "train"), root, "train")
    shard = root / "train" / "shard_00000.safetensors"
    body = bytearray(shard.read_bytes())
    body[-1] ^= 1
    shard.write_bytes(body)

    # An ordinary in-place mutation changes ctime/mtime and must invalidate the
    # durable receipt before any consumer can reuse it.
    _VERIFIED_STORE_BINDINGS.clear()
    with pytest.raises(ValueError, match="checksum"):
        _verify_store_reader_once(StoreReader(root, "train"), root, "train")


def test_ad_hoc_store_verification_uses_process_cache_without_shared_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "store"
    writer = ShardWriter(
        root,
        "train",
        whitener_hash="raw:test",
        sites=(0,),
        d_model=3,
        meta={"site_dims": [3]},
        tokens_per_shard=8,
    )
    writer.add(
        torch.randn(8, 1, 3),
        torch.stack((torch.arange(8), torch.arange(8)), dim=1),
    )
    writer.close()
    monkeypatch.delenv("BSC_VERIFICATION_CACHE_ROOT", raising=False)
    monkeypatch.delenv("BSC_CAMPAIGN_ROOT", raising=False)
    calls = 0
    original_verify = StoreReader.verify

    def counted(reader, **kwargs):
        nonlocal calls
        calls += 1
        return original_verify(reader, **kwargs)

    monkeypatch.setattr(StoreReader, "verify", counted)
    _VERIFIED_STORE_BINDINGS.clear()
    reader = StoreReader(root, "train")
    _verify_store_reader_once(reader, root, "train")
    _verify_store_reader_once(reader, root, "train")
    assert calls == 1

    # A fresh process cannot inherit an unauthenticated receipt from shared
    # temporary storage, so clearing the in-memory cache forces a full verify.
    _VERIFIED_STORE_BINDINGS.clear()
    _verify_store_reader_once(StoreReader(root, "train"), root, "train")
    assert calls == 2


def test_host_gpu_lock_serializes_cuda_cells_across_campaigns(tmp_path, monkeypatch):
    base = _cell(seed=19)
    cuda_cell = replace(
        base,
        decisions=tuple(
            replace(decision, value="cuda")
            if decision.name == "runtime.device"
            else decision
            for decision in base.decisions
        ),
    )
    cell_path = tmp_path / "cell.json"
    cell_path.write_text(json.dumps(cuda_cell.to_manifest()) + "\n")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"test-{tmp_path.name}")
    monkeypatch.setattr(
        implementation_module,
        "physical_cuda_device_key",
        lambda _device: "GPU-test-physical-device",
    )
    lock_path = _gpu_lock_path(torch.device("cuda"))
    lock_path.unlink(missing_ok=True)
    with _host_gpu_execution_lock(cell_path):
        payload = json.loads(lock_path.read_text())
        assert payload["schema"] == "bsc-host-gpu-lock-v2"
        assert payload["owner_id"] == cuda_cell.cell_id
        assert payload["physical_device"] == "GPU-test-physical-device"
        contender = os.open(lock_path, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(contender)
    lock_path.unlink(missing_ok=True)


def test_tampered_checkpoint_fails_before_calibration(tmp_path: Path) -> None:
    cell = _cell(seed=2)
    campaign = _campaign(tmp_path, cell)
    runner = _runner(campaign)
    assert runner.run(stop_after="train").completed_cells == 1
    checkpoint = campaign.record(cell.cell_id).artifact_map["checkpoint"]
    checkpoint_path = checkpoint.resolve(campaign.root)
    body = bytearray(checkpoint_path.read_bytes())
    body[len(body) // 2] ^= 0x01
    checkpoint_path.write_bytes(body)

    summary = runner.run(limit=1)
    assert summary.failed_cells == 1
    assert campaign.record(cell.cell_id).state is RunState.FAILED
    messages = [event["message"] for event in campaign.events(cell.cell_id)]
    assert any("calibrate stage failed" in message for message in messages)
    assert not (
        campaign.cell_dir(cell.cell_id) / "outputs" / "calibration-codec.pt"
    ).exists()


def test_codec_event_memory_ceiling_fails_before_materialization(
    tmp_path: Path,
) -> None:
    base = _cell(seed=23)
    cell = replace(
        base,
        name="phase1.test.codec_memory_preflight.s23",
        decisions=tuple(
            replace(decision, value=1)
            if decision.name == "codec.max_calibration_event_bytes"
            else decision
            for decision in base.decisions
        ),
    )
    campaign = _campaign(tmp_path, cell)
    summary = _runner(campaign).run(limit=1, stop_after="calibrate")
    assert summary.failed_cells == 1
    messages = [event["message"] for event in campaign.events(cell.cell_id)]
    assert any(
        "exceeds its resolved event-memory ceiling before event materialization"
        in message
        for message in messages
    )
    assert not (
        campaign.cell_dir(cell.cell_id) / "outputs" / "calibration-codec.pt"
    ).exists()


def test_phase2_without_configured_store_fails_clearly_in_prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = _cell(phase=Phase.PHASE2)
    missing = tmp_path / "does-not-exist"
    monkeypatch.setenv("BSC_ACTIVATION_STORE", str(missing))
    monkeypatch.setenv("BSC_STORE_ROOT", str(missing))
    with pytest.raises(CellExecutionError, match="has no train/split.json"):
        _resolve_real_store(cell.decision_map)


def test_declared_but_unimplemented_semantics_fail_closed(tmp_path: Path) -> None:
    base = _cell()
    cell = replace(
        base,
        decisions=tuple(
            replace(
                decision,
                value=(
                    "unimplemented_decoder"
                    if decision.name == "model.decoder"
                    else DECODER_RETRACTION_NOT_APPLICABLE
                ),
            )
            if decision.name
            in {
                "model.decoder",
                "implementation.decoder_retraction_implementation",
            }
            else decision
            for decision in base.decisions
        ),
    )
    campaign = _campaign(tmp_path, cell)
    summary = _runner(campaign).run(limit=1)
    assert summary.failed_cells == 1
    assert campaign.record(cell.cell_id).state is RunState.FAILED
    messages = [event["message"] for event in campaign.events(cell.cell_id)]
    assert any("unknown resolved decoder" in message for message in messages)


@pytest.mark.parametrize("target", ("cpu", "cuda"))
def test_chunked_recovery_association_matches_exact_host_counts(target: str) -> None:
    if target == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device(target)
    tokens, factors, groups = 37, 5, 300
    truth = torch.arange(tokens * factors).reshape(tokens, factors).remainder(11) < 3
    predicted = torch.arange(tokens * groups).reshape(tokens, groups).remainder(17) < 2
    coactive = torch.zeros(factors, groups, dtype=torch.float64, device=device)
    truth_count = torch.zeros(factors, dtype=torch.float64, device=device)
    predicted_count = torch.zeros(groups, dtype=torch.float64, device=device)

    _accumulate_chunked_recovery_association(
        truth,
        predicted.to(device),
        coactive,
        truth_count,
        predicted_count,
    )

    assert torch.equal(coactive.cpu(), truth.double().T @ predicted.double())
    assert torch.equal(truth_count.cpu(), truth.sum(dim=0).double())
    assert torch.equal(predicted_count.cpu(), predicted.sum(dim=0).double())


@pytest.mark.parametrize("target", ("cpu", "cuda"))
def test_mapped_support_confusion_keeps_exact_device_counts(target: str) -> None:
    if target == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device(target)
    tokens, factors, groups = 37, 5, 300
    truth = torch.arange(tokens * factors).reshape(tokens, factors).remainder(11) < 3
    block_mask = torch.arange(tokens * groups).reshape(tokens, groups).remainder(17) < 2
    group_to_factor = torch.arange(groups).remainder(factors)
    category_masks = (
        torch.arange(factors).remainder(2) == 0,
        torch.arange(factors).remainder(2) == 1,
    )
    active_events = block_mask.nonzero(as_tuple=False)
    predicted = torch.zeros_like(truth)
    predicted[
        active_events[:, 0],
        group_to_factor[active_events[:, 1]],
    ] = True
    expected = torch.stack(
        (
            (predicted & truth).sum(),
            (predicted & ~truth).sum(),
            (~predicted & truth).sum(),
            (~predicted & ~truth).sum(),
        )
    )
    expected_categories = torch.stack(
        tuple(
            torch.stack(
                (
                    (predicted[:, mask] & truth[:, mask]).sum(),
                    (predicted[:, mask] & ~truth[:, mask]).sum(),
                    (~predicted[:, mask] & truth[:, mask]).sum(),
                    truth[:, mask].sum(),
                )
            )
            for mask in category_masks
        )
    )

    actual, actual_categories = _mapped_support_confusion_counts(
        truth.to(device),
        block_mask.to(device),
        group_to_factor.to(device),
        tuple(mask.to(device) for mask in category_masks),
    )

    assert torch.equal(actual.cpu(), expected)
    assert torch.equal(actual_categories.cpu(), expected_categories)


def test_support_confusion_distinguishes_fdr_from_false_positive_rate() -> None:
    truth = torch.tensor([[True, False, False], [False, True, False]])
    predicted = torch.tensor([[True, True, False], [False, True, False]])
    metrics = _support_confusion(predicted, truth)
    assert metrics["precision"] == 2 / 3
    assert metrics["false_discovery_rate"] == 1 / 3
    assert metrics["false_positive_rate"] == 1 / 4


def test_matching_pathologies_keep_split_and_merge_directions_distinct() -> None:
    def metrics(association):
        return _matching_pathologies(
            association,
            strong_cutoff=0.5,
            weak_cutoff=0.25,
        )

    perfect = metrics(torch.eye(3))
    assert perfect["split_factor_fraction"] == 0
    assert perfect["merge_group_fraction"] == 0

    split = metrics(torch.tensor([[0.9, 0.8, 0.0], [0.0, 0.0, 0.9]]))
    assert split["split_factor_fraction"] == 0.5
    assert split["merge_group_fraction"] == 0

    merge = metrics(torch.tensor([[0.9, 0.0], [0.8, 0.0], [0.0, 0.9]]))
    assert merge["split_factor_fraction"] == 0
    assert merge["merge_group_fraction"] == 0.5


def test_real_source_contract_is_exact_not_shape_only() -> None:
    cell = _cell(phase=Phase.PHASE2)
    expected = _expected_real_source_contract(cell.decision_map)
    assert _verify_real_source_contract(expected, cell.decision_map) == expected

    wrong = {**expected, "corpus_revision": "wrong-but-same-shape"}
    with pytest.raises(CellExecutionError, match="corpus_revision"):
        _verify_real_source_contract(wrong, cell.decision_map)

    wrong_capture = {**expected, "text_field": "body"}
    with pytest.raises(CellExecutionError, match="text_field"):
        _verify_real_source_contract(wrong_capture, cell.decision_map)

    unexpected = {**expected, "undeclared_capture_behavior": True}
    with pytest.raises(CellExecutionError, match="undeclared_capture_behavior"):
        _verify_real_source_contract(unexpected, cell.decision_map)


def test_synthetic_capture_contract_is_exact() -> None:
    cell = _cell()
    source = _synthetic_source_contract(cell.decision_map)["contract"]
    assert source["coordinate_amplitude_law"] == "gaussian"
    assert source["factor_subspace_overlap"] == "uncontrolled"
    changed = replace(
        cell,
        decisions=tuple(
            replace(
                decision,
                value=(("version", "wrong"),),
            )
            if decision.name == "data.capture_contract"
            else decision
            for decision in cell.decisions
        ),
    )
    with pytest.raises(CellExecutionError, match="data.capture_contract"):
        _synthetic_source_contract(changed.decision_map)


@pytest.mark.parametrize(
    ("decision_name", "decision_value", "protocol_key"),
    (
        (
            "data.coordinate_amplitude_law",
            "student_t_df3",
            "coordinate_amplitude_law",
        ),
        (
            "data.factor_subspace_overlap",
            "paired_30deg",
            "factor_subspace_overlap",
        ),
    ),
)
def test_executor_maps_phase1_robustness_geometry_into_the_generator(
    decision_name: str,
    decision_value: str,
    protocol_key: str,
) -> None:
    base = _cell(recipe_index=2)
    cell = replace(
        base,
        decisions=tuple(
            replace(decision, value=decision_value)
            if decision.name == decision_name
            else decision
            for decision in base.decisions
        ),
    )
    dataset = _synthetic_dataset(cell, "confirmation")
    assert dataset.protocol_dict()["sampling"][protocol_key] == decision_value
    assert (
        _synthetic_source_contract(cell.decision_map)["contract"][
            decision_name.removeprefix("data.")
        ]
        == decision_value
    )


def test_rate_envelope_prunes_higher_rate_worse_distortion_points() -> None:
    two = _lower_convex_rate_envelope(
        (
            {"name": "low", "total_bits_per_token": 4.0, "raw_space_fvu": 0.8},
            {"name": "high", "total_bits_per_token": 8.0, "raw_space_fvu": 0.9},
        )
    )
    assert [point["name"] for point in two] == ["low"]

    three = _lower_convex_rate_envelope(
        (
            {"name": "zero", "total_bits_per_token": 0.0, "raw_space_fvu": 1.0},
            {"name": "q4", "total_bits_per_token": 4.0, "raw_space_fvu": 0.8},
            {"name": "q8", "total_bits_per_token": 8.0, "raw_space_fvu": 0.9},
        )
    )
    assert [point["name"] for point in three] == ["zero", "q4"]


def test_balanced_time_sharing_schedule_has_exact_count_without_mode_bits() -> None:
    decisions = [
        _balanced_schedule_uses_upper(index, upper_tokens=3, horizon_tokens=11)
        for index in range(11)
    ]
    assert sum(decisions) == 3
    assert decisions == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        True,
    ]
    with pytest.raises(ValueError, match="invalid balanced"):
        _balanced_schedule_uses_upper(11, upper_tokens=3, horizon_tokens=11)


def test_cached_time_sharing_matches_paired_batch_execution() -> None:
    chunks = (
        torch.tensor(
            [[10.0, 11.0, 12.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=torch.float64,
        ),
        torch.tensor(
            [
                [13.0, 14.0, 15.0, 16.0],
                [7.0, 8.0, 9.0, 10.0],
                [11.0, 12.0, 13.0, 14.0],
            ],
            dtype=torch.float64,
        ),
        torch.tensor(
            [
                [17.0, 18.0, 19.0, 20.0],
                [15.0, 16.0, 17.0, 18.0],
                [19.0, 20.0, 21.0, 22.0],
            ],
            dtype=torch.float64,
        ),
    )
    cache = _RawEndpointErrorCache(
        endpoint_names=("zero_event_calibration_mean", "q4", "q8"),
        chunks=chunks,
        tokens=11,
        pooled_denominator=200.0,
    )
    plan = {
        "schedule": {
            "lower_name": "q4",
            "upper_name": "q8",
            "upper_tokens": 3,
            "horizon_tokens": 11,
        }
    }
    result = _evaluate_cached_time_sharing(cache, plan, device=torch.device("cpu"))
    lower = torch.cat([chunk[1] for chunk in chunks])
    upper = torch.cat([chunk[2] for chunk in chunks])
    mask = torch.tensor(
        [
            _balanced_schedule_uses_upper(
                index,
                upper_tokens=3,
                horizon_tokens=11,
            )
            for index in range(11)
        ]
    )
    expected = float(torch.where(mask, upper, lower).sum() / 200.0)
    assert result["schedule"]["raw_space_fvu"] == expected
    assert result["schedule"]["evaluation_upper_tokens"] == 3
    prefix_plan = {
        "schedule": {
            **plan["schedule"],
            "upper_tokens": 6,
            "horizon_tokens": 22,
        }
    }
    prefix = _evaluate_cached_time_sharing(
        cache,
        prefix_plan,
        device=torch.device("cpu"),
    )
    assert prefix["schedule"]["raw_space_fvu"] == expected
    assert prefix["schedule"]["evaluation_tokens"] == 11
    with pytest.raises(CellExecutionError, match="horizon"):
        _evaluate_cached_time_sharing(
            cache,
            {"schedule": {**plan["schedule"], "horizon_tokens": 10}},
            device=torch.device("cpu"),
        )


def _schedule_bundle(
    tmp_path: Path,
    *,
    cell: CellSpec,
    deployment_hash: str,
    values: dict,
    plans: dict,
) -> tuple[Path, dict, dict]:
    path = tmp_path / "deployment-schedules.bin"
    manifest = _write_deployment_schedule_bundle(
        path,
        cell_id=cell.cell_id,
        deployment_codec_sha256=deployment_hash,
        schedule_contract=values["codec.time_sharing_schedule_contract"],
        plans=plans,
    )
    loaded = _load_deployment_schedule_bundle(
        path,
        manifest,
        cell_id=cell.cell_id,
        deployment_codec_sha256=deployment_hash,
        schedule_contract=values["codec.time_sharing_schedule_contract"],
    )
    return path, manifest, loaded


def test_deployment_schedule_bundle_roundtrip_and_tamper_refusal(
    tmp_path: Path,
) -> None:
    cell = _cell(phase=Phase.PHASE2)
    deployment_hash = "ab" * 32
    budget = 384.0
    horizon = 100_000_000
    upper_tokens = 12_345_678
    schedule_key = _time_sharing_plan_key(
        budget=budget,
        lower_name="q4",
        upper_name="q8",
        upper_tokens=upper_tokens,
        horizon_tokens=horizon,
    )
    path, manifest, loaded = _schedule_bundle(
        tmp_path,
        cell=cell,
        deployment_hash=deployment_hash,
        values=cell.decision_map,
        plans={
            schedule_key: {
                "budget_bits_per_token": budget,
                "lower_name": "q4",
                "lower_q": 4,
                "upper_name": "q8",
                "upper_q": 8,
                "upper_tokens": upper_tokens,
                "horizon_tokens": horizon,
                "upper_mixture_weight": upper_tokens / horizon,
                "achieved_total_bits_per_token": 383.5,
            }
        },
    )
    assert path.stat().st_size == 32
    assert manifest["record_count"] == 1
    assert loaded[schedule_key]["deployment_schedule"]["size_bytes"] == 32

    damaged = bytearray(path.read_bytes())
    damaged[9] ^= 1
    path.write_bytes(damaged)
    with pytest.raises(CellExecutionError, match="bytes/manifest|record hash"):
        _load_deployment_schedule_bundle(
            path,
            manifest,
            cell_id=cell.cell_id,
            deployment_codec_sha256=deployment_hash,
            schedule_contract=cell.decision_map["codec.time_sharing_schedule_contract"],
        )
    forged_manifest = copy.deepcopy(manifest)
    forged_manifest["artifact_sha256"] = hashlib.sha256(damaged).hexdigest()
    forged_manifest["records"][0]["sha256"] = hashlib.sha256(damaged).hexdigest()
    with pytest.raises(CellExecutionError, match="header binding"):
        _load_deployment_schedule_bundle(
            path,
            forged_manifest,
            cell_id=cell.cell_id,
            deployment_codec_sha256=deployment_hash,
            schedule_contract=cell.decision_map["codec.time_sharing_schedule_contract"],
        )


def test_fixed_rate_budget_uses_better_lower_rate_packet(
    tmp_path: Path,
) -> None:
    cell = _cell(phase=Phase.PHASE2)
    values = {
        **cell.decision_map,
        "evaluation.fixed_rate_budgets_bits_per_token": (8.0,),
        "evaluation.side_information_amortization_tokens": 1_000_000,
    }
    deployment = tmp_path / "deployable-codec.pt"
    deployment.write_bytes(b"x")
    deployment_hash = hashlib.sha256(deployment.read_bytes()).hexdigest()
    ctx = SimpleNamespace(cell=cell, values=values)
    rd_payload = {
        "rate_model": "fixed_width_count_plus_block_ids_plus_amplitudes_v1",
        "support_count_width_bits": 5,
        "support_id_width_bits": 8,
        "codec_meta": {
            "packet_contract": values["codec.packet_contract"],
            "side_information_contract": values["codec.side_information_contract"],
            "count_alphabet_max": 16,
        },
        "points": {
            "4": {"rate_bits_per_token": 4.0},
            "8": {"rate_bits_per_token": 8.0},
        },
    }
    raw_payload = {
        "eligible": True,
        "points": {
            "4": {"fvu_pooled": 0.8},
            "8": {"fvu_pooled": 0.9},
        },
    }
    plans = _selected_time_sharing_plans(
        ctx,
        rd=rd_payload,
        raw_space=raw_payload,
        deployment_artifact_size_bytes=deployment.stat().st_size,
    )
    assert len(plans) == 1
    _, schedule_manifest, loaded = _schedule_bundle(
        tmp_path,
        cell=cell,
        deployment_hash=deployment_hash,
        values=values,
        plans=plans,
    )
    schedule_key = next(iter(loaded))
    raw_payload["operational_time_sharing"] = {
        schedule_key: {
            **loaded[schedule_key],
            "evaluation_tokens": 17,
            "evaluation_upper_tokens": 0,
            "raw_space_fvu": 0.8,
            "distortion_measurement": (
                "executed_balanced_schedule_on_paired_raw_evaluation_rows"
            ),
        }
    }
    result = _fixed_rate_raw_score(
        ctx,
        rd=rd_payload,
        raw_space=raw_payload,
        deployment_path=deployment,
        deployment_hash=deployment_hash,
        calibration_hash="0" * 64,
        deployment_schedule_manifest=schedule_manifest,
    )
    assert result["eligible"] is True
    assert result["fixed_budgets"][0]["raw_space_fvu"] == pytest.approx(0.8)
    assert result["fixed_budgets"][0]["bracket"] == ["q4", "q4"]
    assert result["fixed_budgets"][0]["operating_record"]["header_bytes"] == 32


def test_exact_fixed_rate_endpoint_is_selected_as_a_pure_record() -> None:
    cell = _cell(phase=Phase.PHASE2)
    horizon = 1_000
    artifact_bytes = 1
    budget = 4.0 + 8.0 * (artifact_bytes + 32) / horizon
    values = {
        **cell.decision_map,
        "evaluation.fixed_rate_budgets_bits_per_token": (budget,),
        "evaluation.side_information_amortization_tokens": horizon,
    }
    plans = _selected_time_sharing_plans(
        SimpleNamespace(cell=cell, values=values),
        rd={
            "points": {
                "4": {"rate_bits_per_token": 4.0},
                "8": {"rate_bits_per_token": 8.0},
            }
        },
        raw_space={
            "eligible": True,
            "points": {
                "4": {"fvu_pooled": 0.5},
                "8": {"fvu_pooled": 0.4},
            },
        },
        deployment_artifact_size_bytes=artifact_bytes,
    )
    assert len(plans) == 1
    plan = next(iter(plans.values()))
    assert plan["lower_name"] == "q4"
    assert plan["upper_name"] == "q4"
    assert plan["upper_tokens"] == 0
    assert plan["achieved_total_bits_per_token"] == pytest.approx(budget)


def test_ineligible_fixed_rate_endpoint_produces_durable_worst_score(
    tmp_path: Path,
) -> None:
    cell = _cell(phase=Phase.PHASE2)
    values = {
        **cell.decision_map,
        "evaluation.fixed_rate_budgets_bits_per_token": (0.001,),
        "evaluation.side_information_amortization_tokens": 1_000,
    }
    deployment = tmp_path / "deployable-codec.pt"
    deployment.write_bytes(b"x")
    deployment_hash = hashlib.sha256(deployment.read_bytes()).hexdigest()
    _, schedule_manifest, _ = _schedule_bundle(
        tmp_path,
        cell=cell,
        deployment_hash=deployment_hash,
        values=values,
        plans={},
    )
    fixed_rate = _fixed_rate_raw_score(
        SimpleNamespace(cell=cell, values=values),
        rd={
            "rate_model": "fixed_width_decodable_payload_bits_v1",
            "support_count_width_bits": 5,
            "support_id_width_bits": 8,
            "codec_meta": {
                "packet_contract": values["codec.packet_contract"],
                "side_information_contract": values["codec.side_information_contract"],
                "count_alphabet_max": 16,
            },
            "points": {
                "4": {"rate_bits_per_token": 4.0},
                "8": {"rate_bits_per_token": 8.0},
            },
        },
        raw_space={
            "eligible": True,
            "points": {
                "4": {"fvu_pooled": 0.8},
                "8": {"fvu_pooled": 0.7},
            },
        },
        deployment_path=deployment,
        deployment_hash=deployment_hash,
        calibration_hash="0" * 64,
        deployment_schedule_manifest=schedule_manifest,
    )
    assert fixed_rate["eligible"] is False
    assert fixed_rate["selection_score"] is None
    validation = _selection_validation_metrics(
        Phase.PHASE2,
        identification=None,
        fixed_rate=fixed_rate,
    )
    assert validation == {
        PHASE2_SELECTION_METRIC_KEY: PHASE2_INELIGIBLE_SELECTION_SCORE
    }
    policy = build_phase2_plan(seeds=(0,), smoke=True).stages[0].selection_policy
    assert policy is not None
    assert (
        Campaign._policy_metric({"validation": validation}, policy)
        == PHASE2_INELIGIBLE_SELECTION_SCORE
    )


def test_fixed_rate_mixture_prices_reproducible_schedule_header(
    tmp_path: Path,
) -> None:
    cell = _cell(phase=Phase.PHASE2)
    values = {
        **cell.decision_map,
        "evaluation.fixed_rate_budgets_bits_per_token": (6.0,),
        "evaluation.side_information_amortization_tokens": 1_000,
    }
    deployment = tmp_path / "deployable-codec.pt"
    deployment.write_bytes(b"x")
    side_rate = 8.0 * deployment.stat().st_size / 1_000
    schedule_rate = 8.0 * 32 / 1_000
    lower_rate = 4.0 + side_rate
    upper_rate = 8.0 + side_rate
    upper_tokens = int(
        ((6.0 - schedule_rate - lower_rate) / (upper_rate - lower_rate)) * 1_000
    )
    schedule_key = _time_sharing_plan_key(
        budget=6.0,
        lower_name="q4",
        upper_name="q8",
        upper_tokens=upper_tokens,
        horizon_tokens=1_000,
    )
    deployment_hash = hashlib.sha256(deployment.read_bytes()).hexdigest()
    achieved_rate = (
        (1.0 - upper_tokens / 1_000) * lower_rate
        + (upper_tokens / 1_000) * upper_rate
        + schedule_rate
    )
    _, schedule_manifest, loaded_schedules = _schedule_bundle(
        tmp_path,
        cell=cell,
        deployment_hash=deployment_hash,
        values=values,
        plans={
            schedule_key: {
                "budget_bits_per_token": 6.0,
                "lower_name": "q4",
                "lower_q": 4,
                "upper_name": "q8",
                "upper_q": 8,
                "upper_tokens": upper_tokens,
                "horizon_tokens": 1_000,
                "upper_mixture_weight": upper_tokens / 1_000,
                "achieved_total_bits_per_token": achieved_rate,
            }
        },
    )
    rd_payload = {
        "rate_model": "fixed_width_decodable_payload_bits_v1",
        "support_count_width_bits": 5,
        "support_id_width_bits": 8,
        "codec_meta": {
            "packet_contract": values["codec.packet_contract"],
            "side_information_contract": values["codec.side_information_contract"],
            "count_alphabet_max": 16,
        },
        "points": {
            "4": {"rate_bits_per_token": 4.0},
            "8": {"rate_bits_per_token": 8.0},
        },
    }
    raw_payload = {
        "eligible": True,
        "points": {
            "4": {"fvu_pooled": 0.5},
            "8": {"fvu_pooled": 0.4},
        },
        "operational_time_sharing": {
            schedule_key: {
                **loaded_schedules[schedule_key],
                "evaluation_tokens": 17,
                "evaluation_upper_tokens": 7,
                "raw_space_fvu": 0.37,
                "distortion_measurement": (
                    "executed_balanced_schedule_on_paired_raw_evaluation_rows"
                ),
            }
        },
    }
    result = _fixed_rate_raw_score(
        SimpleNamespace(cell=cell, values=values),
        rd=rd_payload,
        raw_space=raw_payload,
        deployment_path=deployment,
        deployment_hash=deployment_hash,
        calibration_hash="0" * 64,
        deployment_schedule_manifest=schedule_manifest,
    )
    point = result["fixed_budgets"][0]
    assert point["eligible"] is True
    # This is the executed schedule result, not the 0.4566... analytic blend
    # of the two aggregate endpoint FVUs.
    assert point["raw_space_fvu"] == pytest.approx(0.37)
    assert point["bracket"] == ["q4", "q8"]
    assert point["achieved_total_bits_per_token"] <= 6.0
    schedule = point["operating_record"]
    assert schedule["contract"] == "balanced_global_token_counter_u64_v1"
    assert schedule["header_bytes"] == 32
    assert schedule["per_token_mode_bits"] == 0
    assert schedule["upper_tokens"] == round(
        point["upper_mixture_weight"] * schedule["horizon_tokens"]
    )
    assert schedule["evaluation_upper_tokens"] == 7
    assert schedule["distortion_measurement"].startswith("executed_")

    worse_schedule = copy.deepcopy(raw_payload)
    worse_schedule["operational_time_sharing"][schedule_key]["raw_space_fvu"] = 0.6
    with pytest.raises(CellExecutionError, match="better lower endpoint"):
        _fixed_rate_raw_score(
            SimpleNamespace(cell=cell, values=values),
            rd=rd_payload,
            raw_space=worse_schedule,
            deployment_path=deployment,
            deployment_hash=deployment_hash,
            calibration_hash="0" * 64,
            deployment_schedule_manifest=schedule_manifest,
        )

    # Holdout evaluation must replay a frozen development policy even when its
    # own distortion would have selected a different endpoint.
    replayed = _fixed_rate_raw_score(
        SimpleNamespace(cell=cell, values=values),
        rd=rd_payload,
        raw_space=worse_schedule,
        deployment_path=deployment,
        deployment_hash=deployment_hash,
        calibration_hash="0" * 64,
        deployment_schedule_manifest=schedule_manifest,
        frozen_operating_policy=result["operating_policy"],
    )["fixed_budgets"][0]
    assert replayed["bracket"] == ["q4", "q8"]
    assert replayed["raw_space_fvu"] == pytest.approx(0.6)


def test_development_time_sharing_retains_better_lower_endpoint() -> None:
    horizon = 1_000
    budget = 6.0
    schedule_rate = 8.0 * 32 / horizon
    lower_rate = 4.0 + 8.0 / horizon
    upper_rate = 8.0 + 8.0 / horizon
    upper_tokens = math.floor(
        ((budget - schedule_rate - lower_rate) / (upper_rate - lower_rate)) * horizon
    )
    key = _time_sharing_plan_key(
        budget=budget,
        lower_name="q4",
        upper_name="q8",
        upper_tokens=upper_tokens,
        horizon_tokens=horizon,
    )
    weight = upper_tokens / horizon
    plans = {
        key: {
            "budget_bits_per_token": budget,
            "lower_name": "q4",
            "lower_q": 4,
            "upper_name": "q8",
            "upper_q": 8,
            "upper_tokens": upper_tokens,
            "horizon_tokens": horizon,
            "upper_mixture_weight": weight,
            "achieved_total_bits_per_token": (
                (1.0 - weight) * lower_rate + weight * upper_rate + schedule_rate
            ),
        }
    }
    measurements = {
        key: {
            **plans[key],
            "evaluation_tokens": 17,
            "evaluation_upper_tokens": 7,
            "raw_space_fvu": 0.6,
            "distortion_measurement": (
                "executed_balanced_schedule_on_paired_raw_evaluation_rows"
            ),
        }
    }
    finalized, selected = _finalize_development_time_sharing(
        plans,
        measurements,
        rd={
            "points": {
                "4": {"rate_bits_per_token": 4.0},
                "8": {"rate_bits_per_token": 8.0},
            }
        },
        raw_space={
            "points": {
                "4": {"fvu_pooled": 0.5},
                "8": {"fvu_pooled": 0.4},
            }
        },
        deployment_artifact_size_bytes=1,
        horizon_tokens=horizon,
    )
    final_key = next(iter(finalized))
    assert finalized[final_key]["lower_name"] == "q4"
    assert finalized[final_key]["upper_name"] == "q4"
    assert finalized[final_key]["upper_tokens"] == 0
    assert finalized[final_key]["achieved_total_bits_per_token"] == pytest.approx(
        lower_rate + schedule_rate
    )
    assert selected[final_key]["raw_space_fvu"] == pytest.approx(0.5)
    assert selected[final_key]["evaluation_upper_tokens"] == 0


def test_frozen_fixed_rate_policy_is_independent_of_holdout_distortion(tmp_path):
    cell = _cell(phase=Phase.PHASE2)
    values = {
        **cell.decision_map,
        "evaluation.fixed_rate_budgets_bits_per_token": (6.0,),
        "evaluation.side_information_amortization_tokens": 1_000,
    }
    ctx = SimpleNamespace(cell=cell, values=values)
    rd = {
        "rate_model": "fixed_width_decodable_payload_bits_v1",
        "support_count_width_bits": 5,
        "support_id_width_bits": 8,
        "codec_meta": {
            "packet_contract": values["codec.packet_contract"],
            "side_information_contract": values["codec.side_information_contract"],
            "count_alphabet_max": 16,
        },
        "points": {
            "4": {"rate_bits_per_token": 4.0},
            "8": {"rate_bits_per_token": 8.0},
        },
    }
    development_raw = {
        "eligible": True,
        "points": {
            "4": {"fvu_pooled": 0.5},
            "8": {"fvu_pooled": 0.4},
        },
    }
    plans = _selected_time_sharing_plans(
        ctx,
        rd=rd,
        raw_space=development_raw,
        deployment_artifact_size_bytes=1,
    )
    deployment = tmp_path / "deployable-codec.pt"
    deployment.write_bytes(b"x")
    deployment_hash = hashlib.sha256(deployment.read_bytes()).hexdigest()
    _, manifest, loaded = _schedule_bundle(
        tmp_path,
        cell=cell,
        deployment_hash=deployment_hash,
        values=values,
        plans=plans,
    )
    key = next(iter(loaded))
    development_raw["operational_time_sharing"] = {
        key: {
            **loaded[key],
            "evaluation_tokens": 17,
            "evaluation_upper_tokens": 7,
            "raw_space_fvu": 0.37,
            "distortion_measurement": (
                "executed_balanced_schedule_on_paired_raw_evaluation_rows"
            ),
        }
    }
    selected = _fixed_rate_raw_score(
        ctx,
        rd=rd,
        raw_space=development_raw,
        deployment_path=deployment,
        deployment_hash=deployment_hash,
        calibration_hash="0" * 64,
        deployment_schedule_manifest=manifest,
    )
    policy = selected["operating_policy"]
    assert policy["rows"][0]["lower_name"] == "q4"
    assert policy["rows"][0]["upper_name"] == "q8"

    # These holdout outcomes would make q8 dominated if they were allowed to
    # rebuild the hull. Endpoint identities must nevertheless remain frozen.
    holdout_raw = {
        "eligible": True,
        "points": {
            "4": {"fvu_pooled": 0.1},
            "8": {"fvu_pooled": 0.9},
        },
    }
    replay = _selected_time_sharing_plans(
        ctx,
        rd=rd,
        raw_space=holdout_raw,
        deployment_artifact_size_bytes=1,
        frozen_operating_policy=policy,
    )
    replay_plan = next(iter(replay.values()))
    assert replay_plan["lower_name"] == "q4"
    assert replay_plan["upper_name"] == "q8"


def test_phase2_evaluator_metric_schema_resolves_through_live_policy() -> None:
    validation = _selection_validation_metrics(
        Phase.PHASE2,
        identification=None,
        fixed_rate={"selection_score": -0.375},
    )
    assert validation == {PHASE2_SELECTION_METRIC_KEY: -0.375}
    policy = build_phase2_plan(seeds=(0,), smoke=True).stages[0].selection_policy
    assert policy is not None
    assert policy.metric_path == PHASE2_SELECTION_METRIC_PATH
    assert policy.direction == "max"
    selection_metrics = {"validation": validation}
    assert Campaign._policy_metric(selection_metrics, policy) == -0.375
    better = {
        "validation": _selection_validation_metrics(
            Phase.PHASE2,
            identification=None,
            fixed_rate={"selection_score": -0.25},
        )
    }
    assert Campaign._policy_metric(better, policy) > Campaign._policy_metric(
        selection_metrics, policy
    )


def test_bsf_released_bridge_wires_shared_group_threshold() -> None:
    base = _cell()
    release = RELEASE_DIAGNOSTIC_RECIPES["bsf_group_lasso_released_paper_mode_drift"]
    threshold_names = {
        "model.threshold_scope",
        "model.threshold_parameterization",
        "model.threshold_raw_init",
        "model.threshold_effective_init",
    }
    cell = replace(
        base,
        decisions=merge_decisions(
            base.decisions,
            tuple(
                decision
                for decision in release.decisions
                if decision.name in threshold_names
            ),
        ),
    )
    config = _model_config(cell)
    assert config.group_threshold_scope == "shared_scalar"
    assert config.group_threshold_parameterization == "exp"
    assert config.group_threshold_raw_init == 0
    assert config.group_threshold_effective_init == 1


def test_anthropic_anchor_maps_only_the_minimal_dense_l1_method() -> None:
    cell = next(
        cell
        for cell in build_phase1_plan(seeds=(0,), smoke=True).cells
        if cell.recipe_name == "anthropic_crosscoder_architecture_bridge"
    )
    config = _model_config(cell)
    assert config.selection == "dense"
    assert config.code_activation == "relu"
    assert config.regularizer == "crosscoder_l1"
    assert config.lambda_regularizer > 0
    # The dense-L1 root is excluded from the sparse-finalist allowlist, but it
    # must remain eligible for its own independently calibrated comparator
    # family chain.
    assert cell.decision_map["qualification.promotable"] is True


def _bound_capture_manifest(
    values: dict[str, object],
    *,
    profile: str,
    split_manifests: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    source = _expected_real_source_contract(values)
    source_hash = hashlib.sha256(canonical_json(source).encode("utf-8")).hexdigest()
    split_order, split_plan = _expected_capture_allocation(values)
    implementation = {
        **capture_implementation_contract(),
        "runtime": {
            "requested_device": "cuda",
            "torch_cuda_version": torch.version.cuda,
            "cuda_device_name": "test-device",
        },
    }
    store_site_count = len(values["data.store_sites"])
    captured_width = max(int(item) for item in values["data.site_dims"])
    writer_pipeline = estimate_writer_residency_bytes(
        [captured_width] * store_site_count,
        tokens_per_shard=128,
        row_id_width=3,
    )
    binding = {
        "schema": CAPTURE_BINDING_SCHEMA,
        "campaign_profile": profile,
        "source_hash": source_hash,
        "split_order": list(split_order),
        "split_plan": split_plan,
        "capture_implementation": implementation,
        "sites": list(range(store_site_count)),
        "site_dims": [captured_width] * store_site_count,
        "d_model": captured_width,
        "physical_store_format_version": STORE_FORMAT_VERSION,
        "batch_rows": 1,
        "write_batch_tokens": 1,
        "tokens_per_shard": 128,
        "writer_pipeline": {
            "contract": "one_pending_shard_v1",
            **writer_pipeline,
            "max_writer_residency_bytes": writer_pipeline["writer_residency_bytes"],
        },
        "capture_transfer_pipeline": estimate_capture_pipeline_residency_bytes(
            writer_pipeline,
            [captured_width] * store_site_count,
            batch_rows=1,
            context=int(source["context"]),
            drop_positions=int(source["drop_positions"]),
            cuda_overlap=True,
        ),
    }
    records = {}
    for split in split_order:
        manifest = None if split_manifests is None else split_manifests[split]
        records[split] = {
            "allocation": copy.deepcopy(split_plan[split]),
            "manifest_file_sha256": (
                "0" * 64
                if manifest is None
                else hashlib.sha256(
                    (json.dumps(manifest, indent=2) + "\n").encode()
                ).hexdigest()
            ),
            "manifest_sha256": (
                "1" * 64 if manifest is None else manifest["manifest_sha256"]
            ),
            "content_stream_sha256": (
                "2" * 64 if manifest is None else manifest["content_stream_sha256"]
            ),
            "row_stream_sha256": (
                "3" * 64 if manifest is None else manifest["row_stream_sha256"]
            ),
            "n_tokens": split_plan[split]["actual_tokens"],
            "sites": list(range(store_site_count)),
            "site_dims": [captured_width] * store_site_count,
            "d_model": captured_width,
            "row_id_width": 3,
            "whitener_hash": f"raw:{source_hash}",
        }
    payload = {
        "schema": CAPTURE_MANIFEST_SCHEMA,
        "source": source,
        "source_hash": source_hash,
        "split_order": list(split_order),
        "split_plan": split_plan,
        "splits": records,
        "capture_implementation": implementation,
        "capture_binding": binding,
        "capture_binding_sha256": hashlib.sha256(
            canonical_json(binding).encode("utf-8")
        ).hexdigest(),
    }
    payload["capture_content_sha256"] = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def _rehash_capture_binding(payload: dict[str, object]) -> None:
    payload["capture_binding_sha256"] = hashlib.sha256(
        canonical_json(payload["capture_binding"]).encode("utf-8")
    ).hexdigest()
    unsigned = dict(payload)
    unsigned.pop("capture_content_sha256", None)
    payload["capture_content_sha256"] = hashlib.sha256(
        canonical_json(unsigned).encode("utf-8")
    ).hexdigest()


def test_capture_contract_refuses_reordered_or_reassigned_split_allocation(
    tmp_path: Path,
) -> None:
    values = _phase3_cell().decision_map
    split_order, split_plan = _expected_capture_allocation(values)
    payload = _bound_capture_manifest(values, profile="phase3")
    tmp_path.mkdir(exist_ok=True)
    capture_path = tmp_path / "capture.json"
    capture_path.write_text(json.dumps(payload, indent=2) + "\n")
    assert validate_capture_manifest(payload)["split_order"] == list(split_order)
    assert payload["split_plan"] == split_plan

    reordered = copy.deepcopy(payload)
    reordered["split_order"][0], reordered["split_order"][1] = (
        reordered["split_order"][1],
        reordered["split_order"][0],
    )
    reordered["capture_binding"]["split_order"] = copy.deepcopy(
        reordered["split_order"]
    )
    _rehash_capture_binding(reordered)
    capture_path.write_text(json.dumps(reordered, indent=2) + "\n")
    with pytest.raises(CellExecutionError, match="split order|embedded binding"):
        _load_capture_contract(tmp_path, values)

    reassigned = copy.deepcopy(payload)
    first, second = split_order[:2]
    (
        reassigned["split_plan"][first]["sequence_start"],
        reassigned["split_plan"][second]["sequence_start"],
    ) = (
        reassigned["split_plan"][second]["sequence_start"],
        reassigned["split_plan"][first]["sequence_start"],
    )
    for split in split_order:
        reassigned["splits"][split]["allocation"] = copy.deepcopy(
            reassigned["split_plan"][split]
        )
    reassigned["capture_binding"]["split_plan"] = copy.deepcopy(
        reassigned["split_plan"]
    )
    _rehash_capture_binding(reassigned)
    capture_path.write_text(json.dumps(reassigned, indent=2) + "\n")
    with pytest.raises(CellExecutionError, match="split plan|split allocation"):
        _load_capture_contract(tmp_path, values)


def test_phase3_single_raw_store_resolves_bound_transform_only_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _phase3_cell()
    values = cell.decision_map
    raw_root = tmp_path / "raw"
    capture_manifest = _bound_capture_manifest(values, profile="phase3")
    source = capture_manifest["source"]
    source_hash = capture_manifest["source_hash"]
    split_order, split_plan = _expected_capture_allocation(values)
    capture_binding_sha256 = capture_manifest["capture_binding_sha256"]
    raw_root.mkdir()
    split_manifests: dict[str, dict[str, object]] = {}
    store_site_count = len(values["data.store_sites"])
    captured_width = max(int(item) for item in values["data.site_dims"])
    for split in split_order:
        allocation = split_plan[split]
        sequence_start = allocation["sequence_start"]
        sequence_stop = allocation["sequence_stop_exclusive"]
        tokens_per_sequence = allocation["tokens_per_sequence"]
        n_tokens = allocation["actual_tokens"]
        generator = torch.Generator().manual_seed(sequence_start + 1)
        x = torch.randn(n_tokens, store_site_count, captured_width, generator=generator)
        row_ids = torch.stack(
            (
                torch.arange(sequence_start, sequence_stop).repeat_interleave(
                    tokens_per_sequence
                ),
                torch.arange(1, tokens_per_sequence + 1).repeat(
                    sequence_stop - sequence_start
                ),
                torch.arange(n_tokens),
            ),
            dim=1,
        )
        writer = ShardWriter(
            raw_root,
            split,
            whitener_hash=f"raw:{source_hash}",
            sites=range(store_site_count),
            d_model=captured_width,
            meta={
                **source,
                "site_dims": [captured_width] * store_site_count,
                "split_requested_tokens": allocation["requested_tokens"],
                "split_actual_tokens": n_tokens,
                "sequence_start": sequence_start,
                "sequence_stop_exclusive": sequence_stop,
                "tokens_per_sequence": tokens_per_sequence,
                "sequence_allocation": "whole_packed_contexts_v1",
                "capture_binding_sha256": capture_binding_sha256,
                "ordered_split_allocation": list(split_order),
            },
            tokens_per_shard=n_tokens,
        )
        writer.add(x, row_ids)
        split_manifests[split] = writer.close()
    capture_manifest = _bound_capture_manifest(
        values,
        profile="phase3",
        split_manifests=split_manifests,
    )
    (raw_root / "capture.json").write_text(
        json.dumps(capture_manifest, indent=2) + "\n"
    )
    fit_transform_artifacts(
        raw_root,
        raw_root / "transforms",
        ("scalar_rms",),
        batch_size=16,
    )
    monkeypatch.setenv("BSC_ACTIVATION_STORE", str(raw_root))
    monkeypatch.setenv("BSC_TRANSFORM_ROOT", str(raw_root / "transforms"))
    resolved = _resolve_real_store(values)
    assert resolved["store_view_policy"].startswith("single_bf16_raw_view")
    assert (
        resolved["bindings"]["train"]["n_tokens"]
        == split_plan["train"]["actual_tokens"]
    )
    assert resolved["normalization"]["application"] == "on_the_fly"
    assert (
        resolved["normalization"]["source_capture_sha256"]
        == resolved["source_contract"]["sha256"]
    )

    # Campaign registration deliberately refuses an isolated Phase-3 stage:
    # the real launch must carry the exact frozen-panel and blueprint
    # manifests.  Exercise the numerical refusal gate directly here while
    # the campaign tests cover manifest binding and forgery rejection.
    preparation = {"data": {"kind": "activation_store", **resolved}}
    assert not _transform_on_cuda(preparation, torch.device("cpu"))
    assert _transform_on_cuda(preparation, torch.device("cuda"))
    preflight_ctx = SimpleNamespace(values=values)
    init_x = next(_training_batches(preflight_ctx, preparation, start_token=0))
    model = BlockCrosscoder(_model_config(cell))
    model.initialize_decoder_bias_(init_x.float())
    model.project_decoder_()
    _apply_encoder_scale_calibration(preflight_ctx, preparation, model)
    precision = _production_precision_preflight(
        preflight_ctx,
        model,
        init_x,
        None,
    )
    assert precision["applicable"] is True
    assert precision["contract"] == "fp32_bf16_initial_forward_v1"
    assert precision["passed"] is True
    assert precision["reconstruction_relative_error"] <= 0.05
    assert precision["support_iou"] >= 0.90

    replay_values = {
        **values,
        "data.unique_tokens": 10,
        "data.train_tokens": 23,
        "optimizer.batch_tokens": 8,
    }
    replay_resolved = _resolve_real_store(replay_values)
    preparation = {"data": {"kind": "activation_store", **replay_resolved}}
    replay_ctx = SimpleNamespace(values=replay_values)
    full_batches = list(_training_batches(replay_ctx, preparation, start_token=0))
    assert [len(batch) for batch in full_batches] == [8, 8, 7]
    resumed_batches = list(_training_batches(replay_ctx, preparation, start_token=16))
    assert [len(batch) for batch in resumed_batches] == [7]
    assert torch.equal(torch.cat(full_batches)[16:], torch.cat(resumed_batches))

    final_reader = StoreReader(raw_root, "final")
    final_batches = list(final_reader.sequential_batches_with_ids(128))
    final_meta = dict(final_reader.manifest["meta"])
    (raw_root / "final").rename(tmp_path / "valid-final")
    bad_writer = ShardWriter(
        raw_root,
        "final",
        whitener_hash=final_reader.whitener_hash,
        sites=final_reader.sites,
        d_model=final_reader.d_model,
        meta=final_meta,
        tokens_per_shard=final_reader.manifest["tokens_per_shard"],
    )
    for acts, row_ids in final_batches:
        shifted_ids = row_ids.clone()
        shifted_ids[:, 0] += 100
        bad_writer.add(acts, shifted_ids)
    bad_writer.close()
    _VERIFIED_STORE_BINDINGS.clear()
    with pytest.raises(
        CellExecutionError,
        match="authenticated capture record|row identity sequence",
    ):
        _resolve_real_store(values)


def test_training_batches_do_not_copy_aligned_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [torch.full((8, 2, 3), float(index)) for index in range(4)]

    class Reader:
        def shuffled_batches(self, *args, **kwargs):
            yield from chunks

    monkeypatch.setattr(run_cell_module, "_store_reader", lambda *args: Reader())
    values = {
        "optimizer.batch_tokens": 8,
        "data.train_tokens": 32,
        "data.unique_tokens": 32,
        "random.train_data_seed": 7,
    }
    preparation = {"data": {"kind": "activation_store"}}
    batches = list(
        _training_batches(
            SimpleNamespace(values=values),
            preparation,
            start_token=0,
            apply_transform=False,
        )
    )
    assert len(batches) == len(chunks)
    assert all(
        batch.untyped_storage().data_ptr() == chunk.untyped_storage().data_ptr()
        for batch, chunk in zip(batches, chunks)
    )

    tails = [torch.arange(60).reshape(10, 2, 3).float() + 60 * i for i in range(3)]
    monkeypatch.setattr(
        run_cell_module,
        "_store_reader",
        lambda *args: type(
            "TailReader",
            (),
            {"shuffled_batches": lambda self, *a, **kw: iter(tails)},
        )(),
    )
    values.update(
        {
            "data.train_tokens": 23,
            "data.unique_tokens": 10,
        }
    )
    full = list(
        _training_batches(
            SimpleNamespace(values=values),
            preparation,
            start_token=0,
            apply_transform=False,
        )
    )
    resumed = list(
        _training_batches(
            SimpleNamespace(values=values),
            preparation,
            start_token=16,
            apply_transform=False,
        )
    )
    assert [len(batch) for batch in full] == [8, 8, 7]
    assert [len(batch) for batch in resumed] == [7]
    assert torch.equal(torch.cat(full)[16:], torch.cat(resumed))


def test_released_mechanics_reach_declared_config_and_dead_auxiliary_refuses() -> None:
    base = _cell()
    sasa = RELEASE_DIAGNOSTIC_RECIPES["sasa_released_code_drift"]
    sasa_cell = replace(
        base,
        decisions=merge_decisions(
            base.decisions,
            tuple(
                decision
                for decision in sasa.decisions
                if decision.name in {"model.encoder", "model.encoder_constraint"}
            ),
        ),
    )
    assert _model_config(sasa_cell).encoder_mode == "untied"
    assert _model_config(sasa_cell).encoder_constraint == "unit_latent"

    adapted_values = {
        "model.block_width": 1,
        "model.selector": "decoder_weighted_batchtopk",
        "objective.auxiliary": "decoder_weighted_token_horizon_residual",
        "objective.auxiliary_reconstruction": ("squared_l2_over_residual_variance"),
        "auxiliary.apply_b_dec_to_input": False,
    }
    with pytest.raises(StudyError, match="not declared by any live study recipe"):
        replace(
            base,
            name="phase1.test.decoder_weighted_token_horizon.s0",
            decisions=tuple(
                replace(decision, value=adapted_values[decision.name])
                if decision.name in adapted_values
                else decision
                for decision in base.decisions
            ),
        )
