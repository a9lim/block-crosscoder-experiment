from __future__ import annotations

import gc
import hashlib
import json
import shutil
import sys
import types
import weakref
from types import SimpleNamespace

import pytest
import torch

from block_crosscoder_experiment.cli.data import (
    _overlap_cuda_capture_copies,
    _canonical_hash,
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
    verify_alignment,
    whole_sequence_split_plan,
)
from block_crosscoder_experiment.cli import data as data_module
from block_crosscoder_experiment.cli import matrix as matrix_module
from block_crosscoder_experiment.cli.matrix import (
    _resolve_phase2_view_dispatch,
    _storage_preflight,
    _verified_existing_input_storage,
)
from block_crosscoder_experiment.cli.matrix import main as matrix_main
from block_crosscoder_experiment.cli.run_cell import _expected_real_source_contract
from block_crosscoder_experiment.store import ShardWriter, StoreReader, Whitener
from block_crosscoder_experiment.studies import (
    FrozenSelection,
    StudyError,
    build_phase2_blueprint,
    build_phase2_plan,
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


def _raw_store(root, *, offset=0.0):
    root.mkdir(parents=True, exist_ok=True)
    source = {
        "store_contract_version": "activation-store-v3-single-view",
        "site_dims": [5, 5],
    }
    source_hash = hashlib.sha256(
        json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    split_sizes = {
        "normalization_fit": 64,
        "calibration": 48,
        "eval": 32,
        "train": 80,
    }
    split_plan = whole_sequence_split_plan(split_sizes, 1)
    (root / "capture.json").write_text(
        json.dumps(
            {
                "source": source,
                "source_hash": source_hash,
                "split_order": list(split_sizes),
                "split_plan": split_plan,
                "splits": split_plan,
            }
        )
        + "\n"
    )
    gen = torch.Generator().manual_seed(4)
    for split, n in split_sizes.items():
        writer = ShardWriter(
            root,
            split,
            whitener_hash=f"raw:{source_hash}",
            sites=(0, 1),
            d_model=5,
            meta={
                "site_dims": [5, 5],
                "split_requested_tokens": n,
                "split_actual_tokens": n,
            },
            tokens_per_shard=17,
            free_space_floor_frac=0,
        )
        x = torch.randn(n, 2, 5, generator=gen) + offset
        ids = torch.stack((torch.arange(n), torch.arange(n) % 7), dim=1)
        writer.add(x, ids)
        writer.close()


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
    assert aligned["eval"]["n_tokens"] == 32
    assert (
        StoreReader(raw, "train").manifest["row_stream_sha256"]
        == StoreReader(out / "scalar_rms", "train").manifest["row_stream_sha256"]
    )


def test_fit_transform_artifact_binds_capture_and_fit_stream_without_shards(tmp_path):
    raw = tmp_path / "raw"
    _raw_store(raw)
    out = tmp_path / "transforms"
    result = fit_transform_artifacts(raw, out, ("scalar_rms",), batch_size=13)
    record = result["scalar_rms"]
    artifact_root = out / "scalar_rms" / record["transform_hash"]
    assert (artifact_root / "whitener.pt").is_file()
    assert (artifact_root / "transform.json").is_file()
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


def test_alignment_refuses_different_rows(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _raw_store(a)
    _raw_store(b)
    # Rebuild one split with different explicit identities.
    split = b / "eval"
    for child in split.iterdir():
        child.unlink()
    writer = ShardWriter(
        b,
        "eval",
        whitener_hash="raw:test",
        sites=(0, 1),
        d_model=5,
        free_space_floor_frac=0,
    )
    writer.add(torch.randn(32, 2, 5), torch.arange(100, 132).view(-1, 1))
    writer.close()
    with pytest.raises(ValueError, match="alignment"):
        verify_alignment((a, b))


def test_split_parser_and_estimate():
    splits = parse_split_sizes(
        ["normalization_fit=2", "calibration=3", "eval=5", "train=7"]
    )
    assert estimate_store_bytes(splits, (4, 6), n_views=2) == 17 * 44 * 2
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
            self.tokenizer = explicit_tokenizer
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
            "test_runtime": "exact",
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
    assert uninterrupted["capture_implementation"]["test_runtime"] == "exact"
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

    for split in uninterrupted["split_order"]:
        left = StoreReader(uninterrupted_root, split)
        right = StoreReader(resumed_root, split)
        assert left.manifest == right.manifest
        for shard in left.manifest["shards"]:
            left_acts, left_ids = left._shard_payload(shard, verify=True)
            right_acts, right_ids = right._shard_payload(shard, verify=True)
            assert torch.equal(left_acts, right_acts)
            assert torch.equal(left_ids, right_ids)


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
            yield torch.full(
                (4, 2, 8),
                index,
                dtype=torch.bfloat16,
                device="cuda",
            ), identities

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
    assert len(verified["inputs"][0]["splits"]) == 4
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
        ("eval", 32),
        ("train", 80),
    )
    cells = {
        "none-cell": SimpleNamespace(
            decision_map={
                "data.normalization": "none",
                "data.split_sizes": split_sizes,
            }
        ),
        "scalar-cell": SimpleNamespace(
            decision_map={
                "data.normalization": "scalar_rms",
                "data.split_sizes": split_sizes,
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

    manifest_path = views / "none" / "train" / "split.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["n_tokens"] += 1
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(StudyError, match="manifest hash mismatch"):
        _resolve_phase2_view_dispatch(views, {"none-cell": cells["none-cell"]})


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
