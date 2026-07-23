import copy

import pytest

from block_crosscoder_experiment.implementation import (
    execution_identity_sha256,
    implementation_identity,
    validate_implementation_identity,
)


def _clean_identity():
    identity = implementation_identity()
    identity["provenance"] = {"git": {"commit": "1" * 40, "source_dirty": False}}
    return identity


def test_execution_digest_excludes_only_git_provenance():
    clean = _clean_identity()
    another_commit = copy.deepcopy(clean)
    another_commit["provenance"]["git"]["commit"] = "2" * 40
    assert execution_identity_sha256(another_commit) == execution_identity_sha256(clean)

    changed_runtime = copy.deepcopy(clean)
    changed_runtime["numerical_runtime"]["cuda_matmul_allow_tf32"] = True
    assert execution_identity_sha256(changed_runtime) != execution_identity_sha256(
        clean
    )


def test_scientific_identity_requires_clean_provenance_but_smoke_authenticates_it():
    dirty = _clean_identity()
    dirty["provenance"]["git"]["source_dirty"] = True
    assert validate_implementation_identity(
        dirty,
        scientific=False,
    ) == execution_identity_sha256(dirty)
    with pytest.raises(ValueError, match="clean committed source tree"):
        validate_implementation_identity(dirty, scientific=True)


def test_identity_schema_has_no_legacy_or_extra_field_tolerance():
    extra = _clean_identity()
    extra["legacy_git_dirty"] = False
    with pytest.raises(ValueError, match="noncanonical field set"):
        validate_implementation_identity(extra, scientific=False)

    missing = _clean_identity()
    missing["dependencies"].pop("triton")
    with pytest.raises(ValueError, match="dependency identity"):
        validate_implementation_identity(missing, scientific=False)
