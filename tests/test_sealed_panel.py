"""The 0.9.9 sealed panel: the seal must be structural (import is fine,
label-map building is not), and the fixtures must be well-formed — all
checkable without any tokenizer run or scan."""

import pytest

from block_crosscoder_experiment.discovery import sealed_panel as sp


def test_seal_blocks_label_maps(monkeypatch):
    monkeypatch.delenv(sp._UNSEAL_VAR, raising=False)
    with pytest.raises(sp.PanelSealedError, match="SEALED"):
        sp.assert_unsealed()
    with pytest.raises(sp.PanelSealedError):
        sp.build_sealed_label_map(object(), "rank")


def test_fixtures_well_formed():
    assert set(sp.SEALED_KINDS) == set(sp.SEALED_FAMILIES)
    for fam, words in sp.SEALED_FAMILIES.items():
        assert len(words) == len(set(words)), f"duplicate class in {fam}"
    for fam, vals in sp.SEALED_ORDER_VALUES.items():
        assert len(vals) == len(sp.SEALED_FAMILIES[fam])
        assert vals == sorted(vals), f"{fam} order values must be monotone"
    assert set(sp.STATE_COORDS) == set(sp.SEALED_FAMILIES["us_state"])
    assert "zodiac" not in sp.SEALED_FAMILIES
    assert len(sp.RELEASED_DEVELOPMENT_FAMILIES["zodiac"]) == 12
    assert "A" not in sp.SEALED_FAMILIES["alphabet"]
    assert "I" not in sp.SEALED_FAMILIES["alphabet"]


def test_unsealed_map_builds_with_stub_tokenizer(monkeypatch):
    """Mapping logic only — a stub vocabulary, not a scan."""
    monkeypatch.setenv(sp._UNSEAL_VAR, "1")

    class Stub:
        def __init__(self):
            self.vocab = {}

        def encode(self, s, add_special_tokens=False):
            return [self.vocab.setdefault(s, len(self.vocab))]

    mapping = sp.build_sealed_label_map(Stub(), "rank")
    assert set(mapping.values()) == set(range(8))
