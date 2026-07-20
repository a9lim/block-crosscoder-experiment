"""The Phase-1 confirmatory panel — DO NOT SCAN BEFORE UNSEALING.

Six concept families preregistered blind (a9-ratified 2026-07-18): fixtures
written from world knowledge only, with
no tokenizer runs, no stream scans, and no dictionary probes at pin time.
The panel exists so at least one capture readout is untouched by every
tuning decision — the calendar/zoo/atlas families are burned as selection
criteria (three analysis passes deep), and this panel is the confirmatory
replacement.

SEAL CONDITION: label maps may only be built (and any scan run) after the
Phase-1 config freeze, or under the single a9-authorized pre-freeze
unsealing budgeted for the learning-rate re-ratification bar. Unsealing is
explicit: set BCC_PANEL_UNSEALED=1 in the environment. Code that imports
this module without the variable set can see the word lists (they are not
secret — they are *untested*); it cannot build label maps. Every metric is
the standing zoo/geo machinery (`zoo_block_tests.py` kinds), and the
mega-block rule applies: top-1 capture is never read without order + FVU.

Known polysemy, worn at pin time exactly as the zoo families wore theirs:
Cancer/Leo (disease/name), rank words as adjectives (General, Major,
Private — cap-only mitigates, sentence-initial residue remains),
Georgia/Washington (country/person), single letters as initials and list
markers. These are part of what the panel measures, not defects to patch
after peeking.
"""

from __future__ import annotations

import os

__all__ = [
    "SEALED_FAMILIES",
    "SEALED_CAP_ONLY",
    "SEALED_KINDS",
    "SEALED_ORDER_VALUES",
    "STATE_COORDS",
    "assert_unsealed",
    "build_sealed_label_map",
]

# -- fixtures (order = class order) -----------------------------------------

ZODIAC = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]

# Single-word states only (two-word states are multi-token by construction);
# the tokenizer drops any of these that fail the single-token rule at
# unsealing. Geo ground truth: approximate state centroids, same rounding
# convention as the atlas tranche's capital-city table.
US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Tennessee", "Texas", "Utah",
    "Vermont", "Virginia", "Washington", "Wisconsin", "Wyoming",
]
STATE_COORDS: dict[str, tuple[float, float]] = {
    "Alabama": (32.8, -86.8), "Alaska": (64.0, -152.0),
    "Arizona": (34.3, -111.7), "Arkansas": (34.8, -92.4),
    "California": (37.2, -119.3), "Colorado": (39.0, -105.5),
    "Connecticut": (41.6, -72.7), "Delaware": (39.0, -75.5),
    "Florida": (28.6, -82.4), "Georgia": (32.6, -83.4),
    "Hawaii": (20.3, -156.4), "Idaho": (44.4, -114.6),
    "Illinois": (40.0, -89.2), "Indiana": (39.9, -86.3),
    "Iowa": (42.0, -93.5), "Kansas": (38.5, -98.4),
    "Kentucky": (37.5, -85.3), "Louisiana": (31.0, -92.0),
    "Maine": (45.4, -69.2), "Maryland": (39.0, -76.8),
    "Massachusetts": (42.3, -71.8), "Michigan": (44.3, -85.4),
    "Minnesota": (46.3, -94.3), "Mississippi": (32.7, -89.7),
    "Missouri": (38.4, -92.5), "Montana": (47.0, -109.6),
    "Nebraska": (41.5, -99.8), "Nevada": (39.3, -116.6),
    "Ohio": (40.3, -82.8), "Oklahoma": (35.6, -97.5),
    "Oregon": (43.9, -120.6), "Pennsylvania": (40.9, -77.8),
    "Tennessee": (35.9, -86.4), "Texas": (31.5, -99.3),
    "Utah": (39.3, -111.7), "Vermont": (44.0, -72.7),
    "Virginia": (37.5, -78.9), "Washington": (47.4, -120.4),
    "Wisconsin": (44.6, -89.7), "Wyoming": (43.0, -107.6),
}

# The clean enlisted->officer ladder, low to high.
MILITARY_RANKS = [
    "Private", "Corporal", "Sergeant", "Lieutenant",
    "Captain", "Major", "Colonel", "General",
]

# Log-line: order values are the powers of ten.
SI_PREFIXES = [
    "pico", "nano", "micro", "milli", "centi", "deci",
    "kilo", "mega", "giga", "tera",
]
SI_EXPONENTS = [-12.0, -9.0, -6.0, -3.0, -2.0, -1.0, 3.0, 6.0, 9.0, 12.0]

SIZE_ADJECTIVES = [
    "tiny", "small", "medium", "large", "huge", "enormous", "gigantic",
]

# Capital letters, alphabetical positions as order values. "A" and "I" are
# excluded at pin time (article and pronoun would dominate every firing);
# the gap is carried in the order values, not renumbered.
ALPHABET = [c for c in "BCDEFGHJKLMNOPQRSTUVWXYZ"]
ALPHABET_POSITIONS = [float(ord(c) - ord("A") + 1) for c in ALPHABET]

SEALED_FAMILIES: dict[str, list[str]] = {
    "zodiac": ZODIAC,
    "us_state": US_STATES,
    "rank": MILITARY_RANKS,
    "si_prefix": SI_PREFIXES,
    "size_adj": SIZE_ADJECTIVES,
    "alphabet": ALPHABET,
}

SEALED_CAP_ONLY = {"zodiac", "us_state", "rank", "alphabet"}

# Metric kind per family (the standing zoo_block_tests machinery).
SEALED_KINDS: dict[str, str] = {
    "zodiac": "ring",        # C=12, adjacency + 20k-perm null
    "us_state": "geo",       # LOO lat/lon decode vs STATE_COORDS
    "rank": "linear",        # Spearman rho along PC1
    "si_prefix": "linear",   # rho on SI_EXPONENTS (log-line by construction)
    "size_adj": "linear",
    "alphabet": "linear",    # rho on ALPHABET_POSITIONS
}

# Non-contiguous order values where class index != order value.
SEALED_ORDER_VALUES: dict[str, list[float]] = {
    "si_prefix": SI_EXPONENTS,
    "alphabet": ALPHABET_POSITIONS,
}

_UNSEAL_VAR = "BCC_PANEL_UNSEALED"


class PanelSealedError(RuntimeError):
    pass


def assert_unsealed() -> None:
    """Loud structural seal. Raises unless BCC_PANEL_UNSEALED=1.

    Setting the variable is an a9 decision (Phase-1 config freeze, or the
    one budgeted tranche-5 unsealing) — never something a script exports
    for convenience.
    """
    if os.environ.get(_UNSEAL_VAR) != "1":
        raise PanelSealedError(
            "The Phase-1 confirmatory panel is SEALED (design §6.4). "
            f"Unsealing is an a9 decision: set {_UNSEAL_VAR}=1 only at the "
            "Phase-1 config freeze or the single authorized tranche-5 use."
        )


def build_sealed_label_map(tokenizer, family: str) -> dict[int, int]:
    """token_id -> class index for a sealed family. Guarded."""
    assert_unsealed()
    from block_crosscoder_experiment.analysis.catalog import surface_forms

    words = SEALED_FAMILIES[family]
    mapping: dict[int, int] = {}
    for k, word in enumerate(words):
        for form in dict.fromkeys(
            surface_forms(word, family in SEALED_CAP_ONLY)
        ):
            ids = tokenizer.encode(form, add_special_tokens=False)
            if len(ids) == 1:
                existing = mapping.get(ids[0])
                if existing is not None and existing != k:
                    raise ValueError(
                        f"{form!r} tokenizes onto a token already claimed "
                        f"by class {existing}"
                    )
                mapping[ids[0]] = k
    if not mapping:
        raise ValueError(f"no single-token surface forms for {family!r}")
    return mapping
