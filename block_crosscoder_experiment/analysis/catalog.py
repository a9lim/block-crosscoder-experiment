"""One registry for labeled probes and the descriptive manifold zoo.

Only single-token surface forms participate: multi-token spellings do not
identify one residual-stream position. The zoo is descriptive evidence and
must never be used for model selection; confirmatory capture uses the sealed
panel in :mod:`block_crosscoder_experiment.discovery.sealed_panel`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "CAP_ONLY",
    "FAMILIES",
    "FamilySpec",
    "ZOO",
    "ZOO_FAMILIES",
    "build_label_map",
    "label_tokens",
    "surface_forms",
]

WEEKDAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
]
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
YEARS = [str(y) for y in range(1900, 2000)]

# Linear families test number-line structure; small cyclic families provide
# topology controls. Known polysemy is
# accepted and noted at analysis time (ordinal "second" = time unit,
# season "spring" = verb/coil, digits ride years and list markers).
ORDINALS = [
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
    "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
    "nineteenth", "twentieth",
]
CARDINALS = [
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
]
DIGITS = [str(d) for d in range(10)]
SEASONS = ["winter", "spring", "summer", "autumn"]
COMPASS = ["north", "east", "south", "west"]

# Non-1D manifold candidates.
# Colors lead with the hue circle in wheel order (classes 0-5; achromatic
# 6-10 join for capture/individuation but sit off the ring — order tests
# use the chromatic prefix only). Countries and planets are
# capitalized-only (lowercase turkey/china/mercury are other words, the
# May lesson applied at the source); elements are listed in atomic-number
# order so the default line statistic reads Z-order directly.
COLORS = [
    "red", "orange", "yellow", "green", "blue", "purple",  # hue ring
    "pink", "brown", "gray", "black", "white",
]
COUNTRIES = [
    "England", "France", "Germany", "Italy", "Spain", "Portugal",
    "Ireland", "Scotland", "Russia", "China", "Japan", "India",
    "Australia", "Canada", "Mexico", "Brazil", "Egypt", "Israel",
    "Iran", "Iraq", "Turkey", "Greece", "Poland", "Sweden", "Norway",
    "Denmark", "Finland", "Austria", "Switzerland", "Netherlands",
    "Belgium", "Argentina", "Chile", "Peru", "Cuba", "Kenya",
    "Nigeria", "Ethiopia", "Vietnam", "Korea", "Thailand", "Indonesia",
    "Pakistan", "Afghanistan", "Ukraine", "Hungary", "Romania",
    "Iceland",
]
ELEMENTS = [
    "hydrogen", "helium", "lithium", "boron", "carbon", "nitrogen",
    "oxygen", "sodium", "magnesium", "aluminum", "silicon",
    "phosphorus", "sulfur", "chlorine", "potassium", "calcium",
    "titanium", "iron", "nickel", "copper", "zinc", "silver", "tin",
    "gold", "mercury", "lead", "uranium",
]
PLANETS = [
    "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn",
    "Uranus", "Neptune", "Pluto",
]

FAMILIES: dict[str, list[str]] = {
    "weekday": WEEKDAYS,
    "month": MONTHS,
    "year": YEARS,
    "ordinal": ORDINALS,
    "cardinal": CARDINALS,
    "digit": DIGITS,
    "season": SEASONS,
    "compass": COMPASS,
    "color": COLORS,
    "country": COUNTRIES,
    "element": ELEMENTS,
    "planet": PLANETS,
}

# Families whose lowercase surface forms are *different words*
# (turkey the bird, china the porcelain, mercury the metal).
CAP_ONLY = {"country", "planet"}


@dataclass(frozen=True)
class FamilySpec:
    """Rendering and ordering contract for one descriptive family."""

    topology: str
    labels: tuple[str, ...]
    fit_count: int | None = None


# Alphabetical order is deliberate: it is also the canonical order in the
# generated figure index. ``year`` belongs to the SAE positive control, not
# the current winner zoo.
ZOO: dict[str, FamilySpec] = {
    "cardinal": FamilySpec("line", tuple(CARDINALS)),
    "color": FamilySpec("ring", tuple(COLORS), fit_count=6),
    "compass": FamilySpec("ring", ("N", "E", "S", "W")),
    "country": FamilySpec("cloud", tuple(COUNTRIES)),
    "digit": FamilySpec("line", tuple(DIGITS)),
    "element": FamilySpec("line", tuple(ELEMENTS)),
    "month": FamilySpec("ring", tuple(MONTHS)),
    "ordinal": FamilySpec(
        "line",
        tuple(
            f"{i}{'th' if 10 < i % 100 < 14 else {1: 'st', 2: 'nd', 3: 'rd'}.get(i % 10, 'th')}"
            for i in range(1, 21)
        ),
    ),
    "planet": FamilySpec("line", tuple(PLANETS)),
    "season": FamilySpec("ring", tuple(SEASONS)),
    "weekday": FamilySpec("ring", tuple(WEEKDAYS)),
}
ZOO_FAMILIES = tuple(ZOO)


def surface_forms(word: str, cap_only: bool = False) -> list[str]:
    if cap_only:
        return [f" {word}", word]
    return [f" {word}", word, f" {word.lower()}", word.lower()]


def build_label_map(tokenizer, family: str) -> dict[int, int]:
    """token_id → class index, single-token surface forms only."""
    words = FAMILIES[family]
    mapping: dict[int, int] = {}
    for k, word in enumerate(words):
        for form in dict.fromkeys(surface_forms(word, family in CAP_ONLY)):
            # add_special_tokens=False: gemma-style tokenizers prepend BOS
            # on bare encode, which made every form look multi-token.
            ids = tokenizer.encode(form, add_special_tokens=False)
            if len(ids) == 1:
                existing = mapping.get(ids[0])
                if existing is not None and existing != k:
                    raise ValueError(
                        f"{form!r} tokenizes onto a token already claimed by "
                        f"class {existing}"
                    )
                mapping[ids[0]] = k
    if not mapping:
        raise ValueError(f"no single-token surface forms for family {family!r}")
    return mapping


def label_tokens(token_ids: torch.Tensor, label_map: dict[int, int]) -> torch.Tensor:
    """(T,) token ids → (T,) class ids, −1 for unlabeled."""
    out = torch.full_like(token_ids, -1, dtype=torch.long)
    for tid, cls in label_map.items():
        out[token_ids == tid] = cls
    return out
