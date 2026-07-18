"""Cyclic-family token labelers for the positive control (Engels families).

Maps token ids to class indices for the known cyclic families on GPT-2:
weekdays (n=7), months (n=12), years of the 20th century (n=100). Only
single-token surface forms participate — multi-token spellings never carry
the class in one residual position.
"""

from __future__ import annotations

import torch

__all__ = ["FAMILIES", "build_label_map", "label_tokens"]

WEEKDAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
]
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
YEARS = [str(y) for y in range(1900, 2000)]

# Zoo tranche (0.9.6 analysis pass): linear families for the number-line
# question, small cyclic families for completeness. Known polysemy is
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

FAMILIES: dict[str, list[str]] = {
    "weekday": WEEKDAYS,
    "month": MONTHS,
    "year": YEARS,
    "ordinal": ORDINALS,
    "cardinal": CARDINALS,
    "digit": DIGITS,
    "season": SEASONS,
    "compass": COMPASS,
}


def _surface_forms(word: str) -> list[str]:
    return [f" {word}", word, f" {word.lower()}", word.lower()]


def build_label_map(tokenizer, family: str) -> dict[int, int]:
    """token_id → class index, single-token surface forms only."""
    words = FAMILIES[family]
    mapping: dict[int, int] = {}
    for k, word in enumerate(words):
        for form in dict.fromkeys(_surface_forms(word)):
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
