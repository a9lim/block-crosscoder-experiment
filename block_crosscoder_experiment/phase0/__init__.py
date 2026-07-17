"""Phase-0 post-hoc blockification: Engels battery + clustering + ring hunt.

Zero training. Everything here is a pure function of decoder directions and
harvested codes, unit-tested against synthetic ground truth before any real
SAE is touched (design §Phase 0; same discipline as Phase −1).
"""

from block_crosscoder_experiment.phase0.indices import (
    epsilon_mixture_index,
    irreducibility_score,
    separability_index,
)

__all__ = [
    "epsilon_mixture_index",
    "irreducibility_score",
    "separability_index",
]
