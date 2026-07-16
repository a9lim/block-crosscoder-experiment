"""Block-sparse crosscoders: unsupervised cross-layer manifold discovery.

Phased experiment (see docs/design.md, v2.2 frozen). Modules land phase by
phase: gram/model are the Phase −1 core; Phase 0 blockification, Phase 0.5
cross-layer coherence, Phase 1 the BSC trainer + scalar baseline follow;
Phase 2's import bridge lands in saklas.
"""

from .gram import (
    block_gram,
    gram_residual,
    init_decoder_stack,
    rank_penalty,
    retract_,
    site_frobenius_shares,
    site_singular_values,
)
from .model import BlockCrosscoder, BSCConfig, BSCOutput, batch_topk_mask, bsc_loss
from .trainer import DeadTracker, TrainConfig, Trainer, aux_loss, tensor_batches

__version__ = "0.1.0"

__all__ = [
    "BSCConfig",
    "BSCOutput",
    "BlockCrosscoder",
    "DeadTracker",
    "TrainConfig",
    "Trainer",
    "aux_loss",
    "batch_topk_mask",
    "tensor_batches",
    "block_gram",
    "bsc_loss",
    "gram_residual",
    "init_decoder_stack",
    "rank_penalty",
    "retract_",
    "site_frobenius_shares",
    "site_singular_values",
    "__version__",
]
