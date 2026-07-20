"""Block-sparse crosscoders and their controlled evaluation stack.

See ``docs/design.md`` for the normative architecture and Phase-1 protocol.
The post-publication consumer bridge lands in saklas, not this package.
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
from .synthetic import BlockSpec, PlantedModel, SyntheticBatch
from .trainer import DeadTracker, TrainConfig, Trainer, aux_loss, tensor_batches

__version__ = "0.1.0"

__all__ = [
    "BSCConfig",
    "BSCOutput",
    "BlockCrosscoder",
    "BlockSpec",
    "DeadTracker",
    "PlantedModel",
    "SyntheticBatch",
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
