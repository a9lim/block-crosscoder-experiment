"""Block-sparse crosscoders and their controlled evaluation stack.

See ``docs/design.md`` for the normative three-phase architecture and protocol.
The post-publication consumer bridge lands in saklas, not this package.
"""

from .gram import (
    block_gram,
    gram_residual,
    init_decoder_stack,
    map_nuclear_penalty,
    project_block_frobenius_,
    retract_,
    site_frobenius_shares,
    site_singular_values,
)
from .model import (
    BlockCrosscoder,
    BSCConfig,
    BSCOutput,
    batch_topk_mask,
    bsc_loss,
    token_topk_mask,
)
from .phase1 import (
    FelSyntheticConfig,
    LadderSyntheticConfig,
    Phase1Batch,
    Phase1Dataset,
)
from .trainer import DeadTracker, TrainConfig, Trainer, aux_loss, tensor_batches

__version__ = "0.1.0"

__all__ = [
    "BSCConfig",
    "BSCOutput",
    "BlockCrosscoder",
    "DeadTracker",
    "FelSyntheticConfig",
    "LadderSyntheticConfig",
    "Phase1Batch",
    "Phase1Dataset",
    "TrainConfig",
    "Trainer",
    "aux_loss",
    "batch_topk_mask",
    "token_topk_mask",
    "tensor_batches",
    "block_gram",
    "bsc_loss",
    "gram_residual",
    "init_decoder_stack",
    "map_nuclear_penalty",
    "project_block_frobenius_",
    "retract_",
    "site_frobenius_shares",
    "site_singular_values",
    "__version__",
]
