"""Content-bound fixed workspace geometry shared with resource preflight."""

TRUSTED_DECODE_Q_CHUNK = 2
EVALUATION_CONCORDANCE_BLOCK_CHUNK = 512
EVALUATION_REDUCTION_TOKEN_CHUNK = 256
EVALUATION_SPARSE_DECODE_DENSITY_DENOMINATOR = 32

# A decoded-energy cell may use the cheaper block-code norm only while the
# effective concatenated decoder is retracted onto the Stiefel carrier after
# every optimizer update.  These strings are serialized in model configs and
# therefore form part of checkpoint/run identity, not an ambient CUDA choice.
DECODED_ENERGY_EXACT_IMPLEMENTATION = "exact_decoder_gram_v1"
DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION = (
    "stiefel_code_norm_bounded_v1"
)
DECODED_ENERGY_IMPLEMENTATIONS = (
    DECODED_ENERGY_EXACT_IMPLEMENTATION,
    DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION,
)

# Runtime refusal bounds for the specialization.  The fp32 master remains on
# the declared manifold; the regenerated bf16 forward copy is only nearby.
DECODED_ENERGY_MASTER_GRAM_RESIDUAL_MAX = 1.0e-4
DECODED_ENERGY_POSTCAST_GRAM_RESIDUAL_MAX = 2.0e-3

# The v10 planner removes only this conservative subset of the measured score
# graph residency.  Four fp32 [tokens, groups, block_width] buffers are less
# than the observed Phase-2/3 peak reduction and so do not over-credit VRAM.
DECODED_ENERGY_STIEFEL_WORKSPACE_CREDIT_BUFFERS = 4

# The mapped signed quadratic's measured net peak reduction safely supports a
# conservative credit of three fp32 [tokens, groups, block_width] buffers.
ISOLATED_LOSS_EXACT_IMPLEMENTATION = "exact_site_gram_quadratic_v1"
ISOLATED_LOSS_MAPPED_IMPLEMENTATION = "mapped_free_decoder_quadratic_v1"
ISOLATED_LOSS_IMPLEMENTATIONS = (
    ISOLATED_LOSS_EXACT_IMPLEMENTATION,
    ISOLATED_LOSS_MAPPED_IMPLEMENTATION,
)
ISOLATED_LOSS_MAPPED_NET_WORKSPACE_CREDIT_BUFFERS = 3

MODEL_IMPLEMENTATION_IDENTITY_FIELDS = (
    "decoded_energy_implementation",
    "isolated_loss_decrease_implementation",
)


def decoded_energy_code_norm_eligible(
    *,
    selection_score: str,
    decoder_constraint: str,
    training_selector: str,
    site_rank: int | None,
    retract_every: int,
) -> bool:
    """Return the complete fail-closed Stiefel score specialization predicate."""

    return (
        selection_score == "decoded_energy"
        and decoder_constraint in {"gram", "qr"}
        and training_selector in {"token_topk", "batch_topk"}
        and site_rank is None
        and retract_every == 1
    )


def isolated_loss_mapped_eligible(
    *,
    selection_score: str,
    decoder_constraint: str,
    decoder_bias: bool,
    reconstruction_loss: str,
) -> bool:
    """Return the complete scientific carrier for the mapped quadratic."""

    return (
        selection_score == "isolated_loss_decrease"
        and decoder_constraint == "free"
        and not decoder_bias
        and reconstruction_loss in {"mean_squared", "squared_l2"}
    )
