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
DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION = "stiefel_code_norm_bounded_v1"
DECODED_ENERGY_IMPLEMENTATIONS = (
    DECODED_ENERGY_EXACT_IMPLEMENTATION,
    DECODED_ENERGY_STIEFEL_CODE_NORM_IMPLEMENTATION,
)

# Runtime refusal bounds for the specialization.  The fp32 master remains on
# the declared manifold; the regenerated bf16 forward copy is only nearby.
DECODED_ENERGY_MASTER_GRAM_RESIDUAL_MAX = 1.0e-4
DECODED_ENERGY_POSTCAST_GRAM_RESIDUAL_MAX = 2.0e-3

# The v14 planner removes only this conservative subset of the measured score
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

# Decoder retraction is part of the serialized model implementation, not an
# ambient device or shape dispatch.  Cholesky-QR and the canonical Householder
# reference share the positive-diagonal QR convention; the latter is retained
# as an exact oracle rather than silently selected at runtime.
DECODER_RETRACTION_CHOLESKY_QR_IMPLEMENTATION = (
    "cholesky_qr1_positive_diagonal_cond64_v1"
)
DECODER_RETRACTION_HOUSEHOLDER_QR_IMPLEMENTATION = "householder_qr_positive_diagonal_v1"
DECODER_RETRACTION_SYMMETRIC_POLAR_IMPLEMENTATION = (
    "symmetric_polar_site_bmm_guard_g1024_w8192_c512_f2_r1e-4_v2"
)
DECODER_RETRACTION_SYMMETRIC_POLAR_REFERENCE_IMPLEMENTATION = (
    "symmetric_polar_eigh_floor_v1"
)
DECODER_RETRACTION_NOT_APPLICABLE = "not_applicable_v1"
DECODER_RETRACTION_IMPLEMENTATIONS = (
    DECODER_RETRACTION_CHOLESKY_QR_IMPLEMENTATION,
    DECODER_RETRACTION_HOUSEHOLDER_QR_IMPLEMENTATION,
    DECODER_RETRACTION_SYMMETRIC_POLAR_IMPLEMENTATION,
    DECODER_RETRACTION_SYMMETRIC_POLAR_REFERENCE_IMPLEMENTATION,
    DECODER_RETRACTION_NOT_APPLICABLE,
)
SYMMETRIC_POLAR_FAST_MIN_GROUPS = 1024
SYMMETRIC_POLAR_FAST_MIN_SITE_BLOCK_WIDTH = 8192
SYMMETRIC_POLAR_FAST_FLOOR_MULTIPLIER = 2.0
SYMMETRIC_POLAR_FAST_SPECTRUM_RATIO_MIN = 1.0e-4

# Site-axis factorization is either absent, evaluated directly in its compact
# rank carrier, or executed through the full materialized tensor only as a
# release oracle. The direct CUDA contraction and sparse TopK decoder
# deliberately change bf16 reduction order and therefore remain one explicit
# serialized identity.
FACTORIZED_EXECUTION_DIRECT_RANK_SPACE_IMPLEMENTATION = (
    "direct_rank_space_sparse_topk_cuda_v3"
)
FACTORIZED_EXECUTION_FACTOR_REGULARIZERS_IMPLEMENTATION = (
    "direct_rank_space_sparse_topk_cuda_factor_regularizers_v4"
)
FACTORIZED_EXECUTION_MATERIALIZED_REFERENCE_IMPLEMENTATION = (
    "materialized_prepacked_core_reference_v2"
)
FACTORIZED_EXECUTION_NOT_APPLICABLE = "not_applicable_v1"
FACTORIZED_EXECUTION_IMPLEMENTATIONS = (
    FACTORIZED_EXECUTION_DIRECT_RANK_SPACE_IMPLEMENTATION,
    FACTORIZED_EXECUTION_FACTOR_REGULARIZERS_IMPLEMENTATION,
    FACTORIZED_EXECUTION_MATERIALIZED_REFERENCE_IMPLEMENTATION,
    FACTORIZED_EXECUTION_NOT_APPLICABLE,
)

SPARSE_DECODE_CUDA_IMPLEMENTATION = (
    "native_or_rank_hard_topk_cuda_else_dense_v1"
)
SPARSE_DECODE_DENSE_REFERENCE_IMPLEMENTATION = "dense_reference_v1"
SPARSE_DECODE_IMPLEMENTATIONS = (
    SPARSE_DECODE_CUDA_IMPLEMENTATION,
    SPARSE_DECODE_DENSE_REFERENCE_IMPLEMENTATION,
)

# The full unfactorized SASA map objective can form each site's small block
# Gram and then reduce sites, instead of asking einsum to contract the site
# axis in the same kernel.  The matmul schedule changes fp32 summation order,
# so it is serialized independently from the scientific regularizer name.
MAP_NUCLEAR_GUARDED_MATMUL_IMPLEMENTATION = (
    "batched_site_gram_reference_guard_d1e-3_e1e-4_v1"
)
MAP_NUCLEAR_EINSUM_REFERENCE_IMPLEMENTATION = "site_reduced_einsum_reference_v1"
MAP_NUCLEAR_IMPLEMENTATIONS = (
    MAP_NUCLEAR_GUARDED_MATMUL_IMPLEMENTATION,
    MAP_NUCLEAR_EINSUM_REFERENCE_IMPLEMENTATION,
)
MAP_NUCLEAR_DECODER_CHOLESKY_DIAGONAL_RATIO_MIN = 1.0e-3
MAP_NUCLEAR_SPECTRUM_RATIO_MIN = 1.0e-4

# The custom CUDA decoder wins only while hard selection retains at most one
# block in this many.  The gate is shape-derived and therefore introduces no
# device synchronization or data-dependent algorithm switch.
CUDA_SPARSE_DECODE_DENSITY_DENOMINATOR = 32
CUDA_SPARSE_DECODE_MIN_BATCH = 2048

# Cholesky-QR1 squares the input condition number.  The admitted fp32 carrier
# is deliberately narrow enough that the post-retraction Gram remains inside
# the existing decoded-energy master bound.  Any bound change requires a new
# implementation identity.
CHOLESKY_QR_GRAM_CONDITION_MAX = 64.0
CHOLESKY_QR_RECONSTRUCTION_RELATIVE_RESIDUAL_MAX = 2.0e-6
CHOLESKY_QR_POST_GRAM_RESIDUAL_MAX = 1.0e-4

MODEL_IMPLEMENTATION_IDENTITY_FIELDS = (
    "decoded_energy_implementation",
    "isolated_loss_decrease_implementation",
    "decoder_retraction_implementation",
    "factorized_execution_implementation",
    "sparse_decode_implementation",
    "map_nuclear_implementation",
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
