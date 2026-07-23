"""Phase-1 declarative DGP invariants and provenance checks."""

import math
from dataclasses import replace

import pytest
import torch

from block_crosscoder_experiment.phase1 import (
    FelSyntheticConfig,
    LADDER_STEPS,
    LadderSyntheticConfig,
    make_fel_dataset,
    make_ladder_dataset,
)


def _fel_config(**overrides) -> FelSyntheticConfig:
    base = FelSyntheticConfig(
        ambient_dim=12,
        n_factors=16,
        active_per_example=4,
        calibration_examples=2048,
        train_unique_examples=16,
        train_presentations=32,
        eval_unique_examples=12,
        eval_presentations=12,
        structure_seed=101,
        train_seed=202,
        eval_seed=303,
    )
    return replace(base, **overrides)


def _ladder_config(step: str = "baseline", **overrides) -> LadderSyntheticConfig:
    base = LadderSyntheticConfig(
        step=step,
        n_sites=3,
        d_model=8,
        n_factors=6,
        block_dim=3,
        base_rank=2,
        active_per_example=2,
        scale_ratio=2.0,
        noise_std=0.2,
        train_unique_examples=512,
        train_presentations=512,
        eval_unique_examples=128,
        eval_presentations=128,
        structure_seed=707,
        train_seed=808,
        eval_seed=909,
    )
    return replace(base, **overrides)


def test_fel_determinism_exact_k_and_batch_partition_invariance():
    first = make_fel_dataset(_fel_config())
    second = make_fel_dataset(_fel_config())
    assert first.stream_digest == second.stream_digest

    whole = first.sample(12)
    again = second.sample(12)
    assert torch.equal(whole.x, again.x)
    assert torch.equal(whole.active, again.active)
    assert torch.equal(whole.event_example, again.event_example)
    assert torch.equal(whole.event_factor, again.event_factor)
    assert torch.equal(whole.coordinates, again.coordinates)
    assert torch.equal(whole.contributions, again.contributions)
    assert torch.equal(whole.active.sum(dim=1), torch.full((12,), 4))
    assert whole.n_events == 12 * 4

    partitioned = list(first.batches(5, stop=12))
    assert torch.equal(whole.x, torch.cat([batch.x for batch in partitioned]))
    assert torch.equal(whole.active, torch.cat([batch.active for batch in partitioned]))


def test_fel_family_balance_calibration_and_orthonormal_embeddings():
    dataset = make_fel_dataset(_fel_config())
    factors = dataset.factors
    assert sum(f.category == "scalar" for f in factors) == len(factors) // 2
    manifold_counts = [
        sum(f.family == family for f in factors)
        for family in (
            "circle",
            "disk",
            "sphere",
            "torus",
            "mobius",
            "swiss_roll",
            "helix",
        )
    ]
    assert max(manifold_counts) - min(manifold_counts) <= 1
    assert float(dataset.calibration_center_residual.max()) < 2e-6
    assert torch.allclose(
        dataset.calibration_rms_after,
        torch.ones(len(factors)),
        atol=2e-6,
    )
    for factor in factors:
        basis = dataset.contribution_maps[factor.index, 0, :, : factor.coordinate_dim]
        assert torch.allclose(
            basis.T @ basis,
            torch.eye(factor.coordinate_dim),
            atol=2e-6,
        )


def test_fel_unique_examples_presentations_and_split_are_exact():
    config = _fel_config()
    train = make_fel_dataset(config, split="train")
    batch = train.sample(32)
    assert train.protocol["counts"]["unique_examples"] == 16
    assert train.protocol["counts"]["presentations"] == 32
    assert torch.equal(batch.example_ids[:16].sort().values, torch.arange(16))
    assert torch.equal(batch.example_ids[16:].sort().values, torch.arange(16))
    assert not torch.equal(batch.example_ids[:16], batch.example_ids[16:])
    first_by_id = batch.x[:16][batch.example_ids[:16].argsort()]
    second_by_id = batch.x[16:][batch.example_ids[16:].argsort()]
    assert torch.equal(first_by_id, second_by_id)
    first_active_by_id = batch.active[:16][batch.example_ids[:16].argsort()]
    second_active_by_id = batch.active[16:][batch.example_ids[16:].argsort()]
    assert torch.equal(first_active_by_id, second_active_by_id)
    assert not torch.equal(batch.presentation_ids[:16], batch.presentation_ids[16:])

    evaluation = make_fel_dataset(config, split="eval")
    assert evaluation.stream_digest != train.stream_digest
    assert not torch.equal(train.sample(12).x, evaluation.sample(12).x)


def test_ladder_steps_change_exactly_one_declared_dgp_axis():
    datasets = {
        step: make_ladder_dataset(_ladder_config(step)) for step in LADDER_STEPS
    }
    baseline = datasets["baseline"]
    baseline_batch = baseline.sample(512)
    assert baseline.protocol["design"]["delta_from_baseline"] is None

    digests = {dataset.stream_digest for dataset in datasets.values()}
    assert len(digests) == len(LADDER_STEPS)
    for step, dataset in datasets.items():
        batch = dataset.sample(512)
        # All steps retain the same example identities and exact support draw.
        assert torch.equal(batch.example_ids, baseline_batch.example_ids)
        assert torch.equal(batch.active, baseline_batch.active)
        assert torch.equal(batch.active.sum(dim=1), torch.full((512,), 2))
        assert torch.equal(batch.event_example, baseline_batch.event_example)
        assert torch.equal(batch.event_factor, baseline_batch.event_factor)
        if step == "baseline":
            continue
        delta = dataset.protocol["design"]["delta_from_baseline"]
        assert set(delta) == {"field", "baseline", "value"}
        assert delta["field"] in {
            "coordinate_truth",
            "site_rotation",
            "site_scale",
            "noise_std",
            "factor_rank",
        }

    shared_support = datasets["shared_support"].sample(512)
    assert torch.equal(
        shared_support.coordinates[:, 0], baseline_batch.coordinates[:, 0]
    )
    assert not torch.equal(
        shared_support.coordinates[:, 1], baseline_batch.coordinates[:, 1]
    )
    assert torch.equal(
        baseline_batch.coordinates[:, 0], baseline_batch.coordinates[:, 1]
    )

    rotated = datasets["site_rotation"].sample(512)
    assert torch.equal(rotated.coordinates, baseline_batch.coordinates)
    assert not torch.equal(
        datasets["site_rotation"].contribution_maps,
        baseline.contribution_maps,
    )

    scaled = datasets["site_scale"].sample(512)
    assert torch.equal(scaled.coordinates, baseline_batch.coordinates)
    assert torch.equal(datasets["site_scale"].site_scales[0], torch.tensor(1.0))
    assert torch.equal(datasets["site_scale"].site_scales[-1], torch.tensor(2.0))

    noisy = datasets["noise"].sample(512)
    assert torch.equal(noisy.clean_x, baseline_batch.clean_x)
    assert not torch.equal(noisy.x, noisy.clean_x)
    assert noisy.observed.all()

    heterogeneous = datasets["rank_heterogeneity"]
    assert torch.equal(
        heterogeneous.factor_ranks,
        torch.tensor([1, 2, 3, 1, 2, 3]),
    )
    assert not torch.equal(heterogeneous.factor_ranks, baseline.factor_ranks)


def test_ladder_site_map_rank_families_have_truth_known_realized_ranks():
    base = _ladder_config(n_sites=4)
    datasets = {
        family: make_ladder_dataset(replace(base, site_map_rank_family=family))
        for family in ("rank1", "rank2", "independent")
    }
    assert torch.equal(
        datasets["rank1"].realized_site_map_ranks,
        torch.ones(base.n_factors, dtype=torch.long),
    )
    assert torch.equal(
        datasets["rank2"].realized_site_map_ranks,
        torch.full((base.n_factors,), 2, dtype=torch.long),
    )
    assert torch.equal(
        datasets["independent"].realized_site_map_ranks,
        torch.full((base.n_factors,), 4, dtype=torch.long),
    )
    assert torch.linalg.matrix_rank(datasets["rank2"].site_map_loadings) == 2
    independent = datasets["independent"]
    assert set(independent.factor_categories) == {"shared_high_rank_site_map"}
    assert bool(independent.ground_truth["shared_feature_claim_eligible"])
    assert (
        independent.protocol["design"]["independent_map_interpretation"]
        == "shared_coordinate_high_rank_site_map_factorization_stress"
    )


@pytest.mark.parametrize(("span", "expected"), (("one", 1), ("two", 2), ("all", 4)))
def test_ladder_site_presence_span_masks_maps_coordinates_and_contributions(
    span: str, expected: int
):
    dataset = make_ladder_dataset(_ladder_config(n_sites=4, site_presence_span=span))
    assert torch.equal(
        dataset.factor_site_mask.sum(dim=1),
        torch.full((len(dataset.factors),), expected),
    )
    assert all(len(factor.active_sites) == expected for factor in dataset.factors)
    map_mask = dataset.factor_site_mask[:, :, None, None]
    assert torch.count_nonzero(dataset.contribution_maps.masked_select(~map_mask)) == 0

    batch = dataset.sample(128)
    event_site_mask = dataset.factor_site_mask[batch.event_factor]
    coordinate_mask = event_site_mask[:, :, None].expand_as(batch.coordinates)
    contribution_mask = event_site_mask[:, :, None].expand_as(batch.contributions)
    assert torch.count_nonzero(batch.coordinates.masked_select(~coordinate_mask)) == 0
    assert (
        torch.count_nonzero(batch.contributions.masked_select(~contribution_mask)) == 0
    )


def test_ladder_frequency_laws_expose_exact_marginals_and_change_supports():
    base = _ladder_config(
        n_sites=4,
        train_unique_examples=20_000,
        train_presentations=20_000,
    )
    uniform = make_ladder_dataset(base)
    zipf = make_ladder_dataset(replace(base, feature_frequency="zipf_alpha_1"))
    expected_uniform = torch.full(
        (base.n_factors,),
        base.active_per_example / base.n_factors,
        dtype=torch.float64,
    )
    assert torch.allclose(
        uniform.factor_inclusion_probabilities, expected_uniform, atol=1e-12
    )
    assert torch.isclose(
        zipf.factor_inclusion_probabilities.sum(),
        torch.tensor(float(base.active_per_example), dtype=torch.float64),
        atol=1e-12,
    )
    ordered = zipf.factor_inclusion_probabilities[zipf.factor_frequency_rank.argsort()]
    assert bool(torch.all(ordered[:-1] > ordered[1:]))
    uniform_batch = uniform.sample(20_000)
    zipf_batch = zipf.sample(20_000)
    observed = zipf_batch.active.to(torch.float64).mean(dim=0)
    assert torch.allclose(observed, zipf.factor_inclusion_probabilities, atol=0.015)
    assert not torch.equal(uniform_batch.active, zipf_batch.active)
    assert zipf.protocol["sampling"]["zipf_alpha"] == 1.0


def test_ladder_coactivation_mixture_is_exact_k_monotone_and_replayable():
    base = _ladder_config(
        n_sites=4,
        active_per_example=2,
        train_unique_examples=20_000,
        train_presentations=20_000,
    )
    paired_rates = []
    for probability in (0.0, 0.5, 0.9):
        config = replace(base, coactivation_probability=probability)
        first = make_ladder_dataset(config)
        second = make_ladder_dataset(config)
        batch = first.sample(20_000)
        replay = second.sample(20_000)
        assert torch.equal(batch.active, replay.active)
        assert torch.equal(batch.x, replay.x)
        assert torch.equal(batch.active.sum(dim=1), torch.full((20_000,), 2))
        paired_rates.append(
            sum(
                float(
                    (batch.active[:, int(pair[0])] & batch.active[:, int(pair[1])])
                    .float()
                    .mean()
                )
                for pair in first.coactivation_groups
            )
        )
        assert torch.isclose(
            first.factor_inclusion_probabilities.sum(),
            torch.tensor(2.0, dtype=torch.float64),
            atol=1e-12,
        )
    assert paired_rates[0] < paired_rates[1] < paired_rates[2]
    assert paired_rates[0] == pytest.approx(0.2, abs=0.02)
    assert paired_rates[1] == pytest.approx(0.6, abs=0.02)
    assert paired_rates[2] == pytest.approx(0.92, abs=0.02)


def test_ladder_new_axes_preserve_partition_invariance_and_truth_fields():
    config = _ladder_config(
        n_sites=4,
        site_map_rank_family="rank2",
        site_presence_span="two",
        feature_frequency="zipf_alpha_1",
        coactivation_probability=0.5,
        train_unique_examples=64,
        train_presentations=128,
    )
    dataset = make_ladder_dataset(config)
    whole = dataset.sample(128)
    pieces = list(dataset.batches(17, stop=128))
    assert torch.equal(whole.x, torch.cat([batch.x for batch in pieces]))
    assert torch.equal(whole.clean_x, torch.cat([batch.clean_x for batch in pieces]))
    assert torch.equal(whole.active, torch.cat([batch.active for batch in pieces]))
    first_epoch = whole.x[:64][whole.example_ids[:64].argsort()]
    second_epoch = whole.x[64:][whole.example_ids[64:].argsort()]
    assert torch.equal(first_epoch, second_epoch)
    evaluation = make_ladder_dataset(config, split="eval")
    assert evaluation.stream_digest != dataset.stream_digest
    assert not torch.equal(evaluation.sample(64).x, whole.x[:64])
    truth = dataset.ground_truth
    for field in (
        "coordinate_truth",
        "site_map_rank_family",
        "site_presence_span",
        "feature_frequency",
        "coactivation_probability",
        "factor_ranks",
        "site_map_bases",
        "site_map_loadings",
        "factor_site_mask",
        "realized_site_map_ranks",
        "factor_frequency_rank",
        "factor_sampling_weights",
        "factor_inclusion_probabilities",
        "coactivation_groups",
        "coactivation_group_probabilities",
        "coordinate_amplitude_law",
        "coordinate_standardization",
        "student_t_degrees_of_freedom",
        "factor_subspace_overlap",
        "factor_overlap_pairs",
        "target_pair_principal_angle_degrees",
        "principal_angle_reference",
        "realized_pair_principal_angles_degrees",
        "shared_feature_claim_eligible",
    ):
        assert field in truth
    assert truth["factor_site_mask"] is dataset.factor_site_mask
    assert truth["factor_inclusion_probabilities"] is (
        dataset.factor_inclusion_probabilities
    )


def test_ladder_coordinate_amplitude_law_is_one_delta_replayable_and_scaled():
    base = _ladder_config(
        train_unique_examples=50_000,
        train_presentations=50_000,
    )
    default = make_ladder_dataset(base)
    explicit_control = make_ladder_dataset(
        replace(base, coordinate_amplitude_law="gaussian")
    )
    assert default.stream_digest == explicit_control.stream_digest
    assert torch.equal(default.sample(512).x, explicit_control.sample(512).x)

    heavy = make_ladder_dataset(replace(base, coordinate_amplitude_law="student_t_df3"))
    replay = make_ladder_dataset(
        replace(base, coordinate_amplitude_law="student_t_df3")
    )
    gaussian_batch = default.sample(50_000)
    heavy_batch = heavy.sample(50_000)
    replay_batch = replay.sample(50_000)
    partitioned = list(heavy.batches(7_777, stop=50_000))
    assert torch.equal(heavy_batch.x, replay_batch.x)
    assert torch.equal(heavy_batch.coordinates, replay_batch.coordinates)
    assert torch.equal(
        heavy_batch.x,
        torch.cat([batch.x for batch in partitioned]),
    )
    assert torch.equal(
        heavy_batch.active,
        torch.cat([batch.active for batch in partitioned]),
    )
    assert torch.equal(heavy_batch.active, gaussian_batch.active)
    assert torch.equal(heavy_batch.event_factor, gaussian_batch.event_factor)
    assert torch.equal(heavy.contribution_maps, default.contribution_maps)
    assert not torch.equal(heavy_batch.coordinates, gaussian_batch.coordinates)

    rank = base.base_rank
    gaussian_coordinates = (
        gaussian_batch.coordinates[:, 0, :rank] * math.sqrt(rank)
    ).flatten()
    heavy_coordinates = (
        heavy_batch.coordinates[:, 0, :rank] * math.sqrt(rank)
    ).flatten()
    assert float(heavy_coordinates.square().mean()) == pytest.approx(1.0, abs=0.2)
    assert torch.quantile(heavy_coordinates.abs(), 0.999) > (
        1.5 * torch.quantile(gaussian_coordinates.abs(), 0.999)
    )
    assert heavy.ground_truth["student_t_degrees_of_freedom"] == 3
    assert (
        heavy.protocol["sampling"]["student_t_construction"]
        == "elliptical_normal_over_sqrt_chi_squared_df3"
    )
    assert (
        heavy.protocol["configuration"]["coordinate_amplitude_law"] == "student_t_df3"
    )
    assert {factor.family for factor in heavy.factors} == {"student_t_df3_subspace"}


def test_ladder_paired_overlap_has_realized_30_degree_principal_angles():
    base = _ladder_config(n_sites=4)
    uncontrolled = make_ladder_dataset(base)
    paired = make_ladder_dataset(replace(base, factor_subspace_overlap="paired_30deg"))
    control_batch = uncontrolled.sample(128)
    paired_batch = paired.sample(128)
    assert torch.equal(paired_batch.active, control_batch.active)
    assert torch.equal(paired_batch.coordinates, control_batch.coordinates)
    assert not torch.equal(paired.contribution_maps, uncontrolled.contribution_maps)

    expected = torch.full_like(paired.realized_pair_principal_angles_degrees, 30.0)
    assert torch.allclose(
        paired.realized_pair_principal_angles_degrees,
        expected,
        atol=2e-5,
    )
    for pair_index, pair in enumerate(paired.factor_overlap_pairs):
        first, second = int(pair[0]), int(pair[1])
        singular_values = torch.linalg.svdvals(
            paired.base_frames[first].to(torch.float64).T
            @ paired.base_frames[second].to(torch.float64)
        ).clamp(0.0, 1.0)
        realized = torch.rad2deg(torch.acos(singular_values)).sort().values
        assert torch.allclose(
            realized,
            paired.realized_pair_principal_angles_degrees[pair_index],
            atol=1e-10,
        )

        # The rank-1 site-map family uses the canonical frame at every site;
        # the realized contribution subspaces therefore carry the same angle.
        for site in range(base.n_sites):
            first_map = paired.contribution_maps[first, site, :, : base.base_rank].to(
                torch.float64
            )
            second_map = paired.contribution_maps[second, site, :, : base.base_rank].to(
                torch.float64
            )
            first_q = torch.linalg.qr(first_map, mode="reduced").Q
            second_q = torch.linalg.qr(second_map, mode="reduced").Q
            site_angles = torch.rad2deg(
                torch.acos(torch.linalg.svdvals(first_q.T @ second_q).clamp(0.0, 1.0))
            )
            assert torch.allclose(
                site_angles,
                torch.full_like(site_angles, 30.0),
                atol=2e-5,
            )

    truth = paired.ground_truth
    assert truth["factor_overlap_pairs"] is paired.factor_overlap_pairs
    assert truth["target_pair_principal_angle_degrees"] == 30.0
    assert (
        truth["principal_angle_reference"]
        == "full_block_dim_canonical_base_frames_before_rank_truncation"
    )
    assert truth["realized_pair_principal_angles_degrees"] is (
        paired.realized_pair_principal_angles_degrees
    )
    assert paired.protocol["configuration"]["factor_subspace_overlap"] == "paired_30deg"
    assert paired.protocol["sampling"]["factor_overlap_pairs"] == (
        paired.factor_overlap_pairs.tolist()
    )


@pytest.mark.parametrize("n_sites", (1, 4))
def test_ladder_orthogonal_truth_is_exact_and_marks_multisite_claims(n_sites: int):
    dataset = make_ladder_dataset(
        _ladder_config(
            n_sites=n_sites,
            d_model=24,
            n_factors=6,
            block_dim=4,
            base_rank=2,
            factor_subspace_overlap="orthogonal",
            site_map_rank_family="independent",
        )
    )
    for site in range(n_sites):
        columns = dataset.site_map_bases[:, site].permute(1, 0, 2).reshape(24, 24)
        assert torch.allclose(
            columns.T @ columns,
            torch.eye(24),
            atol=2e-5,
        )
    expected = torch.full_like(
        dataset.realized_pair_principal_angles_degrees, 90.0
    )
    assert torch.allclose(
        dataset.realized_pair_principal_angles_degrees,
        expected,
        atol=2e-5,
    )
    assert dataset.ground_truth["target_pair_principal_angle_degrees"] == 90.0
    assert bool(dataset.ground_truth["shared_feature_claim_eligible"]) is (
        n_sites > 1
    )


def test_ladder_one_factor_contract_has_empty_pair_diagnostics():
    dataset = make_ladder_dataset(
        _ladder_config(
            n_sites=4,
            d_model=2,
            n_factors=1,
            block_dim=2,
            base_rank=2,
            active_per_example=1,
            factor_subspace_overlap="orthogonal",
            site_map_rank_family="independent",
        )
    )
    batch = dataset.sample(32)
    assert batch.active.all()
    assert dataset.factor_overlap_pairs.shape == (0, 2)
    assert dataset.coactivation_group_probabilities.shape == (0,)
    assert dataset.realized_pair_principal_angles_degrees.shape == (0, 2)


def test_ladder_overlap_geometry_is_independent_of_coactivation_control():
    base = _ladder_config(factor_subspace_overlap="paired_30deg")
    independent = make_ladder_dataset(base)
    coactivated = make_ladder_dataset(replace(base, coactivation_probability=0.5))
    assert torch.equal(
        independent.factor_overlap_pairs, coactivated.factor_overlap_pairs
    )
    assert torch.equal(independent.base_frames, coactivated.base_frames)
    assert torch.equal(
        independent.realized_pair_principal_angles_degrees,
        coactivated.realized_pair_principal_angles_degrees,
    )
    assert not torch.equal(
        independent.sample(512).active,
        coactivated.sample(512).active,
    )


@pytest.mark.parametrize(
    "overrides",
    (
        {"site_map_rank_family": "rank3"},
        {"site_presence_span": "three"},
        {"feature_frequency": "zipf_alpha_2"},
        {"coactivation_probability": 0.25},
        {"coordinate_amplitude_law": "laplace"},
        {"factor_subspace_overlap": "paired_45deg"},
        {
            "factor_subspace_overlap": "orthogonal",
            "n_factors": 4,
            "block_dim": 3,
            "d_model": 11,
        },
        {
            "factor_subspace_overlap": "paired_30deg",
            "d_model": 5,
            "block_dim": 3,
        },
        {"factor_subspace_overlap": "paired_30deg", "n_factors": 1},
        {"n_factors": 5, "coactivation_probability": 0.5},
        {"active_per_example": 1, "coactivation_probability": 0.5},
    ),
)
def test_ladder_new_axis_validation_rejects_ambiguous_protocols(overrides):
    with pytest.raises(ValueError):
        replace(_ladder_config(), **overrides)


def test_all_protocols_have_canonical_lineage_rationale_counts_and_digest():
    datasets = (
        make_fel_dataset(_fel_config()),
        make_ladder_dataset(_ladder_config()),
    )
    for dataset in datasets:
        protocol = dataset.protocol
        assert protocol["schema"] == "block-crosscoder.phase1.synthetic.v1"
        assert protocol["lineage"]["classification"] in {
            "exact",
            "adapted",
            "novel",
        }
        assert protocol["rationale"]
        assert protocol["counts"] == {
            "unique_examples": dataset.unique_examples,
            "presentations": dataset.presentations,
            "presentation_order": "deterministic_epoch_shuffle",
            "shuffle_algorithm": "splitmix64_seeded_affine_bijection_v1",
            "repeat_semantics": "presentations with the same example_id are exact",
        }
        assert protocol["stream_digest_kind"].startswith("sha256-")
        assert len(protocol["stream_digest"]) == 64
        int(protocol["stream_digest"], 16)
        assert dataset.ground_truth["contribution_maps"] is dataset.contribution_maps
        assert len(dataset.ground_truth["factors"]) == len(dataset.factors)
