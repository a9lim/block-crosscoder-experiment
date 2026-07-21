import hashlib
import json
from dataclasses import replace

import pytest

from block_crosscoder_experiment.campaign import Campaign
from block_crosscoder_experiment.cli.run_cell import validate_cell_config
from block_crosscoder_experiment.runtime_limits import (
    EVALUATION_CONCORDANCE_BLOCK_CHUNK,
    EVALUATION_REDUCTION_TOKEN_CHUNK,
    TRUSTED_DECODE_Q_CHUNK,
)
from block_crosscoder_experiment.studies import (
    ANTHROPIC_ARCHITECTURE,
    BRIDGE_RECIPES,
    CONTROL_RECIPES,
    DECODER_WEIGHTED_BATCHTOPK_BRIDGE,
    ESTIMATOR_VERSION,
    PAPER_RECIPES,
    PHASE3_COMPUTE_CEILING_FLOPS,
    PHASE3_PARAMETER_CEILING,
    PHASE3_PANEL_SLOTS,
    PHASE3_PRODUCTION_STABILITY_TOKENS,
    PHASE3_PROVISIONED_STORAGE_BYTES,
    PHASE3_RUNTIME_CEILING_SECONDS,
    PHASE3_STORAGE_CEILING_BYTES,
    PHASE3_TRAINING_TOKEN_CEILING,
    RECIPES,
    RELEASE_DIAGNOSTIC_RECIPES,
    REQUIRED_CELL_DECISIONS,
    Budget,
    BudgetExceeded,
    CellSpec,
    Decision,
    DecisionDomain,
    FrozenPanelDecision,
    FrozenPanelEntry,
    FrozenSelection,
    Lineage,
    Phase,
    Phase1Blueprint,
    Phase2Blueprint,
    Phase3Blueprint,
    SelectionPolicy,
    StudyError,
    StudyPlan,
    _evaluation_workspace_bytes,
    adapted,
    build_phase1_blueprint,
    build_phase1_plan,
    build_phase2_blueprint,
    build_phase2_plan,
    build_phase3_blueprint,
    build_phase3_plan,
    build_plan,
    engineering,
    estimate_cell,
    estimate_plan,
    exact,
    materialize_child_plan,
    materialize_family_child_plan,
    materialize_family_revisit_plan,
    merge_decisions,
    resolved_candidate_execution_signature,
)


def _replace_decision(cell, name, value):
    return replace(
        cell,
        decisions=tuple(
            replace(decision, value=value) if decision.name == name else decision
            for decision in cell.decisions
        ),
    )


def _candidate_groups(stage):
    groups = {}
    for cell in stage.cells:
        groups.setdefault(cell.candidate_id, []).append(cell)
    return {
        candidate_id: tuple(sorted(cells, key=lambda cell: cell.seed))
        for candidate_id, cells in groups.items()
    }


def _eligible_groups(stage):
    policy = stage.selection_policy
    assert policy is not None
    return tuple(
        cells
        for _, cells in sorted(_candidate_groups(stage).items())
        if not policy.eligible_recipe_names
        or cells[0].recipe_name in policy.eligible_recipe_names
        if all(cell.decision_map["qualification.promotable"] is True for cell in cells)
    )


def _hash(label):
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


def _freeze(stage, cells=None, *, reverse=False):
    policy = stage.selection_policy
    assert policy is not None
    selected = _eligible_groups(stage)[0] if cells is None else tuple(cells)
    supplied = tuple(reversed(selected)) if reverse else tuple(selected)
    metrics = tuple(0.5 + index for index in range(len(supplied)))
    qualifications = tuple(_hash(cell.cell_id) for cell in supplied)
    return FrozenSelection.from_cells(
        policy,
        supplied,
        metrics,
        qualifications,
        _hash("complete-ranked-universe:" + stage.name),
    )


def _freeze_with_policy(stage, policy, cells, *, universe_label=None):
    selected = tuple(cells)
    return FrozenSelection.from_cells(
        policy,
        selected,
        tuple(0.5 + index for index in range(len(selected))),
        tuple(_hash(cell.cell_id) for cell in selected),
        _hash(
            universe_label
            or (
                "complete-ranked-universe:"
                + stage.name
                + ":"
                + selected[0].candidate_id
            )
        ),
    )


def _materialize_all(blueprint, prefix):
    plan = prefix
    while plan.stages[-1].selection_policy is not None:
        plan = materialize_child_plan(plan, blueprint, _freeze(plan.stages[-1]))
    return plan


def _panel_decision(blueprint):
    phase2_blueprint = build_phase2_blueprint()
    phase2 = _materialize_all(phase2_blueprint, build_phase2_plan())
    anchors = _candidate_groups(phase2.stages[0])
    comparator_families = {
        family.name: family for family in phase2_blueprint.comparator_families
    }
    confirmation_groups = _candidate_groups(phase2.stages[-1])
    finalist_cells = next(
        cells
        for cells in confirmation_groups.values()
        if cells[0].decision_map["data.normalization"] == "scalar_rms"
    )
    entries = []
    for index, slot in enumerate(blueprint.panel_slots):
        if slot.role == "selected_finalist":
            selection_ids = finalist_cells[0].decision_map[
                "selection.upstream_selection_ids"
            ]
            entries.append(
                FrozenPanelEntry.from_cells(
                    panel_slot=slot.name,
                    role=slot.role,
                    source_cells=finalist_cells,
                    selection_ids=selection_ids,
                    qualification_sha256s=tuple(
                        _hash(f"qualification:finalist:{cell.seed}")
                        for cell in finalist_cells
                    ),
                    confirmation_sha256s=tuple(
                        _hash(f"confirmation:finalist:{cell.seed}")
                        for cell in finalist_cells
                    ),
                )
            )
            continue
        family = comparator_families[slot.comparator_family_name]
        root_cells = next(
            cells
            for cells in anchors.values()
            if cells[0].recipe_name == family.root_recipe_name
        )
        derived_recipe_id = (
            "derived-recipe:"
            + hashlib.sha256(f"family:{slot.name}".encode()).hexdigest()
        )
        comparator_cells = tuple(
            replace(
                cell,
                stage=f"family_{slot.name}_top2_revisit_16m",
                recipe_name=f"derived_family_{slot.name}_revisit_winner",
                recipe_id=derived_recipe_id,
                decisions=merge_decisions(
                    cell.decisions,
                    (
                        engineering(
                            "selection.comparator_family_name",
                            slot.comparator_family_name,
                            rationale="test fixture binds the calibrated family",
                        ),
                        engineering(
                            "selection.comparator_family_blueprint_id",
                            family.family_id,
                            rationale="test fixture binds the family blueprint",
                        ),
                        engineering(
                            "selection.family_root_recipe_id",
                            family.root_recipe_id,
                            rationale="test fixture binds the family root lineage",
                        ),
                    ),
                ),
            )
            for cell in root_cells
        )
        entries.append(
            FrozenPanelEntry.from_cells(
                panel_slot=slot.name,
                role=slot.role,
                source_cells=comparator_cells,
                selection_ids=(
                    "selection:"
                    + hashlib.sha256(
                        f"family-selection:{slot.name}".encode()
                    ).hexdigest(),
                ),
                qualification_sha256s=tuple(
                    _hash(f"qualification:{slot.name}:{cell.seed}")
                    for cell in comparator_cells
                ),
            )
        )
    return FrozenPanelDecision(
        source_phase2_plan_id=phase2.plan_id,
        source_phase2_blueprint_id=phase2_blueprint.blueprint_id,
        phase2_campaign_manifest_sha256=_hash("phase2-campaign"),
        selection_universe_sha256=_hash("phase2-universe"),
        entries=tuple(entries),
    )


def test_lineage_is_mandatory_and_scientific_deltas_require_ablation():
    with pytest.raises(StudyError, match="unclassified lineage"):
        Decision(  # type: ignore[arg-type]
            "model.choice", "x", "exact", DecisionDomain.SCIENTIFIC, citation="paper"
        )
    with pytest.raises(StudyError, match="requires a citation"):
        Decision("model.choice", "x", Lineage.EXACT, DecisionDomain.SCIENTIFIC)
    with pytest.raises(StudyError, match="requires an ablation"):
        Decision(
            "model.choice",
            "x",
            Lineage.NOVEL,
            DecisionDomain.SCIENTIFIC,
            rationale="new hypothesis",
        )
    with pytest.raises(StudyError, match="deeply immutable JSON"):
        engineering("runtime.bad", [1, 2], rationale="mutable")  # type: ignore[arg-type]


def test_scope_registries_exclude_dsf_dfc_and_model_diffing_recipes():
    assert len(PAPER_RECIPES) == 8
    assert len(CONTROL_RECIPES) == 4
    assert set(BRIDGE_RECIPES) == {DECODER_WEIGHTED_BATCHTOPK_BRIDGE.name}
    assert set(RELEASE_DIAGNOSTIC_RECIPES) == {
        "bsf_group_lasso_released_paper_mode_drift",
        "sasa_released_code_drift",
    }
    executable_names = " ".join(RECIPES).lower()
    assert "dfc" not in executable_names
    assert "dsf" not in executable_names
    assert "minder_same_model" not in executable_names
    bridge = DECODER_WEIGHTED_BATCHTOPK_BRIDGE.decisions
    values = {decision.name: decision.value for decision in bridge}
    assert (
        values["protocol.claim_scope"]
        == "adapted_crosslayer_mechanics_not_model_diffing"
    )
    assert values["model.decoder_norm_geometry"] == "sum_l2"
    assert values["objective.auxiliary"] == "none"
    sasa_release = {
        decision.name: decision.value
        for decision in RELEASE_DIAGNOSTIC_RECIPES["sasa_released_code_drift"].decisions
    }
    assert sasa_release["objective.regularizer"] == "decoder_nuclear"
    assert sasa_release["objective.regularizer_coefficient"] == 100.0
    assert sasa_release["objective.regularizer_coefficient_mode"] == "absolute"
    assert sasa_release["objective.regularizer_target_initial_ratio"] is None
    assert sasa_release["qualification.promotable"] is False
    anthro = {decision.name: decision for decision in ANTHROPIC_ARCHITECTURE.decisions}
    assert anthro["model.decoder_norm_geometry"].value == "sum_l2"
    assert anthro["model.decoder_norm_geometry"].lineage is Lineage.EXACT
    paper_cell = build_phase1_plan().stages[0].cells[0]
    qualification = {decision.name: decision for decision in paper_cell.decisions}
    for name in (
        "qualification.profile",
        "qualification.phase1_identification_thresholds",
        "qualification.threshold_basis",
        "qualification.phase1_threshold_sensitivity",
    ):
        assert qualification[name].lineage is Lineage.NOVEL
        assert qualification[name].citation is None
    assert qualification["qualification.phase1_threshold_sensitivity"].value


def test_phase1_blueprint_is_an_honest_conditional_one_factor_campaign():
    blueprint = build_phase1_blueprint()
    prefix = build_phase1_plan()
    assert blueprint.projected_cells == 198
    assert len(prefix.cells) == 51
    assert [(stage.name, len(stage.cells)) for stage in prefix.stages] == [
        ("paper_anchors", 24),
        ("representable_controls", 12),
        ("fusion_identification", 6),
        ("dgp_identification_screen", 9),
    ]
    assert [
        (round_spec.name, len(round_spec.variants)) for round_spec in blueprint.rounds
    ] == [
        ("capacity_identification", 8),
        ("retraction_identification", 2),
        ("site_factorization_identification", 5),
        ("site_mask_fusion_control_identification", 3),
        ("site_masking_identification", 6),
        ("selection_score_identification", 6),
        ("selector_identification", 2),
        ("robustness_confirmation", 17),
    ]
    capability_rounds = tuple(
        round_spec
        for round_spec in blueprint.rounds
        if round_spec.role == "capability_panel"
    )
    assert {round_spec.name for round_spec in capability_rounds} == {
        "capacity_identification",
        "retraction_identification",
        "site_factorization_identification",
        "site_mask_fusion_control_identification",
        "site_masking_identification",
        "selection_score_identification",
        "selector_identification",
    }
    for round_spec in capability_rounds:
        assert round_spec.advancement == "fixed_carrier"
        assert round_spec.fixed_carrier_variant is not None
        for variant in round_spec.variants:
            promotable = next(
                (
                    decision.value
                    for decision in variant.decisions
                    if decision.name == "qualification.promotable"
                ),
                None,
            )
            if variant.name == round_spec.fixed_carrier_variant:
                assert promotable is not False
            else:
                assert promotable is False
    score_round = next(
        round_spec
        for round_spec in blueprint.rounds
        if round_spec.name == "selection_score_identification"
    )
    assert score_round.role == "capability_panel"
    assert score_round.advancement == "fixed_carrier"
    assert score_round.fixed_carrier_variant == "score_decoded_energy"
    confirmation = blueprint.rounds[-1]
    assert confirmation.role == "confirmation"
    assert confirmation.advancement == "none"
    assert confirmation.selection_policy is None
    assert all(
        stage.gate is None or stage.gate.basis == "integrity_complete"
        for stage in prefix.stages[:-1]
    )
    restored = Phase1Blueprint.from_manifest(
        json.loads(json.dumps(blueprint.to_manifest()))
    )
    assert restored == blueprint


def test_scientific_seed_contracts_reject_noncanonical_tuples_and_keep_counts():
    phase1 = build_phase1_blueprint()
    phase2 = build_phase2_blueprint()
    assert phase1.seeds == (0, 1, 2)
    assert phase1.projected_cells == 198
    assert phase2.seeds == (0, 1)
    assert phase2.declared_cell_ceiling == 414

    for builder in (build_phase1_blueprint, build_phase1_plan):
        with pytest.raises(StudyError, match="exact preregistered seeds"):
            builder(seeds=(0,))
    for builder in (build_phase2_blueprint, build_phase2_plan):
        with pytest.raises(StudyError, match="exact preregistered seeds"):
            builder(seeds=(0,))


def test_phase1_selection_uses_qualified_truth_margin_and_untouched_confirmation():
    blueprint = build_phase1_blueprint(seeds=(0, 1), smoke=True)
    plan = _materialize_all(blueprint, build_phase1_plan(seeds=(0, 1), smoke=True))
    assert len(plan.cells) == blueprint.projected_cells == 132
    assert all(stage.selection_policy is not None for stage in plan.stages[3:-1])
    assert plan.stages[-1].selection_policy is None
    for stage in plan.stages[3:-1]:
        policy = stage.selection_policy
        assert policy.metric_path == "validation.phase1_identification_margin"
        assert policy.selection_score == "minimum_normalized_identification_margin"
        assert policy.require_qualification is True
        assert policy.require_scientific_outcome_pass is True
    rounds_by_name = {round_spec.name: round_spec for round_spec in blueprint.rounds}
    for stage in plan.stages:
        round_spec = rounds_by_name.get(stage.name)
        if round_spec is None or round_spec.role != "capability_panel":
            continue
        promotable = {
            cell.recipe_name.removeprefix(f"derived_{stage.name}_"): cell.decision_map[
                "qualification.promotable"
            ]
            for cell in stage.cells
            if cell.seed == 0
        }
        assert promotable[round_spec.fixed_carrier_variant] is True
        assert all(
            value is False
            for variant, value in promotable.items()
            if variant != round_spec.fixed_carrier_variant
        )
    final = plan.stages[-1]
    assert {cell.decision_map["evaluation.split"] for cell in final.cells} == {
        "confirmation"
    }
    for cell in final.cells:
        ranges = dict(
            (item[0], (item[1], item[2]))
            for item in cell.decision_map["data.synthetic_split_ranges"]
        )
        assert ranges["development"][1] <= ranges["confirmation"][0]
        assert (
            cell.decision_map["random.eval_data_seed"]
            != cell.decision_map["random.confirmation_data_seed"]
        )
        for role in (
            "factor_calibration",
            "calibration",
            "development",
            "confirmation",
        ):
            assert cell.decision_map[f"data.synthetic_{role}_examples"] == 64
            assert ranges[role][1] - ranges[role][0] == 64
    robustness = {
        cell.decision_map["factor.robustness"]: cell.decision_map
        for cell in final.cells
        if cell.seed == 0
    }
    assert robustness["baseline"]["data.coordinate_amplitude_law"] == "gaussian"
    assert robustness["baseline"]["data.factor_subspace_overlap"] == "uncontrolled"
    assert (
        robustness["student_t_df3_coordinates"]["data.coordinate_amplitude_law"]
        == "student_t_df3"
    )
    assert (
        robustness["paired_subspaces_30deg"]["data.factor_subspace_overlap"]
        == "paired_30deg"
    )


def test_phase1_non_smoke_sample_roles_are_separate_and_executable():
    cell = build_phase1_plan().cells[0]
    values = cell.decision_map
    ranges = {
        role: (start, stop)
        for role, start, stop in values["data.synthetic_split_ranges"]
    }
    expected = {
        "factor_calibration": 50_000,
        "calibration": 100_000,
        "development": 100_000,
        "confirmation": 100_000,
    }
    assert tuple(ranges) == tuple(expected)
    previous_stop = 0
    for role, count in expected.items():
        start, stop = ranges[role]
        assert start == previous_stop
        assert stop - start == count
        assert values[f"data.synthetic_{role}_examples"] == count
        previous_stop = stop
    assert values["data.unique_tokens"] == 300_000
    assert previous_stop == 350_000


def test_phase2_blueprint_has_main_chain_and_independent_family_calibration_chains():
    blueprint = build_phase2_blueprint()
    prefix = build_phase2_plan()
    assert len(prefix.cells) == 18
    expected = (
        len(prefix.cells)
        + sum(len(item.variants) * len(blueprint.seeds) for item in blueprint.rounds)
        + sum(
            family.projected_candidates * len(blueprint.seeds)
            for family in blueprint.comparator_families
        )
    )
    assert blueprint.declared_cell_ceiling == expected == 414
    assert [(item.name, len(item.variants)) for item in blueprint.rounds] == [
        ("architecture_4m", 5),
        ("capacity_4m", 9),
        ("site_factorization_4m", 5),
        ("site_masking_4m", 7),
        ("site_factorization_revisit_4m", 5),
        ("hard_selector_score_interaction_4m", 6),
        ("group_threshold_method_4m", 4),
        ("learning_rate_4m", 4),
        ("batch_size_4m", 4),
        ("warmup_4m", 3),
        ("schedule_4m", 4),
        ("learning_rate_revisit_4m", 4),
        ("regularization_16m", 8),
        ("auxiliary_16m", 6),
        ("confirmation_16m", 5),
    ]
    assert blueprint.source_phase1_decision_id == "unbound-preview"
    assert blueprint.phase1_transfer_id == "unbound-preview"
    roots = {
        cells[0].recipe_name: cells
        for cells in _candidate_groups(prefix.stages[0]).values()
    }
    inherited_root = roots["phase1_contract_bsc"]
    assert all(
        cell.decision_map["provenance.phase1_decision_id"] == "unbound-preview"
        and cell.decision_map["provenance.phase1_transfer_id"] == "unbound-preview"
        and cell.decision_map["qualification.promotable"] is True
        for cell in inherited_root
    )
    source_only = roots["phase1_contract_source_only_control"]
    assert all(
        cell.decision_map["model.encoder_fusion"] == "source"
        and cell.decision_map["qualification.promotable"] is False
        for cell in source_only
    )
    assert prefix.stages[0].selection_policy.eligible_recipe_names == (
        "phase1_contract_bsc",
    )
    assert all(
        round_spec.role == "phase_local_tuning"
        and round_spec.advancement == "empirical_selection"
        for round_spec in blueprint.rounds[:-1]
    )
    assert blueprint.rounds[-1].role == "confirmation"
    assert blueprint.rounds[-1].advancement == "none"
    assert blueprint.rounds[-1].selection_policy is None
    families = {family.name: family for family in blueprint.comparator_families}
    assert set(families) == {
        "bsc_shared_coordinates",
        "bsf_grassmannian",
        "bsf_group_lasso",
        "sasa",
        "anthropic_dense_l1",
        "decoder_weighted_batchtopk",
        "scalar_relu_batchtopk",
    }
    for family in families.values():
        assert family.root_selection_policy.eligible_recipe_names == (
            family.root_recipe_name,
        )
        assert family.revisit.top_k == 2
        assert family.revisit.nomination_policy.retain_count == 2
        assert family.revisit.selection_policy.retain_count == 1
        assert set(family.revisit.source_rounds) == {
            round_spec.name for round_spec in family.rounds
        }
        assert "global optimum" in family.revisit.order_sensitivity_disclaimer
        assert family.root_selection_policy.require_sharing_guard is False
        assert family.revisit.nomination_policy.require_sharing_guard is False
        assert family.revisit.selection_policy.require_sharing_guard is False
        assert all(
            not round_spec.selection_policy.require_sharing_guard
            for round_spec in family.rounds
        )
        learning_rate_round = next(
            round_spec
            for round_spec in family.rounds
            if round_spec.name.endswith("_learning_rate_4m")
        )
        assert [
            (
                variant.name,
                next(
                    decision.value
                    for decision in variant.decisions
                    if decision.name == "optimizer.learning_rate"
                ),
            )
            for variant in learning_rate_round.variants
        ] == [
            ("lr_3e_minus_5", 3e-5),
            ("lr_1e_minus_4", 1e-4),
            ("lr_2e_minus_4", 2e-4),
            ("lr_3e_minus_4", 3e-4),
        ]
    assert any(
        "coefficient" in item.name for item in families["anthropic_dense_l1"].rounds
    )
    assert any("auxiliary" in item.name for item in families["sasa"].rounds)
    sasa_coefficient_round = next(
        item for item in families["sasa"].rounds if "coefficient" in item.name
    )
    assert len(sasa_coefficient_round.variants) == 4
    assert {
        decision.value
        for variant in sasa_coefficient_round.variants
        for decision in variant.decisions
        if decision.name == "objective.regularizer_target_initial_ratio"
    } == {0.0, 0.01, 0.03, 0.10}
    assert any(
        "batch" in item.name for item in families["scalar_relu_batchtopk"].rounds
    )
    restored = Phase2Blueprint.from_manifest(
        json.loads(json.dumps(blueprint.to_manifest()))
    )
    assert restored == blueprint


def test_phase2_materialization_elides_execution_equivalent_centers() -> None:
    blueprint = build_phase2_blueprint()
    plan = _materialize_all(
        blueprint,
        build_phase2_plan(),
    )
    stages = {stage.name: stage for stage in plan.stages}
    expected_unconditional = {
        "architecture_4m": ("parent_architecture", "selected_parent"),
        "capacity_4m": ("width_4", "selected_parent"),
        "site_masking_4m": ("site_mask_0", "selected_parent"),
        "learning_rate_4m": ("lr_1e_minus_4", "selected_parent"),
        "batch_size_4m": ("batch_4096", "selected_parent"),
        "schedule_4m": ("schedule_constant", "selected_parent"),
        "auxiliary_16m": ("no_auxiliary", "selected_parent"),
    }
    for stage_name, pair in expected_unconditional.items():
        stage = stages[stage_name]
        assert stage.execution_duplicate_policy == ("elide_by_resolved_value_signature")
        assert pair in stage.elided_execution_duplicates

    ignored_prefixes = (
        "factor.",
        "selection.",
        "protocol.",
        "qualification.",
        "provenance.",
    )
    for stage in plan.stages[1:-1]:
        if stage.execution_duplicate_policy != ("elide_by_resolved_value_signature"):
            continue
        signatures = [
            json.dumps(
                {
                    name: value
                    for name, value in cell.decision_map.items()
                    if not name.startswith(ignored_prefixes)
                },
                sort_keys=True,
            )
            for cell in stage.cells
        ]
        assert len(signatures) == len(set(signatures))
    declared_main_ceiling = len(blueprint.initial_stage.cells) + sum(
        len(blueprint.seeds) * len(round_spec.variants)
        for round_spec in blueprint.rounds
    )
    assert len(plan.cells) < declared_main_ceiling


def test_zero_mask_condition_elides_rank_mask_interaction_children() -> None:
    blueprint = build_phase2_blueprint()
    plan = build_phase2_plan()
    while plan.stages[-1].name != "site_masking_4m":
        plan = materialize_child_plan(
            plan,
            blueprint,
            _freeze(plan.stages[-1]),
        )
    mask_stage = plan.stages[-1]
    zero_mask = next(
        group
        for group in _eligible_groups(mask_stage)
        if group[0].decision_map["objective.encoder_site_mask_mode"] == "bernoulli"
        and group[0].decision_map["objective.encoder_site_mask_probability"] == 0.0
    )
    plan = materialize_child_plan(
        plan,
        blueprint,
        _freeze(mask_stage, zero_mask),
    )
    revisit = plan.stages[-1]
    assert revisit.name == "site_factorization_revisit_4m"
    assert len(revisit.cells) == len(blueprint.seeds)
    assert revisit.conditional_elision_reason == (
        "zero_bernoulli_mask_has_no_rank_mask_interaction"
    )
    assert revisit.elided_conditional_variants == (
        "site_rank_full",
        "site_rank_1",
        "site_rank_2",
        "site_rank_4",
    )


def test_every_comparator_family_chain_and_top_two_revisit_materializes():
    blueprint = build_phase2_blueprint(seeds=(0,), smoke=True)
    plan = build_phase2_plan(seeds=(0,), smoke=True)
    for family in blueprint.comparator_families:
        anchor_stage = next(
            stage for stage in plan.stages if stage.name == blueprint.initial_stage.name
        )
        anchors = _candidate_groups(anchor_stage)
        root_cells = next(
            cells
            for cells in anchors.values()
            if cells[0].recipe_name == family.root_recipe_name
        )
        root_selection = _freeze_with_policy(
            anchor_stage, family.root_selection_policy, root_cells
        )
        plan = materialize_family_child_plan(
            plan, blueprint, family.name, root_selection
        )
        while plan.stages[-1].name != family.rounds[-1].name:
            plan = materialize_family_child_plan(
                plan, blueprint, family.name, _freeze(plan.stages[-1])
            )
        finalists = tuple(_candidate_groups(plan.stages[-1]).values())[:2]
        assert len(finalists) == 2
        ranked = tuple(
            _freeze_with_policy(
                plan.stages[-1],
                family.revisit.nomination_policy,
                cells,
                universe_label="family-revisit-universe:" + family.name,
            )
            for cells in finalists
        )
        plan = materialize_family_revisit_plan(plan, blueprint, family.name, ranked)
        revisit = plan.stages[-1]
        assert revisit.name == family.revisit.name
        assert len(revisit.cells) == family.revisit.top_k
        assert {
            cell.decision_map["factor.family_revisit_rank"] for cell in revisit.cells
        } == {1, 2}
        for cell in revisit.cells:
            assert cell.recipe_id.startswith("derived-recipe:")
            assert cell.decision_map["selection.comparator_family_name"] == family.name
            assert (
                cell.decision_map["selection.comparator_family_blueprint_id"]
                == family.family_id
            )
            assert (
                cell.decision_map["selection.family_root_recipe_id"]
                == family.root_recipe_id
            )
            validate_cell_config(cell)
    assert {family.revisit.name for family in blueprint.comparator_families}.issubset(
        {stage.name for stage in plan.stages}
    )


def test_family_revisit_rejects_aliases_of_one_resolved_configuration():
    blueprint = build_phase2_blueprint(seeds=(0,), smoke=True)
    plan = build_phase2_plan(seeds=(0,), smoke=True)
    family = next(
        item
        for item in blueprint.comparator_families
        if item.name == "bsf_grassmannian"
    )
    anchor_stage = plan.stages[0]
    root_cells = next(
        cells
        for cells in _candidate_groups(anchor_stage).values()
        if cells[0].recipe_name == family.root_recipe_name
    )
    plan = materialize_family_child_plan(
        plan,
        blueprint,
        family.name,
        _freeze_with_policy(
            anchor_stage,
            family.root_selection_policy,
            root_cells,
        ),
    )
    while plan.stages[-1].name != family.rounds[-1].name:
        plan = materialize_family_child_plan(
            plan,
            blueprint,
            family.name,
            _freeze(plan.stages[-1]),
        )
    family_stages = {
        stage.name: stage
        for stage in plan.stages
        if stage.name in {item.name for item in family.rounds}
    }
    by_signature = {}
    for stage in family_stages.values():
        for cells in _candidate_groups(stage).values():
            by_signature.setdefault(
                resolved_candidate_execution_signature(cells), []
            ).append((stage, cells))
    aliases = next(items for items in by_signature.values() if len(items) >= 2)
    universe = "family-revisit-universe:" + family.name
    nominations = tuple(
        _freeze_with_policy(
            stage,
            family.revisit.nomination_policy,
            cells,
            universe_label=universe,
        )
        for stage, cells in aliases[:2]
    )
    assert len({item.candidate_id for item in nominations}) == 2
    with pytest.raises(StudyError, match="distinct resolved execution signatures"):
        materialize_family_revisit_plan(
            plan,
            blueprint,
            family.name,
            nominations,
        )


def test_coupled_geometry_and_decoder_weighted_bundles_are_complete():
    phase1 = build_phase1_blueprint()
    retraction_stage = next(
        stage for stage in phase1.rounds if stage.name == "retraction_identification"
    )
    retractions = {
        variant.name: {decision.name: decision.value for decision in variant.decisions}
        for variant in retraction_stage.variants
    }
    assert retractions["qr_retraction"]["model.decoder"] == "concatenated_stiefel"
    assert (
        retractions["symmetric_polar_retraction"]["model.decoder"]
        == "concatenated_stiefel_polar"
    )
    phase1_rounds = {round_spec.name: round_spec for round_spec in phase1.rounds}
    fusion_controls = {
        variant.name: {decision.name: decision.value for decision in variant.decisions}
        for variant in phase1_rounds["site_mask_fusion_control_identification"].variants
    }
    assert set(fusion_controls) == {
        "literal_sum_p0_parent",
        "literal_sum_p010_diagnostic",
        "availability_rescaled_sum_p010",
    }
    assert fusion_controls["literal_sum_p0_parent"]["model.encoder_fusion"] == "sum"
    assert (
        fusion_controls["literal_sum_p010_diagnostic"]["qualification.promotable"]
        is False
    )
    assert (
        fusion_controls["availability_rescaled_sum_p010"]["model.encoder_fusion"]
        == "availability_rescaled_sum"
    )
    scores = {
        decision.value
        for variant in phase1_rounds["selection_score_identification"].variants
        for decision in variant.decisions
        if decision.name == "model.selection_score"
    }
    assert scores == {
        "code_norm",
        "decoded_energy",
        "isolated_loss_decrease",
    }
    score_variants = {
        variant.name: {decision.name: decision.value for decision in variant.decisions}
        for variant in phase1_rounds["selection_score_identification"].variants
    }
    assert {name for name in score_variants if name.startswith("free_score_")} == {
        "free_score_code_norm",
        "free_score_decoded_energy",
        "free_score_isolated_loss_decrease",
    }
    assert all(
        values["model.decoder"] == "free_scale_controlled"
        for name, values in score_variants.items()
        if name.startswith("free_score_")
    )
    assert (
        phase1_rounds["selection_score_identification"].fixed_carrier_variant
        == "score_decoded_energy"
    )
    assert all(
        cell.decision_map["data.missing_probability"] == 0.0
        for cell in build_phase1_plan().cells
    )
    with pytest.raises(StudyError, match="synthetic whitening"):
        _replace_decision(
            build_phase1_plan().cells[0],
            "data.normalization",
            "whiten",
        )

    phase2 = build_phase2_blueprint()
    rounds = {round_spec.name: round_spec for round_spec in phase2.rounds}
    interaction = rounds["hard_selector_score_interaction_4m"]
    assert {variant.name for variant in interaction.variants} == {
        f"score_{score}__signed_{selector}"
        for score in ("code_norm", "decoded_energy", "isolated_loss_decrease")
        for selector in ("token_topk", "batchtopk")
    }
    assert interaction.source_stage == "site_factorization_revisit_4m"
    assert (
        interaction.selection_policy.default_parent_variant
        == "score_decoded_energy__signed_token_topk"
    )
    exact_interaction_parent = next(
        variant
        for variant in interaction.variants
        if variant.name == "score_decoded_energy__signed_token_topk"
    )
    assert exact_interaction_parent.delta_decision_names == ("factor.round_control",)
    assert rounds["site_factorization_revisit_4m"].source_stage == ("site_masking_4m")
    assert rounds["site_factorization_revisit_4m"].activation_condition == (
        "nonzero_or_structured_site_mask"
    )
    assert rounds["group_threshold_method_4m"].source_stage == (
        "hard_selector_score_interaction_4m"
    )
    architecture = {
        variant.name: variant for variant in rounds["architecture_4m"].variants
    }
    assert {
        decision.name: decision.value
        for decision in architecture["tied_grassmann_b4_polar"].decisions
    }["model.decoder"] == "concatenated_stiefel_polar"
    assert "relu_decoder_weighted_batchtopk" not in {
        item.name for item in rounds["group_threshold_method_4m"].variants
    }
    assert "decoder_weighted_residual_aux" not in {
        item.name for item in rounds["auxiliary_16m"].variants
    }
    assert {
        decision.name: decision.value
        for decision in DECODER_WEIGHTED_BATCHTOPK_BRIDGE.decisions
    }["model.decoder_norm_geometry"] == "sum_l2"
    factorization = {
        variant.name: {decision.name: decision.value for decision in variant.decisions}
        for variant in rounds["site_factorization_4m"].variants
    }
    assert factorization["selected_parent_carrier"] == {
        "factor.site_factorization": "selected_parent_carrier"
    }
    assert {
        values["model.site_rank"]
        for name, values in factorization.items()
        if name != "selected_parent_carrier"
    } == {
        None,
        1,
        2,
        4,
    }
    masking_probabilities = {
        decision.value
        for variant in rounds["site_masking_4m"].variants
        for decision in variant.decisions
        if decision.name == "objective.encoder_site_mask_probability"
    }
    assert masking_probabilities == {0.0, 0.02, 0.05, 0.10}
    masking_modes = {
        decision.value
        for variant in rounds["site_masking_4m"].variants
        for decision in variant.decisions
        if decision.name == "objective.encoder_site_mask_mode"
    }
    assert masking_modes == {
        "bernoulli",
        "exactly_one_hidden",
        "exactly_one_retained",
    }
    assert all(
        any(
            decision.name == "model.encoder_fusion"
            and decision.value == "availability_rescaled_sum"
            for decision in variant.decisions
        )
        for variant in rounds["site_masking_4m"].variants
        if variant.name != "selected_parent"
    )
    regularization = {
        variant.name: {decision.name: decision.value for decision in variant.decisions}
        for variant in rounds["regularization_16m"].variants
    }
    assert not any("site_profile" in name for name in regularization)
    map_arms = {
        name: values
        for name, values in regularization.items()
        if name.startswith("map_nuclear_initial_ratio_")
    }
    assert {
        values["objective.regularizer_target_initial_ratio"]
        for values in map_arms.values()
    } == {0.01, 0.03, 0.10}
    assert all(
        values["objective.regularizer_coefficient"] == 0.0
        and values["objective.regularizer_coefficient_mode"] == "initial_loss_ratio"
        and values["objective.regularizer_calibration_contract"]
        == "post_init_train_prefix_true_observation_fp32_v1"
        for values in map_arms.values()
    )
    decoder_diagnostics = {
        name: values
        for name, values in regularization.items()
        if name.startswith("decoder_nuclear_coefficient_")
    }
    assert {
        values["objective.regularizer_coefficient"]
        for values in decoder_diagnostics.values()
    } == {30.0, 100.0, 300.0}
    assert all(
        values["qualification.promotable"] is False
        for values in decoder_diagnostics.values()
    )
    anthro = next(
        cell
        for cell in build_phase2_plan().cells
        if cell.recipe_name == "anthropic_crosscoder_architecture_bridge"
    )
    assert anthro.decision_map["qualification.promotable"] is True
    assert anthro.decision_map["model.decoder_norm_geometry"] == "sum_l2"
    for cell in build_phase2_plan().cells:
        width = cell.decision_map["model.block_width"]
        assert cell.decision_map["model.groups"] == 8_192 // width
        assert cell.decision_map["model.active_blocks"] == 32 // width
    factorization_policy = rounds["site_factorization_4m"].selection_policy
    assert factorization_policy.required_control_variant == "selected_parent_carrier"
    assert factorization_policy.noninferiority_candidate_variant == "site_rank_full"
    assert factorization_policy.control_noninferiority_absolute_tolerance == 0.01
    assert factorization_policy.parsimony_order_variants == (
        "site_rank_1",
        "site_rank_2",
        "site_rank_4",
        "site_rank_full",
    )
    assert factorization_policy.parsimony_noninferiority_absolute_tolerance == 0.01
    assert factorization_policy.parsimony_reduction == ("per_seed_and_median_and_worst")
    assert factorization_policy.sharing_site_only_fvu_degradation_max == 0.02
    assert factorization_policy.sharing_leave_one_out_fvu_degradation_max == 0.02
    assert factorization_policy.sharing_support_iou_drop_max == 0.05
    assert factorization_policy.sharing_coordinate_concordance_min == 0.8
    assert factorization_policy.sharing_intersection_recall_min == 0.75
    assert factorization_policy.sharing_intersection_energy_coverage_min == 0.9
    assert dict(factorization_policy.sharing_guard_metric_paths)["all_site_fvu"] == (
        "selection_metrics.sharing_guard.all_site_fvu_mean"
    )
    assert factorization_policy.sharing_coordinate_concordance_definition == (
        "lin_decoder_gram_concordance_covariance_over_variance_plus_mean_offset"
    )
    assert factorization_policy.sharing_intersection_coverage_definition == (
        "intersection_full_decoded_energy_over_all_full_selected_energy"
    )
    assert factorization_policy.sharing_all_view_advantage_definition == (
        "site_only_heldout_fvu_minus_all_site_fvu_descriptive_only"
    )
    assert factorization_policy.sharing_fvu_absolute_max == 1.0
    assert factorization_policy.sharing_root_site_only_fvu_degradation_max == 0.02
    assert factorization_policy.sharing_root_leave_one_out_fvu_degradation_max == 0.02
    assert factorization_policy.threshold_basis.startswith("novel_preregistered")
    assert dict(factorization_policy.threshold_sensitivity)[
        "sharing_coordinate_concordance_min"
    ] == (0.5, 0.8, 0.9)
    assert (
        SelectionPolicy.from_dict(
            json.loads(json.dumps(factorization_policy.to_dict()))
        )
        == factorization_policy
    )
    for round_spec in phase2.rounds:
        if round_spec.split != "development" or round_spec.name in {
            "site_factorization_4m",
            "hard_selector_score_interaction_4m",
        }:
            continue
        policy = round_spec.selection_policy
        assert policy.default_parent_variant == "selected_parent"
        assert policy.minimum_effect_absolute == 0.002
        assert policy.minimum_effect_reduction == ("per_seed_and_median_and_worst")
        assert (
            SelectionPolicy.from_dict(json.loads(json.dumps(policy.to_dict())))
            == policy
        )
        parent = next(
            variant
            for variant in round_spec.variants
            if variant.name == "selected_parent"
        )
        assert parent.delta_decision_names == ("factor.round_control",)


def test_selection_and_child_evidence_are_hash_bound_and_seed_order_safe():
    stage = build_phase2_plan().stages[-1]
    policy = stage.selection_policy
    restored_policy = SelectionPolicy.from_dict(
        json.loads(json.dumps(policy.to_dict()))
    )
    assert restored_policy == policy
    selection = _freeze(stage, reverse=True)
    assert selection.seeds == (0, 1)
    assert selection.metric_values == (1.5, 0.5)
    restored = FrozenSelection.from_dict(json.loads(json.dumps(selection.to_dict())))
    assert restored == selection
    extended = materialize_child_plan(
        build_phase2_plan(), build_phase2_blueprint(), selection
    )
    for cell in extended.stages[-1].cells:
        values = cell.decision_map
        assert values["selection.id"] == selection.selection_id
        assert (
            values["selection.qualification_sha256s"] == selection.qualification_sha256s
        )
        assert (
            values["selection.universe_sha256"] == selection.selection_universe_sha256
        )
        assert values["selection.source_plan_id"] == build_phase2_plan().plan_id
        assert (
            values["selection.source_blueprint_id"]
            == build_phase2_blueprint().blueprint_id
        )


def test_every_stage_variant_materializes_and_adversarial_parent_routes_resolve():
    def materialize(blueprint, prefix, choices):
        plan = prefix
        while plan.stages[-1].selection_policy is not None:
            stage = plan.stages[-1]
            groups = _eligible_groups(stage)
            wanted = choices.get(stage.name)
            selected = groups[0]
            if wanted is not None:
                matches = [group for group in groups if wanted in group[0].name]
                assert len(matches) == 1, (stage.name, wanted)
                selected = matches[0]
            plan = materialize_child_plan(plan, blueprint, _freeze(stage, selected))
            for cell in plan.stages[-1].cells:
                assert REQUIRED_CELL_DECISIONS.issubset(cell.decision_map)
                validate_cell_config(cell)
        return plan

    phase1 = build_phase1_blueprint(seeds=(0,), smoke=True)
    materialize(phase1, build_phase1_plan(seeds=(0,), smoke=True), {})
    phase2 = build_phase2_blueprint(seeds=(0,), smoke=True)
    for choices in (
        {},
        {"architecture_4m": "tied_grassmann_b4_polar"},
        {
            "hard_selector_score_interaction_4m": (
                "score_isolated_loss_decrease__signed_batchtopk"
            )
        },
        {"group_threshold_method_4m": "group_soft_threshold_1e_minus_3"},
        {"regularization_16m": "map_nuclear_initial_ratio_0p03"},
        {"auxiliary_16m": "sasa_low_weight"},
    ):
        materialize(
            phase2,
            build_phase2_plan(seeds=(0,), smoke=True),
            choices,
        )


def test_phase3_is_a_blueprint_until_a_frozen_panel_decision_is_supplied():
    blueprint = build_phase3_blueprint()
    assert blueprint.projected_cells == 48
    assert blueprint.content_payload()["resource_contract"]["estimator"] == (
        ESTIMATOR_VERSION
    )
    assert blueprint.panel_slots[0].duplicate_policy == "fail"
    assert all(
        slot.duplicate_policy == "next_ranked_nonduplicate"
        for slot in blueprint.panel_slots[1:]
    )
    restored_blueprint = Phase3Blueprint.from_manifest(
        json.loads(json.dumps(blueprint.to_manifest()))
    )
    assert restored_blueprint == blueprint
    stale_estimator = json.loads(json.dumps(blueprint.to_manifest()))
    stale_estimator["resource_contract"]["estimator"] = (
        "dense-linear-memory-v3-q2-c512-t256"
    )
    with pytest.raises(StudyError, match="resource contract mismatch"):
        Phase3Blueprint.from_manifest(stale_estimator)
    with pytest.raises(StudyError, match="requires a frozen Phase-2 panel decision"):
        build_phase3_plan()
    decision = _panel_decision(blueprint)
    projected_configurations = Campaign._projected_scientific_configurations(
        decision,
        smoke=False,
    )
    assert set(projected_configurations) == {
        slot.name for slot in blueprint.panel_slots
    }
    restored_decision = FrozenPanelDecision.from_dict(
        json.loads(json.dumps(decision.to_dict()))
    )
    assert restored_decision == decision
    plan = build_phase3_plan(panel_decision=decision)
    assert len(plan.cells) == 48
    preflight = plan.stages[0]
    frozen_panel = plan.stages[1]
    assert preflight.name == "production_stability_preflight"
    assert len(preflight.cells) == PHASE3_PANEL_SLOTS == 8
    assert frozen_panel.name == "frozen_panel"
    assert len(frozen_panel.cells) == 40
    assert frozen_panel.depends_on == (preflight.name,)
    assert frozen_panel.gate is not None
    assert frozen_panel.gate.minimum_count == PHASE3_PANEL_SLOTS
    assert all(
        cell.decision_map["qualification.promotable"] is True
        and cell.decision_map["evaluation.split"] == "stability"
        and cell.decision_map["data.train_tokens"] == PHASE3_PRODUCTION_STABILITY_TOKENS
        and cell.decision_map["precision.preflight_contract"]
        == "fp32_bf16_initial_forward_v1"
        and cell.decision_map["precision.preflight_min_nonzero_rate_endpoints"] == 2
        and cell.decision_map["evaluation.fixed_rate_budgets_bits_per_token"]
        == (1024.0, 1536.0, 2048.0)
        and cell.decision_map["evaluation.fixed_rate_budget_scale_factor"] == 4.0
        and cell.decision_map["evaluation.fixed_rate_budget_scale_contract"]
        == "phase3_active_coordinate_ratio_128_over_32_v1"
        for cell in preflight.cells
    )
    assert all(
        entry.recipe_id.startswith("derived-recipe:") and entry.selection_ids
        for entry in decision.entries
        if entry.role != "selected_finalist"
    )
    assert {cell.decision_map["selection.id"] for cell in plan.cells} == {
        decision.panel_id
    }
    assert all(
        cell.decision_map["qualification.promotable"] is False
        and cell.decision_map["precision.preflight_contract"] == "not_applicable"
        and cell.decision_map["evaluation.fixed_rate_budgets_bits_per_token"]
        == (1024.0, 1536.0, 2048.0)
        for cell in frozen_panel.cells
    )
    assert {len(cell.decision_map["data.site_dims"]) for cell in plan.cells} == {4}
    assert {
        cell.decision_map["data.store_contract_version"] for cell in plan.cells
    } == {"activation-store-v3-single-view"}
    assert {cell.decision_map["data.generator_version"] for cell in plan.cells} == {
        "activation-store-v3-single-view"
    }
    finalist_entry = next(
        entry for entry in decision.entries if entry.role == "selected_finalist"
    )
    finalist_cell = next(
        cell
        for cell in frozen_panel.cells
        if cell.recipe_name == "phase3_selected_finalist"
    )
    source_values = finalist_entry.source_cells[0].decision_map
    production_values = finalist_cell.decision_map
    for name in (
        "model.block_width",
        "model.selector",
        "objective.regularizer",
        "objective.auxiliary",
        "optimizer.name",
    ):
        assert production_values[name] == source_values[name]


def test_phase3_storage_has_hard_headroom_and_seed_contract_fails_closed():
    blueprint = build_phase3_blueprint()
    decision = _panel_decision(blueprint)
    plan = build_phase3_plan(panel_decision=decision)
    estimate = estimate_plan(plan)
    assert estimate.storage_bytes <= PHASE3_STORAGE_CEILING_BYTES
    assert estimate.training_tokens == PHASE3_TRAINING_TOKEN_CEILING
    assert estimate.parameters <= PHASE3_PARAMETER_CEILING
    assert estimate.compute_flops <= PHASE3_COMPUTE_CEILING_FLOPS
    assert estimate.compute_flops / 20_000_000_000_000 <= PHASE3_RUNTIME_CEILING_SECONDS
    assert (
        PHASE3_PROVISIONED_STORAGE_BYTES - estimate.storage_bytes
    ) / PHASE3_PROVISIONED_STORAGE_BYTES >= 0.15
    with pytest.raises(StudyError, match="exact seeds 0,1,2,3,4"):
        build_phase3_blueprint(seeds=tuple(range(7)))


def test_every_cell_binds_capture_codec_init_random_and_endpoint_contracts():
    plans = (build_phase1_plan(), build_phase2_plan())
    for plan in plans:
        for cell in plan.cells:
            values = cell.decision_map
            assert REQUIRED_CELL_DECISIONS.issubset(values)
            assert values["qualification.endpoint_paths"] == (
                "native_training_rule",
                "saved_codec_deployment_rule",
            )
            assert values["qualification.require_saved_codec_validation"] is True
            assert values["codec.quantizer_bits"] == (
                (4, 6, 8) if cell.phase is Phase.PHASE1 else (2, 4, 6, 8, 12, 16)
            )
            assert (
                values["codec.packet_contract"]
                == "fixed_width_count_compact_block_id_amplitude_v1"
            )
            assert (
                values["codec.side_information_contract"]
                == "exact_deployable_saved_codec_bytes_v1"
            )
            assert values["codec.time_sharing_schedule_contract"] == (
                "not_applicable"
                if cell.phase is Phase.PHASE1
                else "balanced_global_token_counter_u64_v1"
            )
            assert (
                values["model.decoder_init_distribution"]
                == "gaussian_std_inverse_sqrt_d"
            )
            assert (
                values["model.decoder_init_preconditioning"]
                == "concatenated_gram_retraction"
            )
            assert values["model.selector_tie_break"] == "lowest_flat_index_at_cutoff"
            assert (
                len(
                    {
                        values["random.model_seed"],
                        values["random.structure_seed"],
                        values["random.train_data_seed"],
                        values["random.eval_data_seed"],
                        values["random.confirmation_data_seed"],
                    }
                )
                == 5
            )
            assert values["data.capture_contract"]


def test_real_capture_contracts_are_pinned_and_smoke_fit_counts_match_tiny_splits():
    phase2 = build_phase2_plan()
    assert {
        cell.decision_map["data.store_contract_version"] for cell in phase2.cells
    } == {"activation-store-v3-derived-views"}
    assert {cell.decision_map["data.generator_version"] for cell in phase2.cells} == {
        "activation-store-v3-derived-views"
    }
    assert {
        cell.decision_map["data.source_model_revisions"] for cell in phase2.cells
    } == {("607a30d783dfa663caf39e06633721c8d4cfcd7e",)}
    capture = dict(phase2.cells[0].decision_map["data.capture_contract"])
    assert (
        capture["model_loader"] == "transformer_lens_from_pretrained_no_processing_v1"
    )
    assert capture["transformer_lens_model_names"] == ("gpt2",)
    assert capture["tokenizer_vocab_sha256"].startswith("sha256:")
    assert capture["row_identity_columns"] == ("sequence", "position", "token_id")
    for builder in (build_phase1_plan, build_phase2_plan):
        plan = builder(seeds=(0,), smoke=True)
        for cell in plan.cells:
            values = cell.decision_map
            assert values["codec.minimum_active_events_per_block"] == 1
            assert values["codec.bootstrap_replicates"] == 32
            if values["data.normalization_fit_split"] != "not_applicable":
                assert values["data.normalization_fit_count"] == 64
            if values["model.encoder_scale_fit_split"] != "not_applicable":
                assert values["model.encoder_scale_fit_count"] == 64


def test_smoke_preserves_underlying_scientific_promotable_intent():
    for builder in (build_phase1_plan, build_phase2_plan):
        full = builder()
        smoke = builder(seeds=(0,), smoke=True)
        full_intent = {
            (cell.stage, cell.recipe_name): cell.decision_map[
                "qualification.promotable"
            ]
            for cell in full.cells
        }
        smoke_intent = {
            (cell.stage, cell.recipe_name): cell.decision_map[
                "qualification.promotable"
            ]
            for cell in smoke.cells
        }
        assert smoke_intent == full_intent
        assert all(cell.decision_map["runtime.smoke"] is True for cell in smoke.cells)
    assert set(full_intent.values()) == {False, True}
    assert {
        recipe_name
        for (_, recipe_name), promotable in full_intent.items()
        if promotable is False
    } == {"phase1_contract_source_only_control"}

    phase1_intent = {
        cell.decision_map["qualification.promotable"]
        for cell in build_phase1_plan(seeds=(0,), smoke=True).cells
    }
    assert phase1_intent == {False, True}


def test_cell_validation_rejects_hidden_or_incoherent_resolved_values():
    cell = build_phase1_plan().cells[0]
    incomplete = tuple(
        decision
        for decision in cell.decisions
        if decision.name != "codec.quantizer_bits"
    )
    with pytest.raises(StudyError, match="unresolved decisions"):
        replace(cell, decisions=incomplete)
    with pytest.raises(StudyError, match="unknown store normalization"):
        _replace_decision(cell, "data.normalization", "paper_default")
    with pytest.raises(StudyError, match="selector_tie_break"):
        _replace_decision(cell, "model.selector_tie_break", "runtime_default")
    with pytest.raises(StudyError, match="raw/effective"):
        _replace_decision(cell, "model.threshold_effective_init", 0.2)
    with pytest.raises(StudyError, match="distinct seeds"):
        _replace_decision(
            cell,
            "random.confirmation_data_seed",
            cell.decision_map["random.eval_data_seed"],
        )
    with pytest.raises(StudyError, match="clipping quantiles"):
        _replace_decision(cell, "codec.clip_lower_quantile", 1.0)


def test_manifests_are_deterministic_round_trip_and_tamper_evident():
    plan_a = build_phase1_plan(seeds=(7,), smoke=True)
    plan_b = build_plan("synthetic", seeds=(7,), smoke=True)
    assert plan_a.plan_id == plan_b.plan_id
    cell = plan_a.cells[0]
    assert (
        replace(cell, decisions=tuple(reversed(cell.decisions))).cell_id == cell.cell_id
    )
    assert CellSpec.from_manifest(json.loads(json.dumps(cell.to_manifest()))) == cell
    assert (
        StudyPlan.from_manifest(json.loads(json.dumps(plan_a.to_manifest()))) == plan_a
    )
    bad = cell.to_manifest()
    bad["candidate_id"] = "candidate:" + "0" * 64
    with pytest.raises(StudyError, match="candidate ID mismatch"):
        CellSpec.from_manifest(bad)


def test_resource_estimator_reuses_real_capture_and_budget_refuses_overrun():
    phase1_cell = build_phase1_plan(seeds=(0,), smoke=True).cells[0]
    phase1_estimate = estimate_cell(phase1_cell)
    assert phase1_estimate.estimator == (
        "dense-linear-memory-v6"
        f"-q{TRUSTED_DECODE_Q_CHUNK}"
        f"-c{EVALUATION_CONCORDANCE_BLOCK_CHUNK}"
        f"-t{EVALUATION_REDUCTION_TOKEN_CHUNK}"
    )
    assert phase1_estimate.storage_bytes == phase1_estimate.parameters * 16
    assert phase1_estimate.peak_vram_bytes > phase1_estimate.parameters * 28
    assert phase1_estimate.peak_host_ram_bytes >= 8 * 1024**3
    phase2 = build_phase2_plan(seeds=(0,), smoke=True)
    per_cell_store = sum(
        estimate_cell(cell).storage_bytes - estimate_cell(cell).parameters * 16
        for cell in phase2.cells
    )
    estimate = estimate_plan(phase2)
    plan_store = estimate.storage_bytes - sum(
        estimate_cell(cell).parameters * 16 for cell in phase2.cells
    )
    assert 0 < plan_store < per_cell_store
    one_derived_view = (
        estimate_cell(phase2.cells[0]).storage_bytes
        - estimate_cell(phase2.cells[0]).parameters * 16
    )
    assert plan_store == 2 * one_derived_view  # immutable raw + scalar-RMS view
    Budget(max_training_tokens=estimate.training_tokens).enforce(estimate)
    with pytest.raises(BudgetExceeded, match="training_tokens"):
        Budget(max_training_tokens=estimate.training_tokens - 1).enforce(estimate)
    with pytest.raises(BudgetExceeded, match="peak_vram_bytes"):
        Budget(max_peak_vram_bytes=estimate.peak_vram_bytes - 1).enforce(estimate)
    with pytest.raises(BudgetExceeded, match="peak_host_ram_bytes"):
        Budget(max_peak_host_ram_bytes=estimate.peak_host_ram_bytes - 1).enforce(
            estimate
        )


def test_evaluation_workspace_prices_dense_support_and_saturates_q_chunks():
    groups = 16 * EVALUATION_CONCORDANCE_BLOCK_CHUNK
    common = {
        "batch_tokens": 128,
        "groups": groups,
        "block_width": 2,
        "total_dim": 96,
        "operational_decoder_elements": groups * 2 * 96,
        "sites": 3,
        "selection_score": "decoded_energy",
    }
    one_q = _evaluation_workspace_bytes(quantizer_count=1, **common)
    saturated = _evaluation_workspace_bytes(
        quantizer_count=TRUSTED_DECODE_Q_CHUNK,
        **common,
    )
    extra_qs = _evaluation_workspace_bytes(
        quantizer_count=TRUSTED_DECODE_Q_CHUNK + 3,
        **common,
    )
    assert one_q > 0
    assert saturated >= one_q
    assert extra_qs == saturated
    by_score = {
        score: _evaluation_workspace_bytes(
            quantizer_count=TRUSTED_DECODE_Q_CHUNK,
            **{**common, "selection_score": score},
        )
        for score in (
            "code_norm",
            "decoder_weighted",
            "decoded_energy",
            "isolated_loss_decrease",
        )
    }
    base = by_score["code_norm"]
    assert by_score["decoder_weighted"] - base == groups * 4
    assert by_score["decoded_energy"] - base == groups * 2**2 * 4
    assert by_score["isolated_loss_decrease"] - base == 3 * groups * 2**2 * 4


def test_evaluation_workspace_prices_both_frozen_encoder_site_tensors():
    batch_tokens = 128
    groups = 256
    block_width = 2
    sites = 16
    total_dim = 16
    decoder_elements = sites * groups * block_width
    actual = _evaluation_workspace_bytes(
        batch_tokens=batch_tokens,
        groups=groups,
        block_width=block_width,
        total_dim=total_dim,
        operational_decoder_elements=decoder_elements,
        quantizer_count=1,
        sites=sites,
        selection_score="code_norm",
    )
    events = batch_tokens * groups
    latents = groups * block_width
    trusted_decode = (
        batch_tokens * 4
        + events * (32 + 12 * block_width)
        + events * block_width * 16
        + events * block_width * 8
        + (batch_tokens + 1) * 8
        + batch_tokens * total_dim * 4
        + decoder_elements * 4
    )
    output = 2 * batch_tokens * (
        total_dim * 4 + latents * 8 + groups * 5
    )
    concordance_groups = min(groups, EVALUATION_CONCORDANCE_BLOCK_CHUNK)
    concordance = (
        batch_tokens
        * concordance_groups
        * (8 * block_width + 4)
        * 8
    )
    reduction_tokens = min(batch_tokens, EVALUATION_REDUCTION_TOKEN_CHUNK)
    reduction = reduction_tokens * total_dim * 4 * 8
    decoder_grams = (sites + 1) * groups * block_width**2 * 8
    accumulators = (
        (20 * sites * groups + 4 * sites * groups * block_width)
        + groups * block_width**2
        + groups * block_width
    ) * 8
    one_site_tensor = sites * batch_tokens * latents * 4
    shared_code = (
        output
        + concordance
        + reduction
        + decoder_grams
        + accumulators
        + 2 * one_site_tensor
    )
    assert shared_code > trusted_decode
    assert actual == shared_code


def test_factorization_prices_optimizer_savings_but_not_free_dense_compute():
    blueprint = build_phase1_blueprint(seeds=(0,), smoke=True)
    plan = _materialize_all(blueprint, build_phase1_plan(seeds=(0,), smoke=True))
    stage = next(
        item for item in plan.stages if item.name == "site_factorization_identification"
    )
    estimates = {
        cell.decision_map["model.site_rank"]: estimate_cell(cell)
        for cell in stage.cells
    }
    assert (
        estimates[1].parameters < estimates[2].parameters < estimates[None].parameters
    )
    assert estimates[4].parameters > estimates[None].parameters
    assert len({item.compute_flops for item in estimates.values()}) == 1


def test_phase_alias_seed_and_merge_validation():
    assert Phase.parse("pilot") is Phase.PHASE2
    assert Phase.parse("publishable") is Phase.PHASE3
    with pytest.raises(StudyError, match="unknown phase"):
        Phase.parse("phase0")
    with pytest.raises(StudyError, match="nonempty and unique"):
        build_plan("phase1", seeds=())
    with pytest.raises(StudyError, match="nonempty and unique"):
        build_plan("phase1", seeds=(0, 0))
    first = exact("model.selector", "token_topk", "Paper A")
    second = adapted(
        "model.selector",
        "batchtopk",
        citation="Paper B",
        rationale="test allocation",
        ablation="compare at matched rate",
    )
    assert merge_decisions((first,), (second,)) == (second,)
