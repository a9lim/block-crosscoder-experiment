import json

from block_crosscoder_experiment.cli.phase05_matrix import (
    EPOCHS,
    LEARNING_RATES,
    NORMALIZATIONS,
    RECIPES,
    SEEDS,
    SCHEDULES,
    _command,
    build_jobs,
    plan,
    status,
)


def test_screen_covers_every_recipe_and_normalization():
    jobs = build_jobs("screen")
    assert len(jobs) == len(RECIPES) * len(NORMALIZATIONS)
    seen = {(j["recipe"], j["normalization"]) for j in jobs}
    assert seen == {(r.name, n) for r in RECIPES for n in NORMALIZATIONS}
    assert {j["lr"] for j in jobs} == {1e-4}
    assert {j["epochs"] for j in jobs} == {4}


def test_full_matrix_covers_optimizer_factorial():
    jobs = build_jobs("full")
    assert {j["lr"] for j in jobs} == set(LEARNING_RATES)
    assert {j["schedule"] for j in jobs} == set(SCHEDULES)
    assert {j["epochs"] for j in jobs} == set(EPOCHS)
    assert {j["normalization"] for j in jobs} == set(NORMALIZATIONS)
    assert {j["recipe"] for j in jobs} == {r.name for r in RECIPES}
    assert {j["seed"] for j in jobs} == set(SEEDS)
    assert len({j["job_id"] for j in jobs}) == len(jobs)
    assert len(jobs) == 68_220


def test_commands_bind_raw_store_and_all_model_factors(tmp_path):
    job = next(
        j for j in build_jobs("full")
        if j["recipe"] == "sasa_paper_bridge" and j["aux_variant"] == "sasa"
    )
    command = _command(job, tmp_path / "stores", tmp_path / "runs", "cuda")
    joined = " ".join(command)
    for flag in (
        "--raw-store", "--selection-score", "--decoder-constraint",
        "--regularizer", "--dead-window-tokens", "--aux-ratio-cap", "--seed",
    ):
        assert flag in joined


def test_plan_is_stable_and_initializes_state(tmp_path):
    a = plan(tmp_path, "all")
    b = plan(tmp_path, "all")
    assert a == b
    matrix = json.loads((tmp_path / "matrix.json").read_text())
    assert matrix["n_jobs"] == len(a)
    s = status(tmp_path)
    assert s["counts"] == {"pending": len(a)}
