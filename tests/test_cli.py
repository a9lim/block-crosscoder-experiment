from block_crosscoder_experiment.analysis.month_geometry import build_parser
from block_crosscoder_experiment.cli import COMMANDS, _usage


def test_cli_exposes_one_named_surface_per_operator_workflow() -> None:
    assert tuple(COMMANDS) == ("matrix", "data", "cell", "figures", "replicate")
    usage = _usage()
    assert all(f"  {command}" in usage for command in COMMANDS)


def test_publication_figures_default_to_the_canonical_directory() -> None:
    args = build_parser().parse_args(
        ["--deployable", "seed-0.pt", "--deployable", "seed-1.pt"]
    )
    assert args.output == "figures/phase2"
