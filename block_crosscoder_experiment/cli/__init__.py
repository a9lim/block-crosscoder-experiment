"""Unified command line for training, stores, validation, and analysis."""

from __future__ import annotations

import importlib
import sys

COMMANDS: dict[str, tuple[str, str]] = {
    "train": ("block_crosscoder_experiment.cli.train_bsc", "train a joint BSC"),
    "train-single-site": (
        "block_crosscoder_experiment.cli.train_single_site",
        "train the factorial single-site controls",
    ),
    "harvest": ("block_crosscoder_experiment.cli.harvest_store", "harvest a store"),
    "verify-store": (
        "block_crosscoder_experiment.cli.verify_store",
        "verify manifests, shards, and checksums",
    ),
    "drill-resume": (
        "block_crosscoder_experiment.cli.drill_store_resume",
        "exercise interrupted-harvest recovery",
    ),
    "battery": (
        "block_crosscoder_experiment.cli.run_battery",
        "run the synthetic recovery battery",
    ),
    "sweep-bundle": (
        "block_crosscoder_experiment.cli.sweep_bundle",
        "sweep synthetic bundle controls",
    ),
    "sweep-capture": (
        "block_crosscoder_experiment.cli.sweep_capture",
        "sweep synthetic capture conditions",
    ),
    "sweep-manifolds": (
        "block_crosscoder_experiment.cli.sweep_manifolds",
        "run the pre-NVMe real-manifold tuning screen",
    ),
    "validate-codec": (
        "block_crosscoder_experiment.cli.validate_codec",
        "validate the rate-distortion codec",
    ),
    "validate-revival": (
        "block_crosscoder_experiment.cli.validate_revival",
        "validate rare-feature revival",
    ),
    "validate-theta": (
        "block_crosscoder_experiment.cli.validate_theta",
        "validate streaming threshold fitting",
    ),
    "capture-zoo": (
        "block_crosscoder_experiment.analysis.capture_zoo",
        "capture the descriptive zoo activation sample",
    ),
    "probe-families": (
        "block_crosscoder_experiment.analysis.probe_families",
        "evaluate descriptive family capture",
    ),
    "eval-manifolds": (
        "block_crosscoder_experiment.analysis.eval_manifolds",
        "score held-out operational manifold fidelity",
    ),
    "eval-stream-manifolds": (
        "block_crosscoder_experiment.analysis.eval_stream_manifolds",
        "gate tuning rings on held-out source-model geometry",
    ),
    "eval-manifold-sweep": (
        "block_crosscoder_experiment.cli.eval_manifold_sweep",
        "evaluate every completed manifold-sweep cell",
    ),
    "probe-crossarm": (
        "block_crosscoder_experiment.analysis.probe_crossarm",
        "compare primary and renorm codes",
    ),
    "extract-geometry": (
        "block_crosscoder_experiment.analysis.extract_geometry",
        "extract winner weight-space geometry",
    ),
    "activation-stats": (
        "block_crosscoder_experiment.analysis.eval_activation_stats",
        "stream evaluation activation statistics",
    ),
    "dump-frames": (
        "block_crosscoder_experiment.analysis.dump_block_frames",
        "export selected decoder frames",
    ),
    "derive-showcase": (
        "block_crosscoder_experiment.analysis.derive_showcase",
        "derive mega-block-gated showcase identities",
    ),
    "place-single-site": (
        "block_crosscoder_experiment.analysis.place_single_site_rd",
        "place single-site controls on the R-D plane",
    ),
    "figures": (
        "block_crosscoder_experiment.analysis.figures",
        "regenerate the winner-scoped figure catalog",
    ),
    "fig-sweep-frames": (
        "block_crosscoder_experiment.analysis.render_sweep_frames",
        "render temporary split-B sweep frame figures",
    ),
    "fig-capture": (
        "block_crosscoder_experiment.analysis.fig_capture",
        "regenerate global capture and allocation summaries",
    ),
    "fig-geometry": (
        "block_crosscoder_experiment.analysis.fig_geometry",
        "regenerate dictionary geometry summaries",
    ),
    "fig-rd-frontier": (
        "block_crosscoder_experiment.analysis.fig_rd_frontier",
        "regenerate the rate-distortion frontier",
    ),
    "fig-rd-tying": (
        "block_crosscoder_experiment.analysis.fig_rd_tying",
        "regenerate the tying-rate comparison",
    ),
    "refresh-analysis": (
        "block_crosscoder_experiment.analysis.pipeline",
        "refresh winner artifacts and every canonical figure",
    ),
}


def _usage() -> str:
    width = max(map(len, COMMANDS))
    rows = "\n".join(
        f"  {name:<{width}}  {description}"
        for name, (_, description) in COMMANDS.items()
    )
    return f"usage: bsc <command> [options]\n\ncommands:\n{rows}\n"


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(_usage())
        return
    command = sys.argv[1]
    if command not in COMMANDS:
        raise SystemExit(f"unknown command {command!r}\n\n{_usage()}")
    module_name = COMMANDS[command][0]
    module = importlib.import_module(module_name)
    sys.argv = [f"bsc {command}", *sys.argv[2:]]
    module.main()


__all__ = ["COMMANDS", "main"]
