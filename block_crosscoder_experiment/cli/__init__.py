"""Small, explicit command surface for the staged experiment."""

from __future__ import annotations

import importlib
import sys

COMMANDS: dict[str, tuple[str, str]] = {
    "matrix": (
        "block_crosscoder_experiment.cli.matrix",
        "plan, estimate, run, and reconcile staged studies",
    ),
    "data": (
        "block_crosscoder_experiment.cli.data",
        "capture raw activations and derive/verify aligned views",
    ),
    "cell": (
        "block_crosscoder_experiment.cli.run_cell",
        "execute one resolved cell stage",
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
    module = importlib.import_module(COMMANDS[command][0])
    sys.argv = [f"bsc {command}", *sys.argv[2:]]
    module.main()


__all__ = ["COMMANDS", "main"]
