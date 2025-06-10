"""
Top-level command router for the project.

Usage
-----
Run from the repo root (or anywhere on PYTHONPATH):

    python -m fraud_unsup train [--split train|test]
    python -m fraud_unsup infer <input_parquet> [--out_csv scores.csv]

Internally this file maps the verbs `train` and `infer`
to the concrete pipeline modules.

Because the package is installed **editable** (`pip install -e .`)
you can edit any pipeline and simply re-run the same commands.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict

# --------------------------------------------------------------------------- #
_COMMANDS: Dict[str, str] = {
    "train": "fraud_unsup.pipelines.train",
    "infer": "fraud_unsup.pipelines.infer",
}

# --------------------------------------------------------------------------- #
def _usage() -> None:
    cmds = "|".join(_COMMANDS)
    prog = Path(sys.argv[0]).name
    print(f"Usage: {prog} <{cmds}> [args …]")
    print(f"Example: {prog} train --split train")
    sys.exit(1)


def _dispatch() -> None:
    # Need at least one positional token (command)
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        _usage()

    cmd = sys.argv.pop(1)  # ⚡ remove the routing token so argparse sees clean argv

    module_name = _COMMANDS[cmd]
    module: ModuleType = importlib.import_module(module_name)

    # All pipeline modules expose a `main()` for CLI entry
    if not hasattr(module, "main"):
        raise AttributeError(f"Module '{module_name}' has no callable 'main()'")

    module.main()


def main() -> None:  # entry-point when imported by fraud_unsup.__main__
    _dispatch()


# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # entry-point when this file is executed directly
    _dispatch()
