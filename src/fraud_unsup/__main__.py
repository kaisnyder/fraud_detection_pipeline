"""
python -m fraud_unsup <command>

Delegates to the top-level CLI helper.
"""
from importlib import import_module
import sys

# Re-use the command router defined in src/cli.py
def _entry():
    mod = import_module("cli")
    mod.main()

if __name__ == "__main__":
    _entry()
