"""Development-tree import shim.

The installed package is sourced from src/. This shim lets documented
``python -m swepruner_dataset_builder`` commands work directly from a clone.
"""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent / "src" / "swepruner_dataset_builder")]

