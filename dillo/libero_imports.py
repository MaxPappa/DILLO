"""Import helpers for the external LIBERO checkout bundled as a symlink."""

from __future__ import annotations

import sys
import types
from pathlib import Path


def prepare_libero_imports() -> None:
    """
    Make the external LIBERO checkout importable and avoid robosuite's private
    macro fallback writing to a shared /tmp log file.
    """
    sys.modules.setdefault(
        "robosuite.macros_private", types.ModuleType("robosuite.macros_private")
    )

    root = Path(__file__).resolve().parents[1]
    external = root / "LIBERO"
    if external.exists():
        path_str = str(external)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
