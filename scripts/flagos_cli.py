"""Standalone entry point so you can run ``python scripts/flagos_cli.py ...``
without installing the package first."""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flagos_track1.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
