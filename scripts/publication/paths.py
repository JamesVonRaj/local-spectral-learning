"""Repository paths shared by publication scripts."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
DATA_ROOT = Path(
    os.environ.get("LSL_DATA_DIR", REPO_ROOT / "scripts" / "outputs")
).expanduser()
FIGURE_DIR = REPO_ROOT / "artifacts" / "figures"


def dataset(name: str) -> Path:
    """Return one named derived-data directory."""
    return DATA_ROOT / name
