from __future__ import annotations
from pathlib import Path


def project_root(start: Path | None = None) -> Path:
    """
    Walk upward from `start` (or CWD) until we find a project marker.
    Works in scripts, notebooks, and terminals.
    """
    p = (start or Path.cwd()).resolve()
    markers = {".git", "pyproject.toml", "README.md"}
    for _ in range(12):
        if any((p / m).exists() for m in markers):
            return p
        if p.parent == p:
            break
        p = p.parent
    raise RuntimeError(
        "Couldn't locate project root; add a marker like .git or pyproject.toml"
    )


# Standard paths
ROOT = project_root()
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

# Common files
TRAIN_CSV = RAW / "train.csv"
TEST_CSV = RAW / "test.csv"
