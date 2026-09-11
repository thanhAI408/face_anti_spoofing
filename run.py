"""Backward-compatible entry point for the old project command."""

import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from face_antispoofing.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["demo", *sys.argv[1:]]))
