"""Deprecated compatibility entry point.

The old script did not preserve source video and subject metadata. Use the
structured preparation pipeline documented in docs/DATASET_PROTOCOL.md.
"""

from scripts.prepare_dataset import main


if __name__ == "__main__":
    raise SystemExit(main())
