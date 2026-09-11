"""Backward-compatible entry point for the new texture training pipeline."""

from scripts.train_texture import main


if __name__ == "__main__":
    raise SystemExit(main())

