from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass(frozen=True)
class VideoMetadata:
    label: int
    class_name: str
    attack_type: str
    subject_id: str
    session_id: str
    source_video: str


def parse_video_path(root: str | Path, video_path: str | Path) -> VideoMetadata:
    root_path = Path(root).resolve()
    path = Path(video_path).resolve()
    try:
        relative = path.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"Video {path} is outside dataset root {root_path}") from exc
    parts = relative.parts
    if not parts:
        raise ValueError(f"Cannot parse empty relative path for {path}")
    top = parts[0].lower()
    if top in {"live", "real", "bonafide", "bona_fide"}:
        if len(parts) < 4:
            raise ValueError("Live layout must be live/<subject>/<session>/<video>")
        return VideoMetadata(1, "live", "none", parts[1], parts[2], relative.as_posix())
    if top in {"spoof", "fake", "attack"}:
        if len(parts) < 5:
            raise ValueError(
                "Spoof layout must be spoof/<attack_type>/<subject>/<session>/<video>"
            )
        return VideoMetadata(0, "spoof", parts[1], parts[2], parts[3], relative.as_posix())
    raise ValueError(f"Unknown class directory '{parts[0]}' in {relative}")


def assign_subject_splits(
    subject_ids: list[str] | set[str] | tuple[str, ...],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, str]:
    subjects = sorted(set(subject_ids))
    if len(subjects) < 3:
        raise ValueError("At least three subjects are required for train/val/test isolation")
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Split ratios must leave non-empty train, val and test portions")
    random.Random(seed).shuffle(subjects)
    train_count = max(1, round(len(subjects) * train_ratio))
    val_count = max(1, round(len(subjects) * val_ratio))
    if train_count + val_count >= len(subjects):
        train_count = len(subjects) - 2
        val_count = 1
    mapping: dict[str, str] = {}
    for index, subject in enumerate(subjects):
        if index < train_count:
            mapping[subject] = "train"
        elif index < train_count + val_count:
            mapping[subject] = "val"
        else:
            mapping[subject] = "test"
    return mapping


def stable_sample_id(source_video: str, frame_index: int) -> str:
    payload = f"{source_video}|{frame_index}".encode()
    return hashlib.sha256(payload).hexdigest()[:20]


def assert_no_subject_leakage(rows: list[dict[str, str]]) -> None:
    seen: dict[str, str] = {}
    for row in rows:
        subject = row["subject_id"]
        split = row["split"]
        previous = seen.setdefault(subject, split)
        if previous != split:
            raise ValueError(
                f"Subject leakage: '{subject}' occurs in both '{previous}' and '{split}'"
            )
