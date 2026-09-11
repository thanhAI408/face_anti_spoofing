from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from face_antispoofing.config import DetectorConfig  # noqa: E402
from face_antispoofing.dataset import (  # noqa: E402
    VIDEO_EXTENSIONS,
    assign_subject_splits,
    parse_video_path,
    stable_sample_id,
)
from face_antispoofing.detector import OpenCVSSDDetector  # noqa: E402

MANIFEST_FIELDS = (
    "sample_id",
    "dataset_name",
    "image_path",
    "label",
    "class_name",
    "attack_type",
    "subject_id",
    "session_id",
    "source_video",
    "frame_index",
    "timestamp_seconds",
    "split",
    "device_id",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract de-duplicated face crops and create a leakage-safe manifest"
    )
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--sample-fps", type=float, default=3.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence", type=float, default=0.60)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _average_hash(gray, cv2) -> int:
    tiny = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    bits = tiny >= tiny.mean()
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import cv2
    except ImportError:
        print("opencv-python is required. Run: pip install -e .", file=sys.stderr)
        return 2

    input_root = args.input.resolve()
    output_root = args.output.resolve()
    videos = sorted(
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        print(f"No videos found under {input_root}", file=sys.stderr)
        return 2
    metadata = [parse_video_path(input_root, path) for path in videos]
    split_by_subject = assign_subject_splits(
        {item.subject_id for item in metadata}, seed=args.seed
    )
    manifest_path = output_root / "manifest.csv"
    if manifest_path.exists() and not args.overwrite:
        print(f"Manifest exists: {manifest_path}. Use --overwrite to replace it.", file=sys.stderr)
        return 2
    output_root.mkdir(parents=True, exist_ok=True)
    detector = OpenCVSSDDetector(
        DetectorConfig(confidence=args.confidence), project_root=PROJECT_ROOT
    )
    rows: list[dict[str, object]] = []
    for path, meta in zip(videos, metadata, strict=True):
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            print(f"[WARNING] Cannot open video: {path}", file=sys.stderr)
            continue
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
        interval = max(1, round(fps / max(0.1, args.sample_fps)))
        frame_index = -1
        previous_hash: int | None = None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % interval:
                continue
            boxes = detector.detect(frame)
            if not boxes:
                continue
            box = boxes[0]
            face = frame[box.y1 : box.y2, box.x1 : box.x2]
            if face.size == 0:
                continue
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            current_hash = _average_hash(gray, cv2)
            if previous_hash is not None and (current_hash ^ previous_hash).bit_count() <= 1:
                continue
            previous_hash = current_hash
            sample_id = stable_sample_id(meta.source_video, frame_index)
            relative_image = (
                Path(split_by_subject[meta.subject_id])
                / meta.class_name
                / f"{sample_id}.jpg"
            )
            target = output_root / relative_image
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(target),
                face,
                [cv2.IMWRITE_JPEG_QUALITY, max(70, min(100, args.jpeg_quality))],
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "dataset_name": "internal",
                    "image_path": relative_image.as_posix(),
                    "label": meta.label,
                    "class_name": meta.class_name,
                    "attack_type": meta.attack_type,
                    "subject_id": meta.subject_id,
                    "session_id": meta.session_id,
                    "source_video": meta.source_video,
                    "frame_index": frame_index,
                    "timestamp_seconds": round(frame_index / fps, 3),
                    "split": split_by_subject[meta.subject_id],
                    "device_id": "unknown",
                }
            )
        capture.release()
    if not rows:
        print("No face crops were produced; manifest was not written", file=sys.stderr)
        return 2
    for split in ("train", "val", "test"):
        labels = {int(row["label"]) for row in rows if row["split"] == split}
        if labels != {0, 1}:
            raise ValueError(
                f"Split '{split}' must contain both live and spoof samples; found {labels}"
            )
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("train", "val", "test")
    }
    print(f"Wrote {len(rows)} samples to {manifest_path}")
    print(f"Split counts: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
