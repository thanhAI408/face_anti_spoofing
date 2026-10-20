"""Prepare auditable, person-disjoint new-video training data without using labels to filter.

Pixels match pipeline.preprocess: OpenCV BGR, resize to 32 x 32, then divide by
255 when training. The archive stores uint8 pixels to keep that conversion explicit.
The old model's five held-out video stems are a separate regression set; their
people can overlap training and are never presented as an unseen-person test.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
import unicodedata

import cv2
import numpy as np

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
from pipeline import FaceDetector, quality  # noqa: E402

TEST_SUBJECTS = {"nga", "phuc"}
VALIDATION_SUBJECTS = {"b", "thu"}
OLD_SUBJECTS = {"bang", "phuong", "thang", "thanh", "thanh2", "truong", "tuyen"}
GENERIC_OLD_NAMES = {"real", "fake", "gia", "that", "fake-1", "real-1"}
SPLITS = ("train", "validation", "test", "regression_old")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_subject(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value).strip().casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[\s_-]+", "", value)


def new_split(subject_id: str) -> str:
    if subject_id in TEST_SUBJECTS:
        return "test"
    if subject_id in VALIDATION_SUBJECTS:
        return "validation"
    return "train"


def old_identity(stem: str):
    """Infer only documented unambiguous old naming patterns."""
    name = stem.casefold()
    if name in GENERIC_OLD_NAMES:
        return None
    parts = name.split("_")
    if len(parts) >= 2 and parts[0] in OLD_SUBJECTS and parts[1] in {"real", "fake"}:
        subject, label = parts[0], parts[1]
    elif len(parts) == 2 and parts[0] in {"real", "fake"} and parts[1] in OLD_SUBJECTS:
        label, subject = parts
    else:
        return None
    return ("thanh" if subject == "thanh2" else subject), label


def sampling_indices(frame_count: int, fps: float, min_interval: float, maximum: int):
    if frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Video must have a positive frame count and finite positive FPS")
    if not math.isfinite(min_interval) or min_interval <= 0 or maximum <= 0:
        raise ValueError("Sampling interval and maximum count must be positive")
    step = max(1, math.ceil(fps * min_interval))
    candidates = np.arange(0, frame_count, step, dtype=np.int64)
    if len(candidates) > maximum:
        # Uniformly span the clip instead of selecting only its beginning.
        candidates = candidates[np.linspace(0, len(candidates) - 1, maximum, dtype=np.int64)]
    return candidates.tolist()


def safe_source_path(relative_path: str, base: Path) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"Inventory paths must be relative: {relative_path}")
    result = (base / relative).resolve()
    if not result.is_relative_to(base.resolve()):
        raise ValueError(f"Inventory path leaves project directory: {relative_path}")
    return result


def collect_sources(inventory: Path, old_split: Path, base: Path = BASE):
    data = json.loads(inventory.read_text(encoding="utf-8"))
    files = data.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Inventory must contain a nonempty files list")
    regression_stems = set(json.loads(old_split.read_text(encoding="utf-8"))["test"]["videos"])
    sources, exclusions, seen_ids = [], [], set()
    for record in files:
        label = record.get("label")
        subject = canonical_subject(record.get("subject_id", ""))
        drive_id = record.get("id")
        if label not in {"fake", "real"} or not subject or not drive_id:
            raise ValueError(f"Invalid inventory identity/label: {record}")
        video_id = "drive:" + str(drive_id)
        if video_id in seen_ids:
            raise ValueError(f"Repeated Drive ID in inventory: {drive_id}")
        seen_ids.add(video_id)
        source_path = safe_source_path(record["relative_path"], base)
        sources.append({
            "video_id": video_id, "source": "drive-20260930", "drive_id": drive_id,
            "relative_path": source_path.relative_to(base.resolve()).as_posix(),
            "label": label, "subject_id": subject, "split": new_split(subject),
            "expected_bytes": record.get("size"), "title": record.get("title", source_path.name),
        })
    for source_path in sorted((base / "videos").glob("*.mp4")):
        identity = old_identity(source_path.stem)
        relative = source_path.relative_to(base).as_posix()
        if identity is None:
            exclusions.append({"relative_path": relative, "source": "original",
                               "reason": "ambiguous_or_generic_subject_name"})
            continue
        subject, label = identity
        sources.append({
            "video_id": "old:" + source_path.stem, "source": "original",
            "relative_path": relative, "label": label, "subject_id": subject,
            "split": "regression_old" if source_path.stem in regression_stems else "train",
            "title": source_path.name,
        })
    # Assert the intended person boundary independently of extraction success.
    subjects = {split: {v["subject_id"] for v in sources if v["split"] == split}
                for split in ("train", "validation", "test")}
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if subjects[first] & subjects[second]:
            raise ValueError(f"Subject leakage between {first} and {second}: {subjects[first] & subjects[second]}")
    return data, sources, exclusions


def deduplicate_videos(sources: list[dict], base: Path = BASE):
    unique, excluded, seen = [], [], {}
    for source in sources:
        path = safe_source_path(source["relative_path"], base)
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty source video: {path}")
        size = path.stat().st_size
        if source.get("expected_bytes") is not None and size != int(source["expected_bytes"]):
            raise ValueError(f"Downloaded size mismatch: {path}: {size} != {source['expected_bytes']}")
        source["bytes"] = size
        digest = source["sha256"] = sha256_file(path)
        previous = seen.get(digest)
        if previous is not None:
            if previous["label"] != source["label"]:
                raise ValueError(f"Conflicting labels for identical videos: {previous['relative_path']} and {source['relative_path']}")
            if previous["split"] != source["split"]:
                raise ValueError(f"Identical video crosses split boundary: {previous['relative_path']} ({previous['split']}) and {source['relative_path']} ({source['split']})")
            excluded.append({**source, "reason": "duplicate_video_sha256",
                             "duplicate_of": previous["video_id"]})
            continue
        seen[digest] = source
        unique.append(source)
    return unique, excluded


def write_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_jpeg(path: Path, frame):
    success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not success:
        raise RuntimeError(f"Could not encode preview {path}")
    encoded.tofile(str(path))  # Supports Unicode Windows paths.


def summaries(videos: list[dict], frames: list[dict]):
    result = {}
    for split in SPLITS:
        subset = [frame for frame in frames if frame["split"] == split]
        clips = [video for video in videos if video["split"] == split]
        accepted = [video for video in clips if video.get("accepted_frames", 0) > 0]
        result[split] = {
            "frames": len(subset), "classes": dict(Counter(frame["label"] for frame in subset)),
            "videos": len(accepted), "candidate_videos": len(clips),
            "subjects": sorted({frame["subject_id"] for frame in subset}),
            "videos_by_class": dict(Counter(video["label"] for video in accepted)),
        }
    return result


def prepare(args):
    inventory, sources, exclusions = collect_sources(args.inventory, args.old_split, BASE)
    sources, duplicate_exclusions = deduplicate_videos(sources, BASE)
    exclusions.extend(duplicate_exclusions)
    # Never overwrite an existing dataset or reuse stale frame numbering.
    args.output.mkdir(parents=True, exist_ok=False)
    preview_dir = args.output / "preview"
    preview_dir.mkdir()
    cv2.setNumThreads(2)
    detector = FaceDetector(confidence=args.confidence)
    frames, pixels, videos = [], [], []
    global_pixels = {}
    manifest = {
        "schema_version": "1.0", "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "preparing", "inventory": str(args.inventory.resolve()),
        "inventory_sha256": sha256_file(args.inventory),
        "label_source": inventory.get("label_source"), "labels": ["fake", "real"],
        "split_policy": {
            "test_subjects": sorted(TEST_SUBJECTS), "validation_subjects": sorted(VALIDATION_SUBJECTS),
            "train": "All other identified subjects; excludes original held-out test videos",
            "regression_old": "Original 2026-09-15 test video stems; overlapping people allowed only here",
            "test_is_used_for_selection": False,
        },
        "preprocessing": {
            "pixels": "BGR uint8 32x32; divide by 255 at model input",
            "detector": "YuNet face_detection_yunet_2023mar.onnx",
            "detector_confidence": args.confidence,
            "detector_sha256": sha256_file(BASE / "face_detector/face_detection_yunet_2023mar.onnx"),
            "pipeline_sha256": sha256_file(BASE / "pipeline.py"),
            "quality": "pipeline.quality; accept only exactly one detected face passing all gates",
            "orientation_auto": True, "min_interval_seconds": args.min_interval,
            "max_sampled_frames_per_video": args.max_per_video,
            "duplicate_pixels": "Drop identical 32x32 crop pixels within clip; fail cross-split duplicates",
        },
        "runtime": {"python": platform.python_version(), "opencv": cv2.__version__,
                    "numpy": np.__version__, "opencv_threads": 2},
        "videos": videos, "frames": frames, "exclusions": exclusions,
    }
    try:
        for source in sources:
            video = dict(source)
            video["sample_events"] = []
            videos.append(video)
            counts = Counter()
            quality_failures = Counter()
            face_counts = Counter()
            path = safe_source_path(video["relative_path"], BASE)
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise RuntimeError(f"Cannot decode source video: {path}")
            try:
                orientation_set = bool(capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1))
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                indices = sampling_indices(frame_count, fps, args.min_interval, args.max_per_video)
                video.update({
                    "fps": fps, "frame_count": frame_count,
                    "metadata_width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "metadata_height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "orientation_degrees": float(capture.get(cv2.CAP_PROP_ORIENTATION_META)),
                    "orientation_auto_set": orientation_set,
                    "orientation_auto_active": bool(capture.get(cv2.CAP_PROP_ORIENTATION_AUTO)),
                    "duration_seconds": frame_count / fps,
                    "backend": capture.getBackendName(), "sampled_frame_indices": indices,
                })
                local_pixels = {}
                preview_saved = False
                current_index = -1
                for frame_index in indices:
                    event = {"frame_index": frame_index, "time_seconds": frame_index / fps}
                    while current_index + 1 < frame_index:
                        if not capture.grab():
                            video["sample_events"].append({**event, "status": "decode_error"})
                            raise RuntimeError(f"Video ends before advertised frame {frame_index}: {path}")
                        current_index += 1
                    success, frame = capture.read()
                    current_index += 1
                    if not success or frame is None or frame.size == 0:
                        video["sample_events"].append({**event, "status": "decode_error"})
                        raise RuntimeError(f"Cannot read sampled frame {frame_index}: {path}")
                    if current_index != frame_index:
                        raise RuntimeError(f"Unexpected decoder index at {path}: {current_index} != {frame_index}")
                    counts["sampled"] += 1
                    video["decoded_width"], video["decoded_height"] = int(frame.shape[1]), int(frame.shape[0])
                    faces = detector.detect(frame)
                    face_counts[str(len(faces))] += 1
                    event["face_count"] = len(faces)
                    if not faces:
                        counts["no_face"] += 1
                        video["sample_events"].append({**event, "status": "no_face"})
                        continue
                    if len(faces) != 1:
                        counts["multiple_faces"] += 1
                        video["sample_events"].append({**event, "status": "multiple_faces"})
                        continue
                    face = faces[0]
                    event["box"] = list(face["box"])
                    event["detector_score"] = face["detector_score"]
                    issue = quality(frame, face)
                    if issue is not None:
                        counts["quality_rejected"] += 1
                        quality_failures[issue] += 1
                        video["sample_events"].append({**event, "status": "adjust", "quality_reason": issue})
                        continue
                    counts["quality_valid"] += 1
                    x, y, right, bottom = face["box"]
                    crop = frame[y:bottom, x:right]
                    sample = cv2.resize(crop, (32, 32))
                    digest = hashlib.sha256(sample.tobytes()).hexdigest()
                    if digest in local_pixels:
                        counts["duplicate_pixels_within_video"] += 1
                        # Keep the sampled-time event for temporal/coverage evaluation.
                        # Training stores its crop only once, while inference may reuse
                        # exactly the same preprocessed pixels at another timestamp.
                        video["sample_events"].append({**event, "status": "valid",
                                                      "array_index": local_pixels[digest]["array_index"],
                                                      "duplicate_pixels_within_video": True})
                        continue
                    previous = global_pixels.get(digest)
                    if previous and previous["split"] != video["split"]:
                        raise ValueError(f"Identical crop pixels cross split boundary: {previous['video_id']} ({previous['split']}) and {video['video_id']} ({video['split']})")
                    if previous and previous["label"] != video["label"]:
                        raise ValueError(f"Identical crop pixels have conflicting labels: {previous['video_id']} and {video['video_id']}")
                    if previous and previous["video_id"] != video["video_id"]:
                        counts["duplicate_pixels_same_split_other_video"] += 1
                    record = {
                        "array_index": len(pixels), "label": video["label"],
                        "video_id": video["video_id"], "subject_id": video["subject_id"],
                        "source": video["source"], "split": video["split"],
                        "frame_index": frame_index, "time_seconds": frame_index / fps,
                        "box": list(face["box"]), "detector_score": face["detector_score"],
                        "crop_pixel_sha256": digest,
                    }
                    local_pixels[digest] = record
                    global_pixels.setdefault(digest, record)
                    frames.append(record)
                    pixels.append(sample)
                    counts["accepted"] += 1
                    video["sample_events"].append({**event, "status": "valid", "array_index": record["array_index"]})
                    if not preview_saved:
                        preview_id = hashlib.sha256(video["video_id"].encode("utf-8")).hexdigest()[:16]
                        frame_file, crop_file = preview_dir / f"{preview_id}_frame.jpg", preview_dir / f"{preview_id}_crop.jpg"
                        scale = min(1.0, 640 / max(frame.shape[:2]))
                        preview = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
                        save_jpeg(frame_file, preview)
                        save_jpeg(crop_file, cv2.resize(crop, (128, 128)))
                        video["preview"] = {"frame": frame_file.relative_to(args.output).as_posix(),
                                            "crop": crop_file.relative_to(args.output).as_posix(),
                                            "frame_index": frame_index}
                        preview_saved = True
                video["statistics"] = dict(counts)
                video["face_count_distribution"] = dict(face_counts)
                video["quality_failure_reasons"] = dict(quality_failures)
                video["accepted_frames"] = counts["accepted"]
                if not video["accepted_frames"]:
                    exclusions.append({"video_id": video["video_id"], "relative_path": video["relative_path"],
                                       "split": video["split"], "reason": "zero_accepted_quality_frames",
                                       "statistics": dict(counts)})
                print(json.dumps({"event": "video_prepared", "video_id": video["video_id"],
                                  "label": video["label"], "subject": video["subject_id"],
                                  "split": video["split"], "statistics": dict(counts)}, ensure_ascii=False), flush=True)
            finally:
                video["statistics"] = dict(counts)
                video["face_count_distribution"] = dict(face_counts)
                video["quality_failure_reasons"] = dict(quality_failures)
                video["accepted_frames"] = counts["accepted"]
                capture.release()
        manifest["summary"] = summaries(videos, frames)
        for split in SPLITS:
            if set(manifest["summary"][split]["classes"]) != {"fake", "real"}:
                raise ValueError(f"Prepared {split} must contain accepted frames of both classes: {manifest['summary'][split]}")
        array = np.stack(pixels).astype(np.uint8, copy=False)
        np.savez_compressed(args.output / "frames.npz", x=array)
        manifest["array"] = {"path": "frames.npz", "key": "x", "shape": list(array.shape),
                             "dtype": str(array.dtype), "sha256": sha256_file(args.output / "frames.npz")}
        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output / "manifest.json", manifest)
        print(json.dumps({"event": "dataset_complete", "output": str(args.output.resolve()),
                          "summary": manifest["summary"]}, ensure_ascii=False), flush=True)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        manifest["summary"] = summaries(videos, frames)
        write_json(args.output / "manifest.failed.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=BASE / "outputs/drive-20260930/inventory-verified.json")
    parser.add_argument("--old-split", type=Path, default=BASE / "outputs/retrain-20260915/split.json")
    parser.add_argument("--output", type=Path, default=BASE / "dataset/retrain-20260930")
    parser.add_argument("--max-per-video", type=int, default=100)
    parser.add_argument("--min-interval", type=float, default=.2)
    parser.add_argument("--confidence", type=float, default=.85)
    args = parser.parse_args()
    if not 0 < args.confidence < 1:
        parser.error("--confidence must be between 0 and 1")
    if args.max_per_video <= 0 or not math.isfinite(args.min_interval) or args.min_interval <= 0:
        parser.error("Sampling counts/interval must be positive and finite")
    prepare(args)


if __name__ == "__main__":
    main()
