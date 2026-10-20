"""Merge the entire legacy crop cache with uncapped, quality-valid new video crops.

Historical legacy video boundaries remain frozen. Only the new Drive portion has
folder-person-disjoint train/validation/test groups; legacy identity is not
certified and unknown video identities remain explicitly marked. This script
never modifies originals and refuses to reuse an existing output directory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys

import cv2
import numpy as np

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
from pipeline import FaceDetector, quality  # noqa: E402
from scripts.prepare_retraining_data import (  # noqa: E402
    TEST_SUBJECTS, VALIDATION_SUBJECTS, canonical_subject, new_split,
    old_identity, safe_source_path, save_jpeg, sha256_file, write_json,
)

SPLITS = ("train", "validation", "test")
CACHE_NAME = re.compile(r"^(?P<stem>.+)\.mp4(?P<crop>\d+)\.png$", re.IGNORECASE)


def uncapped_indices(frame_count: int, fps: float, interval: float) -> list[int]:
    if frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Video must have positive frame count and finite positive FPS")
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("Sampling interval must be positive and finite")
    return list(range(0, frame_count, max(1, math.ceil(fps * interval))))


def cached_crop_quality(crop: np.ndarray) -> tuple[str | None, dict]:
    """Apply only gates measurable on already-cropped legacy images.

Do not invent full-frame boxes/landmarks or redetect tightly cropped faces.
Thresholds are the same light/sharpness/minimum-size thresholds as pipeline.
"""
    if crop is None or crop.size == 0 or crop.ndim != 3 or crop.shape[2] != 3:
        return "invalid_decode_or_shape", {}
    metadata = {"height": int(crop.shape[0]), "width": int(crop.shape[1])}
    if min(crop.shape[:2]) < 65:
        return "Move closer", metadata
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    metadata.update({"gray_mean": float(gray.mean()),
                     "glare_fraction": float(np.mean(gray > 245)),
                     "gray_std": float(gray.std()),
                     "sharpness": float(cv2.Laplacian(
                         cv2.resize(gray, (128, 128)), cv2.CV_32F).var())})
    if metadata["gray_std"] == 0:
        return "constant_pixels", metadata
    if metadata["gray_mean"] < 35:
        return "Need more light", metadata
    if metadata["glare_fraction"] > .45:
        return "Reduce glare", metadata
    if metadata["sharpness"] < 12:
        return "Hold still / improve focus", metadata
    return None, metadata


def parse_cache_path(relative_path: str) -> tuple[str, str, int]:
    normalized = relative_path.replace("\\", "/")
    path = Path(normalized)
    if len(path.parts) != 3 or path.parts[0] != "dataset" or path.parts[1] not in ("fake", "real"):
        raise ValueError(f"Unexpected cached crop path: {relative_path}")
    match = CACHE_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Cached image has no auditable video lineage: {relative_path}")
    return path.parts[1], match["stem"], int(match["crop"])


def collect_legacy(frame_manifest: Path, split_manifest: Path, base: Path = BASE):
    frame_data = json.loads(frame_manifest.read_text(encoding="utf-8"))
    split_data = json.loads(split_manifest.read_text(encoding="utf-8"))
    records, seen = [], set()
    video_splits = {}
    for split in SPLITS:
        paths = frame_data.get(split)
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"Legacy {split} frame list must be nonempty")
        allowed_stems = set(split_data[split]["videos"])
        for raw_path in paths:
            relative = str(raw_path).replace("\\", "/")
            if relative in seen:
                raise ValueError(f"Image occurs more than once in historical manifest: {relative}")
            seen.add(relative)
            label, stem, crop_index = parse_cache_path(relative)
            if stem not in allowed_stems:
                raise ValueError(f"Legacy frame outside historical {split} video boundary: {relative}")
            prior_split = video_splits.setdefault(stem, split)
            if prior_split != split:
                raise ValueError(f"Legacy video leaks across splits: {stem}")
            identity = old_identity(stem)
            if identity is not None and identity[1] != label:
                raise ValueError(f"Filename label and cache folder conflict: {relative}")
            path = safe_source_path(relative, base)
            if not path.is_file():
                raise FileNotFoundError(f"Historical crop is missing: {relative}")
            records.append({
                "relative_path": relative, "label": label, "video_id": "old:" + stem,
                "subject_id": identity[0] if identity else "unknown:" + stem,
                "identity_unknown": identity is None, "split": split,
                "source": "legacy-cache", "evaluation_group": "old",
                "cached_crop_index": crop_index,
                "cached_crop_index_is_video_frame_index": False,
                "legacy_split": split,
            })
    disk_files = {p.relative_to(base).as_posix()
                  for label in ("fake", "real") for p in (base / "dataset" / label).glob("*.png")}
    if disk_files != seen:
        raise ValueError(f"Legacy disk/manifest mismatch: missing={len(seen - disk_files)}, "
                         f"untracked={len(disk_files - seen)}")
    return records, frame_data, split_data


def collect_drive(inventory_path: Path, base: Path = BASE):
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    sources, seen = [], set()
    for entry in inventory.get("files", []):
        label = entry.get("label")
        subject = canonical_subject(entry.get("subject_id", ""))
        drive_id = str(entry.get("id", ""))
        if label not in ("fake", "real") or not subject or not drive_id:
            raise ValueError(f"Invalid Drive label/identity: {entry}")
        video_id = "drive:" + drive_id
        if video_id in seen:
            raise ValueError(f"Duplicate Drive ID: {drive_id}")
        seen.add(video_id)
        path = safe_source_path(entry["relative_path"], base)
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing/empty Drive video: {path}")
        expected = int(entry["size"])
        if path.stat().st_size != expected:
            raise ValueError(f"Drive download size mismatch: {path}")
        sources.append({
            "video_id": video_id, "source": "drive-20260930", "drive_id": drive_id,
            "relative_path": path.relative_to(base.resolve()).as_posix(), "label": label,
            "subject_id": subject, "identity_unknown": False, "split": new_split(subject),
            "evaluation_group": "new", "bytes": expected, "title": entry.get("title", path.name),
        })
    if not sources:
        raise ValueError("Drive inventory must contain videos")
    return sources, inventory


def frame_summaries(records: list[dict]) -> dict:
    result = {}
    for split in SPLITS:
        subset = [record for record in records if record["split"] == split]
        result[split] = {
            "frames": len(subset), "classes": dict(Counter(r["label"] for r in subset)),
            "videos": len({r["video_id"] for r in subset}),
            "subjects": sorted({r["subject_id"] for r in subset}),
            "identity_unknown_frames": sum(r["identity_unknown"] for r in subset),
            "sources": {},
        }
        for source in ("legacy-cache", "drive-20260930"):
            group = [r for r in subset if r["source"] == source]
            result[split]["sources"][source] = {
                "frames": len(group), "classes": dict(Counter(r["label"] for r in group)),
                "videos": len({r["video_id"] for r in group}),
            }
    return result


def deduplicate_records(records: list[dict], pixels: list[np.ndarray]):
    """Quarantine all conflicting/cross-split copies; keep one same-split copy."""
    if len(records) != len(pixels):
        raise ValueError("Records and pixel rows must align before deduplication")
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[record["crop_pixel_sha256"]].append(index)
    remove = {}
    for indices in groups.values():
        if len(indices) < 2:
            continue
        copies = [records[index] for index in indices]
        if len({r["label"] for r in copies}) > 1:
            reason = "conflicting_label_identical_model_pixels"
            for index in indices:
                remove[index] = reason
        elif len({r["split"] for r in copies}) > 1:
            reason = "cross_split_identical_model_pixels"
            for index in indices:
                remove[index] = reason
        else:
            for index in indices[1:]:
                remove[index] = "duplicate_identical_model_pixels_same_split"
    accepted, accepted_pixels, exclusions, original_to_final = [], [], [], {}
    for index, (record, sample) in enumerate(zip(records, pixels)):
        if index in remove:
            exclusions.append({**record, "reason": remove[index]})
            continue
        final_index = len(accepted)
        original_to_final[index] = final_index
        accepted.append({**record, "array_index": final_index})
        accepted_pixels.append(sample)
    return accepted, accepted_pixels, exclusions, original_to_final


def validate_boundaries(records: list[dict]):
    video_splits = {}
    pixel_splits = {}
    subject_splits = {}
    for record in records:
        prior = video_splits.setdefault(record["video_id"], record["split"])
        if prior != record["split"]:
            raise ValueError("Video crosses final split boundary")
        prior_pixel = pixel_splits.setdefault(record["crop_pixel_sha256"], record["split"])
        if prior_pixel != record["split"]:
            raise ValueError("Identical model pixels cross final split boundary")
        if record["source"] == "drive-20260930":
            prior_subject = subject_splits.setdefault(record["subject_id"], record["split"])
            if prior_subject != record["split"]:
                raise ValueError("New Drive subject crosses final split boundary")
        elif record["subject_id"] in TEST_SUBJECTS | VALIDATION_SUBJECTS:
            raise ValueError("Legacy named identity conflicts with frozen new held-out subject")
    for split in SPLITS:
        if {r["label"] for r in records if r["split"] == split} != {"fake", "real"}:
            raise ValueError(f"Merged {split} needs both classes")


def append_sample(records, pixels, sample, record):
    digest = hashlib.sha256(sample.tobytes()).hexdigest()
    record = {**record, "array_index": len(pixels), "crop_pixel_sha256": digest,
              "frame_pixel_sha256": digest}
    records.append(record)
    pixels.append(sample)
    return record["array_index"]


def extract_drive_video(video, detector, args, records, pixels, exclusions):
    path = safe_source_path(video["relative_path"], BASE)
    video["sha256"] = sha256_file(path)
    video["sample_events"] = []
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source video: {path}")
    counts, failures = Counter(), Counter()
    try:
        orientation_set = bool(capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = uncapped_indices(count, fps, args.min_interval)
        video.update({"fps": fps, "frame_count": count, "duration_seconds": count / fps,
                      "metadata_width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      "metadata_height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      "orientation_degrees": float(capture.get(cv2.CAP_PROP_ORIENTATION_META)),
                      "orientation_auto_set": orientation_set,
                      "orientation_auto_active": bool(capture.get(cv2.CAP_PROP_ORIENTATION_AUTO)),
                      "backend": capture.getBackendName(), "sampled_frame_indices": indices})
        current = -1
        preview_saved = False
        for frame_index in indices:
            event = {"frame_index": frame_index, "time_seconds": frame_index / fps}
            while current + 1 < frame_index:
                if not capture.grab():
                    raise RuntimeError(f"Video ends before advertised frame {frame_index}: {path}")
                current += 1
            success, frame = capture.read()
            current += 1
            if not success or frame is None or frame.size == 0 or current != frame_index:
                raise RuntimeError(f"Cannot decode sampled frame {frame_index}: {path}")
            counts["sampled"] += 1
            video["decoded_width"], video["decoded_height"] = int(frame.shape[1]), int(frame.shape[0])
            faces = detector.detect(frame)
            event["face_count"] = len(faces)
            if len(faces) != 1:
                reason = "no_face" if not faces else "multiple_faces"
                counts[reason] += 1
                video["sample_events"].append({**event, "status": reason})
                continue
            face = faces[0]
            issue = quality(frame, face)
            event.update({"box": list(face["box"]), "detector_score": face["detector_score"]})
            if issue is not None:
                counts["quality_rejected"] += 1
                failures[issue] += 1
                video["sample_events"].append({**event, "status": "adjust", "quality_reason": issue})
                continue
            x, y, right, bottom = face["box"]
            crop = frame[y:bottom, x:right]
            sample = cv2.resize(crop, (32, 32))
            index = append_sample(records, pixels, sample, {
                key: video[key] for key in ("label", "video_id", "subject_id", "identity_unknown",
                                           "source", "split", "evaluation_group")
            } | {"frame_index": frame_index, "time_seconds": frame_index / fps,
                 "box": list(face["box"]), "detector_score": float(face["detector_score"]),
                 "relative_path": video["relative_path"]})
            counts["quality_valid_before_dedup"] += 1
            video["sample_events"].append({**event, "status": "valid", "array_index": index})
            if not preview_saved:
                preview_id = hashlib.sha256(video["video_id"].encode()).hexdigest()[:16]
                frame_file = args.output / "preview" / (preview_id + "_frame.jpg")
                crop_file = args.output / "preview" / (preview_id + "_crop.jpg")
                scale = min(1., 640 / max(frame.shape[:2]))
                preview = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
                save_jpeg(frame_file, preview)
                save_jpeg(crop_file, cv2.resize(crop, (128, 128)))
                video["preview"] = {"frame": frame_file.relative_to(args.output).as_posix(),
                                    "crop": crop_file.relative_to(args.output).as_posix(),
                                    "frame_index": frame_index}
                preview_saved = True
    finally:
        video["statistics"] = dict(counts)
        video["quality_failure_reasons"] = dict(failures)
        capture.release()
    if not counts["quality_valid_before_dedup"]:
        exclusions.append({"video_id": video["video_id"], "source": video["source"],
                           "split": video["split"], "reason": "zero_quality_valid_video_frames"})
    print(json.dumps({"event": "drive_video_processed", "video_id": video["video_id"],
                      "label": video["label"], "split": video["split"],
                      "statistics": dict(counts)}, ensure_ascii=False), flush=True)


def prepare(args):
    legacy, _, old_split = collect_legacy(args.old_frames, args.old_split)
    drive, inventory = collect_drive(args.inventory)
    initial_summary = frame_summaries(legacy)
    if args.plan:
        print(json.dumps({"legacy_candidate_summary": initial_summary,
                          "legacy_candidate_images": len(legacy), "drive_videos": len(drive),
                          "drive_videos_by_split": dict(Counter(v["split"] for v in drive)),
                          "new_sampling": "Every >=0.2 seconds across entire clip; no frame-count cap",
                          "old_test_videos": old_split["test"]["videos"]}, ensure_ascii=False, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "preview").mkdir()
    cv2.setNumThreads(2)
    records, pixels, exclusions, videos = [], [], [], []
    manifest = {
        "schema_version": "merged-1.0", "status": "preparing",
        "created_at": datetime.now(timezone.utc).isoformat(), "labels": ["fake", "real"],
        "source_lineage": {
            "legacy_frame_manifest": str(args.old_frames.resolve()),
            "legacy_frame_manifest_sha256": sha256_file(args.old_frames),
            "legacy_split_manifest": str(args.old_split.resolve()),
            "legacy_split_manifest_sha256": sha256_file(args.old_split),
            "drive_inventory": str(args.inventory.resolve()),
            "drive_inventory_sha256": sha256_file(args.inventory),
            "before_merge_dataset_manifest": str(args.prior_manifest.resolve()),
            "before_merge_dataset_manifest_sha256": sha256_file(args.prior_manifest),
        },
        "label_source": {"legacy-cache": "Cached fake/real folder, checked against named video",
                         "drive-20260930": inventory.get("label_source")},
        "split_policy": {
            "subject_disjoint_sources": ["drive-20260930"],
            "test_subjects": sorted(TEST_SUBJECTS), "validation_subjects": sorted(VALIDATION_SUBJECTS),
            "legacy": "Preserve exact 2026-09-15 frame/video train-validation-test boundaries",
            "legacy_test_videos": old_split["test"]["videos"],
            "legacy_validation_videos": old_split["validation"]["videos"],
            "test": "Common test: all quality-valid legacy held-out crops plus frozen Nga/Phuc videos",
            "validation": "Historical old validation videos plus frozen B/Thu videos",
            "test_is_used_for_selection": False,
            "legacy_person_disjoint_certified": False,
            "unknown_identity_policy": "Retain historical video boundary and mark unknown:<video stem>",
        },
        "preprocessing": {
            "pixels": "BGR uint8 32x32; OpenCV default resize; divide by255 at model input",
            "legacy_candidates": "All cached PNG images from historical manifest, no cap per video",
            "legacy_quality": "Crop-only minimum edge65, graymean>=35, glarefraction<=.45, Laplacian128 variance>=12, nonconstant pixels",
            "legacy_quality_limitation": "Existing crops have no full-frame coordinates/landmarks; no re-detection, clipping/pose gate is unavailable",
            "new_quality": "Exactly one YuNet face passing current pipeline.quality on oriented full frame",
            "detector": "YuNet face_detection_yunet_2023mar.onnx",
            "detector_confidence": args.confidence,
            "detector_sha256": sha256_file(BASE / "face_detector/face_detection_yunet_2023mar.onnx"),
            "pipeline_sha256": sha256_file(BASE / "pipeline.py"),
            "min_interval_seconds": args.min_interval, "max_sampled_frames_per_video": None,
            "duplicate_pixels": "Exact32x32BGR only; quarantine all cross-split or label-conflict copies; keep first same-label same-split copy",
            "near_duplicate_filter_applied": False,
        },
        "runtime": {"python": platform.python_version(), "opencv": cv2.__version__,
                    "numpy": np.__version__, "opencv_threads": 2},
        "candidate_legacy_summary": initial_summary,
        "videos": videos, "frames": records, "exclusions": exclusions,
    }
    try:
        by_video = defaultdict(list)
        legacy_quality_reasons = Counter()
        for position, legacy_record in enumerate(legacy):
            path = safe_source_path(legacy_record["relative_path"], BASE)
            encoded = path.read_bytes()
            crop = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            issue, metadata = cached_crop_quality(crop)
            if issue is not None:
                legacy_quality_reasons[issue] += 1
                exclusions.append({**legacy_record, "reason": "legacy_crop_quality_rejected",
                                   "quality_reason": issue, "quality_measurements": metadata,
                                   "image_file_sha256": hashlib.sha256(encoded).hexdigest()})
            else:
                append_sample(records, pixels, cv2.resize(crop, (32, 32)), {
                    **legacy_record, "original_crop_shape": list(crop.shape),
                    "quality_measurements": metadata,
                    "image_file_sha256": hashlib.sha256(encoded).hexdigest(),
                })
            by_video[legacy_record["video_id"]].append(legacy_record)
            if (position + 1) % 1000 == 0 or position + 1 == len(legacy):
                print(json.dumps({"event": "legacy_cache_processed", "processed": position + 1,
                                  "candidate_images": len(legacy), "quality_valid_before_dedup": len(records),
                                  "quality_rejected": dict(legacy_quality_reasons)}), flush=True)
        manifest["legacy_quality_rejections"] = dict(legacy_quality_reasons)
        for video_id, candidates in by_video.items():
            first = candidates[0]
            stem = video_id.removeprefix("old:")
            videos.append({key: first[key] for key in ("video_id", "source", "subject_id",
                                                      "identity_unknown", "label", "split", "evaluation_group")} |
                          {"relative_path": "videos/" + stem + ".mp4", "cache_only": True,
                           "sample_events": [], "statistics": {"candidate_images": len(candidates)},
                           "title": stem + ".mp4"})
        detector = FaceDetector(confidence=args.confidence)
        for source in drive:
            video = dict(source)
            videos.append(video)
            extract_drive_video(video, detector, args, records, pixels, exclusions)
        manifest["quality_valid_before_dedup_summary"] = frame_summaries(records)
        accepted, accepted_pixels, duplicate_exclusions, remap = deduplicate_records(records, pixels)
        exclusions.extend(duplicate_exclusions)
        # Runtime-event row references must reflect final deduplication.
        removed_reasons = {r["array_index"]: r["reason"] for r in duplicate_exclusions}
        for video in videos:
            for event in video["sample_events"]:
                if "array_index" not in event:
                    continue
                original_index = event["array_index"]
                if original_index in remap:
                    event["array_index"] = remap[original_index]
                else:
                    event.pop("array_index")
                    event["status"] = "excluded_duplicate_pixels"
                    event["exclusion_reason"] = removed_reasons[original_index]
            count = sum(r["video_id"] == video["video_id"] for r in accepted)
            video["accepted_frames"] = count
        validate_boundaries(accepted)
        array = np.stack(accepted_pixels).astype(np.uint8, copy=False)
        np.savez_compressed(args.output / "frames.npz", x=array)
        manifest["frames"] = accepted
        manifest["summary"] = frame_summaries(accepted)
        manifest["deduplication_exclusions"] = dict(Counter(r["reason"] for r in duplicate_exclusions))
        manifest["array"] = {"path": "frames.npz", "key": "x", "shape": list(array.shape),
                             "dtype": str(array.dtype), "sha256": sha256_file(args.output / "frames.npz")}
        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["preparation_script_sha256"] = sha256_file(Path(__file__))
        write_json(args.output / "manifest.json", manifest)
        print(json.dumps({"event": "merged_dataset_complete", "output": str(args.output.resolve()),
                          "summary": manifest["summary"],
                          "legacy_quality_rejections": manifest["legacy_quality_rejections"],
                          "deduplication_exclusions": manifest["deduplication_exclusions"]}, ensure_ascii=False), flush=True)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        manifest["summary"] = frame_summaries(records)
        write_json(args.output / "manifest.failed.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-frames", type=Path, default=BASE / "outputs/retrain-20260915/frame_manifest.json")
    parser.add_argument("--old-split", type=Path, default=BASE / "outputs/retrain-20260915/split.json")
    parser.add_argument("--inventory", type=Path, default=BASE / "outputs/drive-20260930/inventory-verified.json")
    parser.add_argument("--prior-manifest", type=Path, default=BASE / "dataset/retrain-20260930/manifest.json")
    parser.add_argument("--output", type=Path, default=BASE / "dataset/merged-20261001")
    parser.add_argument("--min-interval", type=float, default=.2)
    parser.add_argument("--confidence", type=float, default=.85)
    parser.add_argument("--plan", action="store_true", help="Print source/split plan without decoding or writing")
    args = parser.parse_args()
    if not 0 < args.confidence < 1 or not math.isfinite(args.min_interval) or args.min_interval <= 0:
        parser.error("Confidence must be between0and1; interval positive and finite")
    prepare(args)


if __name__ == "__main__":
    main()
