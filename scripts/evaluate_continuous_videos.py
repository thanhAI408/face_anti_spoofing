"""Evaluate a fixed first-12-second test-video prefix with the actual pipeline.

Selection is fixed before observing scores: sample chronologically at no more
than 5 FPS from the beginning of every frozen new-person test clip. This is an
offline original-video replay, not an HTTP benchmark or live camera FPS test.
Both model versions receive exactly the same SDK-encoded/JPEG-decoded BGR input
after source orientation correction. Detector, quality, and temporal rules run
again. No HTTP request is sent and no network timing is measured.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pipeline import Pipeline, TemporalDecision  # noqa: E402
from clients.antispoof_client import encode_jpeg  # noqa: E402
from scripts.evaluate_retraining_runtime import RUNTIME_CONFIGURATION, sha256_file, summarize  # noqa: E402
from scripts.prepare_retraining_data import safe_source_path  # noqa: E402

PREFIX_SECONDS = 12.0
TARGET_SAMPLE_FPS = 5.0


def prefix_indices(frame_count: int, fps: float):
    """Chronological fixed-rate selection, excluding timestamps at/after 12 s."""
    if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count <= 0:
        raise ValueError("Video frame count must be a positive integer")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Video FPS must be finite and positive")
    step = max(1, math.ceil(fps / TARGET_SAMPLE_FPS))
    stop = min(frame_count, math.ceil(PREFIX_SECONDS * fps))
    return [index for index in range(0, stop, step) if index / fps < PREFIX_SECONDS]


def prepare_api_input(frame):
    """Apply the existing SDK's default transport transform exactly once."""
    encoded = encode_jpeg(frame)  # Existing defaults: max_edge=960, JPEG quality=90.
    decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8),
                           cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if decoded is None or decoded.size == 0:
        raise RuntimeError("Could not decode the SDK-generated JPEG")
    metadata = {"original_shape": list(frame.shape), "processed_shape": list(decoded.shape),
                "jpeg_bytes": len(encoded), "jpeg_sha256": hashlib.sha256(encoded).hexdigest()}
    return decoded, metadata


def classify_sample(engine, temporal, frame, frame_index, fps, video):
    """Run Pipeline and apply the same one-face transaction rule as the API."""
    now = frame_index / fps
    faces = engine.process(frame, now, temporal)
    if not faces:
        status, input_status = "no_face", "no_face"
    elif len(faces) > 1:
        status, input_status = "multiple_faces", "multiple_faces"
        temporal.tracks.clear()
    else:
        status = faces[0]["status"]
        input_status = "valid" if faces[0]["quality"] is None else "adjust"
    if status not in ("fake", "real", "checking", "uncertain", "adjust", "no_face", "multiple_faces"):
        raise ValueError(f"Pipeline returned an unknown status: {status}")
    clean_faces = [{
        "box": [int(value) for value in face["box"]],
        "detector_score": float(face["detector_score"]),
        "quality": face["quality"], "track_id": int(face["track_id"]),
        "status": face["status"],
        "scores": None if face["scores"] is None else {label: float(score) for label, score in face["scores"].items()},
    } for face in faces]
    return {"video_id": video["video_id"], "subject_id": video["subject_id"],
            "true_label": video["label"], "frame_index": frame_index,
            "time_seconds": now, "input_status": input_status, "runtime_status": status,
            "shape": list(frame.shape), "frame_pixel_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
            "face_count": len(faces), "faces": clean_faces}


def selected_test_videos(manifest):
    if manifest.get("status") != "complete":
        raise ValueError("Only a complete prepared manifest may be evaluated")
    videos = [video for video in manifest["videos"] if video["split"] == "test"]
    if not videos or {video["label"] for video in videos} != {"fake", "real"}:
        raise ValueError("Frozen test videos must include both fake and real")
    if len({video["video_id"] for video in videos}) != len(videos):
        raise ValueError("Frozen test video IDs must be unique")
    test_subjects = set(manifest["split_policy"]["test_subjects"])
    if any(video["subject_id"] not in test_subjects for video in videos):
        raise ValueError("A test video is outside the frozen held-out subject groups")
    for video in videos:
        path = safe_source_path(video["relative_path"], ROOT)
        if not path.is_file() or path.stat().st_size != video["bytes"]:
            raise ValueError(f"Video file size differs from preparation: {path}")
        if sha256_file(path) != video["sha256"]:
            raise ValueError(f"Video file changed after the test split was frozen: {path}")
    return videos


def evaluate(manifest, engines):
    videos = selected_test_videos(manifest)
    totals = {phase: [] for phase in engines}
    per_video = {phase: [] for phase in engines}
    decoded_video_metadata = []
    for video in videos:
        path = safe_source_path(video["relative_path"], ROOT)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot open frozen test video: {path}")
        decisions = {phase: [] for phase in engines}
        # Independent histories for each model and clip, with identical defaults.
        temporal = {phase: TemporalDecision(**RUNTIME_CONFIGURATION) for phase in engines}
        try:
            orientation_set = bool(capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = prefix_indices(frame_count, fps)
            metadata = {"video_id": video["video_id"], "relative_path": video["relative_path"],
                        "subject_id": video["subject_id"], "true_label": video["label"],
                        "source_video_sha256": video["sha256"], "fps": fps,
                        "frame_count": frame_count, "duration_seconds": frame_count / fps,
                        "metadata_width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "metadata_height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        "orientation_degrees": float(capture.get(cv2.CAP_PROP_ORIENTATION_META)),
                        "orientation_auto_set": orientation_set,
                        "orientation_auto_active": bool(capture.get(cv2.CAP_PROP_ORIENTATION_AUTO)),
                        "sampled_frame_indices": indices,
                        "effective_sampling_fps": fps / max(1, math.ceil(fps / TARGET_SAMPLE_FPS))}
            if not metadata["orientation_auto_active"]:
                raise RuntimeError(f"Decoder did not enable orientation correction: {path}")
            current_index = -1
            for frame_index in indices:
                while current_index + 1 < frame_index:
                    if not capture.grab():
                        raise RuntimeError(f"Decoder ended before selected frame {frame_index}: {path}")
                    current_index += 1
                success, frame = capture.read()
                current_index += 1
                if not success or frame is None or frame.size == 0 or current_index != frame_index:
                    raise RuntimeError(f"Cannot read selected frame {frame_index}: {path}")
                processed_frame, transport_metadata = prepare_api_input(frame)
                for phase, engine in engines.items():
                    decision = classify_sample(engine, temporal[phase], processed_frame, frame_index, fps, video)
                    decision.update(transport_metadata)
                    decisions[phase].append(decision)
            metadata["sample_count"] = len(indices)
            decoded_video_metadata.append(metadata)
        finally:
            capture.release()
        progress = {"event": "continuous_video_evaluated", "video_id": video["video_id"],
                    "subject_id": video["subject_id"], "label": video["label"], "sample_count": len(indices)}
        for phase in engines:
            result = summarize(decisions[phase])
            result.update({"video_id": video["video_id"], "subject_id": video["subject_id"],
                           "true_label": video["label"], "relative_path": video["relative_path"],
                           "decisions": decisions[phase]})
            per_video[phase].append(result)
            totals[phase].extend(decisions[phase])
            progress[phase] = {key: result[key] for key in
                               ("conclusive_coverage", "correct_conclusive_count", "wrong_conclusive_count", "status_counts")}
        print(json.dumps(progress, ensure_ascii=False), flush=True)
    result = {"video_metadata": decoded_video_metadata}
    for phase in engines:
        result[phase] = {**summarize(totals[phase]), "per_video": per_video[phase],
                         "video_count": len(videos),
                         "subjects": sorted({video["subject_id"] for video in videos})}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset/retrain-20260930")
    parser.add_argument("--baseline-model", type=Path, default=ROOT / "outputs/retrain-20260915/liveness.h5")
    parser.add_argument("--baseline-labels", type=Path, default=ROOT / "outputs/retrain-20260915/le.pickle")
    parser.add_argument("--candidate-model", type=Path, default=ROOT / "outputs/retrain-20260930/liveness.h5")
    parser.add_argument("--candidate-labels", type=Path, default=ROOT / "outputs/retrain-20260930/le.pickle")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/retrain-20260930/continuous_runtime_comparison.json")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite previous continuous comparison: {args.output}")
    for path in (args.baseline_model, args.baseline_labels, args.candidate_model, args.candidate_labels):
        if not path.is_file():
            raise FileNotFoundError(f"Missing model/labels: {path}")
    manifest_path = args.dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pipeline_hash = sha256_file(ROOT / "pipeline.py")
    if manifest["preprocessing"]["pipeline_sha256"] != pipeline_hash:
        raise ValueError("Runtime pipeline changed after dataset preparation")
    # Configure before either model is loaded; no training is performed.
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(args.cpu_threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    cv2.setNumThreads(2)
    engines = {"before": Pipeline(model_path=args.baseline_model, labels_path=args.baseline_labels),
               "after": Pipeline(model_path=args.candidate_model, labels_path=args.candidate_labels)}
    if any(tuple(engine.labels) not in (("fake", "real"), ("real", "fake")) for engine in engines.values()):
        raise ValueError("Unexpected model output labels")
    started = time.perf_counter()
    report = {
        "schema_version": "1.0", "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "Fixed chronological first-12-second replay at no more than 5 FPS through SDK JPEG encoding/decoding and actual Pipeline.process",
        "selection_plan": {"prefix_seconds": PREFIX_SECONDS, "target_sample_fps": TARGET_SAMPLE_FPS,
                           "start_seconds": 0, "end_exclusive": True,
                           "frame_step": "max(1, ceil(source_fps / 5))",
                           "timestamp": "frame_index / source_fps",
                           "chosen_before_observing_scores": True},
        "runtime_configuration": RUNTIME_CONFIGURATION,
        "transport_transform": {"function": "clients.antispoof_client.encode_jpeg with existing defaults",
                                "max_edge_pixels": 960, "jpeg_quality": 90,
                                "source_orientation": "OpenCV CAP_PROP_ORIENTATION_AUTO enabled before sampling",
                                "api_decode_flags": "IMREAD_COLOR | IMREAD_IGNORE_ORIENTATION",
                                "pairing": "One JPEG is encoded and decoded once per sample; both pipelines receive the same pixels"},
        "runtime": {"tensorflow_version": tf.__version__, "opencv_version": cv2.__version__,
                    "numpy_version": np.__version__, "tensorflow_intra_threads": args.cpu_threads,
                    "tensorflow_inter_threads": 1, "opencv_threads": 2},
        "input_hashes": {"manifest_sha256": sha256_file(manifest_path), "pipeline_sha256": pipeline_hash,
                         "evaluation_script_sha256": sha256_file(__file__),
                         "client_source_sha256": sha256_file(ROOT / "clients/antispoof_client.py"),
                         "detector_sha256": sha256_file(ROOT / "face_detector/face_detection_yunet_2023mar.onnx")},
        "models": {phase: {"model_path": str(model.resolve()), "labels_path": str(labels.resolve()),
                           "model_sha256": sha256_file(model), "labels_sha256": sha256_file(labels),
                           "label_order": list(engines[phase].labels)}
                   for phase, model, labels in (("before", args.baseline_model, args.baseline_labels),
                                               ("after", args.candidate_model, args.candidate_labels))},
        "limitations": [
            "Offline replay measures decision outcomes from stored videos; it does not measure HTTP/network latency or live-camera FPS.",
            "Only the fixed first 12 seconds are evaluated, at at most 5 sampled frames per second. Later events in clips are outside this test.",
            "The existing SDK resize/JPEG transform is applied once; both model pipelines receive identical JPEG-decoded pixels and fresh temporal histories per video.",
            "This approximates API input transformation without sending HTTP requests. Matching wire encoding does not remove source-video acquisition bias.",
            "Face detector and input-quality failures remain in all-sample denominators; error rates must be read alongside conclusive coverage.",
            "The frozen test covers only two held-out people and label-dependent recording conditions; it does not establish broad attack or camera robustness.",
        ],
    }
    report.update(evaluate(manifest, engines))
    report["offline_evaluation_wall_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output.resolve()),
                      **{phase: {key: report[phase][key] for key in
                                 ("all_sample_count", "conclusive_coverage", "accuracy_among_conclusive",
                                  "fake_as_real_count", "real_as_fake_count")}
                         for phase in engines}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
