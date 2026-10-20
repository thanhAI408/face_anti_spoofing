"""Replay stored test samples through the camera/API temporal decision rules.

This is an offline replay of previously detected faces, input-quality outcomes,
and saved CNN probabilities. It does not reload models or measure live-camera
FPS, HTTP performance, or presentation-attack security on an external server.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pipeline import TemporalDecision  # noqa: E402

STATUSES = ("fake", "real", "checking", "uncertain", "adjust", "no_face", "multiple_faces")
RUNTIME_CONFIGURATION = {
    "threshold": .8, "min_samples": 6, "min_seconds": .4,
    "max_gap": .5, "window_seconds": 1.5,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def summarize(decisions):
    """Retain abstentions in all-sample denominators and report their coverage."""
    counts = Counter(item["runtime_status"] for item in decisions)
    total = len(decisions)
    conclusive = sum(counts[label] for label in ("fake", "real"))
    correct = sum(item["runtime_status"] == item["true_label"] for item in decisions)
    incorrect = conclusive - correct
    quality_valid = sum(item["input_status"] == "valid" for item in decisions)
    class_summary = {}
    for label, wrong_label in (("fake", "real"), ("real", "fake")):
        subset = [item for item in decisions if item["true_label"] == label]
        class_counts = Counter(item["runtime_status"] for item in subset)
        n = len(subset)
        class_conclusive = class_counts["fake"] + class_counts["real"]
        wrong = class_counts[wrong_label]
        error_fraction = fraction(wrong, n)
        class_summary[label] = {
            "all_sample_count": n, "status_counts": {status: class_counts[status] for status in STATUSES},
            "conclusive_count": class_conclusive, "correct_conclusive_count": class_counts[label],
            "wrong_conclusive_count": wrong, "no_decision_count": n - class_conclusive,
            "conclusive_coverage": fraction(class_conclusive, n),
            "correct_fraction_of_all_samples": fraction(class_counts[label], n),
            "wrong_fraction_of_all_samples": error_fraction,
            "wrong_percent_of_all_samples": 100 * error_fraction if error_fraction is not None else None,
            "accuracy_among_conclusive": fraction(class_counts[label], class_conclusive),
        }
    return {
        "all_sample_count": total,
        "status_counts": {status: counts[status] for status in STATUSES},
        "quality_valid_count": quality_valid,
        "quality_valid_coverage": fraction(quality_valid, total),
        "conclusive_count": conclusive, "no_decision_count": total - conclusive,
        "conclusive_coverage": fraction(conclusive, total),
        "correct_conclusive_count": correct, "wrong_conclusive_count": incorrect,
        "accuracy_among_conclusive": fraction(correct, conclusive),
        "correct_fraction_of_all_samples": fraction(correct, total),
        "wrong_fraction_of_all_samples": fraction(incorrect, total),
        "per_true_class": class_summary,
        "fake_as_real_count": class_summary["fake"]["wrong_conclusive_count"],
        "fake_as_real_fraction_of_all_fake_samples": class_summary["fake"]["wrong_fraction_of_all_samples"],
        "fake_as_real_percent_of_all_fake_samples": class_summary["fake"]["wrong_percent_of_all_samples"],
        "real_as_fake_count": class_summary["real"]["wrong_conclusive_count"],
        "real_as_fake_fraction_of_all_real_samples": class_summary["real"]["wrong_fraction_of_all_samples"],
        "real_as_fake_percent_of_all_real_samples": class_summary["real"]["wrong_percent_of_all_samples"],
        "true_label_order": ["fake", "real"], "decision_order": list(STATUSES),
        "confusion_matrix_including_no_decisions": [
            [class_summary[label]["status_counts"][status] for status in STATUSES] for label in ("fake", "real")
        ],
        "rate_units": {"coverage_and_fraction": "fraction in [0,1]", "percent": "percent in [0,100]"},
        "error_rate_denominator": "All sampled frames of the respective true class, including frames with no decision",
    }


def replay_test(manifest, probabilities, labels=("fake", "real")):
    """Map N x 2 test predictions to global array indices and replay each clip."""
    labels = tuple(labels)
    if len(labels) != 2 or set(labels) != {"fake", "real"}:
        raise ValueError("Output labels must contain fake and real exactly once")
    records = manifest["frames"]
    if any(record.get("array_index") != index for index, record in enumerate(records)):
        raise ValueError("Manifest frame array indices must match their original record order")
    test_records = [record for record in records if record["split"] == "test"]
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(test_records), 2):
        raise ValueError("Saved probabilities must follow the manifest's test-record order, with N x 2 shape")
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Probabilities must be finite and lie in [0,1]")
    if not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-5):
        raise ValueError("Probability rows must sum to 1")
    by_array_index = {record["array_index"]: (record, probabilities[i]) for i, record in enumerate(test_records)}
    test_videos = [video for video in manifest["videos"] if video["split"] == "test"]
    if not test_videos:
        raise ValueError("Manifest must contain test videos")
    if len({video["video_id"] for video in test_videos}) != len(test_videos):
        raise ValueError("Repeated test video ID")
    if {r["video_id"] for r in test_records} - {v["video_id"] for v in test_videos}:
        raise ValueError("Test records reference a video absent from manifest.videos")
    all_decisions, per_video = [], []
    for video in test_videos:
        if video["label"] not in labels:
            raise ValueError("Video has an unknown true label")
        temporal = TemporalDecision(**RUNTIME_CONFIGURATION)
        events = video.get("sample_events")
        if not isinstance(events, list) or not events:
            raise ValueError(f"Video {video['video_id']} has no sampled events")
        previous_time, previous_frame = -math.inf, -1
        decisions, input_rejections = [], Counter()
        used_arrays = set()
        for event in events:
            now, frame_index = event["time_seconds"], event["frame_index"]
            if not isinstance(now, (int, float)) or not math.isfinite(now) or now < 0 or now <= previous_time:
                raise ValueError("Sample times must be finite, nonnegative, and strictly chronological")
            if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index <= previous_frame:
                raise ValueError("Sample frame indices must be strictly chronological nonnegative integers")
            previous_time, previous_frame = now, frame_index
            input_status = event["status"]
            decision = {"video_id": video["video_id"], "subject_id": video["subject_id"],
                        "true_label": video["label"], "frame_index": frame_index,
                        "time_seconds": now, "input_status": input_status}
            if input_status in ("no_face", "multiple_faces"):
                temporal.associate([])
                status, mean = input_status, None
                input_rejections[input_status] += 1
            elif input_status in ("valid", "adjust"):
                box = event.get("box")
                if not isinstance(box, (list, tuple)) or len(box) != 4 or not all(math.isfinite(x) for x in box):
                    raise ValueError("Single-face events require a finite four-coordinate box")
                if box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError("Face box must have positive dimensions")
                track = temporal.associate([tuple(box)])[0]
                probs = None
                if input_status == "valid":
                    array_index = event.get("array_index")
                    if array_index not in by_array_index:
                        raise ValueError("Valid test event references an absent test prediction row")
                    record, probs = by_array_index[array_index]
                    if (record["video_id"], record["label"], record["subject_id"]) != (video["video_id"], video["label"], video["subject_id"]):
                        raise ValueError("Event prediction row belongs to a different video/label/subject")
                    used_arrays.add(array_index)
                    decision["array_index"] = array_index
                    decision["current_scores"] = {label: float(probs[i]) for i, label in enumerate(labels)}
                else:
                    reason = event.get("quality_reason")
                    if not isinstance(reason, str) or not reason:
                        raise ValueError("Adjust events must record their input-quality rejection reason")
                    decision["quality_reason"] = reason
                    input_rejections["adjust: " + reason] += 1
                result, mean = temporal.update(track, probs, now)
                status = labels[result] if isinstance(result, int) else result
                decision["track_id"] = track.id
                decision["box"] = list(box)
            else:
                raise ValueError(f"Cannot replay unrecognized or undecodable input status: {input_status}")
            decision["runtime_status"] = status
            decision["mean_scores"] = None if mean is None else {label: float(mean[i]) for i, label in enumerate(labels)}
            decisions.append(decision)
        expected_arrays = {record["array_index"] for record in test_records if record["video_id"] == video["video_id"]}
        if used_arrays != expected_arrays:
            raise ValueError(f"Sample events do not cover all stored prediction rows of {video['video_id']}")
        summary = summarize(decisions)
        summary.update({"video_id": video["video_id"], "subject_id": video["subject_id"],
                        "true_label": video["label"], "relative_path": video.get("relative_path"),
                        "input_rejection_counts": dict(input_rejections), "decisions": decisions})
        per_video.append(summary)
        all_decisions.extend(decisions)
    result = summarize(all_decisions)
    total_rejections = Counter()
    for video in per_video:
        total_rejections.update(video["input_rejection_counts"])
    result.update({"label_order": list(labels), "video_count": len(test_videos),
                   "subject_count": len({video["subject_id"] for video in test_videos}),
                   "subjects": sorted({video["subject_id"] for video in test_videos}),
                   "per_video": per_video,
                   "input_rejection_counts": dict(total_rejections)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset/retrain-20260930")
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/retrain-20260930")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run / "runtime_comparison.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prior runtime comparison: {output}")
    manifest_path, evaluation_path = args.dataset / "manifest.json", args.run / "evaluation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Only a complete prepared manifest can be evaluated")
    manifest_hash = sha256_file(manifest_path)
    configuration = evaluation["configuration"]
    if configuration["dataset_manifest_sha256"] != manifest_hash:
        raise ValueError("Prepared manifest changed since model evaluation")
    pipeline_hash = sha256_file(ROOT / "pipeline.py")
    if manifest["preprocessing"]["pipeline_sha256"] != pipeline_hash:
        raise ValueError("Runtime pipeline changed since stored detector/quality events were prepared")
    report = {
        "schema_version": "1.0", "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "Offline replay of stored YuNet/quality sample events and saved model predictions through pipeline.TemporalDecision",
        "runtime_configuration": RUNTIME_CONFIGURATION,
        "sampling_configuration": {key: manifest["preprocessing"][key]
                                   for key in ("min_interval_seconds", "max_sampled_frames_per_video")},
        "input_hashes": {"manifest_sha256": manifest_hash,
                         "evaluation_sha256": sha256_file(evaluation_path),
                         "pipeline_sha256": pipeline_hash,
                         "runtime_evaluation_script_sha256": sha256_file(__file__)},
        "models": {"before_model_sha256": configuration["baseline_model_sha256"],
                   "after_model_sha256": evaluation["new_model_sha256"]},
        "limitations": [
            "Same stored detector/quality results and sampled timestamps are replayed before/after; neither models nor videos are reloaded.",
            "This is a sampled offline replay, not a live webcam, HTTP, latency, or FPS measurement.",
            "Uniform sampling is at least 0.2 seconds apart and at most 100 observations per video; larger sample gaps can keep runtime checking.",
            "Error fractions retain all samples including abstentions. Read error rates together with conclusive coverage.",
            "Test includes only the two new held-out people, so the result does not establish generalization across new cameras or attacks.",
        ],
    }
    for phase in ("before", "after"):
        probability_path = args.run / f"{phase}_test_probabilities.npy"
        probabilities = np.load(probability_path, allow_pickle=False)
        labels = evaluation[phase]["test"]["label_order"]
        report[phase] = replay_test(manifest, probabilities, labels)
        report["input_hashes"][phase + "_test_probabilities_sha256"] = sha256_file(probability_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output.resolve()),
                      **{phase: {key: report[phase][key] for key in
                                 ("all_sample_count", "conclusive_coverage", "accuracy_among_conclusive",
                                  "fake_as_real_count", "real_as_fake_count")}
                         for phase in ("before", "after")}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
