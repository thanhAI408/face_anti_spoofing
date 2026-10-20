"""Independently audit saved merged-data training and evaluation evidence.

No TensorFlow, retraining, activation, dataset mutation, or external requests.
Only outputs/merged-20261001/audit.json is written. Missing/invalid evidence
never produces passed:true. Metrics, hashes, counts, boundaries, and history
are checked before slides or reports should consume this run.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.retraining_metrics import evaluate_probabilities, paired_video_bootstrap

RUN = ROOT / "outputs/merged-20261001"
DATA = ROOT / "dataset/merged-20261001"
PRIOR_RUN = ROOT / "outputs/retrain-20260930"
PRIOR_DATA = ROOT / "dataset/retrain-20260930"
LEGACY_RUN = ROOT / "outputs/retrain-20260915"
SPLITS = ("train", "validation", "test")
PHASES = ("before", "after")
LABELS = ("fake", "real")
SOURCES = ("legacy-cache", "drive-20260930")
STATUSES = ("fake", "real", "checking", "uncertain", "adjust", "no_face", "multiple_faces")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def equal(actual, expected):
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(equal(actual[k], expected[k]) for k in actual)
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(equal(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return bool(np.isclose(actual, expected, rtol=1e-8, atol=1e-10))
    return actual == expected


def summary(records):
    return {split: {
        "frames": len(part := [r for r in records if r["split"] == split]),
        "classes": dict(Counter(r["label"] for r in part)),
        "videos": len({r["video_id"] for r in part}),
        "subjects": sorted({r["subject_id"] for r in part}),
        "identity_unknown_frames": sum(r.get("identity_unknown", False) for r in part),
        "sources": {source: {
            "frames": len(group := [r for r in part if r["source"] == source]),
            "classes": dict(Counter(r["label"] for r in group)),
            "videos": len({r["video_id"] for r in group}),
        } for source in SOURCES},
    } for split in SPLITS}


def independent_frame_metrics(probabilities, records, labels):
    """Direct confusion-matrix formula independent of report metric helper."""
    targets = np.asarray([labels.index(r["label"]) for r in records], dtype=np.int64)
    predicted = np.asarray(probabilities).argmax(axis=1)
    matrix = np.bincount(targets * 2 + predicted, minlength=4).reshape(2, 2)
    diagonal = np.diag(matrix).astype(float)
    precision = np.divide(diagonal, matrix.sum(axis=0), out=np.zeros(2), where=matrix.sum(axis=0) != 0)
    recall = np.divide(diagonal, matrix.sum(axis=1), out=np.zeros(2), where=matrix.sum(axis=1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(2), where=(precision + recall) != 0)
    loss = -np.log(np.clip(np.asarray(probabilities, dtype=np.float64)[np.arange(len(targets)), targets], 1e-7, 1 - 1e-7)).mean()
    return {"confusion_matrix": matrix.tolist(), "accuracy": float(diagonal.sum() / len(targets)),
            "loss": float(loss), "macro_precision": float(precision.mean()),
            "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean())}


def runtime_counts(videos):
    decisions = [d for v in videos for d in v["decisions"]]
    statuses = Counter(d["runtime_status"] for d in decisions)
    correct = sum(d["runtime_status"] == d["true_label"] for d in decisions)
    wrong = sum(d["runtime_status"] in LABELS and d["runtime_status"] != d["true_label"] for d in decisions)
    return {"all_sample_count": len(decisions), "status_counts": {s: statuses[s] for s in STATUSES},
            "correct_conclusive_count": correct, "wrong_conclusive_count": wrong,
            "conclusive_count": correct + wrong, "no_decision_count": len(decisions) - correct - wrong,
            "conclusive_coverage": (correct + wrong) / len(decisions),
            "correct_fraction_of_all_samples": correct / len(decisions)}


def weight_fingerprint(path):
    """Read checkpoint/export tensors directly without loading TensorFlow."""
    import h5py
    if Path(path).suffix == ".keras":
        with zipfile.ZipFile(path) as archive:
            content = io.BytesIO(archive.read("model.weights.h5"))
    else:
        content = path
    weights = []
    with h5py.File(content, "r") as handle:
        root = handle["layers"] if "layers" in handle else handle["model_weights"]

        def visit(name, item):
            if isinstance(item, h5py.Dataset):
                array = item[()]
                weights.append((tuple(array.shape), str(array.dtype), hashlib.sha256(array.tobytes()).hexdigest()))

        root.visititems(visit)
    return Counter(weights)


def audit(check, artifact, check_hash):
    manifest = read_json(DATA / "manifest.json")
    evaluation = read_json(RUN / "evaluation.json")
    configuration = read_json(RUN / "configuration.json")
    prior = read_json(PRIOR_DATA / "manifest.json")
    prior_evaluation = read_json(PRIOR_RUN / "evaluation.json")
    legacy_frames = read_json(LEGACY_RUN / "frame_manifest.json")
    legacy_split = read_json(LEGACY_RUN / "split.json")
    frames, videos = manifest["frames"], manifest["videos"]
    check("manifest_completed", manifest["status"] == "complete", {"frames": len(frames), "videos": len(videos)})
    check("configuration_matches_evaluation", equal(configuration, evaluation["configuration"]), {})
    check("manifest_summaries_recomputed", equal(summary(frames), manifest["summary"]), summary(frames))
    check("dataset_summary_in_evaluation", equal(evaluation["dataset_summary"], manifest["summary"]), {})
    check_hash("manifest_configuration_sha", DATA / "manifest.json", configuration["dataset_manifest_sha256"])
    check_hash("archive_manifest_sha", DATA / manifest["array"]["path"], manifest["array"]["sha256"])
    check_hash("baseline_is_correct_sep30_model", PRIOR_RUN / "liveness.h5", configuration["baseline_model_sha256"])
    check("baseline_matches_prior_candidate", configuration["baseline_model_sha256"] == prior_evaluation["new_model_sha256"]
          and (ROOT / configuration["baseline_model_path"]).resolve() == (PRIOR_RUN / "liveness.h5").resolve(), {})
    check_hash("candidate_model_sha", RUN / "liveness.h5", evaluation["new_model_sha256"])
    check("candidate_differs_from_baseline", evaluation["new_model_sha256"] != configuration["baseline_model_sha256"], {})
    check_hash("baseline_labels_sha", PRIOR_RUN / "le.pickle", configuration["baseline_labels_sha256"])
    labels = {phase: tuple(pickle.loads((directory / "le.pickle").read_bytes()).classes_)
              for phase, directory in (("before", PRIOR_RUN), ("after", RUN))}
    check("both_label_orders_compatible", all(value == LABELS for value in labels.values()), {k: list(v) for k, v in labels.items()})
    artifact(RUN / "le.pickle")
    for key, path in (("training_script_sha256", ROOT / "scripts/train_retraining_comparison.py"),
                      ("architecture_source_sha256", ROOT / "model/livenessnet.py")):
        check_hash(key, path, configuration[key])
    check_hash("preparation_script_sha", ROOT / "scripts/prepare_merged_data.py", manifest["preparation_script_sha256"])
    for key, path in (("pipeline_sha256", ROOT / "pipeline.py"),
                      ("detector_sha256", ROOT / "face_detector/face_detection_yunet_2023mar.onnx")):
        check_hash(key, path, manifest["preprocessing"][key])
    for key, path in (("legacy_frame_manifest", LEGACY_RUN / "frame_manifest.json"),
                      ("legacy_split_manifest", LEGACY_RUN / "split.json"),
                      ("drive_inventory", ROOT / "outputs/drive-20260930/inventory-verified.json"),
                      ("before_merge_dataset_manifest", PRIOR_DATA / "manifest.json")):
        check_hash("lineage_" + key, path, manifest["source_lineage"][key + "_sha256"])
    with np.load(DATA / manifest["array"]["path"], allow_pickle=False) as archive:
        pixels = archive[manifest["array"]["key"]]
    check("archive_shape_dtype", pixels.dtype == np.uint8 and list(pixels.shape) == manifest["array"]["shape"]
          and pixels.shape == (len(frames), 32, 32, 3), {"shape": list(pixels.shape), "dtype": str(pixels.dtype)})
    check("array_row_order", all(r["array_index"] == i for i, r in enumerate(frames)), {})
    pixel_hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in pixels]
    check("every_archive_row_matches_pixel_hash", all(d == r["crop_pixel_sha256"] == r["frame_pixel_sha256"]
          for d, r in zip(pixel_hashes, frames)), {"rows": len(pixel_hashes)})
    check("pixels_globally_deduplicated", len(set(pixel_hashes)) == len(pixel_hashes), {})
    del pixels

    grouped = defaultdict(list)
    for record in frames:
        grouped[record["video_id"]].append(record)
    check("video_metadata_consistent", all(len({(r["split"], r["label"], r["subject_id"], r["source"])
          for r in group}) == 1 for group in grouped.values()), {"videos": len(grouped)})
    check("manifest_unique_video_ids", len({v["video_id"] for v in videos}) == len(videos), {})
    check("video_accepted_counts", all(v["accepted_frames"] == len(grouped.get(v["video_id"], [])) for v in videos), {})
    video_sets = {split: {r["video_id"] for r in frames if r["split"] == split} for split in SPLITS}
    check("all_source_videos_disjoint_across_splits", all(not video_sets[a] & video_sets[b]
          for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))),
          {k: sorted(v) for k, v in video_sets.items()})
    subjects = {split: {r["subject_id"] for r in frames if r["split"] == split and r["source"] == "drive-20260930"}
                for split in SPLITS}
    check("drive_folder_ids_disjoint", all(not subjects[a] & subjects[b]
          for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))),
          {k: sorted(v) for k, v in subjects.items()})
    check("drive_frozen_subjects", subjects["validation"] == {"b", "thu"} and subjects["test"] == {"nga", "phuc"}, {})
    original_lookup = {path.replace("\\", "/"): split for split, paths in legacy_frames.items() for path in paths}
    legacy_accepted = [r for r in frames if r["source"] == "legacy-cache"]
    legacy_excluded = [r for r in manifest["exclusions"] if r.get("source") == "legacy-cache"]
    cache_records = legacy_accepted + legacy_excluded
    cache_paths = [r["relative_path"] for r in cache_records]
    disk = {p.relative_to(ROOT).as_posix() for label in LABELS for p in (ROOT / "dataset" / label).glob("*.png")}
    check("all_legacy_images_accounted_for", len(cache_paths) == 24221 and len(set(cache_paths)) == len(cache_paths)
          and set(cache_paths) == set(original_lookup) == disk, {"candidate": len(cache_paths), "accepted": len(legacy_accepted), "excluded": len(legacy_excluded)})
    check("legacy_original_split_preserved", all(original_lookup[r["relative_path"]] == r["split"] == r["legacy_split"]
          and r["video_id"].removeprefix("old:") in legacy_split[r["split"]]["videos"] for r in cache_records), {})
    check("legacy_quality_exclusion_counts", dict(Counter(r["quality_reason"] for r in legacy_excluded
          if r["reason"] == "legacy_crop_quality_rejected")) == manifest["legacy_quality_rejections"], manifest["legacy_quality_rejections"])
    check("accepted_legacy_crop_measurements_pass_gates", all(
        min(r["original_crop_shape"][:2]) >= 65 and r["quality_measurements"]["gray_mean"] >= 35
        and r["quality_measurements"]["glare_fraction"] <= .45
        and r["quality_measurements"]["sharpness"] >= 12 and r["quality_measurements"]["gray_std"] > 0
        for r in legacy_accepted), {"checked_crops": len(legacy_accepted), "scope": "Stored crop measurements; raw full-frame pose/landmarks unavailable"})
    image_mismatches = [r["relative_path"] for r in cache_records if sha(ROOT / r["relative_path"]) != r["image_file_sha256"]]
    check("all_legacy_source_file_hashes", not image_mismatches, {"checked_images": len(cache_records), "mismatches": image_mismatches[:10]})
    drive_videos = [v for v in videos if v["source"] == "drive-20260930"]
    check("all_drive_sources_accounted_for", len(drive_videos) == 56 and Counter(v["label"] for v in drive_videos) == {"fake": 26, "real": 30}, {})
    inventory = read_json(ROOT / "outputs/drive-20260930/inventory-verified.json")
    entries = {"drive:" + r["id"]: r for r in inventory["files"]}
    check("drive_inventory_labels_paths_sizes_match", set(entries) == {v["video_id"] for v in drive_videos}
          and all(v["label"] == entries[v["video_id"]]["label"]
                  and v["subject_id"] == entries[v["video_id"]]["subject_id"]
                  and v["relative_path"] == entries[v["video_id"]]["relative_path"]
                  and v["bytes"] == int(entries[v["video_id"]]["size"]) for v in drive_videos), {})
    source_mismatches = [v["relative_path"] for v in drive_videos
                         if (ROOT / v["relative_path"]).stat().st_size != v["bytes"] or sha(ROOT / v["relative_path"]) != v["sha256"]]
    check("all_drive_video_file_hashes", not source_mismatches, {"checked_videos": len(drive_videos), "mismatches": source_mismatches})
    check("new_sampling_uncapped", manifest["preprocessing"]["max_sampled_frames_per_video"] is None and all(
          v["sampled_frame_indices"] == list(range(0, v["frame_count"], max(1, math.ceil(v["fps"] * manifest["preprocessing"]["min_interval_seconds"]))))
          for v in drive_videos), {})
    sampled = sum(v["statistics"]["sampled"] for v in drive_videos)
    accepted_new = sum(r["source"] == "drive-20260930" for r in frames)
    new_events = Counter(e["status"] for v in drive_videos for e in v["sample_events"])
    check("drive_sampling_accounting", sampled == sum(new_events.values()) and new_events["valid"] == accepted_new
          and sampled == accepted_new + new_events["adjust"] + new_events["no_face"] + new_events["multiple_faces"]
          + new_events["excluded_duplicate_pixels"], {"sampled": sampled, "accepted": accepted_new, "events": dict(new_events)})
    check("drive_event_array_indices_aligned", all(
        0 <= e["array_index"] < len(frames) and frames[e["array_index"]]["video_id"] == v["video_id"]
        and frames[e["array_index"]]["frame_index"] == e["frame_index"]
        for v in drive_videos for e in v["sample_events"] if e["status"] == "valid"), {})

    training = {}
    for phase, directory in (("before", PRIOR_RUN), ("after", RUN)):
        history = read_json(directory / "history.json")
        with (directory / "training.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        best = int(np.argmin(history["val_loss"])) + 1
        stored = prior_evaluation if phase == "before" else evaluation
        coherent = len(rows) == stored["epochs_run"] and best == stored["best_epoch"] and all(
            len(values) == len(rows) and all(equal(float(rows[i][key]), value) for i, value in enumerate(values))
            for key, values in history.items()) and all(int(row["epoch"]) == i for i, row in enumerate(rows))
        training[phase] = {"epochs_run": len(rows), "best_epoch": best,
                           "history_at_best": {k: values[best - 1] for k, values in history.items()},
                           "history_at_last": {k: values[-1] for k, values in history.items()}}
        check(phase + "_history_csv_epochs", coherent, training[phase])
        if phase == "after":
            check("candidate_selected_by_min_validation_loss", equal(training[phase]["history_at_best"], evaluation["history_at_best_epoch"])
                  and equal(training[phase]["history_at_last"], evaluation["history_at_last_epoch"]), {})
    baseline_metadata = configuration["baseline_training"]
    check("baseline_historical_metadata_correct_run", baseline_metadata["status"] == "available"
          and baseline_metadata["epochs_run"] == training["before"]["epochs_run"]
          and baseline_metadata["best_epoch"] == training["before"]["best_epoch"]
          and baseline_metadata["model_sha256"] == configuration["baseline_model_sha256"], {})
    for field in ("evaluation", "history"):
        check_hash("baseline_" + field + "_metadata_hash", PRIOR_RUN / (field + ".json"), baseline_metadata[field + "_sha256"])
    progress = read_json(RUN / "progress.json")
    check("training_complete", progress["phase"] == "complete" and progress["epochs_run"] == evaluation["epochs_run"]
          and progress["best_epoch"] == evaluation["best_epoch"], progress)
    export = configuration["export_compatibility"]
    check("candidate_export_compatibility_evidence", export["passed"] and export["checked_frames"] == 8
          and export["input_shape"] == [None, 32, 32, 3] and export["labels"] == list(LABELS)
          and export["max_probability_difference"] <= 1e-6, export)
    checkpoint_weights = weight_fingerprint(RUN / "best.keras")
    export_weights = weight_fingerprint(RUN / "liveness.h5")
    check("export_weights_exactly_match_selected_checkpoint", bool(checkpoint_weights)
          and checkpoint_weights == export_weights, {"checkpoint_tensors": sum(checkpoint_weights.values()), "export_tensors": sum(export_weights.values())})
    artifact(RUN / "best.keras")

    recomputed = {}
    for phase in PHASES:
        check(phase + "_sidecar_matches_evaluation", equal(read_json(RUN / (phase + ".json")), evaluation[phase]), {})
        recomputed[phase] = {}
        for split in SPLITS:
            records = [r for r in frames if r["split"] == split]
            path = RUN / f"{phase}_{split}_probabilities.npy"
            probabilities = np.load(path, allow_pickle=False)
            actual = evaluate_probabilities(probabilities, records, labels=labels[phase])
            actual["per_source"] = {}
            for source in sorted({r["source"] for r in records}):
                mask = np.asarray([r["source"] == source for r in records])
                actual["per_source"][source] = evaluate_probabilities(probabilities[mask], [r for r in records if r["source"] == source], labels=labels[phase])
            check(f"{phase}_{split}_complete_metrics_reproduced", equal(actual, evaluation[phase][split]),
                  {"frames": len(records), "loss": actual["loss"], "accuracy": actual["accuracy"],
                   "macro": actual["classification_report"]["macro avg"]})
            manual = independent_frame_metrics(probabilities, records, labels[phase])
            check(f"{phase}_{split}_independent_confusion_formula", equal(manual, {
                "confusion_matrix": actual["confusion_matrix"], "accuracy": actual["accuracy"], "loss": actual["loss"],
                "macro_precision": actual["classification_report"]["macro avg"]["precision"],
                "macro_recall": actual["classification_report"]["macro avg"]["recall"],
                "macro_f1": actual["classification_report"]["macro avg"]["f1-score"]}), manual)
            artifact(path)
            recomputed[phase][split] = actual
    for title, stored, before_metrics, after_metrics in [("overall", evaluation["test_video_paired_bootstrap"], recomputed["before"]["test"], recomputed["after"]["test"])] + [
        (source, item, recomputed["before"]["test"]["per_source"][source], recomputed["after"]["test"]["per_source"][source])
        for source, item in evaluation["test_video_paired_bootstrap_by_source"].items()]:
        actual = paired_video_bootstrap(before_metrics, after_metrics, iterations=stored["iterations"], seed=stored["seed"])
        check("bootstrap_" + title, equal(actual, stored), {"iterations": stored["iterations"]})

    check_hash("prior_anchor_archive", PRIOR_DATA / "frames.npz", prior["array"]["sha256"])
    with np.load(PRIOR_DATA / "frames.npz", allow_pickle=False) as archive:
        prior_pixels = archive["x"]
    anchor_records = [r for r in prior["frames"] if r["split"] in ("test", "regression_old")]
    anchor_ids = {r["video_id"] for r in anchor_records}
    train_pixel_hashes = {r["crop_pixel_sha256"] for r in frames if r["split"] == "train"}
    anchor_pixel_hashes = {hashlib.sha256(prior_pixels[r["array_index"]].tobytes()).hexdigest() for r in anchor_records}
    check("all_prior_anchor_videos_out_of_training", not anchor_ids & video_sets["train"], {"anchor_videos": len(anchor_ids)})
    check("all_prior_anchor_pixels_out_of_training", not anchor_pixel_hashes & train_pixel_hashes, {"anchor_frames": len(anchor_records)})
    del prior_pixels
    anchors = read_json(RUN / "anchor_comparison.json")
    check_hash("anchor_prior_manifest_sha", PRIOR_DATA / "manifest.json", anchors["manifest_sha256"])
    check("anchor_model_pairing", anchors["baseline_model_sha256"] == configuration["baseline_model_sha256"]
          and anchors["candidate_model_sha256"] == evaluation["new_model_sha256"], {})
    for key, expected in anchors.get("input_hashes", {}).items():
        check_hash("anchor_input_" + key, {"prior_manifest": PRIOR_DATA / "manifest.json", "prior_archive": PRIOR_DATA / "frames.npz", "merged_manifest": DATA / "manifest.json"}[key], expected)
    for group, split, expected_count in (("new", "test", 1473), ("old", "regression_old", 415)):
        records = [r for r in prior["frames"] if r["split"] == split]
        check("anchor_fixed_count_" + group, len(records) == expected_count, {"frames": len(records)})
        stored = anchors["groups"][group]
        for phase in PHASES:
            path = RUN / stored.get("probability_files", {}).get(phase, f"anchor_{phase}_{group}_probabilities.npy")
            probabilities = np.load(path, allow_pickle=False)
            actual = evaluate_probabilities(probabilities, records, labels=labels[phase])
            check(f"anchor_{phase}_{group}_metrics", equal(actual, stored[phase]), {"frames": len(records), "accuracy": actual["accuracy"], "macro_f1": actual["classification_report"]["macro avg"]["f1-score"]})
            if phase == "before":
                previous = np.load(PRIOR_RUN / f"after_{split}_probabilities.npy", allow_pickle=False)
                check("anchor_baseline_reproduces_prior_" + group, np.allclose(previous, probabilities, rtol=0, atol=1e-6), {})
            artifact(path)
        bootstrap = stored["paired_video_bootstrap"]
        check("anchor_bootstrap_" + group, equal(paired_video_bootstrap(stored["before"], stored["after"], iterations=bootstrap["iterations"], seed=bootstrap["seed"]), bootstrap), {})

    replay = read_json(RUN / "continuous_runtime_comparison.json")
    input_paths = {"manifest_sha256": PRIOR_DATA / "manifest.json", "pipeline_sha256": ROOT / "pipeline.py",
                   "evaluation_script_sha256": ROOT / "scripts/evaluate_continuous_videos.py",
                   "client_source_sha256": ROOT / "clients/antispoof_client.py",
                   "detector_sha256": ROOT / "face_detector/face_detection_yunet_2023mar.onnx"}
    for key, expected in replay["input_hashes"].items():
        check_hash("continuous_input_" + key, input_paths[key], expected)
    replay_counts = {}
    decisions = {}
    for phase in PHASES:
        decisions[phase] = [d for v in replay[phase]["per_video"] for d in v["decisions"]]
        replay_counts[phase] = runtime_counts(replay[phase]["per_video"])
        check("continuous_counts_" + phase, equal(replay_counts[phase], {k: replay[phase][k] for k in replay_counts[phase]}), replay_counts[phase])
        check("continuous_model_" + phase, replay["models"][phase]["model_sha256"] == (configuration["baseline_model_sha256"] if phase == "before" else evaluation["new_model_sha256"]), {})
    pair_fields = ("video_id", "subject_id", "true_label", "frame_index", "time_seconds", "jpeg_sha256", "frame_pixel_sha256", "original_shape", "processed_shape")
    paired = len(decisions["before"]) == len(decisions["after"]) == 942 and all(
        all(a.get(key) == b.get(key) for key in pair_fields) for a, b in zip(decisions["before"], decisions["after"]))
    check("continuous_942_inputs_paired", paired, {"fields": list(pair_fields), "frames": len(decisions["before"])})
    check("continuous_only_frozen_16_new_test_videos", {d["video_id"] for d in decisions["before"]} == {r["video_id"] for r in prior["frames"] if r["split"] == "test"}, {})

    api = read_json(RUN / "api_smoke.json")
    examples = api["examples"]
    check("api_candidate_and_health", api["model_sha256"] == evaluation["new_model_sha256"] and api["health_http_status"] == 200, {})
    check("api_four_fixed_jpeg_examples", len(examples) == 4 and {(r["video_id"], r["true_label"]) for r in examples} == {(r["video_id"], r["true_label"]) for r in read_json(PRIOR_RUN / "api_smoke.json")["examples"]}, {})
    for i, example in enumerate(examples):
        check_hash(f"api_example_{i}_input_file", ROOT / example["input_path"], example["input_file_sha256"])
        response_hashes = example["response_image_hashes"]
        model_hashes = example["response_model_hashes"]
        check(f"api_example_{i}_responses_bound", example["all_http_statuses_ok"] and example["exact_image_hash_binding_ok"]
              and example["session_create_status"] == 201 and example["session_delete_status"] in (200, 204)
              and len(example["http_statuses"]) == len(response_hashes) == len(model_hashes) == 8
              and all(status == 200 for status in example["http_statuses"])
              and all(value == example["jpeg_payload_sha256"] for value in response_hashes)
              and all(value == evaluation["new_model_sha256"] for value in model_hashes)
              and len(example["session_statuses"]) == 7 and example["single_status"] in STATUSES
              and all(status in STATUSES for status in example["session_statuses"]),
              {"single_status": example["single_status"], "session_statuses": example["session_statuses"]})
    check("candidate_not_silently_activated", api["model_activated_for_current_api"] is False, {})
    for path in (RUN / "evaluation.json", RUN / "configuration.json", RUN / "history.json", RUN / "training.csv",
                 RUN / "progress.json", RUN / "before.json", RUN / "after.json", RUN / "anchor_comparison.json",
                 RUN / "continuous_runtime_comparison.json", RUN / "api_smoke.json", Path(__file__)):
        artifact(path)
    return {"dataset": {"accepted_frames": len(frames), "source_candidates": {"legacy-cache": len(cache_records), "drive-20260930": sampled},
                        "summary": manifest["summary"], "legacy_quality_rejections": manifest["legacy_quality_rejections"],
                        "deduplication_exclusions": manifest["deduplication_exclusions"]},
            "training": training, "runtime": replay_counts,
            "api_examples": [{"video_id": r["video_id"], "true_label": r["true_label"], "single_status": r["single_status"], "session_statuses": r["session_statuses"]} for r in examples]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = {"schema_version": "1.0", "created_at": datetime.now(timezone.utc).isoformat(),
              "method": "No-TensorFlow saved-evidence audit with independent confusion formulas, probability recomputation, source-file and archive hashes, history and boundary checks",
              "checks": [], "artifacts": {}, "passed": False,
              "limits": ["Does not rerun TensorFlow model inference, training, or external/live HTTP requests.",
                         "Recomputes metrics from saved probabilities; model/export inference evidence is verified by provenance and hashes.",
                         "API responses are audited from saved actual integration evidence; repeated JPEGs do not prove liveness accuracy.",
                         "Folder/person names are supplied identifiers, not verified identities; legacy test people overlap training people.",
                         "These previously inspected tests are development benchmarks, not an independent final test.",
                         "Crop screening cannot verify lost full-frame landmarks/clipping or exclude every false detector crop."]}

    def check(name, passed, details):
        result["checks"].append({"name": name, "passed": bool(passed), "details": details})

    def artifact(path):
        path = Path(path)
        key = path.resolve().relative_to(ROOT.resolve()).as_posix()
        if key not in result["artifacts"]:
            result["artifacts"][key] = {"bytes": path.stat().st_size, "sha256": sha(path)}
        return result["artifacts"][key]["sha256"]

    def check_hash(name, path, expected):
        actual = artifact(path)
        check(name, actual == expected, {"path": Path(path).resolve().relative_to(ROOT.resolve()).as_posix(), "actual": actual, "expected": expected})

    try:
        result.update(audit(check, artifact, check_hash))
        result["passed"] = bool(result["checks"]) and all(item["passed"] for item in result["checks"])
        result["status"] = "complete" if result["passed"] else "failed"
    except Exception as error:
        result["status"] = "incomplete" if isinstance(error, FileNotFoundError) else "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        check("audit_completed_without_exception", False, {"error": result["error"]})
    output = RUN / "audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "passed": result["passed"], "checks": len(result["checks"]),
                      "failed_checks": [c["name"] for c in result["checks"] if not c["passed"]],
                      "error": result.get("error"), "output": str(output)}, ensure_ascii=False), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
