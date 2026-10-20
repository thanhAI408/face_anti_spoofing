"""Audit the corrected merged run from saved evidence without TensorFlow.

Only the corrected run's audit.json is written. This auditor recomputes metrics
and selection losses, checks the actual sampler indices, and verifies frozen
model/data provenance. Missing evidence can never produce passed:true.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import pickle
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.audit_merged_run import (
    equal, independent_frame_metrics, runtime_counts, sha, summary,
    weight_fingerprint,
)
from scripts.retraining_metrics import evaluate_probabilities, paired_video_bootstrap

RUN = ROOT / "outputs/merged-corrected-20261001"
DATA = ROOT / "dataset/merged-20261001"
PRIOR_RUN = ROOT / "outputs/retrain-20260930"
PRIOR_DATA = ROOT / "dataset/retrain-20260930"
FAILED_RUN = ROOT / "outputs/merged-20261001"
LEGACY_RUN = ROOT / "outputs/retrain-20260915"
BEFORE_SHA = "fb1945fcf2832a843b1b94830d969fa95ce5aa57f31e47ce8110aae80741209b"
FAILED_SHA = "6b6851d7b8e6497a18e4644b700f22c01f969337c3f18ff955052796764c5b08"
MANIFEST_SHA = "5cb8e15f1a30425e0a1cc6c5806f5586c25f3774c138880c951906b44003d169"
LABELS = ("fake", "real")
SOURCES = ("legacy-cache", "drive-20260930")
SPLITS = ("train", "validation", "test")
PHASES = ("before", "failed", "after")
GROUPS = tuple((source, label) for source in SOURCES for label in LABELS)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def groups_metrics(probabilities, records, labels=LABELS):
    actual = evaluate_probabilities(probabilities, records, labels=labels)
    actual["per_source"] = {}
    for source in sorted({r["source"] for r in records}):
        ids = [i for i, r in enumerate(records) if r["source"] == source]
        actual["per_source"][source] = evaluate_probabilities(
            probabilities[ids], [records[i] for i in ids], labels=labels)
    return actual


def balanced_validation_loss(probabilities, records):
    """Independent mean frame CE -> video -> source/class formula."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    grouped = defaultdict(list)
    for index, record in enumerate(records):
        target = LABELS.index(record["label"])
        grouped[(record["source"], record["label"], record["video_id"])].append(
            -np.log(np.clip(probabilities[index, target], 1e-7, 1 - 1e-7)))
    source_class = defaultdict(list)
    for (source, label, _), losses in grouped.items():
        source_class[(source, label)].append(float(np.mean(losses)))
    if set(source_class) != set(GROUPS):
        raise ValueError("Balanced validation needs all source/class groups")
    return float(np.mean([np.mean(source_class[key]) for key in GROUPS]))


def balanced_validation_group_losses(probabilities, records):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    videos = defaultdict(list)
    for index, record in enumerate(records):
        videos[(record["source"], record["label"], record["video_id"])].append(
            -np.log(np.clip(probabilities[index, LABELS.index(record["label"])], 1e-7, 1 - 1e-7)))
    cells = defaultdict(list)
    for (source, label, _), losses in videos.items():
        cells[source + ":" + label].append(float(np.mean(losses)))
    return {key: float(np.mean(values)) for key, values in cells.items()}


def audit_data(check, artifact, check_hash, manifest, configuration):
    frames, videos = manifest["frames"], manifest["videos"]
    check("manifest_complete", manifest["status"] == "complete", {})
    check_hash("frozen_merged_manifest", DATA / "manifest.json", MANIFEST_SHA)
    check("configuration_uses_frozen_manifest", configuration["dataset_manifest_sha256"] == MANIFEST_SHA, {})
    check("summaries_recomputed", equal(summary(frames), manifest["summary"]), summary(frames))
    check_hash("archive_sha256", DATA / manifest["array"]["path"], manifest["array"]["sha256"])
    with np.load(DATA / manifest["array"]["path"], allow_pickle=False) as archive:
        pixels = archive[manifest["array"]["key"]]
    check("pixel_shape_dtype", pixels.dtype == np.uint8 and pixels.shape == (len(frames), 32, 32, 3)
          and list(pixels.shape) == manifest["array"]["shape"], {"shape": list(pixels.shape)})
    hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in pixels]
    check("all_pixel_hashes_and_array_order", all(r["array_index"] == i and
          r["crop_pixel_sha256"] == r["frame_pixel_sha256"] == hashes[i] for i, r in enumerate(frames)),
          {"checked_rows": len(frames)})
    check("all_pixels_unique", len(hashes) == len(set(hashes)), {})
    del pixels
    grouped = defaultdict(list)
    for record in frames:
        grouped[record["video_id"]].append(record)
    check("video_metadata_single_split", all(len({(r["split"], r["label"], r["subject_id"], r["source"])
          for r in group}) == 1 for group in grouped.values()), {"video_count": len(grouped)})
    check("video_acceptance_counts", all(v["accepted_frames"] == len(grouped.get(v["video_id"], []))
          for v in videos), {})
    split_videos = {split: {r["video_id"] for r in frames if r["split"] == split} for split in SPLITS}
    check("all_video_boundaries_disjoint", all(not split_videos[a] & split_videos[b]
          for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))), {})
    drive_subjects = {split: {r["subject_id"] for r in frames if r["split"] == split and r["source"] == SOURCES[1]}
                      for split in SPLITS}
    check("drive_subject_boundaries_frozen", drive_subjects["validation"] == {"b", "thu"}
          and drive_subjects["test"] == {"nga", "phuc"} and all(not drive_subjects[a] & drive_subjects[b]
          for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))),
          {k: sorted(v) for k, v in drive_subjects.items()})
    legacy_frames = read_json(LEGACY_RUN / "frame_manifest.json")
    legacy_splits = read_json(LEGACY_RUN / "split.json")
    lookup = {str(path).replace("\\", "/"): split for split, paths in legacy_frames.items() for path in paths}
    accepted = [r for r in frames if r["source"] == SOURCES[0]]
    excluded = [r for r in manifest["exclusions"] if r.get("source") == SOURCES[0]]
    old_records = accepted + excluded
    disk = {p.relative_to(ROOT).as_posix() for label in LABELS for p in (ROOT / "dataset" / label).glob("*.png")}
    check("all_24221_legacy_images_accounted", len(old_records) == len({r["relative_path"] for r in old_records}) == 24221
          and {r["relative_path"] for r in old_records} == set(lookup) == disk, {})
    check("legacy_boundaries_unchanged", all(lookup[r["relative_path"]] == r["split"] == r["legacy_split"]
          and r["video_id"].removeprefix("old:") in legacy_splits[r["split"]]["videos"] for r in old_records), {})
    check("legacy_source_hashes_unchanged", all(sha(ROOT / r["relative_path"]) == r["image_file_sha256"]
          for r in old_records), {"checked_images": len(old_records)})
    drive = [v for v in videos if v["source"] == SOURCES[1]]
    inventory = read_json(ROOT / "outputs/drive-20260930/inventory-verified.json")
    lookup_drive = {"drive:" + entry["id"]: entry for entry in inventory["files"]}
    check("all_56_drive_labels_unchanged", len(drive) == 56 and set(lookup_drive) == {v["video_id"] for v in drive}
          and all(v["label"] == lookup_drive[v["video_id"]]["label"] and v["subject_id"] == lookup_drive[v["video_id"]]["subject_id"]
                  and v["relative_path"] == lookup_drive[v["video_id"]]["relative_path"] for v in drive), {})
    check("drive_source_hashes_unchanged", all((ROOT / v["relative_path"]).stat().st_size == v["bytes"]
          and sha(ROOT / v["relative_path"]) == v["sha256"] for v in drive), {"checked_videos": len(drive)})
    for field, path in (("preparation_script_sha256", ROOT / "scripts/prepare_merged_data.py"),):
        check_hash(field, path, manifest[field])
    for field, path in (("pipeline_sha256", ROOT / "pipeline.py"),
                        ("detector_sha256", ROOT / "face_detector/face_detection_yunet_2023mar.onnx")):
        check_hash(field, path, manifest["preprocessing"][field])
    for field, path in (("legacy_frame_manifest", LEGACY_RUN / "frame_manifest.json"),
                        ("legacy_split_manifest", LEGACY_RUN / "split.json"),
                        ("drive_inventory", ROOT / "outputs/drive-20260930/inventory-verified.json"),
                        ("before_merge_dataset_manifest", PRIOR_DATA / "manifest.json")):
        check_hash("lineage_" + field, path, manifest["source_lineage"][field + "_sha256"])
    return frames, split_videos


def audit_metrics(check, artifact, evaluation, frames, labels):
    recomputed = {}
    for phase in PHASES:
        sidecar = RUN / (phase + ".json")
        if sidecar.exists():
            check(phase + "_json_matches_evaluation", equal(read_json(sidecar), evaluation[phase]), {})
            artifact(sidecar)
        recomputed[phase] = {}
        for split in SPLITS:
            records = [r for r in frames if r["split"] == split]
            path = RUN / f"{phase}_{split}_probabilities.npy"
            probabilities = np.load(path, allow_pickle=False)
            actual = groups_metrics(probabilities, records, labels[phase])
            check(f"{phase}_{split}_all_metrics_reproduced", equal(actual, evaluation[phase][split]),
                  {"frames": len(records), "loss": actual["loss"], "accuracy": actual["accuracy"],
                   "macro": actual["classification_report"]["macro avg"]})
            manual = independent_frame_metrics(probabilities, records, labels[phase])
            check(f"{phase}_{split}_independent_confusion", equal(manual, {
                "confusion_matrix": actual["confusion_matrix"], "accuracy": actual["accuracy"], "loss": actual["loss"],
                "macro_precision": actual["classification_report"]["macro avg"]["precision"],
                "macro_recall": actual["classification_report"]["macro avg"]["recall"],
                "macro_f1": actual["classification_report"]["macro avg"]["f1-score"]}), manual)
            if phase in ("before", "failed"):
                frozen = np.load(FAILED_RUN / f"{'before' if phase == 'before' else 'after'}_{split}_probabilities.npy", allow_pickle=False)
                check(f"{phase}_{split}_frozen_probabilities", np.allclose(frozen, probabilities, rtol=0, atol=1e-6), {})
            artifact(path)
            recomputed[phase][split] = actual
    return recomputed


def audit_anchors(check, artifact, check_hash, frames, split_videos, labels, model_hashes):
    prior = read_json(PRIOR_DATA / "manifest.json")
    check_hash("prior_anchor_archive", PRIOR_DATA / "frames.npz", prior["array"]["sha256"])
    with np.load(PRIOR_DATA / "frames.npz", allow_pickle=False) as archive:
        pixels = archive["x"]
    anchors = [r for r in prior["frames"] if r["split"] in ("test", "regression_old")]
    train_hashes = {r["crop_pixel_sha256"] for r in frames if r["split"] == "train"}
    check("anchors_never_in_training", not {r["video_id"] for r in anchors} & split_videos["train"]
          and not {hashlib.sha256(pixels[r["array_index"]].tobytes()).hexdigest() for r in anchors} & train_hashes, {})
    del pixels
    evidence = read_json(RUN / "anchor_comparison.json")
    check_hash("anchor_prior_manifest", PRIOR_DATA / "manifest.json", evidence["manifest_sha256"])
    check("anchor_before_after_model_pair", evidence["baseline_model_sha256"] == model_hashes["before"]
          and evidence["candidate_model_sha256"] == model_hashes["after"], {})
    for group, split, expected_count in (("new", "test", 1473), ("old", "regression_old", 415)):
        records = [r for r in prior["frames"] if r["split"] == split]
        stored = evidence["groups"][group]
        check("anchor_exact_count_" + group, len(records) == expected_count, {})
        for phase in ("before", "after"):
            path = RUN / stored.get("probability_files", {}).get(phase, f"anchor_{phase}_{group}_probabilities.npy")
            probabilities = np.load(path, allow_pickle=False)
            actual = evaluate_probabilities(probabilities, records, labels=labels[phase])
            check(f"anchor_{phase}_{group}_metrics", equal(actual, stored[phase]),
                  {"count": len(records), "f1": actual["classification_report"]["macro avg"]["f1-score"]})
            if phase == "before":
                frozen = np.load(PRIOR_RUN / f"after_{split}_probabilities.npy", allow_pickle=False)
                check("anchor_before_frozen_" + group, np.allclose(frozen, probabilities, rtol=0, atol=1e-6), {})
            artifact(path)
        bootstrap = stored["paired_video_bootstrap"]
        check("anchor_bootstrap_" + group, equal(paired_video_bootstrap(stored["before"], stored["after"],
              iterations=bootstrap["iterations"], seed=bootstrap["seed"]), bootstrap), {})
    artifact(RUN / "anchor_comparison.json")


def audit_runtime_api(check, artifact, check_hash, model_hashes):
    prior = read_json(PRIOR_DATA / "manifest.json")
    replay = read_json(RUN / "continuous_runtime_comparison.json")
    input_paths = {"manifest_sha256": PRIOR_DATA / "manifest.json", "pipeline_sha256": ROOT / "pipeline.py",
                   "evaluation_script_sha256": ROOT / "scripts/evaluate_continuous_videos.py",
                   "client_source_sha256": ROOT / "clients/antispoof_client.py",
                   "detector_sha256": ROOT / "face_detector/face_detection_yunet_2023mar.onnx"}
    for field, expected in replay["input_hashes"].items():
        check_hash("runtime_input_" + field, input_paths[field], expected)
    decisions, counts = {}, {}
    for phase in ("before", "after"):
        decisions[phase] = [d for v in replay[phase]["per_video"] for d in v["decisions"]]
        counts[phase] = runtime_counts(replay[phase]["per_video"])
        check("runtime_counts_" + phase, equal(counts[phase], {k: replay[phase][k] for k in counts[phase]}), counts[phase])
        check("runtime_model_" + phase, replay["models"][phase]["model_sha256"] == model_hashes[phase], {})
    paired_fields = ("video_id", "subject_id", "true_label", "frame_index", "time_seconds", "jpeg_sha256",
                     "frame_pixel_sha256", "original_shape", "processed_shape")
    check("runtime_same_942_jpeg_inputs", len(decisions["before"]) == len(decisions["after"]) == 942
          and all(all(a.get(k) == b.get(k) for k in paired_fields)
                  for a, b in zip(decisions["before"], decisions["after"])), {})
    check("runtime_only_16_frozen_new_test_videos", {d["video_id"] for d in decisions["before"]}
          == {r["video_id"] for r in prior["frames"] if r["split"] == "test"}, {})
    api = read_json(RUN / "api_smoke.json")
    examples = api["examples"]
    check("api_health_matches_corrected_model", api["health_http_status"] == 200
          and api["model_sha256"] == model_hashes["after"], {})
    check("api_same_4_jpeg_examples", len(examples) == 4 and
          {(r["video_id"], r["true_label"]) for r in examples} ==
          {(r["video_id"], r["true_label"]) for r in read_json(PRIOR_RUN / "api_smoke.json")["examples"]}, {})
    for index, example in enumerate(examples):
        check_hash(f"api_{index}_input_file", ROOT / example["input_path"], example["input_file_sha256"])
        check(f"api_{index}_actual_responses", example["all_http_statuses_ok"] and example["exact_image_hash_binding_ok"]
              and example["session_create_status"] == 201 and example["session_delete_status"] in (200, 204)
              and len(example["http_statuses"]) == len(example["response_model_hashes"]) == len(example["response_image_hashes"]) == 8
              and all(status == 200 for status in example["http_statuses"])
              and all(value == model_hashes["after"] for value in example["response_model_hashes"])
              and all(value == example["jpeg_payload_sha256"] for value in example["response_image_hashes"])
              and len(example["session_statuses"]) == 7, {})
    check("corrected_candidate_not_silently_activated", api["model_activated_for_current_api"] is False, {})
    artifact(RUN / "continuous_runtime_comparison.json")
    artifact(RUN / "api_smoke.json")
    return counts


def audit_loopback_http(check, artifact, check_hash, model_hashes):
    smoke = read_json(RUN / "http_smoke.json")
    benchmark = read_json(RUN / "http_benchmark.json")
    internal = read_json(RUN / "api_smoke.json")
    prior = read_json(PRIOR_RUN / "api_smoke.json")
    health = smoke["health"]
    check("http_actual_server_frozen_candidate", smoke["passed"] is True
          and smoke["url"] == "http://127.0.0.1:8001" and smoke["model_sha256"] == model_hashes["after"]
          and health["status"] == "ok" and health["model"]["sha256"] == model_hashes["after"]
          and health["model"]["labels"] == list(LABELS) and health["threshold"] == .8, {})
    check("http_test_server_stopped_and_default_unchanged", smoke["server_stopped"] is True
          and smoke["model_activated_for_default_api"] is False, {})
    expected = {(r["video_id"], r["true_label"]) for r in prior["examples"]}
    inputs = {r["video_id"]: r for r in internal["examples"]}
    check("http_same_four_original_examples", len(smoke["examples"]) == 4
          and {(r["video_id"], r["true_label"]) for r in smoke["examples"]} == expected, {})
    allowed_statuses = {"fake", "real", "checking", "uncertain", "adjust", "no_face", "multiple_faces"}
    for index, example in enumerate(smoke["examples"]):
        check(f"http_example_{index}_jpeg_and_model_bound", example["analyze_and_session_routes_passed"] is True
              and example["jpeg_sha256"] == inputs[example["video_id"]]["jpeg_payload_sha256"]
              and len(example["statuses"]) == len(example["response_image_hashes"])
              == len(example["response_model_hashes"]) == 8
              and set(example["statuses"]).issubset(allowed_statuses)
              and all(value == example["jpeg_sha256"] for value in example["response_image_hashes"])
              and all(value == model_hashes["after"] for value in example["response_model_hashes"]), {})
    model = benchmark["model"]
    check("http_benchmark_model_path_sha_labels", model["sha256"] == model_hashes["after"]
          and Path(model["path"]).resolve() == (RUN / "liveness.h5").resolve()
          and model["labels"] == list(LABELS), {})
    check_hash("http_benchmark_labels_sha", RUN / "le.pickle", model["labels_sha256"])
    first = smoke["examples"][0]
    check("http_benchmark_same_first_jpeg", benchmark["input"]["sha256"] == first["jpeg_sha256"]
          == inputs[first["video_id"]]["jpeg_payload_sha256"]
          and benchmark["input"]["name"] == Path(inputs[first["video_id"]]["input_path"]).name
          and benchmark["input"]["jpeg_bytes"] > 0 and benchmark["input"]["width"] > 0
          and benchmark["input"]["height"] > 0, {})
    count = benchmark["iterations"]
    check("http_benchmark_method_and_parity", benchmark["kind"] == "local_single_frame_serial_latency"
          and benchmark["warmup_per_path"] == 5 and count == 10
          and benchmark["configuration"]["threshold"] == .8
          and benchmark["configuration"]["cpu_threads"] == 2
          and benchmark["configuration"]["tf_inter_op_threads"] == 1
          and all(value is True for value in benchmark["comparison"].values())
          and set(benchmark["comparison"]) == {"same_model_hash", "same_label_order", "same_jpeg_hash", "matching_boxes_statuses_scores"}, {})
    for key in ("direct_decode_and_pipeline", "http_client", "server_processing", "server_total"):
        item = benchmark[key]
        measures = [item[field] for field in ("min_ms", "median_ms", "p95_ms", "max_ms")]
        check("http_benchmark_summary_sanity_" + key, item["count"] == count and all(np.isfinite(measures))
              and measures[0] >= 0 and measures == sorted(measures), item)
        if "elapsed_seconds" in item:
            check("http_benchmark_throughput_arithmetic_" + key, item["elapsed_seconds"] > 0
                  and equal(count / item["elapsed_seconds"], item["achieved_serial_requests_per_second"]), {})
    check("http_benchmark_status_count_accounting", sum(benchmark["status_counts"].values()) == count
          and set(benchmark["status_counts"]).issubset(allowed_statuses)
          and all(isinstance(value, int) and value >= 0 for value in benchmark["status_counts"].values())
          and all(isinstance(value, int) and value >= 0 for value in benchmark["face_status_counts"].values())
          and set(benchmark["face_status_counts"]).issubset(allowed_statuses), benchmark["status_counts"])
    for path in (RUN / "http_smoke.json", RUN / "http_benchmark.json", RUN / "http_server.log",
                 ROOT / "scripts/verify_corrected_api_http.py", ROOT / "scripts/benchmark_api.py"):
        artifact(path)
    return {"examples": len(smoke["examples"]), "server_stopped": smoke["server_stopped"],
            "direct_median_ms": benchmark["direct_decode_and_pipeline"]["median_ms"],
            "http_median_ms": benchmark["http_client"]["median_ms"],
            "iterations": count, "status_counts": benchmark["status_counts"],
            "scope": "Saved real loopback HTTP provenance and aggregate consistency; raw latency vectors are not saved, so median/p95 cannot be independently recomputed."}


def audit_bn_ablation(check, artifact, check_hash, frames):
    """Verify the rejected validation-only calibration evidence independently."""
    import h5py
    directory = ROOT / "outputs/merged-investigation-20261001/bn"
    diagnosis = read_json(directory / "diagnosis.json")
    validation = [r for r in frames if r["split"] == "validation"]

    def trainable_fingerprint(path):
        tensors = []
        with h5py.File(path, "r") as handle:
            def visit(name, item):
                if isinstance(item, h5py.Dataset) and not any(
                        key in name for key in ("moving_mean", "moving_variance")):
                    array = item[()]
                    tensors.append((array.shape, str(array.dtype), hashlib.sha256(array.tobytes()).hexdigest()))
            handle["model_weights"].visititems(visit)
        return Counter(tensors)

    check("bn_ablation_uses_only_failed_model_validation", diagnosis["model_sha256"] == FAILED_SHA
          and diagnosis["dataset_manifest_sha256"] == MANIFEST_SHA
          and diagnosis["test_evaluated"] is False, {})
    baseline_probabilities = np.load(FAILED_RUN / "after_validation_probabilities.npy", allow_pickle=False)
    baseline_loss = balanced_validation_loss(baseline_probabilities, validation)
    def calibrated_loss_matches(probabilities, stored):
        # This diagnostic saved float32 group reductions. Its tiny rounding
        # differences are distinct from a changed metric definition.
        groups = balanced_validation_group_losses(probabilities, validation)
        return (set(groups) == set(stored["groups"])
                and np.isclose(balanced_validation_loss(probabilities, validation), stored["mean"], rtol=0, atol=2e-7)
                and all(np.isclose(value, stored["groups"][key], rtol=0, atol=2e-7) for key, value in groups.items()))
    check("bn_ablation_original_validation_loss", calibrated_loss_matches(baseline_probabilities,
          diagnosis["original"]["balanced_validation_loss"]), {"aggregation_rounding_tolerance": 2e-7})
    original_trainable = trainable_fingerprint(FAILED_RUN / "liveness.h5")
    details = {}
    for name in ("uniform", "source_class_video_balanced"):
        trial = diagnosis["trials"][name]
        check_hash("bn_" + name + "_model", directory / (name + ".h5"), trial["model_sha256"])
        check("bn_" + name + "_all_trainable_tensors_unchanged", trial["trainable_weights_unchanged"] is True
              and bool(original_trainable) and original_trainable == trainable_fingerprint(directory / (name + ".h5")),
              {"checked_trainable_tensors": sum(original_trainable.values())})
        path = directory / (name + "_validation_probabilities.npy")
        probabilities = np.load(path, allow_pickle=False)
        manual = independent_frame_metrics(probabilities, validation, LABELS)
        stored = trial["validation"]
        check("bn_" + name + "_independent_metrics", equal(manual, {
            "confusion_matrix": stored["confusion_matrix"], "accuracy": stored["accuracy"], "loss": stored["loss"],
            "macro_precision": stored["classification_report"]["macro avg"]["precision"],
            "macro_recall": stored["classification_report"]["macro avg"]["recall"],
            "macro_f1": stored["classification_report"]["macro avg"]["f1-score"]}), manual)
        actual_balanced = balanced_validation_loss(probabilities, validation)
        check("bn_" + name + "_balanced_loss", calibrated_loss_matches(probabilities, trial["balanced_validation_loss"]),
              {"aggregation_rounding_tolerance": 2e-7})
        check("bn_" + name + "_rejected_using_validation", actual_balanced > baseline_loss,
              {"original_balanced_ce": baseline_loss, "calibrated_balanced_ce": actual_balanced})
        artifact(path)
        details[name] = {"raw_validation_loss": manual["loss"], "balanced_validation_loss": actual_balanced,
                         "macro_f1": manual["macro_f1"], "trainable_weights_unchanged": True}
    artifact(directory / "diagnosis.json")
    return {"original_balanced_ce": baseline_loss, "trials": details,
            "conclusion": "Calibration-only remedies worsened validation; BN statistics alone is not an established explanation or remedy."}


def audit_selection(check, artifact, check_hash, frames):
    selection = read_json(RUN / "selection.json")
    candidates = selection["candidates"]
    expected = ("outputs/merged-balanced-20261001", "outputs/merged-groupnorm-20261001")
    check("exactly_two_predeclared_validation_trials", tuple(c["run_directory"] for c in candidates) == expected
          and selection["metric"] == "val_source_class_video_loss"
          and selection["test_used_for_selection"] is False, {})
    validation = [r for r in frames if r["split"] == "validation"]
    configurations, actual_losses, schedules = [], [], []
    for candidate in candidates:
        directory = ROOT / candidate["run_directory"]
        configuration = read_json(directory / "configuration.json")
        summary = read_json(directory / "training_summary.json")
        label = directory.name
        check_hash("trial_" + label + "_model", directory / "liveness.h5", candidate["candidate_model_sha256"])
        check_hash("trial_" + label + "_configuration", directory / "configuration.json", candidate["configuration_sha256"])
        check_hash("trial_" + label + "_summary", directory / "training_summary.json", candidate["training_summary_sha256"])
        check("trial_" + label + "_validation_only_complete", summary["validation_only_trial"] is True
              and summary["test_predictions_or_metrics_created"] is False
              and summary["model_sha256"] == candidate["candidate_model_sha256"]
              and summary["configuration_sha256"] == candidate["configuration_sha256"]
              and summary["epochs_run"] == candidate["epochs_run"] and summary["best_epoch"] == candidate["best_epoch"]
              and configuration["dataset_manifest_sha256"] == MANIFEST_SHA, {})
        path = directory / "selected_validation_probabilities.npy"
        probabilities = np.load(path, allow_pickle=False)
        check("trial_" + label + "_validation_probability_shape", probabilities.shape == (len(validation), 2)
              and np.isfinite(probabilities).all() and np.all((probabilities >= 0) & (probabilities <= 1))
              and np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-5), {})
        loss = balanced_validation_loss(probabilities, validation)
        check("trial_" + label + "_selection_loss_independent", equal(loss, candidate["validation_source_class_video_loss"])
              and equal(loss, summary["minimum_weighted_validation_ce"]), {"validation_balanced_ce": loss})
        history = read_json(directory / "history.json")
        check("trial_" + label + "_checkpoint_minimum_validation_epoch", len(history["val_source_class_video_loss"])
              == summary["epochs_run"] and int(np.argmin(history["val_source_class_video_loss"])) + 1 == summary["best_epoch"]
              and equal(min(history["val_source_class_video_loss"]), loss), {})
        check("trial_" + label + "_export_exact_checkpoint", weight_fingerprint(directory / "best.keras")
              == weight_fingerprint(directory / "liveness.h5"), {})
        for field, fallback in (("training_script", "scripts/train_balanced_merged.py"),
                                ("architecture_source", "model/livenessnet.py")):
            check_hash("trial_" + label + "_" + field, ROOT / configuration.get(field + "_path", fallback),
                       configuration[field + "_sha256"])
        if "balanced_helper_source_sha256" in configuration:
            check_hash("trial_" + label + "_balanced_helper", ROOT / configuration["balanced_helper_source_path"],
                       configuration["balanced_helper_source_sha256"])
        actual_losses.append(loss)
        configurations.append(configuration)
        with np.load(directory / "balanced_schedules.npz", allow_pickle=False) as archive:
            schedules.append(archive["indices"])
        artifact(path)
        artifact(directory / "history.json")
        artifact(directory / "best.keras")
        artifact(directory / "balanced_schedules.npz")
    same_fields = ("seed", "input", "optimizer", "initial_learning_rate", "fresh_initialization",
                   "baseline_or_warmstart_loaded", "dataset_manifest_sha256", "dataset_archive_sha256",
                   "batch_size", "warmup", "balanced", "augmentation", "early_stopping", "reduce_lr",
                   "split_policy", "train_validation_summary")
    check("normalization_ablation_other_protocol_fields_identical", all(
          equal(configurations[0][field], configurations[1][field]) for field in same_fields), {})
    shared_epochs = min(len(schedules[0]), len(schedules[1]))
    check("normalization_ablation_same_actual_sampling_prefix", shared_epochs >= 1
          and np.array_equal(schedules[0][:shared_epochs], schedules[1][:shared_epochs]),
          {"shared_balanced_epochs": shared_epochs})
    winner = candidates[int(np.argmin(actual_losses))]
    check("winner_selected_only_by_minimum_validation_ce", selection["selected_run_directory"] == winner["run_directory"]
          and selection["candidate_model_sha256"] == winner["candidate_model_sha256"],
          {"selected_trial": winner["run_directory"], "losses": actual_losses})
    for filename in ("liveness.h5", "best.keras", "le.pickle", "configuration.json", "training_summary.json",
                     "validation_selection.json", "history.json", "training.csv", "sample_exposure.npz",
                     "balanced_schedules.npz", "warmup_order.npy", "selected_train_probabilities.npy",
                     "selected_validation_probabilities.npy"):
        original = ROOT / winner["run_directory"] / filename
        check("selected_copy_identical_" + filename, artifact(original) == artifact(RUN / filename), {})
    artifact(ROOT / "scripts/select_corrected_merged.py")
    return {"candidate_losses": {c["run_directory"]: loss for c, loss in zip(candidates, actual_losses)},
            "selected_run_directory": winner["run_directory"]}


def audit_training(check, artifact, check_hash, configuration, evaluation, frames):
    summary = read_json(RUN / "training_summary.json")
    validation_selection = read_json(RUN / "validation_selection.json")
    selection = read_json(RUN / "selection.json")
    history = read_json(RUN / "history.json")
    monitor = "val_source_class_video_loss"
    epochs, best = summary["epochs_run"], summary["best_epoch"]
    train = [r for r in frames if r["split"] == "train"]
    validation = [r for r in frames if r["split"] == "validation"]
    group_videos = {key: sorted({r["video_id"] for r in train
                    if (r["source"], r["label"]) == key}) for key in GROUPS}
    video_indices = {video: np.asarray([i for i, r in enumerate(train) if r["video_id"] == video])
                     for video in sorted({r["video_id"] for r in train})}
    check("training_summary_matches_validation_selection", equal(summary, validation_selection)
          and equal(summary, evaluation["training_summary"]), {})
    check("training_summary_uses_original_configuration", equal(summary["configuration"], configuration), {})
    fields = configuration["protocol_hashed_fields"]
    canonical = json.dumps({key: configuration[key] for key in fields}, sort_keys=True,
                           ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    check("protocol_configuration_hash", hashlib.sha256(canonical).hexdigest() == configuration["protocol_sha256"],
          {"protocol_sha256": configuration["protocol_sha256"]})
    for key, path in (("training_script_sha256", ROOT / configuration.get("training_script_path", "scripts/train_balanced_merged.py")),
                      ("architecture_source_sha256", ROOT / configuration.get("architecture_source_path", "model/livenessnet.py")),
                      ("dataset_archive_sha256", DATA / "frames.npz")):
        check_hash(key, path, configuration[key])
    if "balanced_helper_source_sha256" in configuration:
        check_hash("balanced_helper_source_sha256", ROOT / configuration["balanced_helper_source_path"],
                   configuration["balanced_helper_source_sha256"])
    check("fresh_model_no_warmstart", configuration["fresh_initialization"] is True
          and configuration["baseline_or_warmstart_loaded"] is False, {})
    check("training_only_train_validation_summary", set(configuration["train_validation_summary"])
          == set(SOURCES) and all(set(v) == {"train", "validation"}
          for v in configuration["train_validation_summary"].values()), {})
    check("no_test_in_training_or_selection", summary["validation_only_trial"] is True
          and summary["test_predictions_or_metrics_created"] is False
          and selection["test_used_for_selection"] is False,
          {"scope": "Saved provenance and trainer implementation; not a dynamic execution trace"})
    check("selection_frozen_model_manifest", selection["candidate_model_sha256"] == summary["model_sha256"]
          == evaluation["new_model_sha256"] and selection["dataset_manifest_sha256"] == MANIFEST_SHA, {})
    check("selection_matches_evaluation", equal(selection, evaluation["selection"]), {})
    selection_ns = (RUN / "selection.json").stat().st_mtime_ns
    check("selection_written_before_test_sidecar", selection_ns <= (RUN / "after_test_probabilities.npy").stat().st_mtime_ns,
          {"scope": "Saved artifact creation order, supported by evaluator requiring selection before inference"})
    check("only_balanced_epochs_eligible", configuration["warmup"]["eligible_for_checkpoint"] is False
          and configuration["warmup"]["epochs"] == summary["warmup_epochs"] == 1
          and summary["balanced_epochs_run"] == epochs
          and summary["selected_balanced_epoch"] == best
          and summary["total_passes_including_warmup"] == epochs + 1, {})
    check("selection_monitor_validation_balanced_ce", configuration["early_stopping"]["monitor"] == monitor
          and configuration["reduce_lr"]["monitor"] == monitor
          and summary["selection_monitor"] == summary["selection_metric"] == monitor, {})
    with (RUN / "training.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    check("history_csv_all_epochs", isinstance(epochs, int) and epochs >= 1
          and len(rows) == epochs and all(len(values) == epochs
          and all(equal(float(rows[i][key]), value) for i, value in enumerate(values))
          for key, values in history.items()) and all(int(row["epoch"]) == i for i, row in enumerate(rows)),
          {"epochs": epochs, "balanced_best_epoch": best})
    check("selected_epoch_minimum_validation_ce", best == int(np.argmin(history[monitor])) + 1
          and equal(summary["minimum_weighted_validation_ce"], history[monitor][best - 1])
          and equal(summary["history_at_best_epoch"], {k: v[best - 1] for k, v in history.items()})
          and equal(summary["history_at_last_epoch"], {k: v[-1] for k, v in history.items()}), {})
    for key, path in (("model_sha256", RUN / "liveness.h5"),
                      ("best_keras_sha256", RUN / "best.keras"),
                      ("labels_sha256", RUN / "le.pickle"),
                      ("configuration_sha256", RUN / "configuration.json"),
                      ("sample_exposure_sha256", RUN / "sample_exposure.npz"),
                      ("balanced_schedules_sha256", RUN / "balanced_schedules.npz"),
                      ("warmup_order_sha256", RUN / "warmup_order.npy")):
        check_hash(key, path, summary[key])
    warmup = np.load(RUN / "warmup_order.npy", allow_pickle=False)
    check("warmup_exposes_every_train_frame_once", warmup.dtype.kind in "iu" and warmup.shape == (len(train),)
          and np.array_equal(np.sort(warmup), np.arange(len(train))), {"frames": len(train)})
    check("warmup_order_configuration_hash", hashlib.sha256(warmup.astype("<i8").tobytes()).hexdigest()
          == configuration["warmup"]["order_sha256"], {})
    warmup_history = read_json(RUN / "warmup_history.json")
    check("warmup_history_one_train_only_pass", bool(warmup_history)
          and all(len(v) == 1 for v in warmup_history.values())
          and not any(k.startswith("val_") for k in warmup_history), {})
    with np.load(RUN / "balanced_schedules.npz", allow_pickle=False) as archive:
        schedules = archive["indices"]
    with np.load(RUN / "sample_exposure.npz", allow_pickle=False) as archive:
        exposure_warmup, exposure_epochs = archive["warmup"], archive["balanced_by_epoch"]
    draws_per_epoch = configuration["balanced"]["draws_per_epoch"]
    batch_size = configuration["batch_size"]
    steps = configuration["balanced"]["steps_per_epoch"]
    check("batch_protocol_exactly_4_groups_8_each", batch_size == 32
          and configuration["balanced"]["draws_per_group_per_batch"] == 8
          and {tuple(v) for v in configuration["balanced"]["groups"]} == set(GROUPS)
          and configuration["balanced"]["sample_weights"] is False
          and steps == (len(train) + 31) // 32 and draws_per_epoch == steps * 32,
          {"groups": [list(g) for g in GROUPS], "steps": steps})
    check("actual_schedule_shapes_bounds", schedules.dtype.kind in "iu"
          and schedules.shape == (epochs, draws_per_epoch)
          and np.all((schedules >= 0) & (schedules < len(train)))
          and exposure_epochs.shape == (epochs, len(train))
          and exposure_epochs.dtype.kind in "iu", {"schedule_shape": list(schedules.shape)})
    check("warmup_exposure_counts_recomputed", np.array_equal(exposure_warmup,
          np.bincount(warmup, minlength=len(train))) and np.all(exposure_warmup == 1), {})
    group_codes = np.asarray([GROUPS.index((r["source"], r["label"])) for r in train])
    cumulative = np.zeros(len(train), dtype=np.int64)
    sampler_summary = []
    for epoch, schedule in enumerate(schedules, 1):
        actual_counts = np.bincount(schedule, minlength=len(train))
        cumulative += actual_counts
        batch_groups = group_codes[schedule].reshape(-1, batch_size)
        check(f"epoch_{epoch:03d}_every_actual_batch_balanced", all(
              np.all((batch_groups == group).sum(axis=1) == 8) for group in range(4)), {})
        check(f"epoch_{epoch:03d}_actual_exposure_counts", np.array_equal(actual_counts, exposure_epochs[epoch - 1]), {})
        digest = hashlib.sha256(schedule.astype("<i8").tobytes()).hexdigest()
        check(f"epoch_{epoch:03d}_actual_order_hash", digest == summary["epoch_schedule_sha256"][epoch - 1], {})
        video_draws = {video: int(actual_counts[ids].sum()) for video, ids in video_indices.items()}
        cumulative_video_draws = {video: int(cumulative[ids].sum()) for video, ids in video_indices.items()}
        # Two partial video rounds can straddle an epoch: per-epoch range <=2;
        # the cumulative prefix of persistent rounds has range <=1.
        check(f"epoch_{epoch:03d}_uniform_video_cycles", all(
              max(video_draws[v] for v in videos) - min(video_draws[v] for v in videos) <= 2
              and max(cumulative_video_draws[v] for v in videos) - min(cumulative_video_draws[v] for v in videos) <= 1
              for videos in group_videos.values()), {})
        check(f"epoch_{epoch:03d}_persistent_frame_cycles", all(
              int(cumulative[ids].max()) - int(cumulative[ids].min()) <= 1 for ids in video_indices.values()), {})
        saved = read_json(RUN / f"validation_epoch_{epoch:03d}.json")
        per_video = saved["metrics"]["per_video"]
        video_to_group = {r["video_id"]: (r["source"], r["label"]) for r in validation}
        weighted = float(np.mean([np.mean([v["frame_loss"] for v in per_video
                     if video_to_group[v["video_id"]] == group]) for group in GROUPS]))
        check(f"epoch_{epoch:03d}_validation_only_selection_loss", saved["balanced_epoch"] == epoch
              and saved["schedule_sha256"] == digest and equal(weighted, saved["selection_loss"])
              and equal(weighted, history[monitor][epoch - 1])
              and equal(saved["metrics"]["loss"], history["val_loss"][epoch - 1]),
              {"validation_source_class_video_ce": weighted})
        sampler_summary.append({"balanced_epoch": epoch, "draws": len(schedule),
                                "unique_frames_exposed": int(np.count_nonzero(actual_counts)),
                                "unique_cumulative": int(np.count_nonzero(cumulative))})
        artifact(RUN / f"validation_epoch_{epoch:03d}.json")
    optimizer = summary["optimizer_steps"]
    check("optimizer_step_counts", optimizer["warmup"] == configuration["warmup"]["steps"] == steps
          and optimizer["balanced"] == epochs * steps
          and optimizer["total"] == (epochs + 1) * steps
          and optimizer["selected_checkpoint"] == (best + 1) * steps, optimizer)
    check("all_train_frames_exposed_before_selection", np.all(exposure_warmup
          + exposure_epochs[:best].sum(axis=0) >= 1), {})
    for key, counts in (("exposure_at_selected_epoch", exposure_warmup + exposure_epochs[:best].sum(axis=0)),
                        ("balanced_exposure_at_selected_epoch", exposure_epochs[:best].sum(axis=0)),
                        ("exposure_after_training", exposure_warmup + exposure_epochs.sum(axis=0)),
                        ("balanced_exposure_after_training", exposure_epochs.sum(axis=0))):
        stored = summary[key]
        check(key + "_counts_reproduced", stored["frames"] == len(train) and stored["draws"] == int(counts.sum())
              and stored["unique_frames_exposed"] == int(np.count_nonzero(counts))
              and equal(stored["fraction_exposed"], np.count_nonzero(counts) / len(train))
              and stored["minimum_draws_per_frame"] == int(counts.min())
              and stored["maximum_draws_per_frame"] == int(counts.max()), {})
        check(key + "_per_video_reproduced", set(stored["videos"]) == set(video_indices)
              and all(stored["videos"][video]["frames"] == len(ids)
                      and stored["videos"][video]["draws"] == int(counts[ids].sum())
                      and stored["videos"][video]["unique_frames_exposed"] == int(np.count_nonzero(counts[ids]))
                      and stored["videos"][video]["minimum_draws_per_frame"] == int(counts[ids].min())
                      and stored["videos"][video]["maximum_draws_per_frame"] == int(counts[ids].max())
                      for video, ids in video_indices.items()), {})
    export = configuration["export_compatibility"]
    check("export_probability_parity_evidence", export["passed"] is True
          and export["frames_checked"] == 8 and export["labels"] == list(LABELS)
          and export["max_probability_difference"] <= 1e-6, export)
    original_weights, exported_weights = weight_fingerprint(RUN / "best.keras"), weight_fingerprint(RUN / "liveness.h5")
    check("export_exact_checkpoint_tensors", bool(original_weights) and original_weights == exported_weights,
          {"tensors": sum(original_weights.values())})
    import h5py
    with zipfile.ZipFile(RUN / "best.keras") as archive:
        with h5py.File(io.BytesIO(archive.read("model.weights.h5")), "r") as handle:
            checkpoint_step = int(handle["optimizer/vars/0"][()])
    check("checkpoint_actual_optimizer_iteration", checkpoint_step == optimizer["selected_checkpoint"],
          {"checkpoint_iteration": checkpoint_step})
    with h5py.File(RUN / "liveness.h5", "r") as handle:
        model_config = json.loads(handle.attrs["model_config"])
    layers = model_config["config"]["layers"]
    first = layers[0]["config"]
    check("h5_actual_architecture_input_output", first.get("batch_shape", first.get("batch_input_shape")) == [None, 32, 32, 3]
          and layers[-1]["class_name"] == "Dense" and layers[-1]["config"]["units"] == 2
          and layers[-1]["config"]["activation"] == "softmax", {})
    norm_counts = Counter(layer["class_name"] for layer in layers)
    if configuration.get("norm_type", "batch") == "group-layer":
        check("h5_group_layer_normalization_architecture", norm_counts["BatchNormalization"] == 0
              and norm_counts["GroupNormalization"] == 5 and norm_counts["LayerNormalization"] == 1
              and all(layer["config"]["groups"] == 8 for layer in layers if layer["class_name"] == "GroupNormalization"),
              {"norm_type": "group-layer", "normalization_layers": dict(norm_counts)})
    else:
        check("h5_batch_normalization_architecture", norm_counts["BatchNormalization"] == 6
              and norm_counts["GroupNormalization"] == norm_counts["LayerNormalization"] == 0,
              {"norm_type": "batch", "normalization_layers": dict(norm_counts)})
    for split in ("train", "validation"):
        records = train if split == "train" else validation
        path = RUN / f"selected_{split}_probabilities.npy"
        probabilities = np.load(path, allow_pickle=False)
        actual = groups_metrics(probabilities, records)
        if split == "validation":
            actual["source_class_video_loss"] = balanced_validation_loss(probabilities, records)
            check("selected_checkpoint_validation_loss_independent", equal(actual["source_class_video_loss"],
                  history[monitor][best - 1]), {"selected_loss": actual["source_class_video_loss"]})
        check("selected_" + split + "_complete_metrics", equal(actual, summary["selected"][split]), {})
        after = np.load(RUN / f"after_{split}_probabilities.npy", allow_pickle=False)
        check("selected_" + split + "_unchanged_after_freeze", np.allclose(after, probabilities, rtol=0, atol=1e-6), {})
        artifact(path)
    progress = read_json(RUN / "progress.json")
    check("training_complete", progress["phase"] == "complete" and progress["test_evaluated"] is False
          and progress["balanced_epochs_run"] == epochs and progress["selected_balanced_epoch"] == best
          and progress["model_sha256"] == summary["model_sha256"], {})
    for path in (RUN / "history.json", RUN / "training.csv", RUN / "warmup_history.json",
                 RUN / "training_summary.json", RUN / "validation_selection.json", RUN / "selection.json", RUN / "progress.json"):
        artifact(path)
    return {"warmup_epochs": 1, "balanced_epochs": epochs, "selected_balanced_epoch": best,
            "selected_validation_source_class_video_ce": history[monitor][best - 1],
            "optimizer_steps": optimizer, "sampler_exposure": sampler_summary}


def audit(check, artifact, check_hash):
    manifest = read_json(DATA / "manifest.json")
    evaluation = read_json(RUN / "evaluation.json")
    configuration = read_json(RUN / "configuration.json")
    check("configuration_matches_evaluation", all(equal(value, evaluation["configuration"].get(key))
          for key, value in configuration.items()), {})
    frames, split_videos = audit_data(check, artifact, check_hash, manifest, configuration)
    check("evaluation_dataset_summary", equal(evaluation["dataset_summary"], manifest["summary"]), {})
    model_hashes = {"before": BEFORE_SHA, "failed": FAILED_SHA, "after": evaluation["new_model_sha256"]}
    directories = {"before": PRIOR_RUN, "failed": FAILED_RUN, "after": RUN}
    labels = {}
    for phase, directory in directories.items():
        check_hash("model_" + phase, directory / "liveness.h5", model_hashes[phase])
        labels[phase] = tuple(pickle.loads((directory / "le.pickle").read_bytes()).classes_)
        check("label_order_" + phase, labels[phase] == LABELS, {"labels": list(labels[phase])})
        artifact(directory / "le.pickle")
    check("frozen_baseline_and_failed_models", evaluation["configuration"]["baseline_model_sha256"] == BEFORE_SHA
          and evaluation["configuration"]["failed_merge_model_sha256"] == FAILED_SHA, {})
    check("baseline_path_is_actual_sep30_run", (ROOT / evaluation["configuration"]["baseline_model_path"]).resolve()
          == (PRIOR_RUN / "liveness.h5").resolve(), {})
    check("candidate_is_distinct", model_hashes["after"] not in (BEFORE_SHA, FAILED_SHA), {})
    ablation = audit_bn_ablation(check, artifact, check_hash, frames)
    selection_audit = audit_selection(check, artifact, check_hash, frames)
    training = audit_training(check, artifact, check_hash, configuration, evaluation, frames)
    metrics = audit_metrics(check, artifact, evaluation, frames, labels)
    bootstrap = evaluation["test_video_paired_bootstrap"]
    check("overall_video_bootstrap_reproduced", equal(paired_video_bootstrap(metrics["before"]["test"], metrics["after"]["test"],
          iterations=bootstrap["iterations"], seed=bootstrap["seed"]), bootstrap), {})
    for source, bootstrap in evaluation["test_video_paired_bootstrap_by_source"].items():
        check("source_video_bootstrap_" + source, equal(paired_video_bootstrap(
              metrics["before"]["test"]["per_source"][source], metrics["after"]["test"]["per_source"][source],
              iterations=bootstrap["iterations"], seed=bootstrap["seed"]), bootstrap), {})
    audit_anchors(check, artifact, check_hash, frames, split_videos, labels, model_hashes)
    runtime = audit_runtime_api(check, artifact, check_hash, model_hashes)
    http = audit_loopback_http(check, artifact, check_hash, model_hashes)
    for path in (RUN / "evaluation.json", RUN / "configuration.json", Path(__file__),
                 ROOT / "scripts/audit_merged_run.py", ROOT / "scripts/retraining_metrics.py"):
        artifact(path)
    return {"dataset": {"frames": len(frames), "summary": manifest["summary"]},
            "models": model_hashes, "training": training, "runtime": runtime,
            "loopback_http": http,
            "bn_ablation": ablation,
            "validation_trial_selection": selection_audit,
            "test_metrics": {phase: {"loss": metrics[phase]["test"]["loss"],
                "macro": metrics[phase]["test"]["classification_report"]["macro avg"],
                "attack_rates": metrics[phase]["test"]["presentation_attack_rates"]} for phase in PHASES}}


def main():
    global RUN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="outputs/merged-corrected-20261001")
    args = parser.parse_args()
    RUN = (ROOT / args.run).resolve()
    if not RUN.is_relative_to((ROOT / "outputs").resolve()):
        parser.error("run must stay within this workspace's outputs directory")
    result = {"schema_version": "1.0", "created_at": datetime.now(timezone.utc).isoformat(),
              "method": "Independent saved-evidence audit; no TensorFlow, training, inference, or test-driven selection",
              "checks": [], "artifacts": {}, "passed": False,
              "limits": ["Metrics are recomputed from saved probabilities; inference provenance and checkpoint tensors are audited.",
                         "Test was previously inspected and remains a development benchmark, not an untouched final holdout.",
                         "Legacy videos are disjoint; legacy identities may overlap across splits.",
                         "Camera/network throughput is not measured by this audit or repeated-JPEG API smoke tests.",
                         "Loopback latency summaries are checked for provenance and consistency; raw timing vectors were not saved for independent percentile recomputation.",
                         "Balancing fixes the measured sampling mismatch; calibration-only changes were rejected on validation."]}

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
        check(name, actual == expected, {"path": Path(path).resolve().relative_to(ROOT.resolve()).as_posix(),
                                      "actual": actual, "expected": expected})

    try:
        result.update(audit(check, artifact, check_hash))
        result["passed"] = bool(result["checks"]) and all(item["passed"] for item in result["checks"])
        result["status"] = "complete" if result["passed"] else "failed"
    except Exception as error:
        result["status"] = "incomplete" if isinstance(error, FileNotFoundError) else "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        check("audit_completed_without_exception", False, {"error": result["error"]})
    output = RUN / "audit.json"
    if not RUN.is_dir():
        raise FileNotFoundError("Training output directory must exist before this auditor can run")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "passed": result["passed"], "checks": len(result["checks"]),
                      "failed_checks": [c["name"] for c in result["checks"] if not c["passed"]],
                      "error": result.get("error"), "output": str(output)}, ensure_ascii=False), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
