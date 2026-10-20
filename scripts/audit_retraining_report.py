"""Recompute the evidence used by the 30 September retraining report.

This audit reads saved artifacts and source image files. It does not train,
activate a model, change the dataset, or call external services. The only
generated file is outputs/retrain-20260930/report_audit.json.
"""

from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from retraining_metrics import evaluate_probabilities, paired_video_bootstrap


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/retrain-20260930"
OLD_RUN = ROOT / "outputs/retrain-20260915"
DATASET = ROOT / "dataset/retrain-20260930"
SPLITS = ("train", "validation", "test", "regression_old")
PHASES = ("before", "after")
LABELS = ("fake", "real")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numerically_equal(actual, expected):
    """Compare nested artifacts without rounding numbers for presentation."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            numerically_equal(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            numerically_equal(a, b) for a, b in zip(actual, expected)
        )
    if isinstance(actual, (float, int)) and isinstance(expected, (float, int)):
        return bool(np.isclose(actual, expected, rtol=1e-9, atol=1e-12))
    return actual == expected


def count_group(frames, videos):
    return {
        "frames": len(frames),
        "videos": len(videos),
        "classes": {label: sum(f["label"] == label for f in frames) for label in LABELS},
    }


def csv_training(directory):
    with (directory / "training.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    history = read_json(directory / "history.json")
    best_index = min(range(len(rows)), key=lambda i: float(rows[i]["val_loss"]))
    best_row = {key: int(value) if key == "epoch" else float(value)
                for key, value in rows[best_index].items()}
    consistent = all(len(history[key]) == len(rows) for key in history)
    consistent = consistent and all(
        int(row["epoch"]) == i and all(
            numerically_equal(float(row[key]), history[key][i]) for key in history
        ) for i, row in enumerate(rows)
    )
    return {
        "epochs_run": len(rows),
        "best_epoch": best_index + 1,
        "csv_epoch_index_base": 0,
        "csv_best_row": best_row,
        "history_at_best": {key: values[best_index] for key, values in history.items()},
        "consistency_passed": consistent,
    }


def runtime_counts(videos):
    decisions = [item for video in videos for item in video["decisions"]]
    statuses = Counter(item["runtime_status"] for item in decisions)
    correct = sum(item["runtime_status"] == item["true_label"] for item in decisions)
    wrong = sum(item["runtime_status"] in LABELS and
                item["runtime_status"] != item["true_label"] for item in decisions)
    conclusive = correct + wrong
    return {
        "all_sample_count": len(decisions),
        "status_counts": {status: statuses[status] for status in
                          ("fake", "real", "checking", "uncertain", "adjust", "no_face", "multiple_faces")},
        "correct_conclusive_count": correct,
        "wrong_conclusive_count": wrong,
        "conclusive_count": conclusive,
        "no_decision_count": len(decisions) - conclusive,
        "conclusive_coverage": conclusive / len(decisions),
        "correct_fraction_of_all_samples": correct / len(decisions),
    }


def main():
    manifest_path = DATASET / "manifest.json"
    evaluation_path = RUN / "evaluation.json"
    manifest = read_json(manifest_path)
    evaluation = read_json(evaluation_path)
    configuration = read_json(RUN / "configuration.json")
    prior_split = read_json(OLD_RUN / "split.json")
    prior_evaluation = read_json(OLD_RUN / "evaluation.json")
    assertions = []
    artifacts = {}

    def check(name, passed, details):
        assertions.append({"name": name, "passed": bool(passed), "details": details})

    def artifact(path):
        relative = path.resolve().relative_to(ROOT).as_posix()
        if relative not in artifacts:
            artifacts[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        return artifacts[relative]["sha256"]

    def check_hash(name, path, expected):
        actual = artifact(path)
        check(name, actual == expected, {"path": path.relative_to(ROOT).as_posix(),
                                       "actual_sha256": actual, "expected_sha256": expected})

    cache_classes = {
        label: sum(path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                   for path in (ROOT / "dataset" / label).rglob("*"))
        for label in LABELS
    }
    cache_count = sum(cache_classes.values())
    prior_split_counts = {
        split: {"frames": item["frames"], "videos": len(item["videos"]),
                "classes": item["classes"]} for split, item in prior_split.items()
    }
    prior_total = sum(item["frames"] for item in prior_split_counts.values())
    prior_classes = {label: sum(item["classes"][label] for item in prior_split_counts.values())
                     for label in LABELS}
    check("original_cache_matches_prior_split_counts", cache_count == prior_total and
          cache_classes == prior_classes, {"cache": cache_count, "prior_split": prior_total,
                                           "cache_classes": cache_classes})

    frames, videos = manifest["frames"], manifest["videos"]
    source_counts = {}
    sampling = {}
    for source in sorted({video["source"] for video in videos}):
        source_frames = [frame for frame in frames if frame["source"] == source]
        source_videos = [video for video in videos if video["source"] == source]
        source_counts[source] = {
            "total_frames": len(source_frames), "total_videos": len(source_videos),
            "splits": {split: count_group(
                [frame for frame in source_frames if frame["split"] == split],
                [video for video in source_videos if video["split"] == split],
            ) for split in SPLITS},
        }
        stats = Counter()
        for video in source_videos:
            stats.update(video["statistics"])
        rejected = stats["no_face"] + stats["multiple_faces"] + stats["quality_rejected"]
        sampling[source] = {key: stats[key] for key in
                            ("sampled", "accepted", "no_face", "multiple_faces", "quality_rejected")}
        sampling[source]["rejected_total"] = rejected
        check(f"sampling_accounting_{source}", stats["sampled"] == stats["accepted"] + rejected,
              sampling[source])
        event_counts = Counter(event["status"] for video in source_videos for event in video["sample_events"])
        check(f"sampling_events_accounting_{source}",
              sum(event_counts.values()) == stats["sampled"] and
              event_counts["valid"] == stats["quality_valid"] and
              event_counts["adjust"] == stats["quality_rejected"] and
              event_counts["no_face"] == stats["no_face"] and
              event_counts["multiple_faces"] == stats["multiple_faces"], dict(event_counts))

    for split in SPLITS:
        actual = count_group([f for f in frames if f["split"] == split],
                             [v for v in videos if v["split"] == split])
        expected = {key: manifest["summary"][split][key] for key in actual}
        check(f"manifest_split_summary_{split}", actual == expected, actual)
    check("accepted_frames_match_array_order", len(frames) == manifest["array"]["shape"][0] and
          [frame["array_index"] for frame in frames] == list(range(len(frames))),
          {"frames": len(frames), "array_shape": manifest["array"]["shape"]})
    check("sample_cap", all(v["statistics"]["sampled"] <=
          manifest["preprocessing"]["max_sampled_frames_per_video"] for v in videos),
          {"max_sampled_frames_per_video": manifest["preprocessing"]["max_sampled_frames_per_video"]})
    new_sources = [video for video in videos if video["source"] == "drive-20260930"]
    check("new_video_labels", len(new_sources) == 56 and
          Counter(v["label"] for v in new_sources) == {"real": 30, "fake": 26},
          {"videos": len(new_sources), "classes": dict(Counter(v["label"] for v in new_sources))})
    subject_groups = {split: {f["subject_id"] for f in frames if f["split"] == split}
                      for split in ("train", "validation", "test")}
    check("person_identifiers_disjoint_train_validation_test", all(
        not subject_groups[a].intersection(subject_groups[b])
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))),
        {key: sorted(value) for key, value in subject_groups.items()})
    regression_overlap = sorted(subject_groups["train"].intersection(
        f["subject_id"] for f in frames if f["split"] == "regression_old"))

    check_hash("manifest_hash_configuration", manifest_path, configuration["dataset_manifest_sha256"])
    check_hash("array_hash_manifest", DATASET / manifest["array"]["path"], manifest["array"]["sha256"])
    check_hash("before_model_hash", ROOT / configuration["baseline_model_path"],
               configuration["baseline_model_sha256"])
    check_hash("after_model_hash", RUN / "liveness.h5", evaluation["new_model_sha256"])
    check_hash("before_label_hash", OLD_RUN / "le.pickle", configuration["baseline_labels_sha256"])
    artifact(RUN / "le.pickle")
    check_hash("training_source_hash", ROOT / "scripts/train_retraining_comparison.py",
               configuration["training_script_sha256"])
    check_hash("architecture_source_hash", ROOT / "model/livenessnet.py",
               configuration["architecture_source_sha256"])
    check_hash("preparation_pipeline_hash", ROOT / "pipeline.py",
               manifest["preprocessing"]["pipeline_sha256"])
    check_hash("preparation_detector_hash", ROOT / "face_detector/face_detection_yunet_2023mar.onnx",
               manifest["preprocessing"]["detector_sha256"])
    check("configuration_sidecar_matches_evaluation", configuration == evaluation["configuration"],
          {"evaluation_loss": configuration["evaluation_loss"]})

    training = {"before": csv_training(OLD_RUN), "after": csv_training(RUN)}
    for phase in PHASES:
        stored_training = configuration["baseline_training"] if phase == "before" else evaluation
        check(f"{phase}_epochs_history_csv", training[phase]["consistency_passed"] and
              training[phase]["epochs_run"] == stored_training["epochs_run"] and
              training[phase]["best_epoch"] == stored_training["best_epoch"], training[phase])
    check("baseline_epochs_prior_evaluation", training["before"]["epochs_run"] == prior_evaluation["epochs_run"]
          and training["before"]["best_epoch"] == prior_evaluation["best_epoch"],
          {"epochs_run": prior_evaluation["epochs_run"], "best_epoch": prior_evaluation["best_epoch"]})
    check("candidate_best_history_matches_evaluation", numerically_equal(
        training["after"]["history_at_best"], evaluation["history_at_best_epoch"]),
        {"best_epoch": training["after"]["best_epoch"]})
    recomputed_metrics = {}
    for phase in PHASES:
        stored_sidecar = read_json(RUN / f"{phase}.json")
        check(f"{phase}_evaluation_sidecar", numerically_equal(stored_sidecar, evaluation[phase]),
              {"splits": list(stored_sidecar)})
        recomputed_metrics[phase] = {}
        for split in SPLITS:
            probability_path = RUN / f"{phase}_{split}_probabilities.npy"
            probabilities = np.load(probability_path, allow_pickle=False)
            records = [frame for frame in frames if frame["split"] == split]
            metrics = evaluate_probabilities(probabilities, records, labels=LABELS)
            recomputed_metrics[phase][split] = metrics
            check(f"{phase}_{split}_metrics_from_saved_probabilities", numerically_equal(
                metrics, evaluation[phase][split]),
                {"frames": len(records), "loss": metrics["loss"], "accuracy": metrics["accuracy"],
                 "macro_f1": metrics["classification_report"]["macro avg"]["f1-score"]})
            artifact(probability_path)
    bootstrap = evaluation["test_video_paired_bootstrap"]
    reproduced_bootstrap = paired_video_bootstrap(
        recomputed_metrics["before"]["test"], recomputed_metrics["after"]["test"],
        iterations=bootstrap["iterations"], seed=bootstrap["seed"])
    check("video_bootstrap_reproduced", numerically_equal(reproduced_bootstrap, bootstrap),
          {"iterations": bootstrap["iterations"], "resampling_unit": bootstrap["resampling_unit"]})
    f1_bootstrap = bootstrap["metrics"]["macro_f1"]

    runtime_evidence = {}
    for filename in ("runtime_comparison.json", "continuous_runtime_comparison.json"):
        path = RUN / filename
        replay = read_json(path)
        artifact(path)
        input_paths = {
            "manifest_sha256": manifest_path, "evaluation_sha256": evaluation_path,
            "pipeline_sha256": ROOT / "pipeline.py",
            "evaluation_script_sha256": ROOT / "scripts/evaluate_continuous_videos.py",
            "runtime_evaluation_script_sha256": ROOT / "scripts/evaluate_retraining_runtime.py",
            "client_source_sha256": ROOT / "clients/antispoof_client.py",
            "detector_sha256": ROOT / "face_detector/face_detection_yunet_2023mar.onnx",
            "before_test_probabilities_sha256": RUN / "before_test_probabilities.npy",
            "after_test_probabilities_sha256": RUN / "after_test_probabilities.npy",
        }
        for key, expected in replay["input_hashes"].items():
            check_hash(f"{filename}:{key}", input_paths[key], expected)
        before_decisions = [item for video in replay["before"]["per_video"] for item in video["decisions"]]
        after_decisions = [item for video in replay["after"]["per_video"] for item in video["decisions"]]
        keys = ("video_id", "subject_id", "true_label", "frame_index", "time_seconds")
        if filename == "continuous_runtime_comparison.json":
            keys += ("jpeg_sha256", "frame_pixel_sha256", "original_shape", "processed_shape")
        else:
            keys += ("array_index", "input_status", "box")
        paired = len(before_decisions) == len(after_decisions) and all(
            all(a.get(key) == b.get(key) for key in keys)
            for a, b in zip(before_decisions, after_decisions))
        check(f"{filename}:before_after_inputs_paired", paired,
              {"samples": len(before_decisions), "fields_compared": list(keys)})
        runtime_evidence[filename] = {"method": replay["method"], "input_pairing_passed": paired,
                                      "measurement_scope": "offline replay, not live camera FPS or network latency"}
        for phase in PHASES:
            counts = runtime_counts(replay[phase]["per_video"])
            expected_counts = {key: replay[phase][key] for key in counts}
            check(f"{filename}:{phase}_counts_reproduced", numerically_equal(counts, expected_counts), counts)
            runtime_evidence[filename][phase] = counts
        for phase in PHASES:
            model_hash = (replay["models"][phase]["model_sha256"]
                          if filename == "continuous_runtime_comparison.json"
                          else replay["models"][f"{phase}_model_sha256"])
            expected_hash = configuration["baseline_model_sha256"] if phase == "before" else evaluation["new_model_sha256"]
            check(f"{filename}:{phase}_model_pairing", model_hash == expected_hash,
                  {"model_sha256": model_hash})

    api = read_json(RUN / "api_smoke.json")
    artifact(RUN / "api_smoke.json")
    check("api_candidate_integration_smoke", api["model_sha256"] == evaluation["new_model_sha256"] and
          api["health_http_status"] == 200 and
          all(item["all_http_statuses_ok"] and item["exact_image_hash_binding_ok"] for item in api["examples"]),
          {"method": api["method"], "examples": len(api["examples"]),
           "candidate_activated": api["model_activated_for_current_api"]})
    for path in (manifest_path, evaluation_path, RUN / "before.json", RUN / "after.json",
                 RUN / "configuration.json", RUN / "history.json", RUN / "training.csv",
                 OLD_RUN / "split.json", OLD_RUN / "history.json", OLD_RUN / "training.csv",
                 OLD_RUN / "evaluation.json", Path(__file__).resolve()):
        artifact(path)

    result = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "Read-only count/hash/history/metric audit; saved model probabilities and replay decisions recomputed",
        "original_cache": {"image_count": cache_count, "classes": cache_classes,
                           "prior_split": prior_split_counts, "preserved": cache_count == prior_total,
                           "interpretation": "The original cached images were retained. New extraction from videos is a separate dataset."},
        "fresh_dataset": {"total_frames": len(frames), "total_videos": len(videos),
                          "by_source": source_counts, "splits": manifest["summary"],
                          "excluded_original_videos": sum(item.get("source") == "original" for item in manifest["exclusions"]),
                          "regression_subjects_also_in_train": regression_overlap},
        "sampling": sampling,
        "training": training,
        "metrics": {
            "definition": "Frame argmax classification on accepted BGR crops; rates are fractions, not percentages",
            "clean_inference_loss": {phase: {split: evaluation[phase][split]["loss"] for split in SPLITS} for phase in PHASES},
            "loss_distinction": "Clean loss is unweighted cross-entropy in inference mode on the same original frames. Historical augmented/weighted training loss uses different datasets and is not directly comparable.",
            "historical_augmented_train_loss_at_best": {phase: training[phase]["history_at_best"]["loss"] for phase in PHASES},
            "test_frame_metrics": {phase: {"macro": evaluation[phase]["test"]["classification_report"]["macro avg"],
                                           "accuracy": evaluation[phase]["test"]["accuracy"],
                                           "errors": evaluation[phase]["test"]["presentation_attack_rates"]} for phase in PHASES},
            "video_bootstrap": {"point_before": f1_bootstrap["before"], "point_after": f1_bootstrap["after"],
                                "delta": f1_bootstrap["delta"], "interval": f1_bootstrap["delta_95_percentile_interval"],
                                "unit": "fraction", "decision_unit": "one mean-probability argmax decision per video",
                                "iterations": bootstrap["iterations"], "seed": bootstrap["seed"],
                                "caveat": bootstrap["caveat"]},
        },
        "runtime": runtime_evidence,
        "api_smoke": {"method": api["method"], "examples": len(api["examples"]),
                      "model_activated_for_current_api": api["model_activated_for_current_api"],
                      "limitations": api["limitations"]},
        "artifacts": artifacts,
        "assertions": assertions,
        "passed": all(item["passed"] for item in assertions),
        "limits": ["Does not rerun model inference or live HTTP/camera requests.",
                   "Identity grouping and attack labels rely on supplied folder/video names.",
                   "Cache preservation confirms current counts against prior split totals, not unchanged bytes of every cached image."],
    }
    output = RUN / "report_audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": output.relative_to(ROOT).as_posix(), "passed": result["passed"],
                      "assertions": len(assertions), "failed_assertions": [item["name"] for item in assertions if not item["passed"]],
                      "original_cache_images": cache_count, "fresh_frames": len(frames)}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
