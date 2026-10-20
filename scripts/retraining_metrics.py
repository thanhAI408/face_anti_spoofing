"""Reproducible frame and video metrics for anti-spoofing retraining.

This module deliberately has no TensorFlow dependency. Class decisions use
argmax, and confidence coverage uses a fixed 0.8 threshold; neither is fitted on
the test set. All rates are fractions in [0, 1], not percentages. APCER/BPCER here
describe errors of the supplied cropped-frame classifier, not a certification of
the complete camera/API pipeline. Video decisions average probabilities within
each video before taking argmax; they are not the runtime temporal decision.
"""

from collections import defaultdict
from collections.abc import Mapping

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


LOSS_EPSILON = 1e-7
CONFIDENCE_THRESHOLD = 0.8


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _losses(probabilities, targets):
    # Match the explicitly documented clipped categorical cross-entropy, using
    # float64 for aggregation so float32 saturation does not create NaN/inf.
    selected = probabilities[np.arange(len(targets)), targets]
    return -np.log(np.clip(selected, LOSS_EPSILON, 1.0 - LOSS_EPSILON))


def _fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _attack_rates(targets, predictions, labels):
    fake_index, real_index = labels.index("fake"), labels.index("real")
    fake_count = int(np.count_nonzero(targets == fake_index))
    real_count = int(np.count_nonzero(targets == real_index))
    fake_as_real = int(np.count_nonzero((targets == fake_index) & (predictions == real_index)))
    real_as_fake = int(np.count_nonzero((targets == real_index) & (predictions == fake_index)))
    apcer, bpcer = _fraction(fake_as_real, fake_count), _fraction(real_as_fake, real_count)
    return {
        "unit": "fraction",
        "fake_count": fake_count,
        "real_count": real_count,
        "fake_as_real_count": fake_as_real,
        "real_as_fake_count": real_as_fake,
        "fake_as_real": apcer,
        "real_as_fake": bpcer,
        "APCER": apcer,
        "BPCER": bpcer,
        "ACER": (apcer + bpcer) / 2.0 if apcer is not None and bpcer is not None else None,
    }


def _confidence_summary(probabilities, targets, labels):
    predictions = probabilities.argmax(axis=1)
    confident = probabilities.max(axis=1) >= CONFIDENCE_THRESHOLD
    correct = predictions == targets
    count = len(targets)
    confident_count = int(confident.sum())
    correct_count = int(np.count_nonzero(confident & correct))
    incorrect_count = confident_count - correct_count
    # The third column explicitly retains abstentions rather than counting them
    # as correct or silently removing them from the full-data denominator.
    decisions = np.where(confident, predictions, 2)
    matrix = np.zeros((2, 3), dtype=np.int64)
    np.add.at(matrix, (targets, decisions), 1)
    fake_index, real_index = labels.index("fake"), labels.index("real")
    fake_as_real = int(matrix[fake_index, real_index])
    real_as_fake = int(matrix[real_index, fake_index])
    return {
        "threshold": CONFIDENCE_THRESHOLD,
        "unit": "fraction",
        "count": count,
        "confident_count": confident_count,
        "uncertain_count": count - confident_count,
        "coverage": confident_count / count,
        "correct_confident_count": correct_count,
        "incorrect_confident_count": incorrect_count,
        "confident_accuracy": _fraction(correct_count, confident_count),
        "error_rate_among_confident": _fraction(incorrect_count, confident_count),
        "error_fraction_of_all": incorrect_count / count,
        "true_label_order": list(labels),
        "decision_order": [*labels, "uncertain"],
        "confusion_matrix_with_uncertain": matrix.tolist(),
        "uncertain_by_true_label": {label: int(matrix[i, 2]) for i, label in enumerate(labels)},
        "fake_as_real_count": fake_as_real,
        "real_as_fake_count": real_as_fake,
        "fake_as_real_fraction_of_all_fake": _fraction(fake_as_real, int((targets == fake_index).sum())),
        "real_as_fake_fraction_of_all_real": _fraction(real_as_fake, int((targets == real_index).sum())),
    }


def _summary(probabilities, targets, labels):
    predictions = probabilities.argmax(axis=1)
    return {
        "count": len(targets),
        "loss": float(_losses(probabilities, targets).mean()),
        "accuracy": float((targets == predictions).mean()),
        "classification_report": classification_report(
            targets, predictions, labels=[0, 1], target_names=list(labels),
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=[0, 1]).tolist(),
        "presentation_attack_rates": _attack_rates(targets, predictions, labels),
        "confident_acceptance": _confidence_summary(probabilities, targets, labels),
    }


def evaluate_probabilities(probabilities, records, labels=("fake", "real")):
    """Return JSON-ready metrics with explicit input/label validation.

    ``probabilities`` is N x 2, in the supplied ``labels`` order. ``records`` has
    one mapping per row with ``label``, ``video_id``, ``subject_id``, and ``split``.
    Subject/video identifiers must be nonempty strings. A video has one true
    label, one subject, and one split; violating this signals a manifest error.
    Class order may be reversed explicitly, and is preserved in every matrix.

    ``equal_video_weight_accuracy`` averages frame accuracy per video, giving
    long and short videos equal weight. ``video_level`` instead evaluates one
    mean-probability decision per video. Loss clips each true-class probability
    to [1e-7, 1-1e-7] before natural logarithm, then averages across observations.
    Undefined class error rates and confidence accuracy are None (JSON null).
    """
    labels = tuple(labels)
    _require(len(labels) == 2 and set(labels) == {"fake", "real"},
             "labels must contain fake and real exactly once in model output order")
    try:
        probabilities = np.asarray(probabilities, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("probabilities must be a numeric N x 2 array") from error
    records = list(records)
    _require(probabilities.ndim == 2 and probabilities.shape == (len(records), 2),
             "probabilities must have shape (number of records, 2)")
    _require(len(records) > 0, "at least one record is required")
    _require(bool(np.isfinite(probabilities).all()), "probabilities must be finite")
    _require(bool(((probabilities >= 0) & (probabilities <= 1)).all()),
             "probabilities must lie in [0, 1]")
    _require(bool(np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-5)),
             "each probability row must sum to 1 within absolute tolerance 1e-5")
    video_groups, subject_groups = defaultdict(list), defaultdict(list)
    for i, record in enumerate(records):
        _require(isinstance(record, Mapping), f"record {i} must be a mapping")
        _require(record.get("label") in labels, f"record {i} has an unknown label")
        for field in ("video_id", "subject_id", "split"):
            _require(isinstance(record.get(field), str) and bool(record[field].strip()),
                     f"record {i} needs a nonempty {field}")
        video_groups[record["video_id"]].append(i)
        subject_groups[record["subject_id"]].append(i)
    for video_id, indices in video_groups.items():
        for field in ("label", "subject_id", "split"):
            _require(len({records[i][field] for i in indices}) == 1,
                     f"video {video_id!r} has inconsistent {field}")
    targets = np.asarray([labels.index(record["label"]) for record in records], dtype=np.int64)
    result = _summary(probabilities, targets, labels)
    result.update({
        "label_order": list(labels),
        "probability_clipping_epsilon": LOSS_EPSILON,
        "loss_definition": "mean(-ln(clip(true_class_probability, 1e-7, 1-1e-7)))",
        "frame_count": len(records),
        "video_count": len(video_groups),
        "subject_count": len(subject_groups),
        "split_counts": {name: sum(record["split"] == name for record in records)
                         for name in sorted({record["split"] for record in records})},
    })
    per_video, video_probabilities, video_targets = [], [], []
    for video_id in sorted(video_groups):
        indices = video_groups[video_id]
        record = records[indices[0]]
        mean_probabilities = probabilities[indices].mean(axis=0)
        frame_summary = _summary(probabilities[indices], targets[indices], labels)
        per_video.append({
            "video_id": video_id,
            "subject_id": record["subject_id"],
            "split": record["split"],
            "true_label": record["label"],
            "predicted_label": labels[int(mean_probabilities.argmax())],
            "mean_probabilities": {label: float(mean_probabilities[j]) for j, label in enumerate(labels)},
            "frame_count": len(indices),
            "frame_accuracy": frame_summary["accuracy"],
            "frame_loss": frame_summary["loss"],
            "confident_coverage": frame_summary["confident_acceptance"]["coverage"],
        })
        video_probabilities.append(mean_probabilities)
        video_targets.append(targets[indices[0]])
    result["per_video"] = per_video
    result["equal_video_weight_accuracy"] = float(np.mean([item["frame_accuracy"] for item in per_video]))
    result["video_level"] = _summary(np.asarray(video_probabilities), np.asarray(video_targets), labels)
    result["video_level"]["decision_definition"] = "argmax(mean class probabilities within each video)"
    result["per_subject"] = []
    for subject_id in sorted(subject_groups):
        indices = subject_groups[subject_id]
        summary = _summary(probabilities[indices], targets[indices], labels)
        summary.update({
            "subject_id": subject_id,
            "frame_count": len(indices),
            "video_count": len({records[i]["video_id"] for i in indices}),
        })
        result["per_subject"].append(summary)
    return result


def _video_confusion_metrics(fake_as_real, real_as_fake, fake_count, real_count):
    """Compute scalar or vector metrics without sklearn bootstrap overhead."""
    fake_tp = fake_count - fake_as_real
    real_tp = real_count - real_as_fake
    fake_f1 = 2 * fake_tp / (2 * fake_tp + fake_as_real + real_as_fake)
    real_f1 = 2 * real_tp / (2 * real_tp + fake_as_real + real_as_fake)
    apcer = fake_as_real / fake_count
    bpcer = real_as_fake / real_count
    return {
        "macro_f1": (fake_f1 + real_f1) / 2,
        "accuracy": (fake_tp + real_tp) / (fake_count + real_count),
        "APCER": apcer,
        "BPCER": bpcer,
        "ACER": (apcer + bpcer) / 2,
    }


def paired_video_bootstrap(before_metrics, after_metrics, iterations=5000, seed=42):
    """Estimate paired after-minus-before differences using videos as units.

    Inputs are results of ``evaluate_probabilities``. Each video contributes one
    mean-probability argmax prediction. Videos are sampled with replacement
    separately within each true class; the exact same draws evaluate both models.
    This retains class counts and pairing, and avoids falsely treating adjacent
    frames as independent observations. The 95% percentile intervals condition
    on the observed people and videos, not unseen-person generalization.
    """
    _require(isinstance(iterations, (int, np.integer)) and not isinstance(iterations, bool)
             and iterations > 0, "iterations must be a positive integer")
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, bool) and seed >= 0,
             "seed must be a nonnegative integer")

    def indexed_videos(metrics, name):
        _require(isinstance(metrics, Mapping), f"{name} metrics must be a mapping")
        videos = metrics.get("per_video")
        _require(isinstance(videos, list) and bool(videos), f"{name} needs a nonempty per_video list")
        indexed = {}
        for video in videos:
            _require(isinstance(video, Mapping), f"{name} video must be a mapping")
            video_id = video.get("video_id")
            _require(isinstance(video_id, str) and bool(video_id.strip()), f"{name} video needs a nonempty video_id")
            _require(video_id not in indexed, f"{name} contains repeated video_id {video_id!r}")
            _require(video.get("true_label") in {"fake", "real"}, f"{name} video has an unknown true_label")
            _require(video.get("predicted_label") in {"fake", "real"}, f"{name} video has an unknown predicted_label")
            subject = video.get("subject_id")
            _require(isinstance(subject, str) and bool(subject.strip()), f"{name} video needs a nonempty subject_id")
            indexed[video_id] = video
        return indexed

    before, after = indexed_videos(before_metrics, "before"), indexed_videos(after_metrics, "after")
    _require(set(before) == set(after), "before and after video IDs must match exactly")
    for video_id in before:
        for field in ("true_label", "subject_id"):
            _require(before[video_id][field] == after[video_id][field],
                     f"before and after {field} differ for video {video_id!r}")
    fake_ids = sorted(video_id for video_id, video in before.items() if video["true_label"] == "fake")
    real_ids = sorted(video_id for video_id, video in before.items() if video["true_label"] == "real")
    _require(bool(fake_ids) and bool(real_ids), "bootstrap needs at least one video of each true class")
    subjects = sorted({video["subject_id"] for video in before.values()})
    rng = np.random.default_rng(int(seed))
    # Index arrays are shared between before/after so every replicate is paired.
    # Sampling every video of each class once per replicate preserves supports.
    fake_draws = rng.integers(0, len(fake_ids), size=(int(iterations), len(fake_ids)))
    real_draws = rng.integers(0, len(real_ids), size=(int(iterations), len(real_ids)))
    points, replicates = [], []
    for indexed in (before, after):
        fake_errors = np.asarray([indexed[video_id]["predicted_label"] == "real" for video_id in fake_ids])
        real_errors = np.asarray([indexed[video_id]["predicted_label"] == "fake" for video_id in real_ids])
        points.append(_video_confusion_metrics(int(fake_errors.sum()), int(real_errors.sum()),
                                              len(fake_ids), len(real_ids)))
        replicates.append(_video_confusion_metrics(fake_errors[fake_draws].sum(axis=1),
                                                  real_errors[real_draws].sum(axis=1),
                                                  len(fake_ids), len(real_ids)))
    metrics = {}
    for metric in points[0]:
        differences = replicates[1][metric] - replicates[0][metric]
        interval = np.percentile(differences, [2.5, 97.5])
        metrics[metric] = {
            "before": float(points[0][metric]),
            "after": float(points[1][metric]),
            "delta": float(points[1][metric] - points[0][metric]),
            "delta_95_percentile_interval": [float(value) for value in interval],
        }
    return {
        "method": "paired stratified video bootstrap",
        "resampling_unit": "video; one mean-probability argmax prediction per video",
        "stratification": "true fake/real label, preserving each class video count",
        "difference_definition": "after minus before",
        "unit": "fraction",
        "confidence_level": .95,
        "iterations": int(iterations),
        "seed": int(seed),
        "video_count": len(before),
        "videos_by_class": {"fake": len(fake_ids), "real": len(real_ids)},
        "subject_count": len(subjects),
        "subjects": subjects,
        "metrics": metrics,
        "caveat": (f"Intervals are conditional on the {len(subjects)} tested people and their videos. "
                   "They do not quantify generalization to unseen people or recording conditions; "
                   "videos of the same person can remain correlated. Few videos or constant outcomes "
                   "can produce deceptively narrow or zero-width intervals."),
    }
