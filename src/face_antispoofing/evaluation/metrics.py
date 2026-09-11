from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def _zero_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def roc_auc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return math.nan
    indexed = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average_rank = ((cursor + 1) + end) / 2.0
        for rank_index in range(cursor, end):
            ranks[indexed[rank_index][0]] = average_rank
        cursor = end
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label)
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    )


def equal_error_rate(labels: list[int], scores: list[float]) -> tuple[float, float]:
    if len(set(labels)) < 2:
        return math.nan, 0.5
    thresholds = sorted({0.0, 1.0, *scores})
    best = (float("inf"), math.nan, 0.5)
    for threshold in thresholds:
        metrics = compute_binary_metrics(labels, scores, threshold, include_eer=False)
        fpr = metrics["apcer"]
        fnr = metrics["bpcer"]
        distance = abs(fpr - fnr)
        candidate = (distance, (fpr + fnr) / 2.0, threshold)
        if candidate < best:
            best = candidate
    return float(best[1]), float(best[2])


def compute_binary_metrics(
    labels: Iterable[int],
    scores: Iterable[float],
    threshold: float = 0.5,
    *,
    include_eer: bool = True,
) -> dict[str, Any]:
    y_true = [int(item) for item in labels]
    y_score = [float(item) for item in scores]
    if len(y_true) != len(y_score) or not y_true:
        raise ValueError("labels and scores must have the same non-zero length")
    if any(item not in {0, 1} for item in y_true):
        raise ValueError("labels must use 0=spoof and 1=live")
    predictions = [1 if score >= threshold else 0 for score in y_score]
    pairs = zip(y_true, predictions, strict=True)
    pairs_list = list(pairs)
    tp = sum(actual == 1 and predicted == 1 for actual, predicted in pairs_list)
    tn = sum(actual == 0 and predicted == 0 for actual, predicted in pairs_list)
    fp = sum(actual == 0 and predicted == 1 for actual, predicted in pairs_list)
    fn = sum(actual == 1 and predicted == 0 for actual, predicted in pairs_list)
    apcer = _safe_div(fp, fp + tn)
    bpcer = _safe_div(fn, fn + tp)
    result: dict[str, Any] = {
        "threshold": float(threshold),
        "samples": len(y_true),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": _safe_div(tp + tn, len(y_true)),
        "precision_live": _zero_div(tp, tp + fp),
        "recall_live": _safe_div(tp, tp + fn),
        "f1_live": _zero_div(2 * tp, 2 * tp + fp + fn),
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": (apcer + bpcer) / 2.0,
        "roc_auc": roc_auc(y_true, y_score),
    }
    if include_eer:
        eer, eer_threshold = equal_error_rate(y_true, y_score)
        result["eer"] = eer
        result["eer_threshold"] = eer_threshold
    return result


def find_best_threshold(labels: Iterable[int], scores: Iterable[float]) -> float:
    y_true = [int(item) for item in labels]
    y_score = [float(item) for item in scores]
    candidates = sorted({0.0, 1.0, *y_score})
    ranked: list[tuple[float, float, float]] = []
    for threshold in candidates:
        metrics = compute_binary_metrics(y_true, y_score, threshold, include_eer=False)
        ranked.append((metrics["acer"], abs(threshold - 0.5), threshold))
    return float(min(ranked)[2])


def aggregate_video_scores(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_video"])].append(row)
    aggregated: list[dict[str, Any]] = []
    for source_video, items in sorted(grouped.items()):
        labels = {int(item["label"]) for item in items}
        if len(labels) != 1:
            raise ValueError(f"Mixed labels found for video {source_video}")
        aggregated.append(
            {
                "source_video": source_video,
                "label": labels.pop(),
                "score": sum(float(item["score"]) for item in items) / len(items),
                "frames": len(items),
                "attack_type": str(items[0].get("attack_type", "unknown")),
            }
        )
    return aggregated
