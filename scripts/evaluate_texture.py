from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from face_antispoofing.dataset import assert_no_subject_leakage  # noqa: E402
from face_antispoofing.evaluation.metrics import (  # noqa: E402
    aggregate_video_scores,
    compute_binary_metrics,
    find_best_threshold,
)
from face_antispoofing.texture import OpenCVDNNTextureModel  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the texture model at frame and source-video levels"
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument(
        "--model", type=Path, default=Path("artifacts/models/texture_mobilenetv3.onnx")
    )
    parser.add_argument("--threshold", default="auto", help="auto or a live threshold in [0,1]")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/evaluation"))
    return parser


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Manifest is empty")
    assert_no_subject_leakage(rows)
    return rows


def _score_rows(rows, data_root: Path, model: OpenCVDNNTextureModel, cv2):
    scored = []
    latencies = []
    for row in rows:
        image = cv2.imread(str(data_root / row["image_path"]))
        if image is None:
            raise FileNotFoundError(data_root / row["image_path"])
        signal = model.predict(image)
        if signal.score is None:
            raise RuntimeError(f"Model failed for {row['image_path']}: {signal.reason}")
        scored.append(
            {
                **row,
                "label": int(row["label"]),
                "score": float(signal.score),
            }
        )
        latencies.append(float(signal.metadata.get("latency_ms", 0.0)))
    return scored, latencies


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


def _render_markdown(report: dict) -> str:
    frame = report["frame_metrics"]
    video = report["video_metrics"]

    def metric_row(name: str, metrics: dict) -> str:
        values = (
            metrics["samples"],
            metrics["apcer"],
            metrics["bpcer"],
            metrics["acer"],
            metrics["roc_auc"],
            metrics["f1_live"],
        )
        return (
            f"| {name} | {values[0]} | {values[1]:.4f} | {values[2]:.4f} | "
            f"{values[3]:.4f} | {values[4]:.4f} | {values[5]:.4f} |"
        )

    lines = [
        "# Evaluation Report",
        "",
        "> This file contains measured results from the recorded command; it is not an estimate.",
        "",
        "## Reproducibility",
        "",
        f"- Created (UTC): `{report['created_at_utc']}`",
        f"- Manifest SHA-256: `{report['manifest_sha256']}`",
        f"- Model SHA-256: `{report['model_sha256']}`",
        f"- Threshold selected on validation: `{report['threshold']:.6f}`",
        f"- Command: `{report['command']}`",
        "",
        "## Results",
        "",
        "| Level | Samples | APCER | BPCER | ACER | ROC-AUC | F1 live |",
        "|---|---:|---:|---:|---:|---:|---:|",
        metric_row("Frame", frame),
        metric_row("Video", video),
        "",
        "## Confusion matrix (video level)",
        "",
        f"- TP={video['tp']}, TN={video['tn']}, FP={video['fp']}, FN={video['fn']}",
        "",
        "## Attack-specific APCER",
        "",
        "| Attack type | Attack videos | False accepts | APCER |",
        "|---|---:|---:|---:|",
    ]
    for name, item in sorted(report["attack_apcer"].items()):
        lines.append(
            f"| {name} | {item['attack_videos']} | {item['false_accepts']} | {item['apcer']:.4f} |"
        )
    latency = report["latency_ms"]
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            f"- Median model latency: `{latency['median']:.2f} ms/frame`",
            f"- P95 model latency: `{latency['p95']:.2f} ms/frame`",
            "",
            "## Interpretation guardrails",
            "",
            "- Test subjects do not occur in train/validation according to the manifest audit.",
            "- The reported model metrics cover the passive texture branch only.",
            "- Challenge-response and full fusion use the separate manual-case evaluation sheet.",
            "- Cross-device/cross-dataset claims require their own holdout protocol; "
            "this report does not prove them.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import cv2
    except ImportError:
        print("opencv-python is required. Run: pip install -e .", file=sys.stderr)
        return 2

    manifest_path = args.manifest.resolve()
    model_path = args.model.resolve()
    rows = _read_manifest(manifest_path)
    model = OpenCVDNNTextureModel(model_path)
    if not model.available:
        print(f"ONNX model not found: {model_path}", file=sys.stderr)
        return 2
    validation_rows = [row for row in rows if row["split"] == "val"]
    test_rows = [row for row in rows if row["split"] == "test"]
    if not validation_rows or not test_rows:
        raise ValueError("Validation and test splits must both be non-empty")
    for name, subset in (("val", validation_rows), ("test", test_rows)):
        labels = {int(row["label"]) for row in subset}
        if labels != {0, 1}:
            raise ValueError(f"Split '{name}' must contain both spoof and live; found {labels}")
    data_root = manifest_path.parent
    if args.threshold == "auto":
        validation_scored, _ = _score_rows(validation_rows, data_root, model, cv2)
        validation_videos = aggregate_video_scores(validation_scored)
        threshold = find_best_threshold(
            [item["label"] for item in validation_videos],
            [item["score"] for item in validation_videos],
        )
    else:
        threshold = float(args.threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0,1]")
    scored, latencies = _score_rows(test_rows, data_root, model, cv2)
    frame_metrics = compute_binary_metrics(
        [item["label"] for item in scored],
        [item["score"] for item in scored],
        threshold,
    )
    videos = aggregate_video_scores(scored)
    video_metrics = compute_binary_metrics(
        [item["label"] for item in videos],
        [item["score"] for item in videos],
        threshold,
    )
    attack_groups = defaultdict(list)
    for item in videos:
        if item["label"] == 0:
            attack_groups[item["attack_type"]].append(item)
    attack_apcer = {}
    for attack_type, items in attack_groups.items():
        false_accepts = sum(item["score"] >= threshold for item in items)
        attack_apcer[attack_type] = {
            "attack_videos": len(items),
            "false_accepts": false_accepts,
            "apcer": false_accepts / len(items),
        }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "manifest_sha256": _hash(manifest_path),
        "model": str(model_path),
        "model_sha256": _hash(model_path),
        "threshold": threshold,
        "command": (
            f"python scripts/evaluate_texture.py --manifest {manifest_path} "
            f"--model {model_path} --threshold {args.threshold}"
        ),
        "frame_metrics": frame_metrics,
        "video_metrics": video_metrics,
        "attack_apcer": attack_apcer,
        "latency_ms": {
            "median": statistics.median(latencies),
            "p95": _percentile(latencies, 0.95),
        },
    }
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"texture-{timestamp}.json"
    markdown_path = output_dir / f"texture-{timestamp}.md"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    print(f"Saved JSON report: {json_path}")
    print(f"Saved Markdown report: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
