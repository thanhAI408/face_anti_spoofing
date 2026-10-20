"""Compare local HTTP latency with direct inference on identical JPEG bytes.

This measures single-client serial throughput and transport overhead, not model
accuracy, video tracking, camera FPS, or production capacity.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from clients.antispoof_client import AntiSpoofClient, encode_jpeg, is_loopback_url
from pipeline import Pipeline, TemporalDecision


def overall_status(faces):
    if not faces:
        return "no_face"
    if len(faces) > 1:
        return "multiple_faces"
    return faces[0]["status"]


def metrics(samples, elapsed_seconds=None):
    values = np.asarray(samples, dtype=float)
    result = {"count": len(samples), "median_ms": float(np.median(values)),
              "p95_ms": float(np.percentile(values, 95)),
              "min_ms": float(values.min()), "max_ms": float(values.max())}
    if elapsed_seconds is not None:
        result["elapsed_seconds"] = elapsed_seconds
        result["achieved_serial_requests_per_second"] = len(samples) / elapsed_seconds
    return result


def require(condition, message):
    # Do not use assert: comparisons must still run with python -O.
    if not condition:
        raise ValueError(message)


def compare_faces(direct, response):
    require(response.get("mode") == "single_frame", "HTTP response is not single_frame")
    require(response.get("status") == overall_status(direct), "direct/HTTP status mismatch")
    remote = response.get("faces", [])
    require(len(direct) == len(remote), "direct/HTTP face count mismatch")
    for expected, actual in zip(direct, remote):
        require(list(expected["box"]) == actual.get("box"), "direct/HTTP face box mismatch")
        require(expected["status"] == actual.get("status"), "direct/HTTP face status mismatch")
        expected_scores, actual_scores = expected["scores"], actual.get("scores")
        require((expected_scores is None) == (actual_scores is None),
                "direct/HTTP score availability mismatch")
        if expected_scores is not None:
            require(isinstance(actual_scores, dict) and set(actual_scores) == set(expected_scores),
                    "direct/HTTP score labels mismatch")
            for label in expected_scores:
                require(np.isclose(expected_scores[label], actual_scores[label], atol=1e-4, rtol=1e-4),
                        f"direct/HTTP probability mismatch for {label}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="Local image used for every request")
    parser.add_argument("--model", required=True, type=Path, help="Same model loaded by the API")
    parser.add_argument("--labels", required=True, type=Path, help="Same label encoder loaded by the API")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--cpu-threads", type=int, default=2,
                        help="Match the API's OpenCV/TF intra-op thread setting")
    parser.add_argument("--output", type=Path, required=True, help="New JSON result file; never overwrites")
    args = parser.parse_args()
    require(args.iterations >= 1, "iterations must be positive")
    require(args.cpu_threads >= 1, "cpu-threads must be positive")
    require(is_loopback_url(args.url), "benchmark only sends samples to a loopback API")
    require(not args.output.exists(), "output already exists; choose a new result path")

    cv2.setNumThreads(args.cpu_threads)
    if args.model.suffix.lower() != ".tflite":
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(args.cpu_threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    decode_flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    image = cv2.imdecode(np.frombuffer(args.image.read_bytes(), dtype=np.uint8), decode_flags)
    require(image is not None, "cannot decode input image")
    jpeg = encode_jpeg(image)
    image_hash = hashlib.sha256(jpeg).hexdigest()
    model_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
    labels_hash = hashlib.sha256(args.labels.read_bytes()).hexdigest()

    def direct_call():
        start = time.perf_counter()
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), decode_flags)
        decision = TemporalDecision(threshold=threshold, min_samples=1, min_seconds=0)
        faces = pipeline.process(frame, time.monotonic(), temporal=decision)
        return faces, (time.perf_counter() - start) * 1000

    with AntiSpoofClient(args.url) as client:
        health = client.health()
        threshold = health.get("threshold")
        require(isinstance(threshold, (int, float)) and 0.5 < threshold <= 1,
                "health omitted a valid decision threshold")
        api_model = health.get("model", {})
        require(isinstance(api_model, dict) and api_model.get("sha256") == model_hash,
                "health model hash differs from the direct model")
        pipeline = Pipeline(model_path=args.model, labels_path=args.labels, threshold=threshold)
        api_labels = api_model.get("labels", health.get("labels"))
        require(api_labels == pipeline.labels, "health label order differs from direct labels")

        # Discard startup costs from both paths equally.
        for _ in range(5):
            expected_faces, _ = direct_call()
            response = client.analyze(jpeg)
            compare_faces(expected_faces, response)
            require(response.get("model", {}).get("sha256") == model_hash,
                    "HTTP response model hash mismatch")
            require(response.get("image", {}).get("sha256") == image_hash,
                    "HTTP service did not process the exact submitted JPEG bytes")

        direct_times, direct_results = [], []
        start = time.perf_counter()
        for _ in range(args.iterations):
            faces, elapsed = direct_call()
            direct_times.append(elapsed)
            direct_results.append(faces)
        direct_elapsed = time.perf_counter() - start

        http_times, responses = [], []
        start = time.perf_counter()
        for _ in range(args.iterations):
            request_start = time.perf_counter()
            response = client.analyze(jpeg)
            http_times.append((time.perf_counter() - request_start) * 1000)
            responses.append(response)
        http_elapsed = time.perf_counter() - start

    server_times, server_total_times = [], []
    for direct, response in zip(direct_results, responses):
        compare_faces(direct, response)
        require(response.get("model", {}).get("sha256") == model_hash, "HTTP model changed during benchmark")
        require(response.get("image", {}).get("sha256") == image_hash, "HTTP image hash mismatch")
        for field in ("processing_ms", "total_ms"):
            require(isinstance(response.get(field), (int, float))
                    and np.isfinite(response[field]) and response[field] >= 0,
                    f"API omitted valid {field}")
        server_times.append(response["processing_ms"])
        server_total_times.append(response["total_ms"])

    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), decode_flags)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "local_single_frame_serial_latency",
        "warmup_per_path": 5,
        "iterations": args.iterations,
        "configuration": {"cpu_threads": args.cpu_threads, "tf_inter_op_threads": 1,
                          "threshold": threshold, "ignore_exif_orientation": True},
        "input": {"name": args.image.name, "jpeg_bytes": len(jpeg), "sha256": image_hash,
                  "width": int(decoded.shape[1]), "height": int(decoded.shape[0])},
        "model": {"path": str(args.model), "sha256": model_hash,
                  "labels_sha256": labels_hash, "labels": pipeline.labels,
                  "api_id": api_model.get("id")},
        "direct_decode_and_pipeline": metrics(direct_times, direct_elapsed),
        "http_client": metrics(http_times, http_elapsed),
        "server_processing": metrics(server_times),
        "server_total": metrics(server_total_times),
        "status_counts": dict(Counter(response["status"] for response in responses)),
        "face_status_counts": dict(Counter(face["status"] for response in responses
                                           for face in response["faces"])),
        "comparison": {"same_model_hash": True, "same_label_order": True,
                       "same_jpeg_hash": True, "matching_boxes_statuses_scores": True},
        "limitations": [
            "One image repeated sequentially; no concurrency, camera, or full video measurement.",
            "No labels or ground truth: this is not an accuracy evaluation.",
            "Direct timing includes JPEG decode and pipeline; HTTP includes request/response overhead.",
            "Encoding happens once outside timing, equally for both paths.",
            "Results depend on CPU load and server configuration at the time of the run.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "direct": result["direct_decode_and_pipeline"],
                      "http": result["http_client"], "status_counts": result["status_counts"]}, indent=2))


if __name__ == "__main__":
    main()
