"""Extract face crops from 'video cam4k' into 'dataset_cam4k' preserving train/validate/test structure.

Handles the 3 fake videos (fake_REC-0001-A, fake_REC-0002-A, fake_REC-0003-A) where the
person holding the phone appears on the right edge of the frame (x > 420) so only the
spoofed face on the phone screen is extracted.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
import cv2
import numpy as np

HOLDER_ON_RIGHT_VIDEOS = {
    "fake_REC-0001-A.mp4",
    "fake_REC-0002-A.mp4",
    "fake_REC-0003-A.mp4",
}


def process_one_video(args_tuple):
    video_path_str, output_dir_str, detector_dir_str, confidence_thresh, skip = args_tuple
    video_path = Path(video_path_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    proto_path = str(Path(detector_dir_str) / "deploy.prototxt")
    model_path = str(Path(detector_dir_str) / "res10_300x300_ssd_iter_140000.caffemodel")
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

    vs = cv2.VideoCapture(str(video_path))
    if not vs.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    read_count = 0
    saved_count = 0
    stem = video_path.stem
    mask_right_holder = video_path.name in HOLDER_ON_RIGHT_VIDEOS

    while True:
        grabbed, frame = vs.read()
        if not grabbed or frame is None:
            break

        read_count += 1
        if read_count % skip != 0:
            continue

        h, w = frame.shape[:2]
        # For fake_REC-0001..0003, mask out the real person holding the phone on the right (x >= 420)
        detect_frame = frame.copy()
        if mask_right_holder:
            detect_frame[:, int(w * 0.65):] = 0

        blob = cv2.dnn.blobFromImage(
            cv2.resize(detect_frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
        )
        net.setInput(blob)
        detections = net.forward()

        if detections.shape[2] > 0:
            # Filter out invalid/out-of-bounds detections that Caffe SSD occasionally emits
            best_conf = -1.0
            best_box = None
            for idx in range(detections.shape[2]):
                conf = float(detections[0, 0, idx, 2])
                if conf <= confidence_thresh:
                    continue
                raw_box = detections[0, 0, idx, 3:7]
                if np.any(raw_box < -0.1) or np.any(raw_box > 1.1):
                    continue
                box = (raw_box * np.array([w, h, w, h])).astype("int")
                startX, startY, endX, endY = box
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)
                if endX - startX < 20 or endY - startY < 20:
                    continue
                if mask_right_holder and startX > int(w * 0.62):
                    continue
                if conf > best_conf:
                    best_conf = conf
                    best_box = (startX, startY, endX, endY)

            if best_box is not None:
                startX, startY, endX, endY = best_box
                face = frame[startY:endY, startX:endX]
                out_file = output_dir / f"{stem}__frame_{read_count}.png"
                if not cv2.imwrite(str(out_file), face):
                    raise IOError(f"Cannot save image: {out_file}")
                saved_count += 1

    vs.release()
    return {
        "video": video_path.name,
        "split": video_path.parent.parent.name,
        "label": video_path.parent.name,
        "frames_read": read_count,
        "faces_saved": saved_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="video cam4k")
    parser.add_argument("--output-root", default="dataset_cam4k")
    parser.add_argument("--detector", default="face_detector")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--skip", type=int, default=10)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    tasks = []
    for split in ("train", "validate", "test"):
        for label in ("real", "fake"):
            in_dir = input_root / split / label
            out_dir = output_root / split / label
            for video_path in sorted(in_dir.glob("*.mp4")):
                tasks.append((
                    str(video_path),
                    str(out_dir),
                    args.detector,
                    args.confidence,
                    args.skip,
                ))

    print(f"[INFO] Starting extraction of {len(tasks)} videos with skip={args.skip}, workers={args.workers}...")
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one_video, t): t[0] for t in tasks}
        for idx, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            if idx % 10 == 0 or idx == len(tasks):
                print(
                    f"[INFO] Processed {idx}/{len(tasks)} videos "
                    f"(latest: {res['split']}/{res['label']}/{res['video']} -> {res['faces_saved']} faces)",
                    flush=True,
                )

    elapsed = time.time() - t0
    summary = {"skip": args.skip, "elapsed_seconds": round(elapsed, 2), "splits": {}}
    for split in ("train", "validate", "test"):
        summary["splits"][split] = {}
        for label in ("real", "fake"):
            sub = [r for r in results if r["split"] == split and r["label"] == label]
            summary["splits"][split][label] = {
                "videos": len(sub),
                "frames_read": sum(r["frames_read"] for r in sub),
                "faces_saved": sum(r["faces_saved"] for r in sub),
            }
    summary["details"] = sorted(results, key=lambda x: (x["split"], x["label"], x["video"]))
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "extraction_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== EXTRACTION SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "details"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
