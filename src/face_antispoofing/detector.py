from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import DetectorConfig
from .domain import BoundingBox


class OpenCVSSDDetector:
    """OpenCV SSD face detector adapter using the model already shipped by the project."""

    def __init__(self, config: DetectorConfig, project_root: str | Path = ".") -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - exercised only without runtime deps
            raise RuntimeError("opencv-python is required for camera inference") from exc

        root = Path(project_root)
        prototxt = root / config.prototxt
        weights = root / config.weights
        if not prototxt.is_file() or not weights.is_file():
            raise FileNotFoundError(
                f"Face detector files are missing: {prototxt} / {weights}"
            )
        self._cv2 = cv2
        self._net = cv2.dnn.readNetFromCaffe(str(prototxt), str(weights))
        self._confidence = config.confidence
        self._padding = config.padding

    def detect(self, frame: np.ndarray) -> list[BoundingBox]:
        height, width = frame.shape[:2]
        blob = self._cv2.dnn.blobFromImage(
            self._cv2.resize(frame, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
        )
        self._net.setInput(blob)
        detections = self._net.forward()
        boxes: list[BoundingBox] = []
        for index in range(detections.shape[2]):
            confidence = float(detections[0, 0, index, 2])
            if confidence < self._confidence:
                continue
            raw = detections[0, 0, index, 3:7] * np.array(
                [width, height, width, height]
            )
            x1, y1, x2, y2 = raw.astype(int)
            pad_x = int(max(0, x2 - x1) * self._padding)
            pad_y = int(max(0, y2 - y1) * self._padding)
            box = BoundingBox(
                x1=max(0, x1 - pad_x),
                y1=max(0, y1 - pad_y),
                x2=min(width, x2 + pad_x),
                y2=min(height, y2 + pad_y),
                confidence=confidence,
            )
            if box.width > 0 and box.height > 0:
                boxes.append(box)
        return sorted(boxes, key=lambda item: item.area, reverse=True)
