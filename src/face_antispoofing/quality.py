from __future__ import annotations

import numpy as np

from .config import QualityConfig
from .domain import BoundingBox, QualityResult


class QualityGate:
    def __init__(self, config: QualityConfig) -> None:
        self._config = config

    def evaluate(
        self, frame: np.ndarray, face: np.ndarray, box: BoundingBox
    ) -> QualityResult:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("opencv-python is required for quality analysis") from exc

        if face.size == 0:
            return QualityResult(False, ("empty_face",), {})
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        frame_area = max(1, int(frame.shape[0] * frame.shape[1]))
        face_area_ratio = float(box.area / frame_area)

        reasons: list[str] = []
        if brightness < self._config.min_brightness:
            reasons.append("too_dark")
        if brightness > self._config.max_brightness:
            reasons.append("too_bright")
        if blur_variance < self._config.min_blur_variance:
            reasons.append("too_blurry")
        if face_area_ratio < self._config.min_face_area_ratio:
            reasons.append("face_too_small")
        if face_area_ratio > self._config.max_face_area_ratio:
            reasons.append("face_too_close")
        return QualityResult(
            passed=not reasons,
            reasons=tuple(reasons),
            metrics={
                "brightness": brightness,
                "blur_variance": blur_variance,
                "face_area_ratio": face_area_ratio,
            },
        )
