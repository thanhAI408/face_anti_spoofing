from __future__ import annotations

from collections import deque

import numpy as np

from .config import MotionConfig
from .domain import Signal

MOTION_LANDMARKS = (10, 33, 61, 133, 152, 234, 263, 291, 362, 454)


def _normalize_shape(points: np.ndarray) -> np.ndarray | None:
    centered = points - np.mean(points, axis=0, keepdims=True)
    scale = float(np.linalg.norm(centered))
    return None if scale <= 1e-8 else centered / scale


def procrustes_residual(reference: np.ndarray, current: np.ndarray) -> float:
    """Similarity-aligned residual; weak cue for non-planar/non-rigid motion."""
    ref = _normalize_shape(reference)
    cur = _normalize_shape(current)
    if ref is None or cur is None:
        return 0.0
    u, _, vt = np.linalg.svd(cur.T @ ref)
    rotation = u @ vt
    aligned = cur @ rotation
    return float(np.sqrt(np.mean(np.sum((aligned - ref) ** 2, axis=1))))


class LandmarkMotionAnalyzer:
    def __init__(self, config: MotionConfig) -> None:
        self._config = config
        self._reference: np.ndarray | None = None
        self._residuals: deque[float] = deque(maxlen=config.window_size)

    def reset(self) -> None:
        self._reference = None
        self._residuals.clear()

    def update(self, landmarks: np.ndarray | None) -> Signal:
        if landmarks is None or landmarks.shape[0] < 468:
            return Signal("motion", None, 0.0, "landmarks_unavailable")
        shape = landmarks[list(MOTION_LANDMARKS), :2].astype(np.float64)
        if self._reference is None:
            self._reference = shape
            return Signal("motion", None, 0.0, "collecting_motion")
        residual = procrustes_residual(self._reference, shape)
        self._residuals.append(residual)
        if len(self._residuals) < self._config.min_frames:
            return Signal(
                "motion",
                None,
                len(self._residuals) / self._config.min_frames,
                "collecting_motion",
                {"residual": residual},
            )
        observed = max(self._residuals)
        span = max(1e-8, self._config.high_residual - self._config.low_residual)
        score = float(np.clip((observed - self._config.low_residual) / span, 0.0, 1.0))
        reliability = min(1.0, len(self._residuals) / self._config.window_size)
        return Signal(
            "motion",
            score,
            reliability,
            "non_rigid_landmark_motion",
            {"max_residual": observed},
        )
