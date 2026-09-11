from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .domain import LandmarkFeatures

LEFT_EYE = (362, 385, 387, 263, 373, 380)
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
FACE_LEFT = 234
FACE_RIGHT = 454
NOSE_TIP = 1
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def eye_aspect_ratio(points: np.ndarray, indices: Sequence[int]) -> float:
    p1, p2, p3, p4, p5, p6 = (points[index] for index in indices)
    horizontal = 2.0 * _distance(p1, p4)
    if horizontal <= 1e-8:
        return 0.0
    return (_distance(p2, p6) + _distance(p3, p5)) / horizontal


def extract_features(points: np.ndarray) -> LandmarkFeatures:
    if points.ndim != 2 or points.shape[0] < 468 or points.shape[1] < 2:
        raise ValueError("Expected at least 468 MediaPipe landmarks with x/y coordinates")
    left_ear = eye_aspect_ratio(points, LEFT_EYE)
    right_ear = eye_aspect_ratio(points, RIGHT_EYE)
    face_width = _distance(points[FACE_LEFT], points[FACE_RIGHT])
    eye_mid_x = float((points[33, 0] + points[263, 0]) / 2.0)
    yaw_proxy = 0.0 if face_width <= 1e-8 else (float(points[NOSE_TIP, 0]) - eye_mid_x) / face_width
    mouth_width = _distance(points[MOUTH_LEFT], points[MOUTH_RIGHT])
    mouth_ratio = (
        0.0
        if mouth_width <= 1e-8
        else _distance(points[MOUTH_TOP], points[MOUTH_BOTTOM]) / mouth_width
    )
    return LandmarkFeatures(
        eye_aspect_ratio=(left_ear + right_ear) / 2.0,
        yaw_proxy=yaw_proxy,
        mouth_ratio=mouth_ratio,
    )


class MediaPipeFaceMesh:
    """468-point landmark provider. MediaPipe bundles the required Face Mesh model."""

    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("mediapipe and opencv-python are required for landmarks") from exc
        self._cv2 = cv2
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    def estimate(self, frame: np.ndarray) -> np.ndarray | None:
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        return np.asarray(
            [(item.x, item.y, item.z) for item in result.multi_face_landmarks[0].landmark],
            dtype=np.float32,
        )

    def close(self) -> None:
        self._mesh.close()
