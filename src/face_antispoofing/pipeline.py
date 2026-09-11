from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .challenge import ChallengeEngine
from .config import AppConfig
from .detector import OpenCVSSDDetector
from .domain import FrameResult, Signal, Verdict
from .fusion import FusionEngine
from .landmarks import MediaPipeFaceMesh, extract_features
from .motion import LandmarkMotionAnalyzer
from .quality import QualityGate
from .texture import OpenCVDNNTextureModel


class AntiSpoofPipeline:
    """Coordinates detection, quality, passive/active signals and score fusion."""

    def __init__(self, config: AppConfig, project_root: str | Path = ".") -> None:
        root = Path(project_root)
        self.config = config
        self.detector = OpenCVSSDDetector(config.detector, root)
        self.landmarks = MediaPipeFaceMesh(config.detector.confidence)
        self.quality = QualityGate(config.quality)
        self.challenge = ChallengeEngine(config.challenge)
        self.motion = LandmarkMotionAnalyzer(config.motion)
        model_path = Path(config.texture.model_path)
        if not model_path.is_absolute():
            model_path = root / model_path
        self.texture = OpenCVDNNTextureModel(model_path, config.texture.input_size)
        self.fusion = FusionEngine(config.fusion)
        self._frame_index = 0
        self._missed_face_frames = 0
        self._last_texture = Signal("texture", None, 0.0, "not_sampled_yet")

    @property
    def texture_model_available(self) -> bool:
        return self.texture.available

    def reset(self, *, now: float | None = None) -> None:
        self.challenge.reset(now=now)
        self.motion.reset()
        self.fusion.reset()
        self._missed_face_frames = 0

    def close(self) -> None:
        self.landmarks.close()

    def process(self, frame: np.ndarray, *, now: float | None = None) -> FrameResult:
        started = time.perf_counter()
        timestamp = time.monotonic() if now is None else now
        self._frame_index += 1
        boxes = self.detector.detect(frame)
        if not boxes:
            self.fusion.reset()
            self._missed_face_frames += 1
            if self._missed_face_frames >= self.config.challenge.face_loss_reset_frames:
                self.reset(now=timestamp)
            return self._result(started, Verdict.NO_FACE, "no_face")
        if len(boxes) > 1:
            self.reset(now=timestamp)
            return self._result(
                started,
                Verdict.RETRY,
                "multiple_faces",
                box=boxes[0],
            )

        self._missed_face_frames = 0
        box = boxes[0]
        face = frame[box.y1 : box.y2, box.x1 : box.x2]
        quality = self.quality.evaluate(frame, face, box)
        if not quality.passed:
            outcome = self.fusion.decide(quality, {}, self.challenge.status)
            return self._result(
                started,
                outcome.verdict,
                outcome.reason,
                box=box,
                quality=quality,
            )

        landmarks = self.landmarks.estimate(frame)
        features = extract_features(landmarks) if landmarks is not None else None
        challenge = self.challenge.update(features, now=timestamp)
        motion_signal = self.motion.update(landmarks)

        stride = max(1, self.config.texture.run_every_n_frames)
        if self._frame_index % stride == 1 % stride or self._last_texture.score is None:
            self._last_texture = self.texture.predict(face)
        signals = {
            "texture": self._last_texture,
            "challenge": self.challenge.as_signal(),
            "motion": motion_signal,
        }
        outcome = self.fusion.decide(quality, signals, challenge.status)
        return self._result(
            started,
            outcome.verdict,
            outcome.reason,
            box=box,
            quality=quality,
            signals=signals,
            challenge=challenge,
            live_score=outcome.live_score,
            evidence_weight=outcome.evidence_weight,
        )

    @staticmethod
    def _result(
        started: float,
        verdict: Verdict,
        reason: str,
        **kwargs: object,
    ) -> FrameResult:
        return FrameResult(
            verdict=verdict,
            reason=reason,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            **kwargs,
        )
