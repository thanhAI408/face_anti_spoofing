from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    mirror: bool = True


@dataclass(frozen=True)
class DetectorConfig:
    prototxt: str = "face_detector/deploy.prototxt"
    weights: str = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    confidence: float = 0.60
    padding: float = 0.18


@dataclass(frozen=True)
class TextureConfig:
    model_path: str = "artifacts/models/texture_mobilenetv3.onnx"
    input_size: int = 224
    run_every_n_frames: int = 2


@dataclass(frozen=True)
class QualityConfig:
    min_brightness: float = 45.0
    max_brightness: float = 220.0
    min_blur_variance: float = 45.0
    min_face_area_ratio: float = 0.06
    max_face_area_ratio: float = 0.72


@dataclass(frozen=True)
class ChallengeConfig:
    enabled: bool = True
    steps: int = 2
    actions: tuple[str, ...] = ("blink", "turn_left", "turn_right")
    calibration_frames: int = 8
    timeout_seconds: float = 12.0
    closed_eye_ratio: float = 0.19
    open_eye_ratio: float = 0.23
    turn_delta: float = 0.18
    neutral_delta: float = 0.08
    consecutive_frames: int = 2
    face_loss_reset_frames: int = 5


@dataclass(frozen=True)
class MotionConfig:
    window_size: int = 24
    min_frames: int = 8
    low_residual: float = 0.003
    high_residual: float = 0.020


@dataclass(frozen=True)
class FusionConfig:
    texture_weight: float = 0.55
    challenge_weight: float = 0.30
    motion_weight: float = 0.15
    min_live_evidence_weight: float = 0.65
    min_spoof_evidence_weight: float = 0.45
    live_threshold: float = 0.72
    spoof_threshold: float = 0.35
    hard_spoof_texture_threshold: float = 0.12
    hard_spoof_frames: int = 3
    smoothing_window: int = 10
    min_stable_frames: int = 5
    require_challenge_for_live: bool = True


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = CameraConfig()
    detector: DetectorConfig = DetectorConfig()
    texture: TextureConfig = TextureConfig()
    quality: QualityConfig = QualityConfig()
    challenge: ChallengeConfig = ChallengeConfig()
    motion: MotionConfig = MotionConfig()
    fusion: FusionConfig = FusionConfig()


T = TypeVar("T")


def _from_dict(cls: type[T], values: dict[str, Any] | None) -> T:
    values = values or {}
    allowed = {item.name for item in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown keys in [{cls.__name__}]: {names}")
    if cls is ChallengeConfig and "actions" in values:
        values = {**values, "actions": tuple(values["actions"])}
    return cls(**values)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    known_sections = {item.name for item in fields(AppConfig)}
    unknown_sections = set(raw) - known_sections
    if unknown_sections:
        names = ", ".join(sorted(unknown_sections))
        raise ValueError(f"Unknown configuration sections: {names}")
    config = AppConfig(
        camera=_from_dict(CameraConfig, raw.get("camera")),
        detector=_from_dict(DetectorConfig, raw.get("detector")),
        texture=_from_dict(TextureConfig, raw.get("texture")),
        quality=_from_dict(QualityConfig, raw.get("quality")),
        challenge=_from_dict(ChallengeConfig, raw.get("challenge")),
        motion=_from_dict(MotionConfig, raw.get("motion")),
        fusion=_from_dict(FusionConfig, raw.get("fusion")),
    )
    _validate_config(config)
    return config


def _validate_config(config: AppConfig) -> None:
    if config.texture.run_every_n_frames < 1:
        raise ValueError("texture.run_every_n_frames must be at least 1")
    if config.challenge.calibration_frames < 1 or config.challenge.consecutive_frames < 1:
        raise ValueError("challenge frame counts must be at least 1")
    if config.challenge.face_loss_reset_frames < 1:
        raise ValueError("challenge.face_loss_reset_frames must be at least 1")
    if config.challenge.timeout_seconds <= 0:
        raise ValueError("challenge.timeout_seconds must be positive")
    fusion = config.fusion
    weights = (fusion.texture_weight, fusion.challenge_weight, fusion.motion_weight)
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("fusion weights must be non-negative with a positive sum")
    if not 0.0 <= fusion.spoof_threshold < fusion.live_threshold <= 1.0:
        raise ValueError("fusion thresholds must satisfy 0 <= spoof < live <= 1")
    if fusion.hard_spoof_frames < 1 or fusion.min_stable_frames < 1:
        raise ValueError("fusion frame counts must be at least 1")
    if config.quality.min_brightness >= config.quality.max_brightness:
        raise ValueError("quality brightness range is invalid")
