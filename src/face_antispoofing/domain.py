from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    NO_FACE = "NO_FACE"
    RETRY = "RETRY"
    VERIFYING = "VERIFYING"
    LIVE = "LIVE"
    SPOOF = "SPOOF"
    UNCERTAIN = "UNCERTAIN"
    ERROR = "ERROR"


class ChallengeStatus(str, Enum):
    DISABLED = "DISABLED"
    CALIBRATING = "CALIBRATING"
    IN_PROGRESS = "IN_PROGRESS"
    PASSED = "PASSED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class QualityResult:
    passed: bool
    reasons: tuple[str, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Signal:
    """A live probability in [0, 1] plus confidence in this observation."""

    name: str
    score: float | None
    reliability: float
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LandmarkFeatures:
    eye_aspect_ratio: float
    yaw_proxy: float
    mouth_ratio: float


@dataclass(frozen=True)
class ChallengeSnapshot:
    status: ChallengeStatus
    prompt: str
    prompt_ascii: str
    completed_steps: int
    total_steps: int
    current_action: str | None = None


@dataclass(frozen=True)
class FusionOutcome:
    verdict: Verdict
    live_score: float | None
    evidence_weight: float
    reason: str


@dataclass(frozen=True)
class FrameResult:
    verdict: Verdict
    reason: str
    box: BoundingBox | None = None
    quality: QualityResult | None = None
    signals: dict[str, Signal] = field(default_factory=dict)
    challenge: ChallengeSnapshot | None = None
    live_score: float | None = None
    evidence_weight: float = 0.0
    latency_ms: float = 0.0
