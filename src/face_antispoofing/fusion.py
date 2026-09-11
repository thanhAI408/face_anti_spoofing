from __future__ import annotations

from collections import deque
from statistics import median

from .config import FusionConfig
from .domain import (
    ChallengeStatus,
    FusionOutcome,
    QualityResult,
    Signal,
    Verdict,
)


class FusionEngine:
    """Reliability-aware weighted fusion with conservative fail-closed behavior."""

    def __init__(self, config: FusionConfig) -> None:
        self._config = config
        self._weights = {
            "texture": config.texture_weight,
            "challenge": config.challenge_weight,
            "motion": config.motion_weight,
        }
        self._scores: deque[float] = deque(maxlen=config.smoothing_window)
        self._hard_spoof_count = 0

    def reset(self) -> None:
        self._scores.clear()
        self._hard_spoof_count = 0

    def decide(
        self,
        quality: QualityResult,
        signals: dict[str, Signal],
        challenge_status: ChallengeStatus,
    ) -> FusionOutcome:
        if not quality.passed:
            self.reset()
            return FusionOutcome(Verdict.RETRY, None, 0.0, ",".join(quality.reasons))

        texture = signals.get("texture")
        hard_spoof_observation = (
            texture is not None
            and texture.score is not None
            and texture.reliability >= 0.5
            and texture.score <= self._config.hard_spoof_texture_threshold
        )
        self._hard_spoof_count = self._hard_spoof_count + 1 if hard_spoof_observation else 0
        if self._hard_spoof_count >= self._config.hard_spoof_frames:
            self._scores.append(texture.score)
            return FusionOutcome(
                Verdict.SPOOF,
                texture.score,
                self._weights["texture"],
                "strong_texture_attack_evidence",
            )

        if challenge_status is ChallengeStatus.FAILED:
            self.reset()
            return FusionOutcome(Verdict.RETRY, None, 0.0, "challenge_timeout")

        weighted_sum = 0.0
        evidence_weight = 0.0
        for name, base_weight in self._weights.items():
            signal = signals.get(name)
            if signal is None or signal.score is None or signal.reliability <= 0.0:
                continue
            effective_weight = base_weight * min(1.0, max(0.0, signal.reliability))
            weighted_sum += min(1.0, max(0.0, signal.score)) * effective_weight
            evidence_weight += effective_weight

        if evidence_weight <= 0.0:
            return FusionOutcome(Verdict.VERIFYING, None, 0.0, "collecting_evidence")
        score = weighted_sum / evidence_weight
        self._scores.append(score)
        stable_score = float(median(self._scores))

        if (
            self._config.require_challenge_for_live
            and challenge_status not in {ChallengeStatus.PASSED, ChallengeStatus.DISABLED}
        ):
            return FusionOutcome(
                Verdict.VERIFYING, stable_score, evidence_weight, "challenge_in_progress"
            )
        if len(self._scores) < self._config.min_stable_frames:
            return FusionOutcome(
                Verdict.VERIFYING, stable_score, evidence_weight, "stabilizing_score"
            )
        if (
            evidence_weight >= self._config.min_live_evidence_weight
            and stable_score >= self._config.live_threshold
        ):
            return FusionOutcome(Verdict.LIVE, stable_score, evidence_weight, "live_consensus")
        if (
            evidence_weight >= self._config.min_spoof_evidence_weight
            and stable_score <= self._config.spoof_threshold
        ):
            return FusionOutcome(Verdict.SPOOF, stable_score, evidence_weight, "spoof_consensus")
        return FusionOutcome(
            Verdict.UNCERTAIN, stable_score, evidence_weight, "insufficient_consensus"
        )
