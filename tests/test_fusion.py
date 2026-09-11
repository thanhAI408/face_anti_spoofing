import unittest

from face_antispoofing.config import FusionConfig
from face_antispoofing.domain import ChallengeStatus, QualityResult, Signal, Verdict
from face_antispoofing.fusion import FusionEngine


GOOD_QUALITY = QualityResult(True)


class FusionEngineTests(unittest.TestCase):
    def config(self, **overrides) -> FusionConfig:
        values = {
            "smoothing_window": 3,
            "min_stable_frames": 1,
        }
        values.update(overrides)
        return FusionConfig(**values)

    def test_live_requires_challenge(self) -> None:
        engine = FusionEngine(self.config())
        signals = {
            "texture": Signal("texture", 0.95, 1.0, "test"),
            "motion": Signal("motion", 0.90, 1.0, "test"),
        }
        result = engine.decide(GOOD_QUALITY, signals, ChallengeStatus.IN_PROGRESS)
        self.assertEqual(result.verdict, Verdict.VERIFYING)

    def test_full_live_evidence_passes(self) -> None:
        engine = FusionEngine(self.config())
        signals = {
            "texture": Signal("texture", 0.90, 1.0, "test"),
            "challenge": Signal("challenge", 1.0, 1.0, "test"),
            "motion": Signal("motion", 0.80, 1.0, "test"),
        }
        result = engine.decide(GOOD_QUALITY, signals, ChallengeStatus.PASSED)
        self.assertEqual(result.verdict, Verdict.LIVE)

    def test_strong_texture_can_reject_early(self) -> None:
        engine = FusionEngine(self.config(hard_spoof_frames=3))
        signal = {"texture": Signal("texture", 0.05, 1.0, "test")}
        first = engine.decide(GOOD_QUALITY, signal, ChallengeStatus.IN_PROGRESS)
        second = engine.decide(GOOD_QUALITY, signal, ChallengeStatus.IN_PROGRESS)
        result = engine.decide(GOOD_QUALITY, signal, ChallengeStatus.IN_PROGRESS)
        self.assertEqual(first.verdict, Verdict.VERIFYING)
        self.assertEqual(second.verdict, Verdict.VERIFYING)
        self.assertEqual(result.verdict, Verdict.SPOOF)

    def test_low_evidence_does_not_approve_live(self) -> None:
        engine = FusionEngine(self.config(require_challenge_for_live=False))
        result = engine.decide(
            GOOD_QUALITY,
            {"texture": Signal("texture", 0.90, 1.0, "test")},
            ChallengeStatus.DISABLED,
        )
        self.assertEqual(result.verdict, Verdict.UNCERTAIN)

    def test_quality_failure_returns_retry(self) -> None:
        engine = FusionEngine(self.config())
        quality = QualityResult(False, ("too_dark",))
        result = engine.decide(quality, {}, ChallengeStatus.CALIBRATING)
        self.assertEqual(result.verdict, Verdict.RETRY)
        self.assertEqual(result.reason, "too_dark")

    def test_challenge_timeout_returns_retry_not_false_accusation(self) -> None:
        engine = FusionEngine(self.config())
        result = engine.decide(GOOD_QUALITY, {}, ChallengeStatus.FAILED)
        self.assertEqual(result.verdict, Verdict.RETRY)

    def test_reliability_reduces_effective_evidence(self) -> None:
        engine = FusionEngine(self.config(require_challenge_for_live=False))
        signals = {
            "texture": Signal("texture", 0.9, 0.1, "test"),
            "challenge": Signal("challenge", 1.0, 0.1, "test"),
            "motion": Signal("motion", 0.9, 0.1, "test"),
        }
        result = engine.decide(GOOD_QUALITY, signals, ChallengeStatus.DISABLED)
        self.assertEqual(result.verdict, Verdict.UNCERTAIN)
        self.assertLess(result.evidence_weight, 0.2)


if __name__ == "__main__":
    unittest.main()
