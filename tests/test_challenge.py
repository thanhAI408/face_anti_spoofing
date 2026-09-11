import random
import unittest

from face_antispoofing.challenge import ChallengeEngine
from face_antispoofing.config import ChallengeConfig
from face_antispoofing.domain import ChallengeStatus, LandmarkFeatures


def features(ear: float = 0.30, yaw: float = 0.0) -> LandmarkFeatures:
    return LandmarkFeatures(eye_aspect_ratio=ear, yaw_proxy=yaw, mouth_ratio=0.1)


class ChallengeEngineTests(unittest.TestCase):
    def test_blink_requires_open_close_reopen(self) -> None:
        config = ChallengeConfig(
            actions=("blink",),
            steps=1,
            calibration_frames=1,
            consecutive_frames=1,
        )
        engine = ChallengeEngine(config, rng=random.Random(1), now=0.0)
        self.assertEqual(engine.update(features(), now=0.1).status, ChallengeStatus.IN_PROGRESS)
        engine.update(features(ear=0.30), now=0.2)
        engine.update(features(ear=0.10), now=0.3)
        result = engine.update(features(ear=0.30), now=0.4)
        self.assertEqual(result.status, ChallengeStatus.PASSED)

    def test_turn_is_relative_to_calibrated_pose(self) -> None:
        config = ChallengeConfig(
            actions=("turn_right",),
            steps=1,
            calibration_frames=1,
            consecutive_frames=1,
            turn_delta=0.18,
        )
        engine = ChallengeEngine(config, rng=random.Random(1), now=0.0)
        engine.update(features(yaw=0.10), now=0.1)
        engine.update(features(yaw=0.10), now=0.2)
        result = engine.update(features(yaw=0.29), now=0.3)
        self.assertEqual(result.status, ChallengeStatus.PASSED)

    def test_timeout_fails_closed(self) -> None:
        config = ChallengeConfig(
            actions=("blink",), steps=1, calibration_frames=1, timeout_seconds=1.0
        )
        engine = ChallengeEngine(config, rng=random.Random(1), now=0.0)
        result = engine.update(None, now=1.1)
        self.assertEqual(result.status, ChallengeStatus.FAILED)

    def test_random_sequence_contains_no_duplicate(self) -> None:
        config = ChallengeConfig(steps=3)
        engine = ChallengeEngine(config, rng=random.Random(5), now=0.0)
        self.assertEqual(len(engine.sequence), len(set(engine.sequence)))

    def test_disabled_challenge_emits_no_score(self) -> None:
        engine = ChallengeEngine(ChallengeConfig(enabled=False), now=0.0)
        self.assertEqual(engine.status, ChallengeStatus.DISABLED)
        self.assertIsNone(engine.as_signal().score)

    def test_unknown_action_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ChallengeEngine(ChallengeConfig(actions=("nod",)), now=0.0)


if __name__ == "__main__":
    unittest.main()
