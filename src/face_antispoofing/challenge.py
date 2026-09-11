from __future__ import annotations

import random
import time
from collections import deque

from .config import ChallengeConfig
from .domain import ChallengeSnapshot, ChallengeStatus, LandmarkFeatures, Signal

PROMPTS = {
    "blink": ("Hãy chớp mắt", "CHOP MAT"),
    "turn_left": ("Hãy quay đầu sang trái", "QUAY SANG TRAI"),
    "turn_right": ("Hãy quay đầu sang phải", "QUAY SANG PHAI"),
}


class ChallengeEngine:
    """Randomized, landmark-based challenge-response state machine."""

    def __init__(
        self,
        config: ChallengeConfig,
        *,
        rng: random.Random | None = None,
        now: float | None = None,
    ) -> None:
        self._config = config
        self._rng = rng or random.SystemRandom()
        self._initial_time = time.monotonic() if now is None else now
        self.reset(now=self._initial_time)

    def reset(self, *, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        actions = list(dict.fromkeys(self._config.actions))
        invalid = set(actions) - set(PROMPTS)
        if invalid:
            raise ValueError(f"Unsupported challenge actions: {sorted(invalid)}")
        step_count = min(max(0, self._config.steps), len(actions))
        self._sequence = self._rng.sample(actions, step_count) if step_count else []
        self._index = 0
        self._phase = "wait_neutral"
        self._consecutive = 0
        self._yaw_samples: deque[float] = deque(maxlen=self._config.calibration_frames)
        self._baseline_yaw: float | None = None
        self._step_started_at = current
        if not self._config.enabled:
            self._status = ChallengeStatus.DISABLED
        elif not self._sequence:
            self._status = ChallengeStatus.PASSED
        else:
            self._status = ChallengeStatus.CALIBRATING

    @property
    def status(self) -> ChallengeStatus:
        return self._status

    @property
    def sequence(self) -> tuple[str, ...]:
        return tuple(self._sequence)

    def _snapshot(self) -> ChallengeSnapshot:
        action = self._sequence[self._index] if self._index < len(self._sequence) else None
        if self._status is ChallengeStatus.CALIBRATING:
            prompt, prompt_ascii = "Nhìn thẳng vào camera", "NHIN THANG VAO CAMERA"
        elif action:
            prompt, prompt_ascii = PROMPTS[action]
        elif self._status is ChallengeStatus.PASSED:
            prompt, prompt_ascii = "Đã hoàn thành thử thách", "DA HOAN THANH"
        elif self._status is ChallengeStatus.FAILED:
            prompt, prompt_ascii = "Hết thời gian, nhấn R để thử lại", "HET GIO - NHAN R"
        else:
            prompt, prompt_ascii = "", ""
        return ChallengeSnapshot(
            status=self._status,
            prompt=prompt,
            prompt_ascii=prompt_ascii,
            completed_steps=self._index,
            total_steps=len(self._sequence),
            current_action=action,
        )

    def update(
        self, features: LandmarkFeatures | None, *, now: float | None = None
    ) -> ChallengeSnapshot:
        current = time.monotonic() if now is None else now
        if self._status in {
            ChallengeStatus.DISABLED,
            ChallengeStatus.PASSED,
            ChallengeStatus.FAILED,
        }:
            return self._snapshot()
        if current - self._step_started_at > self._config.timeout_seconds:
            self._status = ChallengeStatus.FAILED
            return self._snapshot()
        if features is None:
            return self._snapshot()

        if self._status is ChallengeStatus.CALIBRATING:
            self._yaw_samples.append(features.yaw_proxy)
            if len(self._yaw_samples) >= self._config.calibration_frames:
                ordered = sorted(self._yaw_samples)
                self._baseline_yaw = ordered[len(ordered) // 2]
                self._status = ChallengeStatus.IN_PROGRESS
                self._step_started_at = current
            return self._snapshot()

        action = self._sequence[self._index]
        completed = False
        if action == "blink":
            completed = self._update_blink(features.eye_aspect_ratio)
        else:
            completed = self._update_turn(features.yaw_proxy, action)
        if completed:
            self._index += 1
            self._phase = "wait_neutral"
            self._consecutive = 0
            self._step_started_at = current
            if self._index >= len(self._sequence):
                self._status = ChallengeStatus.PASSED
        return self._snapshot()

    def _condition_for_frames(self, condition: bool) -> bool:
        self._consecutive = self._consecutive + 1 if condition else 0
        return self._consecutive >= self._config.consecutive_frames

    def _update_blink(self, ear: float) -> bool:
        if self._phase == "wait_neutral":
            if self._condition_for_frames(ear >= self._config.open_eye_ratio):
                self._phase = "wait_closed"
                self._consecutive = 0
        elif self._phase == "wait_closed":
            if self._condition_for_frames(ear <= self._config.closed_eye_ratio):
                self._phase = "wait_reopen"
                self._consecutive = 0
        elif self._phase == "wait_reopen":
            return self._condition_for_frames(ear >= self._config.open_eye_ratio)
        return False

    def _update_turn(self, yaw: float, action: str) -> bool:
        baseline = self._baseline_yaw or 0.0
        delta = yaw - baseline
        if self._phase == "wait_neutral":
            if self._condition_for_frames(abs(delta) <= self._config.neutral_delta):
                self._phase = "wait_turn"
                self._consecutive = 0
            return False
        if action == "turn_left":
            expected = delta <= -self._config.turn_delta
        else:
            expected = delta >= self._config.turn_delta
        return self._condition_for_frames(expected)

    def as_signal(self) -> Signal:
        if self._status is ChallengeStatus.PASSED:
            return Signal("challenge", 1.0, 1.0, "random_challenge_passed")
        if self._status is ChallengeStatus.FAILED:
            return Signal("challenge", 0.0, 1.0, "challenge_timeout")
        if self._status is ChallengeStatus.DISABLED:
            return Signal("challenge", None, 0.0, "challenge_disabled")
        return Signal("challenge", None, 0.0, "challenge_pending")
