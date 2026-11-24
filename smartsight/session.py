"""Test-session state machine.

Pure Python and thread-safe: the camera thread calls :meth:`AcuitySession.update`
once per processed frame, while Flask request threads call ``start``/``reset``/
``snapshot``. Keeping it free of OpenCV and YOLO makes it fully unit-testable.
"""
from __future__ import annotations

import random
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from .config import DIRECTIONS, VISION_LEVELS, Settings
from .config import settings as default_settings

READY = "SYSTEM READY"
TRACKING = "TRACKING ACTIVE"
COMPLETE = "ASSESSMENT COMPLETE"
TOO_CLOSE = "TOO CLOSE"
TOO_FAR = "TOO FAR"
FACE_MISSING = "ANTI-CHEAT: FACE NOT DETECTED"


@dataclass
class FrameObservation:
    """What the vision stack saw in one frame."""

    face_found: bool
    distance_cm: float | None = None
    direction: str | None = None  # gesture detected with enough confidence


def vision_level(score: int) -> str:
    return VISION_LEVELS[max(0, min(score, len(VISION_LEVELS) - 1))]


class AcuitySession:
    def __init__(self, cfg: Settings = default_settings, clock: Callable[[], float] = time.monotonic,
                 rng: random.Random | None = None):
        self.cfg = cfg
        self._clock = clock
        self._rng = rng or random.Random()
        self._lock = threading.Lock()
        self._votes: deque[str] = deque(maxlen=cfg.vote_window)
        self.privacy_mode = False
        self.distance_cm = (cfg.min_distance_cm + cfg.max_distance_cm) / 2
        self._reset_locked()

    # ----- commands (request threads) ---------------------------------------------------------
    def start(self, max_attempts: int | None = None) -> None:
        n = self.cfg.default_attempts if max_attempts is None else int(max_attempts)
        n = max(self.cfg.min_attempts, min(self.cfg.max_attempts, n))
        with self._lock:
            self._reset_locked()
            self.max_attempts = n
            self.is_active = True
            self.alert = TRACKING
            self._next_target_locked()

    def reset(self) -> None:
        with self._lock:
            self._reset_locked()

    def toggle_privacy(self) -> bool:
        with self._lock:
            self.privacy_mode = not self.privacy_mode
            return self.privacy_mode

    # ----- per-frame update (camera thread) ---------------------------------------------------
    def update(self, obs: FrameObservation) -> None:
        now = self._clock()
        cfg = self.cfg
        with self._lock:
            if obs.distance_cm is not None:
                # Light exponential smoothing so the number on screen doesn't flicker.
                self.distance_cm = round(self.distance_cm * 0.7 + obs.distance_cm * 0.3, 1)

            if not obs.face_found:
                self._missing_frames += 1
                in_range = False
                if self._missing_frames > cfg.face_missing_frames:
                    self.alert = FACE_MISSING
            else:
                self._missing_frames = 0
                in_range = cfg.min_distance_cm <= self.distance_cm <= cfg.max_distance_cm
                if self.distance_cm < cfg.min_distance_cm:
                    self.alert = TOO_CLOSE
                elif self.distance_cm > cfg.max_distance_cm:
                    self.alert = TOO_FAR
                elif self.test_finished:
                    self.alert = COMPLETE
                else:
                    self.alert = TRACKING if self.is_active else READY

            if not self.is_active or self.test_finished:
                return

            if not in_range:
                # Pause the trial clock while the patient is out of position.
                self._trial_start = now - (cfg.trial_seconds - self.time_left)
                return

            elapsed = now - self._trial_start
            self.time_left = max(0, cfg.trial_seconds - int(elapsed))
            if self.time_left == 0:
                self._finish_trial_locked(correct=False, response_time=None)
                return

            if obs.direction:
                self._votes.append(obs.direction)
            if self._votes.count(self.target) >= cfg.votes_needed:
                self._finish_trial_locked(correct=True, response_time=elapsed)

    # ----- read model -------------------------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "is_active": self.is_active,
                "test_finished": self.test_finished,
                "max_attempts": self.max_attempts,
                "target_direction": self.target,
                "score": self.score,
                "attempts": self.attempts,
                "time_left": self.time_left,
                "distance_cm": self.distance_cm,
                "vision_level": vision_level(self.score),
                "response_time": self.last_response_time,
                "hesitation": self.hesitation,
                "alert": self.alert,
                "privacy_mode": self.privacy_mode,
            }

    # ----- internals (lock held) --------------------------------------------------------------
    def _reset_locked(self) -> None:
        self.is_active = False
        self.test_finished = False
        self.max_attempts = self.cfg.default_attempts
        self.score = 0
        self.attempts = 0
        self.time_left = self.cfg.trial_seconds
        self.last_response_time = 0.0
        self.hesitation = False
        self.alert = READY
        self.target = self._rng.choice(DIRECTIONS)
        self._trial_start = self._clock()
        self._missing_frames = 0
        self._votes.clear()

    def _next_target_locked(self) -> None:
        self.target = self._rng.choice(DIRECTIONS)
        self._trial_start = self._clock()
        self.time_left = self.cfg.trial_seconds
        self._votes.clear()

    def _finish_trial_locked(self, correct: bool, response_time: float | None) -> None:
        self.attempts += 1
        if correct:
            self.score += 1
            self.last_response_time = round(response_time or 0.0, 2)
            self.hesitation = self.last_response_time > self.cfg.hesitation_seconds
        if self.attempts >= self.max_attempts:
            self.test_finished = True
            self.alert = COMPLETE
        else:
            self._next_target_locked()
