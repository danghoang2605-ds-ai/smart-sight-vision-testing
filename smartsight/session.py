"""Test-session state machine.

Acuity is measured with a simple line-by-line staircase, the way a clinician reads a chart:
start on the largest line (20/200); two correct answers on a line pass it and move to the
next, smaller line; two misses on a line end the test. The result is the smallest line passed.
A miss is either a timeout or a sustained gesture in the wrong direction.


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


PASS_PER_LINE = 2  # correct answers needed to pass a line
FAIL_PER_LINE = 2  # misses on one line that end the test


def vision_level(level: int) -> str | None:
    """Snellen fraction for a line index, or None when no line was passed."""
    if level < 0:
        return None
    return VISION_LEVELS[min(level, len(VISION_LEVELS) - 1)]


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
            self._started_at = self._clock()
            self.alert = TRACKING
            self._next_target_locked()

    def reset(self) -> None:
        with self._lock:
            self._reset_locked()

    def toggle_pause(self) -> bool:
        """Freeze the trial clock and scoring; resuming continues the same trial."""
        with self._lock:
            if self.is_active and not self.test_finished:
                self.paused = not self.paused
            return self.paused

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

            if not in_range or self.paused:
                # Freeze the trial clock while paused or while the patient is out of position.
                self._trial_start = now - (cfg.trial_seconds - self.time_left)
                return

            elapsed = now - self._trial_start
            self.time_left = max(0, cfg.trial_seconds - int(elapsed))
            if self.time_left == 0:
                self._finish_trial_locked(correct=False, response_time=None)
                return

            if obs.direction:
                self._votes.append(obs.direction)
            if self._votes:
                answer = max(set(self._votes), key=self._votes.count)
                if self._votes.count(answer) >= cfg.votes_needed:
                    self._finish_trial_locked(correct=answer == self.target, response_time=elapsed)

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
                "vision_level": vision_level(self.best_level),
                "level": self.level,
                "level_snellen": VISION_LEVELS[self.level],
                "best_level": self.best_level,
                "hesitations": self.hesitations,
                "avg_response_time": round(sum(self._response_times) / len(self._response_times), 2)
                if self._response_times else 0.0,
                "duration_s": self._duration_locked(),
                "response_time": self.last_response_time,
                "hesitation": self.hesitation,
                "alert": self.alert,
                "privacy_mode": self.privacy_mode,
                "paused": self.paused,
            }

    # ----- internals (lock held) --------------------------------------------------------------
    def _reset_locked(self) -> None:
        self.is_active = False
        self.paused = False
        self.test_finished = False
        self.max_attempts = self.cfg.default_attempts
        self.score = 0
        self.attempts = 0
        self.hesitations = 0
        self._response_times: list[float] = []
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self.level = 0  # line currently shown (0 = 20/200)
        self.best_level = -1  # smallest line passed so far
        self._line_correct = 0
        self._line_missed = 0
        self.time_left = self.cfg.trial_seconds
        self.last_response_time = 0.0
        self.hesitation = False
        self.alert = READY
        self.target = self._rng.choice(DIRECTIONS)
        self._trial_start = self._clock()
        self._missing_frames = 0
        self._votes.clear()

    def _duration_locked(self) -> int:
        if self._started_at is None:
            return 0
        end = self._finished_at if self._finished_at is not None else self._clock()
        return int(end - self._started_at)

    def _next_target_locked(self) -> None:
        # Never repeat the previous direction: a hand still held up would score instantly.
        self.target = self._rng.choice([d for d in DIRECTIONS if d != self.target])
        self._trial_start = self._clock()
        self.time_left = self.cfg.trial_seconds
        self._votes.clear()

    def _finish_trial_locked(self, correct: bool, response_time: float | None) -> None:
        self.attempts += 1
        if response_time is not None:
            self.last_response_time = round(response_time, 2)
            self.hesitation = self.last_response_time > self.cfg.hesitation_seconds
            self.hesitations += int(self.hesitation)
            self._response_times.append(response_time)
        line_ended = False
        if correct:
            self.score += 1
            self._line_correct += 1
            if self._line_correct >= PASS_PER_LINE:
                self.best_level = self.level
                line_ended = True
        else:
            self._line_missed += 1
            if self._line_missed >= FAIL_PER_LINE:
                self.test_finished = True
        if line_ended:
            self._line_correct = self._line_missed = 0
            if self.level == len(VISION_LEVELS) - 1:
                self.test_finished = True  # passed the smallest line on the chart
            else:
                self.level += 1
        if self.test_finished or self.attempts >= self.max_attempts:
            self.test_finished = True
            self._finished_at = self._clock()
            self.alert = COMPLETE
        else:
            self._next_target_locked()
