import random

import pytest

from smartsight.config import Settings
from smartsight.session import (
    COMPLETE,
    FACE_MISSING,
    READY,
    TOO_CLOSE,
    TOO_FAR,
    AcuitySession,
    FrameObservation,
    vision_level,
)


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def session(clock):
    return AcuitySession(Settings(), clock=clock, rng=random.Random(0))


def seen(direction=None, dist=60.0):
    return FrameObservation(face_found=True, distance_cm=dist, direction=direction)


def answer(s, n=10):
    for _ in range(n):
        s.update(seen(s.target))


def test_vision_level_is_capped():
    assert vision_level(0) == "20/200"
    assert vision_level(7) == "20/20"
    assert vision_level(99) == "20/20"
    assert vision_level(-3) == "20/200"


def test_start_clamps_attempts(session):
    session.start(0)
    assert session.max_attempts == 1
    session.start(10_000)
    assert session.max_attempts == Settings().max_attempts


def test_majority_vote_needs_ten_matching_frames(session):
    session.start(5)
    answer(session, 9)
    assert session.score == 0
    session.update(seen(session.target))
    assert session.score == 1 and session.attempts == 1


def test_wrong_directions_do_not_score(session):
    session.start(5)
    wrong = next(d for d in ("UP", "DOWN", "LEFT", "RIGHT") if d != session.target)
    for _ in range(20):
        session.update(seen(wrong))
    assert session.score == 0


def test_timeout_counts_as_miss(session, clock):
    session.start(5)
    first = session.attempts
    clock.t += 15.1
    session.update(seen())
    assert session.attempts == first + 1 and session.score == 0
    assert session.time_left == 15


def test_clock_pauses_when_out_of_range(session, clock):
    session.start(5)
    clock.t += 5
    session.update(seen())
    assert session.time_left == 10
    for _ in range(30):  # patient steps away for 60 s; distance smoothing settles above the limit
        clock.t += 2
        session.update(seen(dist=150))
    assert session.alert == TOO_FAR
    for _ in range(30):
        session.update(seen(dist=60))
    assert session.time_left == 10 and session.attempts == 0


def test_hesitation_flag(session, clock):
    session.start(5)
    clock.t += 3
    answer(session)
    assert session.hesitation and session.last_response_time == pytest.approx(3.0)


def test_session_completes(session):
    session.start(3)
    for _ in range(3):
        answer(session)
    snap = session.snapshot()
    assert snap["test_finished"] and snap["alert"] == COMPLETE and snap["score"] == 3
    answer(session)  # no scoring after completion
    assert session.snapshot()["score"] == 3


def test_face_missing_alert(session):
    for _ in range(11):
        session.update(FrameObservation(face_found=False))
    assert session.alert == FACE_MISSING
    for _ in range(10):
        session.update(seen())
    assert session.alert == READY


def test_too_close_alert(session):
    for _ in range(20):
        session.update(seen(dist=20))
    assert session.alert == TOO_CLOSE


def test_reset_keeps_privacy(session):
    session.toggle_privacy()
    session.start(5)
    answer(session)
    session.reset()
    snap = session.snapshot()
    assert snap["score"] == 0 and not snap["is_active"] and snap["privacy_mode"]
