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
    assert vision_level(-1) is None


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


def wrong_answer(s, n=10):
    # A fresh wrong direction: neither the target nor the hand still up from the last trial.
    wrong = next(d for d in ("UP", "DOWN", "LEFT", "RIGHT") if d not in (s.target, s._carryover))
    for _ in range(n):
        s.update(seen(wrong))


def test_sustained_wrong_direction_is_a_miss(session):
    session.start(5)
    wrong_answer(session)
    assert session.score == 0 and session.attempts == 1


def test_staircase_passes_lines_with_two_correct(session):
    session.start(20)
    answer(session)
    assert session.level == 0 and session.best_level == -1
    answer(session)
    assert session.level == 1 and session.best_level == 0
    assert session.snapshot()["vision_level"] == "20/200"
    assert session.snapshot()["level_snellen"] == "20/100"


def test_two_misses_on_a_line_end_the_test(session):
    session.start(20)
    answer(session, 10)
    answer(session, 10)  # pass 20/200
    wrong_answer(session)
    answer(session)
    assert not session.test_finished
    wrong_answer(session)
    snap = session.snapshot()
    assert snap["test_finished"] and snap["vision_level"] == "20/200"


def test_passing_every_line_finishes_at_20_20(session):
    session.start(50)
    for _ in range(16):
        answer(session)
    snap = session.snapshot()
    assert snap["test_finished"] and snap["vision_level"] == "20/20" and snap["attempts"] == 16


def test_no_line_passed_reports_none(session):
    session.start(20)
    wrong_answer(session)
    wrong_answer(session)
    snap = session.snapshot()
    assert snap["test_finished"] and snap["vision_level"] is None


def test_target_never_repeats(session):
    session.start(50)
    previous = session.target
    for _ in range(12):
        answer(session)
        assert session.target != previous
        previous = session.target


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


def test_pause_freezes_clock_and_scoring(session, clock):
    session.start(5)
    clock.t += 4
    session.update(seen())
    assert session.toggle_pause() is True
    clock.t += 30
    answer(session, 20)
    assert session.score == 0 and session.attempts == 0
    assert session.toggle_pause() is False
    session.update(seen())
    assert session.time_left == 11


def test_pause_ignored_when_idle(session):
    assert session.toggle_pause() is False


def test_summary_metrics(session, clock):
    session.start(3)
    clock.t += 3
    answer(session)  # 3 s: hesitation
    clock.t += 1
    answer(session)  # 1 s
    wrong_answer(session)
    snap = session.snapshot()
    assert snap["test_finished"]
    assert snap["hesitations"] == 1
    assert snap["avg_response_time"] == pytest.approx((3 + 1 + 0) / 3, abs=0.01)
    clock.t += 100
    assert session.snapshot()["duration_s"] == 4  # frozen at the moment the test ended


def test_hand_still_up_from_last_answer_is_not_a_miss(session):
    session.start(20)
    previous = session.target
    answer(session)  # correct; a new, different target appears
    for _ in range(30):  # the patient keeps holding the previous direction
        session.update(seen(previous))
    assert session.attempts == 1 and session.score == 1
    session.update(seen(None))  # hand lowered
    answer(session)
    assert session.score == 2


def test_changing_to_another_wrong_direction_still_counts(session):
    session.start(20)
    previous = session.target
    answer(session)
    wrong = next(d for d in ("UP", "DOWN", "LEFT", "RIGHT") if d not in (previous, session.target))
    for _ in range(10):
        session.update(seen(wrong))
    assert session.attempts == 2 and session.score == 1
