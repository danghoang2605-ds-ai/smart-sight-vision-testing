from collections import deque

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from smartsight.camera import placeholder_jpeg  # noqa: E402
from smartsight.config import Settings  # noqa: E402
from smartsight.vision import blur_region, estimate_distance_cm, head_stability, normalize_label  # noqa: E402


def test_distance_pinhole_model():
    cfg = Settings()
    assert estimate_distance_cm(143, cfg) == pytest.approx(60.0)
    assert estimate_distance_cm(0, cfg) is None


def test_normalize_label():
    assert normalize_label(" left ") == "LEFT"
    assert normalize_label("thumbs") is None


def test_head_stability():
    pts = deque([(100, 100)] * 15, maxlen=15)
    assert head_stability(pts, 640) == 100.0
    assert head_stability(deque([(1, 1)], maxlen=15), 640) is None


def test_blur_region_clips_to_frame():
    frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    before = frame.copy()
    blur_region(frame, (80, 80, 50, 50))
    assert not np.array_equal(frame, before)


def test_placeholder_is_jpeg():
    assert placeholder_jpeg("hello")[:2] == b"\xff\xd8"


def test_focal_round_trip(tmp_path):
    from smartsight.vision import focal_from_measurement, load_focal

    cfg = Settings(calibration_path=tmp_path / "cal.json")
    focal = focal_from_measurement(200, 50, cfg)
    assert estimate_distance_cm(200, cfg, focal) == pytest.approx(50)
    assert load_focal(cfg) == cfg.focal_length_px  # no file yet
    (tmp_path / "cal.json").write_text('{"focal_length_px": 812.5}')
    assert load_focal(cfg) == 812.5
    (tmp_path / "cal.json").write_text("garbage")
    assert load_focal(cfg) == cfg.focal_length_px


def test_engine_calibrate_needs_a_face(tmp_path):
    from smartsight.vision import VisionEngine

    cfg = Settings(calibration_path=tmp_path / "cal.json")
    engine = VisionEngine(cfg, model=None)
    with pytest.raises(LookupError):
        engine.calibrate(60)
    engine._smoothed_width = 150.0
    engine._last_face_ts = __import__("time").perf_counter()
    focal = engine.calibrate(60)
    assert focal == pytest.approx(150 * 60 / cfg.known_face_width_cm, abs=0.1)
    assert (tmp_path / "cal.json").exists()
