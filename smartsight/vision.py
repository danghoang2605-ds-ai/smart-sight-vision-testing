"""Per-frame computer vision: face tracking, distance, lighting, privacy blur and gesture detection."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import DIRECTIONS, Settings

DETECT_SIZE = (320, 240)  # face detection runs on a downsampled frame for speed
BOX_COLOR = (238, 211, 34)  # BGR, matches the dashboard accent


def estimate_distance_cm(face_width_px: float, cfg: Settings) -> float | None:
    if face_width_px <= 0:
        return None
    return cfg.known_face_width_cm * cfg.focal_length_px / face_width_px


def estimate_lux(gray: np.ndarray) -> int:
    """Rough ambient-light proxy from mean frame brightness (not a calibrated lux reading)."""
    return int(float(np.mean(gray)) * 2.5)


def blur_region(frame: np.ndarray, box: tuple[int, int, int, int], margin: int = 20) -> None:
    h_img, w_img = frame.shape[:2]
    x, y, w, h = box
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(w_img, x + w + margin), min(h_img, y + h + margin)
    if x2 > x1 and y2 > y1:
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 25)


def head_stability(points: deque, frame_width: int) -> float | None:
    if len(points) < points.maxlen:
        return None
    arr = np.asarray(points, dtype=float)
    jitter = (arr[:, 0].std() + arr[:, 1].std()) * 1000 / frame_width
    return round(float(np.clip(100.0 - jitter, 0.0, 100.0)), 1)


def normalize_label(name: str) -> str | None:
    label = str(name).strip().upper()
    return label if label in DIRECTIONS else None


@dataclass
class Telemetry:
    fps: float = 30.0
    inference_ms: int = 0
    ambient_lux: int = 0
    head_stability_pct: float = 100.0
    gesture: str | None = None  # last gesture seen by YOLO, for the HUD
    gesture_conf: float = 0.0

    def as_dict(self) -> dict:
        return {"fps": self.fps, "inference_ms": self.inference_ms,
                "ambient_lux": self.ambient_lux, "head_stability_pct": self.head_stability_pct,
                "gesture": self.gesture, "gesture_conf": round(self.gesture_conf, 2)}


@dataclass
class FaceResult:
    box: tuple[int, int, int, int] | None
    distance_cm: float | None


@dataclass
class VisionEngine:
    """Holds the detectors and the small amount of cross-frame state they need."""

    cfg: Settings
    model: object  # ultralytics.YOLO
    face_cascade: cv2.CascadeClassifier = field(default_factory=lambda: cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"))
    telemetry: Telemetry = field(default_factory=Telemetry)
    _smoothed_width: float | None = None
    _head: deque = field(default_factory=lambda: deque(maxlen=15))
    _last_ts: float = field(default_factory=time.perf_counter)

    def tick_fps(self) -> None:
        now = time.perf_counter()
        dt = now - self._last_ts
        self._last_ts = now
        if dt > 0:
            self.telemetry.fps = round(self.telemetry.fps * 0.9 + (1.0 / dt) * 0.1, 1)

    def find_face(self, frame: np.ndarray) -> FaceResult:
        h_img, w_img = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.telemetry.ambient_lux = estimate_lux(gray)
        small = cv2.resize(gray, DETECT_SIZE)
        faces = self.face_cascade.detectMultiScale(small, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            return FaceResult(None, None)

        xs, ys, ws, hs = max(faces, key=lambda b: b[2] * b[3])
        sx, sy = w_img / DETECT_SIZE[0], h_img / DETECT_SIZE[1]
        box = (int(xs * sx), int(ys * sy), int(ws * sx), int(hs * sy))
        x, y, w, h = box

        self._smoothed_width = w if self._smoothed_width is None else self._smoothed_width * 0.88 + w * 0.12
        self._head.append((x + w // 2, y + h // 2))
        stability = head_stability(self._head, w_img)
        if stability is not None:
            self.telemetry.head_stability_pct = stability
        return FaceResult(box, estimate_distance_cm(self._smoothed_width, self.cfg))

    def warmup(self) -> None:
        """The first YOLO call is ~1 s slower (graph setup); pay that before the patient is on screen."""
        self.model(np.zeros((240, 320, 3), dtype=np.uint8), imgsz=self.cfg.imgsz, verbose=False)

    def detect_gesture(self, frame: np.ndarray) -> str | None:
        """Run YOLO, draw the best box and return its direction label (or None)."""
        start = time.perf_counter()
        results = self.model(frame, imgsz=self.cfg.imgsz, conf=self.cfg.conf_threshold, verbose=False)
        self.telemetry.inference_ms = int((time.perf_counter() - start) * 1000)

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            self.telemetry.gesture, self.telemetry.gesture_conf = None, 0.0
            return None
        best = int(boxes.conf.argmax())
        conf = float(boxes.conf[best])
        direction = normalize_label(self.model.names[int(boxes.cls[best])])
        self.telemetry.gesture, self.telemetry.gesture_conf = direction, conf
        x1, y1, x2, y2 = map(int, boxes.xyxy[best])
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        cv2.putText(frame, f"{direction or '?'} {conf:.2f}", (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, BOX_COLOR, 2)
        return direction
