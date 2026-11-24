"""One background thread owns the webcam and the AI pipeline.

The original design opened the camera and ran inference inside each
``/video_feed`` generator, so a second browser tab fought over the device and
doubled the CPU load. Here a single worker processes frames and every client
just receives the latest JPEG.
"""
from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

from .config import Settings
from .session import AcuitySession, FrameObservation
from .vision import VisionEngine, blur_region

log = logging.getLogger(__name__)


def placeholder_jpeg(message: str, size: tuple[int, int] = (640, 480)) -> bytes:
    w, h = size
    img = np.full((h, w, 3), 24, dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(img, message, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    return cv2.imencode(".jpg", img)[1].tobytes()


class CameraWorker:
    def __init__(self, cfg: Settings, session: AcuitySession, engine: VisionEngine):
        self.cfg = cfg
        self.session = session
        self.engine = engine
        self._frame = placeholder_jpeg("Starting camera...")
        self._seq = 0
        self._cond = threading.Condition()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="smartsight-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _publish(self, jpeg: bytes) -> None:
        with self._cond:
            self._frame = jpeg
            self._seq += 1
            self._cond.notify_all()

    def _run(self) -> None:
        self.engine.warmup()
        cap = cv2.VideoCapture(self.cfg.camera_index)
        if not cap.isOpened():
            log.error("Could not open camera %s", self.cfg.camera_index)
            self._publish(placeholder_jpeg(f"Camera {self.cfg.camera_index} unavailable"))
            return
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    self._publish(placeholder_jpeg("Camera disconnected"))
                    time.sleep(0.5)
                    continue
                self._publish(self.process(frame))
        finally:
            cap.release()

    def process(self, frame: np.ndarray) -> bytes:
        frame = cv2.flip(frame, 1)  # mirror view feels natural for the patient
        self.engine.tick_fps()
        face = self.engine.find_face(frame)

        snap = self.session.snapshot()
        in_range = (face.distance_cm is not None
                    and self.cfg.min_distance_cm <= snap["distance_cm"] <= self.cfg.max_distance_cm)
        direction = None
        if snap["is_active"] and not snap["test_finished"] and in_range:
            direction = self.engine.detect_gesture(frame)

        self.session.update(FrameObservation(face.box is not None, face.distance_cm, direction))
        if face.box is not None and snap["privacy_mode"]:
            blur_region(frame, face.box)
        return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])[1].tobytes()

    def frames(self):
        """MJPEG generator: yields each new frame once, per connected client."""
        self.start()
        last = -1
        while True:
            with self._cond:
                self._cond.wait_for(lambda seen=last: self._seq != seen, timeout=2.0)
                jpeg, last = self._frame, self._seq
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
