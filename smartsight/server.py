"""Flask app factory. The HTTP contract is unchanged from v2.1, so the dashboard works as before."""
from __future__ import annotations

import logging

from flask import Flask, Response, jsonify, render_template, request

from .config import ROOT, Settings
from .config import settings as default_settings
from .session import AcuitySession

log = logging.getLogger(__name__)


def build_worker(cfg: Settings, session: AcuitySession):
    """Load YOLO and wire up the camera worker. Imported lazily so tests don't need torch."""
    from ultralytics import YOLO

    from .camera import CameraWorker
    from .vision import VisionEngine

    if not cfg.model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {cfg.model_path} (set SMARTSIGHT_MODEL)")
    model = YOLO(str(cfg.model_path))
    log.info("Loaded gesture model %s with classes %s", cfg.model_path.name, model.names)
    return CameraWorker(cfg, session, VisionEngine(cfg, model))


def create_app(cfg: Settings = default_settings, session: AcuitySession | None = None, worker=None) -> Flask:
    app = Flask(__name__, template_folder=str(ROOT / "templates"))
    session = session or AcuitySession(cfg)
    app.config["session"] = session

    def get_worker():
        nonlocal worker
        if worker is None:
            worker = build_worker(cfg, session)
        return worker

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/video_feed")
    def video_feed():
        return Response(get_worker().frames(), mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={"Cache-Control": "no-store"})

    @app.post("/start")
    def start_test():
        data = request.get_json(silent=True) or {}
        try:
            attempts = int(data.get("max_attempts", cfg.default_attempts))
        except (TypeError, ValueError):
            return jsonify({"error": "max_attempts must be an integer"}), 400
        session.start(attempts)
        return jsonify({"status": "started", "max_attempts": session.max_attempts})

    @app.post("/reset")
    def reset_session():
        session.reset()
        return jsonify({"status": "reset"})

    @app.post("/pause")
    def toggle_pause():
        return jsonify({"paused": session.toggle_pause()})

    @app.post("/calibrate_distance")
    def calibrate_distance():
        data = request.get_json(silent=True) or {}
        try:
            distance = float(data.get("distance_cm"))
        except (TypeError, ValueError):
            return jsonify({"error": "distance_cm must be a number"}), 400
        if not 20 <= distance <= 200:
            return jsonify({"error": "distance_cm must be between 20 and 200"}), 400
        if worker is None:
            return jsonify({"error": "Camera is not running yet"}), 409
        try:
            focal = worker.engine.calibrate(distance)
        except LookupError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify({"focal_length_px": focal, "distance_cm": distance})

    @app.post("/toggle_privacy")
    def toggle_privacy():
        return jsonify({"privacy_mode": session.toggle_privacy()})

    @app.get("/test_status")
    def test_status():
        payload = session.snapshot()
        payload.update(worker.engine.telemetry.as_dict() if worker is not None else
                       {"fps": 0.0, "inference_ms": 0, "ambient_lux": 0, "head_stability_pct": 100.0,
                        "gesture": None, "gesture_conf": 0.0})
        return jsonify(payload)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "camera_started": worker is not None})

    return app
