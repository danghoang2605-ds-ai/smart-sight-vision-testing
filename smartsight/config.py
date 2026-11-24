"""Runtime settings. Every value can be overridden with an environment variable."""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")

# Snellen fraction reached after N correct answers (capped at the last entry).
VISION_LEVELS = ("20/200", "20/100", "20/70", "20/50", "20/40", "20/30", "20/25", "20/20")


def _env(name: str, default, cast=str):
    value = os.getenv(name)
    return default if value in (None, "") else cast(value)


@dataclass(frozen=True)
class Settings:
    model_path: Path = field(default_factory=lambda: Path(_env("SMARTSIGHT_MODEL", str(ROOT / "best.pt"))))
    camera_index: int = field(default_factory=lambda: _env("SMARTSIGHT_CAMERA", 0, int))
    host: str = field(default_factory=lambda: _env("SMARTSIGHT_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env("SMARTSIGHT_PORT", 5050, int))

    # Inference
    imgsz: int = 320
    conf_threshold: float = field(default_factory=lambda: _env("SMARTSIGHT_CONF", 0.50, float))
    jpeg_quality: int = 80

    # Distance estimation (pinhole model: distance = real_width * focal / pixel_width)
    known_face_width_cm: float = 14.3
    focal_length_px: float = field(default_factory=lambda: _env("SMARTSIGHT_FOCAL", 600.0, float))
    min_distance_cm: float = 50.0
    max_distance_cm: float = 70.0

    # Test protocol
    trial_seconds: int = 15
    default_attempts: int = 15
    min_attempts: int = 1
    max_attempts: int = 50
    vote_window: int = 15  # frames kept for majority voting
    votes_needed: int = 10  # matching frames required to accept an answer
    hesitation_seconds: float = 2.5
    face_missing_frames: int = 10  # ~1 s at 10+ FPS before the anti-cheat alert


settings = Settings()
