# SmartSight v2.2 — AI-Powered Clinical Vision Testing System

A telehealth vision-testing web app that uses **YOLOv8** for real-time hand gesture recognition and **OpenCV** for face detection and distance estimation. Patients point their hand in the direction of the opening in a rotating Landolt C optotype; the system scores their visual acuity automatically using the international Snellen chart scale.

---

## Features

| Feature | Description |
|---|---|
| **YOLOv8 Gesture Recognition** | Detects hand direction in real-time with optimized downsampled inference for high FPS. |
| **Distance Estimation** | Estimates physical distance (cm) from the camera using dynamic face-width bounding boxes. |
| **15-Second Timeout** | Automatically flags incorrect attempts and changes the target if no answer is provided. |
| **Cognitive Delay Tracking** | Measures response time and flags hesitation if the patient takes > 2.5 seconds to answer. |
| **Anti-Cheat / Gaze Tracking** | Pauses scoring and alerts the clinician if the patient's face disappears for > 1 second. |
| **HIPAA Privacy Mode** | Live Gaussian-blur face anonymization that can be toggled on/off by the clinician. |
| **Majority Voting Filter** | Requires a 10/15 frame confidence consensus before scoring to eliminate noise/jitter. |
| **Snellen Acuity Mapping** | Automatic Score → `20/200` to `20/20` clinical vision level conversion. |

---

## Gesture Navigation Guide

Due to the specific pattern recognition of the underlying YOLOv8 dataset, the AI distinguishes directions based on specific finger shapes rather than spatial movement. Please use the following gestures to interact with the Landolt C chart:

| Gesture | Direction |
|---|---|
| 👈 Index Finger pointing left (`<--`) | **LEFT** |
| 👉 Index Finger pointing right (`-->`) | **LEFT** |
| 👆 Index Finger pointing up | **UP** |
| 👇 Index Finger pointing down | **DOWN** |
| 👍 Horizontal Thumb | **RIGHT** |

> **Note:** The camera feed is horizontally flipped (mirror mode) by default to make hand-eye coordination more natural for the patient.

---

## How it works

```text
webcam ──► CameraWorker (1 background thread) ──► latest JPEG ──► /video_feed (any number of tabs)
             │  mirror · face (Haar) · distance · lux · head stability
             │  YOLOv8 gesture (only while a test runs and the patient is 50–70 cm away)
             ▼
        AcuitySession (thread-safe state machine) ──► /test_status (polled every 250 ms)
             majority vote 10/15 frames · 15 s trial timer · pauses when out of position
```

One worker owns the camera, so opening the dashboard in a second tab no longer opens the device twice or doubles the inference load.

## Project structure

```text
smart-sight-vision-testing/
├── app.py                 # entry point: python app.py
├── webcam.py              # standalone OpenCV window to check the model
├── best.pt                # custom-trained YOLOv8 weights (classes: UP, DOWN, LEFT, RIGHT)
├── smartsight/
│   ├── config.py          # every tunable in one place, env-overridable
│   ├── session.py         # test protocol state machine (no OpenCV / torch)
│   ├── vision.py          # face, distance, lux, blur, YOLO gesture
│   ├── camera.py          # single capture + processing thread, MJPEG fan-out
│   └── server.py          # Flask app factory and routes
├── templates/index.html   # clinician dashboard
├── tests/                 # pytest suite, runs without a webcam or torch
└── ROADMAP.md
```

## Quick start

Requires **Python 3.9+** and a webcam.

```bash
git clone https://github.com/danghoang2605-ds-ai/smart-sight-vision-testing.git
cd smart-sight-vision-testing
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open **http://localhost:5050** in Chrome or Edge.

> `opencv-python` is pinned below 5: OpenCV 5 removed `cv2.CascadeClassifier`, which the face tracker uses.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SMARTSIGHT_CAMERA` | `0` | Webcam index |
| `SMARTSIGHT_MODEL` | `best.pt` | Path to the YOLO weights |
| `SMARTSIGHT_CONF` | `0.50` | Minimum detection confidence |
| `SMARTSIGHT_FOCAL` | `600` | Camera focal length in pixels, used for distance; calibrate per webcam |
| `SMARTSIGHT_HOST` / `SMARTSIGHT_PORT` | `127.0.0.1` / `5050` | Bind address |

## API

| Method | Path | Body / result |
|---|---|---|
| `GET` | `/` | Dashboard |
| `GET` | `/video_feed` | MJPEG stream |
| `POST` | `/start` | `{"max_attempts": 15}` (clamped to 1–50) |
| `POST` | `/reset` | Ends the session |
| `POST` | `/toggle_privacy` | `{"privacy_mode": true}` |
| `GET` | `/test_status` | Session state and telemetry |
| `GET` | `/health` | Liveness |

## Development

```bash
pip install -r requirements-dev.txt
ruff check .
pytest -q
```

CI runs the same checks on every push.

## Disclaimer

SmartSight is a research and portfolio prototype. It is not a medical device and its result is not a diagnosis. Distance is estimated from face width and the brightness figure is a proxy, not a calibrated lux reading.

See [ROADMAP.md](ROADMAP.md) for planned work.
