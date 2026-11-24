# Roadmap

Work is grouped into small phases so each one can ship on its own.

## Done in v2.2
- [x] Split the 260-line `app.py` into `config`, `session`, `vision`, `camera`, `server` modules
- [x] One camera thread shared by all clients (was one capture + one YOLO loop per open tab)
- [x] Thread-safe session state; `/start` validates and clamps `max_attempts`
- [x] Stale "TOO CLOSE / TOO FAR" alert no longer sticks after the patient moves back while idle
- [x] Pick the highest-confidence box, filter by confidence inside YOLO, warm up the model before the first patient
- [x] Fail loudly when `best.pt` is missing instead of silently loading a COCO model that can never score
- [x] `requirements.txt` (README referenced it but it did not exist), OpenCV pinned below 5
- [x] 21 unit/API tests, ruff, GitHub Actions CI
- [x] Dashboard log no longer uses `innerHTML` for server text

## Phase 1: measurement quality (next)
- [x] Per-webcam focal-length calibration (tape-measured distance, saved to `calibration.json`)
- [ ] Replace the Haar cascade with OpenCV's `FaceDetectorYN` or MediaPipe for better distance under side lighting (also removes the OpenCV < 5 pin)
- [x] Line-by-line staircase: 2 correct pass a line, 2 misses end the test, result = smallest line passed
- [x] True-size SVG Landolt C with bank-card screen calibration
- [x] Pause/resume on the server; result screen shows real averages, hesitations and duration
- [x] Target direction is no longer highlighted or logged on the patient-facing screen
- [ ] Separate left-eye / right-eye runs

## Phase 2: product
- [ ] Session report export (PDF/CSV) with per-trial response times
- [ ] Multi-patient support: one session per browser instead of one global session
- [ ] Docker image and a one-command demo
- [ ] Short demo GIF in the README

## Phase 3: model
- [ ] Publish the training notebook and dataset card for `best.pt`
- [ ] Retrain so "index finger pointing right" means RIGHT (today the dataset maps it to LEFT)
- [ ] ONNX export for CPU-only machines
