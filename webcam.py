"""Standalone OpenCV window for checking the gesture model without the web dashboard.

Usage: python webcam.py [--camera 0] [--conf 0.5]   (press q to quit)
"""
import argparse

import cv2
from ultralytics import YOLO

from smartsight.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=settings.camera_index)
    parser.add_argument("--conf", type=float, default=settings.conf_threshold)
    parser.add_argument("--model", default=str(settings.model_path))
    args = parser.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")
    print(f"Classes: {model.names}. Press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            results = model.predict(frame, imgsz=settings.imgsz, conf=args.conf, verbose=False)
            cv2.imshow("SmartSight - gesture check", results[0].plot())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
