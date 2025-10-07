"""Run real-time microplastic detection on a live video stream."""

from __future__ import annotations

import argparse
import sys

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Microplastic detection on a live video feed.")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/microplastics/weights/best.pt",
        help="Path to the trained model weights.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source (webcam index like '0' or path/URL to a stream).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Confidence threshold for predictions.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Computation device to use ('', 'cpu', 'cuda', '0', etc.).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Unable to open video source: {args.source}", file=sys.stderr)
        sys.exit(1)

    model = YOLO(args.weights)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from source.", file=sys.stderr)
            break

        results = model.predict(
            source=frame,
            conf=args.conf,
            device=args.device or None,
            verbose=False,
        )

        annotated_frame = results[0].plot()
        cv2.imshow("Microplastic Detection", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
