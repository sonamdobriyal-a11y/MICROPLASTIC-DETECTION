"""Train a YOLOv8 model to detect microplastics."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model on the microplastics dataset.")
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/microplastics.yaml"),
        help="Path to the dataset YAML describing train/val splits.",
    )
    parser.add_argument(
        
        
        help="Base model checkpoint to fine-tune (e.g. yolov8n.pt, yolov8s.pt).",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to train on ('' for auto, or specify 'cpu', 'cuda', '0', '0,1', etc.).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the last training run instead of starting from scratch.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs"),
        help="Directory where Ultralytics stores training artifacts.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="microplastics",
        help="Name for this training run (used as subdirectory under --project).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.resume:
        model = YOLO(args.model)
        model.train(
            resume=True,
            project=str(args.project),
            name=args.name,
        )
        return

    model = YOLO(args.model)
    model.train(
        data=str(args.data_config),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device or None,
        project=str(args.project),
        name=args.name,
    )


if __name__ == "__main__":
    main()
