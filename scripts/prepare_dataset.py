"""Utilities to convert Roboflow CSV annotations into YOLO format.

This script creates a YOLO-compatible directory layout under the provided
target directory (default: ``dataset``). Images are copied into
``{target}/images/{split}`` and label text files are generated in
``{target}/labels/{split}``.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


BoundingBox = Tuple[float, float, float, float]


def load_annotations(csv_path: Path) -> Dict[str, List[BoundingBox]]:
    """Parse a Roboflow-style CSV and return YOLO-normalised boxes per image."""
    annotations: Dict[str, List[BoundingBox]] = defaultdict(list)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            width = float(row["width"])
            height = float(row["height"])
            xmin = float(row["xmin"])
            xmax = float(row["xmax"])
            ymin = float(row["ymin"])
            ymax = float(row["ymax"])

            # Convert to YOLO format (normalised centre x/y and width/height).
            x_c = ((xmin + xmax) / 2.0) / width
            y_c = ((ymin + ymax) / 2.0) / height
            box_w = (xmax - xmin) / width
            box_h = (ymax - ymin) / height

            # Clamp values to the valid range in case of minor rounding issues.
            x_c = min(max(x_c, 0.0), 1.0)
            y_c = min(max(y_c, 0.0), 1.0)
            box_w = min(max(box_w, 0.0), 1.0)
            box_h = min(max(box_h, 0.0), 1.0)

            annotations[row["filename"]].append((x_c, y_c, box_w, box_h))

    return annotations


def copy_images(image_paths: Iterable[Path], destination_dir: Path) -> None:
    """Copy images into the YOLO directory, skipping files that already exist."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        target_path = destination_dir / image_path.name
        if target_path.exists():
            continue
        shutil.copy2(image_path, target_path)


def write_label_file(label_path: Path, boxes: Iterable[BoundingBox], class_id: int) -> None:
    """Write YOLO label file for a single image."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as handle:
        for box in boxes:
            x_c, y_c, box_w, box_h = box
            handle.write(f"{class_id} {x_c:.6f} {y_c:.6f} {box_w:.6f} {box_h:.6f}\n")


def process_split(
    source_dir: Path,
    target_root: Path,
    split_name: str,
    class_id: int,
) -> Tuple[int, int]:
    """Convert a dataset split into YOLO format.

    Returns a tuple: (number of images found, number of label files written).
    """
    csv_path = source_dir / "_annotations.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing annotations file: {csv_path}")

    annotations = load_annotations(csv_path)
    image_paths = sorted(p for p in source_dir.glob("*.jpg"))

    copy_images(image_paths, target_root / "images" / split_name)

    labels_written = 0
    for image_path in image_paths:
        boxes = annotations.get(image_path.name, [])
        label_path = target_root / "labels" / split_name / f"{image_path.stem}.txt"
        write_label_file(label_path, boxes, class_id)
        labels_written += 1

    return len(image_paths), labels_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare microplastic dataset for YOLO training.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory containing the Roboflow exports (train/ and valid/).",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("dataset"),
        help="Directory where YOLO-formatted images/labels will be stored.",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="Class id to assign to all bounding boxes (default: 0).",
    )
    args = parser.parse_args()

    splits = ("train", "valid")
    for split in splits:
        image_count, label_count = process_split(
            source_dir=args.data_root / split,
            target_root=args.target_root,
            split_name=split,
            class_id=args.class_id,
        )
        print(f"{split}: {image_count} images processed, {label_count} label files written.")

    print(f"Dataset ready at: {args.target_root}")


if __name__ == "__main__":
    main()
