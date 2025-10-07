# Microplastic Detection

YOLOv8-based pipeline to detect microplastics from microscopy imagery.

## Project structure

- `data/` – raw Roboflow export with `train` and `valid` splits (`_annotations.csv` + images).
- `dataset/` – generated YOLO-format dataset (created by `scripts/prepare_dataset.py`).
- `configs/microplastics.yaml` – Ultralytics dataset definition.
- `train.py` – training entrypoint for YOLOv8.
- `live_inference.py` – real-time webcam/video inference.
- `requirements.txt` – Python dependencies.

## Setup

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Install PyTorch with CPU or GPU support (follow the selector on <https://pytorch.org/get-started/locally/>).

## Dataset preparation

Convert the Roboflow CSV annotations into YOLO format (images are copied, labels generated):

```powershell
python scripts/prepare_dataset.py
```

This populates `dataset/images/{train,valid}` and `dataset/labels/{train,valid}`.

## Training

Fine-tune YOLOv8 (defaults to `yolov8n.pt`, 50 epochs, batch size 16):

```powershell
python train.py --model yolov8n.pt --epochs 50 --batch 16 --imgsz 640
```

Artifacts are stored under `runs/microplastics/`. Modify `--project` or `--name` to change the output directory.

## Evaluate on images

Run a trained model on a single image:

```python
from ultralytics import YOLO

model = YOLO("runs/microplastics/weights/best.pt")
model.predict("path/to/image.jpg", conf=0.3, save=True)
```

Annotated images are written to `runs/detect/predict/`.

## Live inference

Launch webcam/stream detection:

```powershell
python live_inference.py --weights runs/microplastics/weights/best.pt --source 0
```

Press `q` to exit the viewer window.

## Git workflow

```powershell
git add .
git commit -m "Message"
git push
```
