# SafeSleep

Real-time infant sleep safety monitor using computer vision. Detects potentially hazardous objects and conditions in a crib via webcam and alerts caregivers.

## What It Detects

| Class | Label | Risk |
|---|---|---|
| 0 | safe | Baby sleeping safely — no action needed |
| 1 | unsafe | Hazardous position detected |
| 2 | toy | Toy present in crib (choking / SIDS risk) |
| 3 | blanket | Bedding present (suffocation risk) |

## Model

- Architecture: YOLOv8 Nano (fine-tuned from COCO pretrained weights)
- Training data: 500 annotated crib images (400 train / 100 val), 80/20 random split
- Training: 50 epochs, batch size 16, image size 640×640
- Weights: `runs/detect/train3/weights/best.pt`

**Validation performance:**

| Metric | Value |
|---|---|
| Precision | 99.0% |
| Recall | 100% |
| mAP50 | 99.5% |
| mAP50-95 | 77.2% |

## Project Structure

```
SafeSleep/
├── datafiles/
│   ├── images/
│   │   ├── train/          # 400 training images (.jpg)
│   │   └── val/            # 100 validation images (.jpg)
│   ├── labels/
│   │   ├── train/          # YOLO-format annotations for training images
│   │   └── val/            # YOLO-format annotations for validation images
│   ├── data.yaml           # Dataset config (paths, class names)
│   ├── webcam_detect.py    # Local webcam inference with OpenCV display
│   ├── safesleep.py        # FastAPI MJPEG streaming server
│   ├── sortimages.py       # Splits dataset into train/val folders (80/20)
│   └── ranamefiles.py      # Normalizes .jpeg → .jpg before training
├── runs/
│   └── detect/
│       └── train3/         # Final training run — metrics, plots, weights
│           └── weights/
│               ├── best.pt # Best checkpoint (use this for inference)
│               └── last.pt # Final epoch checkpoint
└── yolov8n.pt              # Base pretrained weights (COCO)
```

## Setup

**Requirements:**
```
ultralytics
opencv-python
torch
fastapi
uvicorn
```

Install:
```bash
pip install ultralytics opencv-python torch fastapi uvicorn
```

## Running Inference

### Local webcam (OpenCV window)
```bash
python datafiles/webcam_detect.py
```
Press `q` to quit.

### Web streaming server (browser-accessible)
```bash
uvicorn datafiles.safesleep:app --host 0.0.0.0 --port 8000
```
Then open `http://localhost:8000/video_feed` in a browser. The feed is served as an MJPEG stream.

## Training From Scratch

**1. Prepare images**

Rename any `.jpeg` files to `.jpg`:
```bash
python datafiles/ranamefiles.py
```

Split into train/val folders:
```bash
python datafiles/sortimages.py
```

**2. Annotate**

Annotations must be in YOLO format — one `.txt` file per image, with the same filename stem. Each line describes one object:
```
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates are normalized (0.0–1.0) relative to image dimensions.

**3. Train**
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="datafiles/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
)
```

Best weights will be saved to `runs/detect/trainN/weights/best.pt`.

## Annotation Format

Each label file corresponds to one image and contains one row per detected object:

```
0 0.544935 0.520746 0.440943 0.408088
```

Fields: `class  x_center  y_center  width  height` — all as fractions of the image dimensions, so the format is resolution-independent.

## Known Limitations

- **Class imbalance:** `toy` represents only 5% of training data — detection of toys in novel scenarios may be less reliable than other classes.
- **Single collection environment:** Training data was captured in one setting. Generalization to different lighting, camera angles, or crib types is untested.
- **No held-out test set:** Validation metrics reflect in-distribution performance only.
- **Hardcoded paths:** `webcam_detect.py` and `safesleep.py` contain absolute paths that must be updated when running on a different machine.
