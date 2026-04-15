# SafeSleep AI

**AI-powered infant sleep safety detection using computer vision and deep learning.**

SafeSleep AI monitors infant sleep in real time, classifying sleep positions and flagging potential hazards in the crib. Built on YOLOv8, it's designed for integration with baby monitor hardware and SaaS platforms to give parents and caregivers peace of mind.

---

## Features

- **Sleep Position Classification** — Detects back, stomach, and side sleeping positions to alert caregivers when an infant shifts to an unsafe orientation.
- **Hazard Detection** — Identifies loose bedding, toys, pillows, and other objects that pose suffocation or entrapment risks.
- **Real-Time Processing** — Optimized inference pipeline delivers immediate feedback, suitable for live video feeds.
- **Cloud Integration** — Supports AWS S3 for image storage and retrieval, enabling scalable data pipelines.
- **SaaS-Ready API** — Designed as an API-first product for seamless integration with baby monitor companies and smart nursery platforms.

---

## Model Details

| Component | Detail |
|---|---|
| Architecture | YOLOv8 (Ultralytics) |
| Framework | PyTorch |
| Training Data | 1,000+ images with augmentation for improved robustness |
| Annotation Format | YOLO format |
| Task | Object detection + classification |

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLOv8

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/safesleep-ai.git
cd safesleep-ai

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO("path/to/safesleep_model.pt")

# Run inference on an image
results = model.predict(source="path/to/crib_image.jpg", conf=0.5)

# Display results
results[0].show()
```

### Running Inference on a Video Feed

```python
# Real-time detection from a webcam or monitor feed
results = model.predict(source=0, stream=True, conf=0.5)

for result in results:
    annotated_frame = result.plot()
    # Process or display annotated_frame as needed
```

---

## Project Structure

```
safesleep-ai/
├── data/               # Training and validation datasets
├── models/             # Trained model weights
├── notebooks/          # Exploration and training notebooks
├── src/                # Source code and utilities
│   ├── train.py        # Model training script
│   ├── predict.py      # Inference script
│   └── utils.py        # Helper functions
├── api/                # API server for SaaS integration
├── requirements.txt
└── README.md
```

---

## Roadmap

- [ ] Expand training dataset beyond 1,000 images
- [ ] Add audio-based cry detection as a complementary signal
- [ ] Build REST API with FastAPI for production deployment
- [ ] Add support for edge inference (Raspberry Pi, NVIDIA Jetson)
- [ ] Develop a parent-facing mobile dashboard

---

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Disclaimer

SafeSleep AI is a supplementary monitoring tool and is **not a substitute for direct adult supervision**. Always follow the safe sleep guidelines recommended by the American Academy of Pediatrics (AAP).
