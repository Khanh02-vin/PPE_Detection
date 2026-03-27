# YOLOv8 PPE Detection

End-to-end pipeline for training, evaluating, and tracking Personal Protective Equipment (PPE) using YOLOv8 on a custom dataset.

## Overview

- **Training** — Fine-tune YOLOv8 on PPE dataset (3 classes: helmet, head, vest)
- **Evaluation** — Export metrics (mAP, Precision, Recall) to CSV automatically
- **Detection** — Run inference on images/video
- **Tracking** — Multi-object tracking across frames

## Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | `helmet` | Safety helmet / hard hat — PPE compliant |
| 1 | `head` | Exposed human head (no helmet) — PPE violation |
| 2 | `vest` | Safety reflective vest — PPE compliant |

## Dataset

> **Source:** https://huggingface.co/datasets/jhboyo/ppe-dataset
>
> **Stats:** 15,500 images | 60,991 objects
> - Train: 9,999 images
> - Val: 2,750 images
> - Test: 2,751 images

## Setup

```powershell
# 1. Clone repo
git clone https://github.com/Khanh02-vin/PPE_Detection.git
cd PPE_Detection

# 2. Clone dataset (git LFS — ~1.7 GB)
git clone https://huggingface.co/datasets/jhboyo/ppe-dataset

# 3. Create venv (Python 3.9+)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 4. Install ultralytics
pip install ultralytics
```

## Reproduce Results

### 1. Train

```powershell
python src/train.py
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `yolov8n.pt` | Pretrained YOLOv8 model (n/s/m/l/x) |
| `--data` | `data/PPE_Detection/data.yaml` | Path to dataset config |
| `--epochs` | `5` | Number of epochs |
| `--batch` | `2` | Batch size (reduce if OOM) |
| `--imgsz` | `416` | Image size |
| `--device` | `""` | `""`=auto, `cpu`, `0`=GPU0 |

### 2. Evaluate (standalone)

```powershell
python src/eval.py --weights "runs/ppe_yolo/weights/best.pt" --data "data/PPE_Detection/data.yaml"
```

### 3. Detect

```powershell
python src/detect.py --weights "runs/ppe_yolo/weights/best.pt" --source "data/test/images" --conf 0.4 --save
```

### 4. Track

```powershell
python src/track.py --weights "runs/ppe_yolo/weights/best.pt" --source "data/test/images" --conf 0.4 --save
```

## Output Files

```
PPE_Detection/
├── data/
│   └── PPE_Detection/
│       └── data.yaml            # Dataset config (3 classes)
├── src/
│   ├── train.py                 # Training script
│   ├── eval.py                  # Standalone evaluation + CSV export
│   ├── detect.py                # Detection script
│   └── track.py                 # Tracking script
├── runs/                        # Training outputs
│   └── ppe_yolo4/               # Latest run (epoch 5)
│       ├── weights/
│       │   ├── best.pt          # Best model checkpoint
│       │   └── last.pt          # Last epoch checkpoint
│       └── results.csv          # Training metrics
├── output/
│   └── results.csv               # Evaluation metrics
└── README.md
```

## Training Results

**Config:** YOLOv8 nano, 5 epochs, batch=2, imgsz=416, CPU

### Validation (2,750 images)

| Class | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|-----------|--------|---------|--------------|
| **helmet** | 6,793 | 0.795 | 0.822 | 0.838 | 0.527 |
| **head** | 1,144 | 0.795 | 0.693 | 0.762 | 0.423 |
| **vest** | 2,737 | 0.802 | 0.747 | 0.838 | 0.589 |
| **all** | 10,674 | **0.797** | **0.754** | **0.812** | **0.513** |

**Inference speed:** 67.9 ms/image (CPU)

### Notes

- `head` (no helmet) has lower recall — class imbalance (only 1,144 vs 6,793 helmets)
- Increase epochs to 20–50 for better convergence
- For GPU training, use `--device 0`
- Reduce `--batch` if encountering OOM errors
