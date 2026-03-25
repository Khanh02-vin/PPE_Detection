# YOLOv8 PPE Detection

A lightweight project for training, detecting, and tracking Personal Protective Equipment (PPE) using YOLOv8 on custom datasets.

## Overview

This project provides an end-to-end pipeline for PPE object detection, including model training, inference, and multi-object tracking — all built on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).

## Features

- **Training** — Fine-tune YOLOv8 models on custom PPE datasets
- **Detection** — Run inference with adjustable confidence and image size
- **Tracking** — Multi-object tracking across video frames
- **Metrics** — Evaluation metrics including mAP, Precision, and Recall

## Dataset

> **Dataset:** [Khanh510/PPE_Detection](https://huggingface.co/datasets/Khanh510/PPE_Detection/tree/main) on Hugging Face

The dataset should be organized as follows:

```
data/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Installation

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

### Training

```powershell
python src/train.py --data "data/data.yaml" --epochs 5 --batch 8 --imgsz 416 --model yolov8n.pt
```

### Detection

```powershell
python src/detect.py --weights "output/detect/runs/ppe_yolo4/weights/best.pt" --source "data/test/images" --imgsz 416 --conf 0.4 --max-det 50
```

### Tracking

```powershell
python src/track.py --weights "output/detect/runs/ppe_yolo4/weights/best.pt" --source "data/test/images" --imgsz 416 --conf 0.4 --max-det 50 --save
```

## Evaluation Metrics

| Metric        | Description                                             |
|---------------|---------------------------------------------------------|
| mAP@0.5       | Mean Average Precision at IoU threshold 0.5             |
| mAP@0.5:0.95  | Mean Average Precision across IoU thresholds             |
| Precision     | Ratio of true positives to all positive predictions     |
| Recall        | Ratio of true positives to all actual objects          |

Results are saved in `output/detect/runs/ppe_yolo4/results.csv`.

## Project Structure

```
PPE_Detection/
├── data/               # Dataset directory
│   └── data.yaml       # Dataset configuration
├── src/                # Source scripts
│   ├── train.py        # Training script
│   ├── detect.py       # Detection script
│   └── track.py        # Tracking script
├── output/             # Model outputs and results
└── README.md
```

## Requirements

See `requirements.txt` for all dependencies.
