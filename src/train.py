#!/usr/bin/env python3
"""
Train YOLOv8 for PPE Detection (simpler version).
1. Load dataset
2. Train model
3. Validate & export results
"""
from pathlib import Path
from ultralytics import YOLO
import os
import yaml

# -------------------------------
# CONFIG
# -------------------------------
MODEL = "yolov8n.pt"                  # pretrained model
DATA_YAML = "data/PPE_Detection/data.yaml"  # path to your dataset yaml
EPOCHS = 5
BATCH_SIZE = 2
IMGSZ = 416
RUN_NAME = "ppe_yolo"
DEVICE = ""  # '' for auto, or 'cpu', '0' for GPU 0

# Output CSV
OUTPUT_CSV = "output/results.csv"

# -------------------------------
# 1️⃣ Check dataset
# -------------------------------
data_path = Path(DATA_YAML)
if not data_path.exists():
    print(f"[ERROR] data.yaml not found at {DATA_YAML}")
    print("Hint: Download dataset from HuggingFace or set the correct path")
    exit(1)

# -------------------------------
# 2️⃣ Train YOLO
# -------------------------------
print("Starting YOLOv8 training...")
model = YOLO(MODEL)

model.train(
    data=str(data_path),
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMGSZ,
    name=RUN_NAME,
    device=DEVICE,
    project="runs",
    cache=False,
)

# Best weights path
best_weights = f"runs/detect/runs/{RUN_NAME}/weights/best.pt"

# -------------------------------
# 3️⃣ Validate & export metrics
# -------------------------------
if Path(best_weights).exists():
    print("Evaluating best model...")
    results = model.val(data=str(data_path), imgsz=IMGSZ)

    # Simple CSV export
    import csv, yaml

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    with open(data_path) as f:
        names = yaml.safe_load(f).get("names", [])
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Precision", "Recall", "mAP50", "mAP50-95"])
        for idx, metrics in results.box.items():
            class_name = names[idx] if idx < len(names) else str(idx)
            writer.writerow([
                class_name,
                round(metrics.get("P", 0), 4),
                round(metrics.get("R", 0), 4),
                round(metrics.get("mAP50", 0), 4),
                round(metrics.get("mAP50-95", 0), 4)
            ])
    print(f"Metrics saved to {OUTPUT_CSV}")
else:
    print(f"[WARN] Best weights not found at {best_weights}")