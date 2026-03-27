#!/usr/bin/env python3
"""
Evaluate YOLOv8 PPE model and export key metrics to CSV.
Usage:
  python src/eval.py --weights runs/detect/ppe_yolo/weights/best.pt
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 PPE model")
    parser.add_argument("--weights", required=True, type=str, help="Path to trained YOLOv8 weights (.pt)")
    parser.add_argument("--data", default="data/data.yaml", type=str, help="Path to dataset YAML")
    parser.add_argument("--imgsz", default=416, type=int, help="Inference image size")
    parser.add_argument("--conf", default=0.25, type=float, help="Confidence threshold")
    parser.add_argument("--device", default="", type=str, help="Device: '', 'cpu', or '0' (GPU)")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--output", default="runs/eval/results.csv", type=str, help="CSV output path")
    return parser.parse_args()

def export_metrics_csv(metrics, output_path: str, meta: dict):
    """Save key metrics to CSV."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Extract per-class maps if available
    maps = getattr(metrics.box, "maps", [])
    metrics_dict = {
        "timestamp": datetime.now().isoformat(),
        "split": meta["split"],
        "imgsz": str(meta["imgsz"]),
        "conf": str(meta["conf"]),
        "mAP50": f"{metrics.box.map50:.4f}",
        "mAP50-95": f"{metrics.box.map:.4f}",
        "Precision": f"{metrics.box.mp:.4f}",
        "Recall": f"{metrics.box.mr:.4f}",
        "mAP_head": f"{maps[0]:.4f}" if len(maps) > 0 else "N/A",
        "mAP_helmet": f"{maps[1]:.4f}" if len(maps) > 1 else "N/A",
        "mAP_person": f"{maps[2]:.4f}" if len(maps) > 2 else "N/A",
    }

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["key", "value"])
        writer.writeheader()
        for k, v in metrics_dict.items():
            writer.writerow({"key": k, "value": v})

def main():
    args = parse_args()

    print(f"Evaluating model: {args.weights}")
    print(f"Dataset YAML: {args.data} | Split: {args.split}")
    
    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        split=args.split,
    )

    export_metrics_csv(metrics, args.output, {"split": args.split, "imgsz": args.imgsz, "conf": args.conf})
    print(f"[OK] Metrics exported to: {args.output}")
    print(f"mAP50={metrics.box.map50:.4f} | mAP50-95={metrics.box.map:.4f} | P={metrics.box.mp:.4f} | R={metrics.box.mr:.4f}")

if __name__ == "__main__":
    main()