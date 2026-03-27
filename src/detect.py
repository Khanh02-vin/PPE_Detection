#!/usr/bin/env python3
"""
Run YOLOv8 detection on images or videos.
Usage:
  python detect.py --weights runs/detect/ppe_yolo/weights/best.pt --source data/images --save
"""

import argparse
from ultralytics import YOLO
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv8 detection")
    parser.add_argument("--weights", type=str, required=True, help="Path to YOLOv8 weights (.pt)")
    parser.add_argument("--source", type=str, required=True, help="Image, folder, video, or webcam source")
    parser.add_argument("--imgsz", type=int, default=416, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold for detections")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum number of detections per image")
    parser.add_argument("--device", type=str, default="", help="Device: '', 'cpu', or '0' (GPU 0)")
    parser.add_argument("--save", action="store_true", help="Save annotated outputs to runs/detect/")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"[INFO] Running YOLOv8 detection")
    print(f"       Weights : {args.weights}")
    print(f"       Source  : {args.source}")
    print(f"       Img Size: {args.imgsz}, Conf: {args.conf}, Max detections: {args.max_det}\n")

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        max_det=args.max_det,
        device=args.device,
        save=args.save,
    )

    if args.save:
        save_path = Path("runs/detect")
        print(f"[OK] Annotated outputs saved to: {save_path.resolve()}")

if __name__ == "__main__":
    main()