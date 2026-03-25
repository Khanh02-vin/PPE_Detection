#!/usr/bin/env python3
import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv8 detection (fast)")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--save", action="store_true", help="Save annotated outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        max_det=args.max_det,
        device=args.device,
        save=args.save,
    )


if __name__ == "__main__":
    main()
