#!/usr/bin/env python3
import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for PPE detection (fast)")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--data", type=str, default="archive (1)/data.yaml")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--name", type=str, default="ppe_yolo")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--cache", action="store_true", help="Cache images in RAM for faster training")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        name=args.name,
        device=args.device,
        project="runs",
        cache=args.cache,
        workers=4,
    )


if __name__ == "__main__":
    main()
