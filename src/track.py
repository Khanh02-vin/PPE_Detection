#!/usr/bin/env python3
"""
YOLOv8 Tracking Script
Tracks objects in a video or webcam using a trained YOLOv8 model.
"""
import argparse
from ultralytics import YOLO
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv8 tracking")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained YOLOv8 weights (.pt)")
    parser.add_argument("--source", type=str, required=True, help="Video file, folder of images, or webcam index (0)")
    parser.add_argument("--imgsz", type=int, default=416, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold for detections")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum detections per frame")
    parser.add_argument("--device", type=str, default="", help="Device to run on: '' (auto), 'cpu', '0' (GPU 0)")
    parser.add_argument("--save", action="store_true", help="Save annotated video/images to 'runs/track'")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)
    
    print(f"Starting tracking on source: {args.source}")
    result = model.track(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        max_det=args.max_det,
        device=args.device,
        save=args.save
    )
    
    if args.save:
        output_folder = Path("runs/track") / Path(args.source).stem
        print(f"Annotated outputs saved to: {output_folder}")

    print("Tracking complete!")

if __name__ == "__main__":
    main()