#!/usr/bin/env python3
import os
import random
from pathlib import Path
from PIL import Image, ImageDraw

DEMO_COUNT = 30
TEST_COUNT = 10
IMAGE_SIZE = 416
CLASSES = ["head", "helmet", "person"]
NC = 3
DATA_ROOT = Path("data_demo")
COLORS = [(255, 100, 100), (100, 200, 100), (100, 100, 255)]


def make_image(idx, split="train"):
    bg = random.choice([(120, 140, 160), (100, 120, 100), (160, 140, 120), (140, 160, 140)])
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), bg)
    draw = ImageDraw.Draw(img)
    lines = []
    for _ in range(random.randint(1, 4)):
        cls = random.randint(0, NC - 1)
        x1 = random.randint(10, IMAGE_SIZE - 100)
        y1 = random.randint(10, IMAGE_SIZE - 100)
        w = random.randint(40, 120)
        h = random.randint(40, 120)
        x2 = min(x1 + w, IMAGE_SIZE - 1)
        y2 = min(y1 + h, IMAGE_SIZE - 1)
        draw.rectangle([x1, y1, x2, y2], outline=COLORS[cls], width=3)
        cx = ((x1 + x2) / 2) / IMAGE_SIZE
        cy = ((y1 + y2) / 2) / IMAGE_SIZE
        bw = (x2 - x1) / IMAGE_SIZE
        bh = (y2 - y1) / IMAGE_SIZE
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    subdir = "train" if split == "train" else "test"
    img.save(DATA_ROOT / subdir / "images" / f"ppe_{idx:04d}.jpg")
    label_path = DATA_ROOT / subdir / "labels" / f"ppe_{idx:04d}.txt"
    label_path.write_text("\n".join(lines) + "\n")


def main():
    for sub in ["train", "test"]:
        os.makedirs(DATA_ROOT / sub / "images", exist_ok=True)
        os.makedirs(DATA_ROOT / sub / "labels", exist_ok=True)
    print(f"[DEMO] Generating {DEMO_COUNT} train images...")
    for i in range(DEMO_COUNT):
        make_image(i, "train")
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{DEMO_COUNT}]")
    print(f"[DEMO] Generating {TEST_COUNT} test images...")
    for i in range(TEST_COUNT):
        make_image(i, "test")
    yaml_lines = [
        f"path: {str(Path.cwd() / DATA_ROOT)}",
        "train: train/images",
        "val: test/images",
        f"nc: {NC}",
        "names:",
    ]
    for i, name in enumerate(CLASSES):
        yaml_lines.append(f"  {i}: {name}")
    (DATA_ROOT / "data.yaml").write_text("\n".join(yaml_lines) + "\n")
    print(f"\n[DEMO] Done!")
    print(f"  train: {DEMO_COUNT} images, test: {TEST_COUNT} images")
    print(f"\nNext: python src/train.py --data 'data/data.yaml' --epochs 10 --batch 4 --imgsz 416")


if __name__ == "__main__":
    main()
