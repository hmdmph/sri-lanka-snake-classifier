#!/usr/bin/env python
"""Run inference on a single image.

Usage:
    .venv/bin/python scripts/predict.py --checkpoint outputs/run/best.pt path/to/image.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from PIL import Image

from snake_classifier.model import build_model
from snake_classifier.transforms import build_eval_transform
from snake_classifier.utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to an image file")
    parser.add_argument("--checkpoint", default="outputs/run/best.pt")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = load_checkpoint(args.checkpoint)
    class_names = ckpt["class_names"]
    config = ckpt["config"]

    model = build_model(config["model_name"], len(class_names), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    transform = build_eval_transform(config["image_size"])
    with Image.open(args.image) as im:
        tensor = transform(im.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    top_k = min(args.top_k, len(class_names))
    values, indices = probs.topk(top_k)
    for rank, (value, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        print(f"{rank}. {class_names[idx]:<30s} {value * 100:5.1f}%")


if __name__ == "__main__":
    main()
