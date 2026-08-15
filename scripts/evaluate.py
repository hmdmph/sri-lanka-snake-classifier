#!/usr/bin/env python
"""Evaluate a checkpoint on the held-out test split.

Usage:
    .venv/bin/python scripts/evaluate.py --checkpoint outputs/run/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from snake_classifier.dataset import SnakeDataset
from snake_classifier.engine import evaluate
from snake_classifier.model import build_model
from snake_classifier.transforms import build_eval_transform
from snake_classifier.utils import AverageMeter, load_checkpoint, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/run/best.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--tta", action="store_true", help="Average predictions with a horizontal-flip view"
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate_with_tta(model, loader, device, desc="eval"):
    """Same as engine.evaluate, but averages softmax probabilities over the
    original image and its horizontal flip -- a cheap, free-lunch accuracy
    bump that needs no retraining."""
    model.eval()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    all_preds, all_targets = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        probs = torch.softmax(model(images), dim=1)
        probs = probs + torch.softmax(model(images.flip(dims=[3])), dim=1)
        probs = probs / 2
        loss = torch.nn.functional.nll_loss(torch.log(probs.clamp_min(1e-8)), labels)
        preds = probs.argmax(dim=1)
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update((preds == labels).float().mean().item(), images.size(0))
        all_preds.extend(preds.tolist())
        all_targets.extend(labels.tolist())
    return loss_meter.avg, acc_meter.avg, all_preds, all_targets


def main() -> None:
    args = parse_args()
    ckpt = load_checkpoint(args.checkpoint)
    class_names = ckpt["class_names"]
    config = ckpt["config"]
    device = torch.device("cpu")

    model = build_model(config["model_name"], len(class_names), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    ds = SnakeDataset(
        args.data_dir, args.split, build_eval_transform(config["image_size"]), config["bbox_padding"]
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    print(f"{args.split} samples: {len(ds)}")

    if args.tta:
        loss, acc, preds, targets = evaluate_with_tta(model, loader, device, desc=args.split)
    else:
        criterion = nn.CrossEntropyLoss()
        loss, acc, preds, targets = evaluate(model, loader, criterion, device, desc=args.split)
    print(f"{args.split}_loss={loss:.4f} {args.split}_acc={acc:.4f}" + (" (TTA)" if args.tta else ""))

    present_labels = sorted(set(targets) | set(preds))
    present_names = [class_names[i] for i in present_labels]
    report = classification_report(
        targets, preds, labels=present_labels, target_names=present_names,
        zero_division=0, output_dict=True,
    )
    print(
        classification_report(
            targets, preds, labels=present_labels, target_names=present_names, zero_division=0
        )
    )

    cm = confusion_matrix(targets, preds, labels=present_labels).tolist()

    out_dir = Path(args.checkpoint).parent
    save_json(out_dir / f"{args.split}_report.json", {
        "loss": loss,
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": {"labels": present_names, "matrix": cm},
    })
    print(f"Saved report to {out_dir / f'{args.split}_report.json'}")


if __name__ == "__main__":
    main()
