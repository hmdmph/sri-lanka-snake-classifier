#!/usr/bin/env python
"""Bootstrap curation for freshly-downloaded external images (e.g. from
scripts/fetch_inat_images.py): run the current trained model over them and
flag anything it disagrees with for human review, before merging into the
training manifests.

This is a sanity check, not a ground-truth oracle -- the model itself is
only ~59-66% accurate, so plenty of correct images will get flagged too
(especially for classes the model is already weak on). Flagged does not
mean wrong; it means "look at this one before trusting it." Never
auto-deletes anything.

Usage:
    .venv/bin/python scripts/curate_external_images.py --checkpoint outputs/run/best.pt
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from PIL import Image

from snake_classifier.model import build_model
from snake_classifier.transforms import build_eval_transform
from snake_classifier.utils import load_checkpoint

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/run/best.pt")
    parser.add_argument("--external-dir", default="data/external/inaturalist")
    parser.add_argument("--low-conf-threshold", type=float, default=0.15)
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

    external_dir = REPO_ROOT / args.external_dir
    rows = []
    mismatch_count = 0
    low_conf_count = 0

    for class_dir in sorted(external_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        expected_class = class_dir.name
        if expected_class not in class_names:
            print(f"skip {expected_class}: not a known class")
            continue
        for img_path in sorted(class_dir.glob("*.jpg")):
            try:
                with Image.open(img_path) as im:
                    tensor = transform(im.convert("RGB")).unsqueeze(0)
            except Exception as exc:  # noqa: BLE001
                print(f"  unreadable {img_path}: {exc}")
                continue
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)[0]
            top_prob, top_idx = probs.max(dim=0)
            predicted_class = class_names[top_idx.item()]
            expected_prob = probs[class_names.index(expected_class)].item()

            mismatch = predicted_class != expected_class
            low_conf = expected_prob < args.low_conf_threshold
            if mismatch:
                mismatch_count += 1
            if low_conf:
                low_conf_count += 1

            rows.append(
                {
                    "file": str(img_path.relative_to(DATA_DIR)),
                    "expected_class": expected_class,
                    "predicted_class": predicted_class,
                    "predicted_confidence": f"{top_prob.item():.4f}",
                    "expected_class_confidence": f"{expected_prob:.4f}",
                    "flag": "mismatch" if mismatch else ("low_confidence" if low_conf else ""),
                }
            )

    out_path = external_dir / "curation_report.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    flagged = mismatch_count + low_conf_count
    print(f"Scanned {len(rows)} images: {mismatch_count} model-disagrees, "
          f"{low_conf_count} low-confidence-on-expected-class ({flagged} total flagged, "
          f"{flagged / max(len(rows),1)*100:.0f}%)")
    print(f"Report: {out_path} -- sort by 'flag' column, review those first")


if __name__ == "__main__":
    main()
