#!/usr/bin/env python
"""Build stratified 5-fold cross-validation manifests over the *entire*
dataset (not just the 1044-image train split) so every image eventually
gets evaluated by a model that never trained on it.

The fixed train/val/test split in data/manifests/{train,val,test}.csv
(built by prepare_data.py) stays untouched and is still what a plain
`scripts/train.py` run uses by default. This script writes a *parallel* set
of manifests under data/manifests/cv/fold{0..4}/{train,val}.csv for
scripts/cross_validate.py to use instead.

Fold assignment is done at the *image* level (grouping every bounding-box
row of a multi-box image together) so a single photo's crops can never
straddle the train/val boundary of a fold -- that would leak.

Usage:
    .venv/bin/python scripts/make_cv_folds.py
    .venv/bin/python scripts/make_cv_folds.py --n-splits 5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sklearn.model_selection import StratifiedKFold

from snake_classifier.data_prep import scan_dataset, scan_external

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
FIELDNAMES = ["image_path", "class_name", "class_id", "x_center", "y_center", "width", "height", "source", "license_code"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-external", action="store_true",
        help="Skip data/external/inaturalist images, use only the original hand-collected dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scanned = scan_dataset(DATA_DIR)
    class_names = scanned["class_names"]
    class_mapping = {name: idx for idx, name in enumerate(class_names)}

    all_rows = list(scanned["rows"])
    if not args.no_external:
        external_rows = scan_external(DATA_DIR)
        print(f"Including {len(external_rows)} external (iNaturalist) images")
        all_rows.extend(external_rows)

    rows_by_image: dict[str, list[dict]] = defaultdict(list)
    image_class: dict[str, str] = {}
    for row in all_rows:
        row["class_id"] = class_mapping[row["class_name"]]
        rows_by_image[row["image_path"]].append(row)
        image_class[row["image_path"]] = row["class_name"]

    images = sorted(image_class)  # sorted for determinism across runs
    labels = [image_class[img] for img in images]

    min_class_count = min(Counter(labels).values())
    if min_class_count < args.n_splits:
        raise SystemExit(
            f"Class with only {min_class_count} images can't support "
            f"{args.n_splits}-fold stratification (need >= {args.n_splits} "
            "images/class). Lower --n-splits or fix the data."
        )

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    cv_dir = DATA_DIR / "manifests" / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)

    summary = {"n_splits": args.n_splits, "folds": []}
    warnings: list[str] = []

    for fold_idx, (train_pos, val_pos) in enumerate(skf.split(images, labels)):
        train_images = {images[i] for i in train_pos}
        val_images = {images[i] for i in val_pos}

        train_rows = [r for img in train_images for r in rows_by_image[img]]
        val_rows = [r for img in val_images for r in rows_by_image[img]]

        fold_dir = cv_dir / f"fold{fold_idx}"
        fold_dir.mkdir(exist_ok=True)
        for name, rows in (("train", train_rows), ("val", val_rows)):
            with (fold_dir / f"{name}.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)

        train_class_counts = Counter(image_class[img] for img in train_images)
        val_class_counts = Counter(image_class[img] for img in val_images)
        missing_in_train = [c for c in class_names if train_class_counts.get(c, 0) == 0]
        if missing_in_train:
            warnings.append(f"fold{fold_idx}: classes with 0 train images: {missing_in_train}")

        summary["folds"].append(
            {
                "fold": fold_idx,
                "train_images": len(train_images),
                "val_images": len(val_images),
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "train_class_counts": dict(sorted(train_class_counts.items())),
                "val_class_counts": dict(sorted(val_class_counts.items())),
            }
        )
        print(
            f"fold {fold_idx}: {len(train_images)} train images "
            f"({len(train_rows)} rows), {len(val_images)} val images "
            f"({len(val_rows)} rows)"
        )

    summary["warnings"] = warnings
    (cv_dir / "folds_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  {w}")
    else:
        print("All folds have >=1 train example of every class.")
    print(f"Wrote fold manifests to {cv_dir}")


if __name__ == "__main__":
    main()
