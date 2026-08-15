#!/usr/bin/env python
"""Train the model that actually ships: one MobileNetV3 trained on 100% of
the dataset (no held-out split), using the epoch count validated by
scripts/cross_validate.py's out-of-fold results.

There's no val set for this run (skip_validation=True -- see
TrainConfig/train.run_training), so it can't early-stop or self-report
accuracy: its expected real-world accuracy is whatever
scripts/cross_validate.py's oof_report.json says, since that's the same
recipe trained the same way, just on one held-out fold at a time.

Usage:
    .venv/bin/python scripts/train_final.py
    .venv/bin/python scripts/train_final.py --epochs 30 --set model_name=mobilenet_v3_large
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from snake_classifier.config import TrainConfig
from snake_classifier.data_prep import scan_dataset, scan_external

from train import apply_overrides, run_training

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
FIELDNAMES = ["image_path", "class_name", "class_id", "x_center", "y_center", "width", "height", "source", "license_code"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="key=value")
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Fixed epoch count. Defaults to cv-report's median_best_epoch.",
    )
    parser.add_argument("--cv-report", default="outputs/cv/oof_report.json")
    parser.add_argument("--output-dir", default="outputs/final")
    parser.add_argument(
        "--no-external", action="store_true",
        help="Skip data/external/inaturalist images, use only the original hand-collected dataset.",
    )
    return parser.parse_args()


def build_all_manifest(include_external: bool = True) -> Path:
    """data/manifests/all.csv: every image in the dataset, no held-out split."""
    scanned = scan_dataset(DATA_DIR)
    class_mapping = {name: idx for idx, name in enumerate(scanned["class_names"])}
    rows = list(scanned["rows"])
    if include_external:
        external_rows = scan_external(DATA_DIR)
        print(f"Including {len(external_rows)} external (iNaturalist) images")
        rows.extend(external_rows)
    for row in rows:
        row["class_id"] = class_mapping[row["class_name"]]

    out_path = DATA_DIR / "manifests" / "all.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} rows, {len(scanned['class_names'])} classes)")
    return out_path


def resolve_epochs(args: argparse.Namespace) -> int:
    if args.epochs is not None:
        return args.epochs
    cv_report_path = Path(args.cv_report)
    if not cv_report_path.exists():
        raise SystemExit(
            f"{cv_report_path} not found -- run scripts/cross_validate.py first, "
            "or pass --epochs explicitly."
        )
    cv_report = json.loads(cv_report_path.read_text())
    epochs = cv_report["median_best_epoch"]
    print(f"Using median_best_epoch={epochs} from {cv_report_path}")
    return epochs


def main() -> None:
    args = parse_args()
    base_config = TrainConfig.from_yaml(args.config)
    base_config = apply_overrides(base_config, args.overrides)

    all_manifest = build_all_manifest(include_external=not args.no_external)
    epochs = resolve_epochs(args)

    final_config = dataclasses.replace(
        base_config,
        train_manifest=str(all_manifest),
        val_manifest=None,
        skip_validation=True,
        epochs=epochs,
        early_stop_patience=epochs,  # unused when skip_validation, but keep consistent
        output_dir=args.output_dir,
    )
    result = run_training(final_config)
    print(f"Final model trained on 100% of the data ({epochs} epochs).")
    print(f"Checkpoint: {Path(result['output_dir']) / 'best.pt'}")
    print("Expected accuracy: see the OOF report from scripts/cross_validate.py, not a self-eval.")


if __name__ == "__main__":
    main()
