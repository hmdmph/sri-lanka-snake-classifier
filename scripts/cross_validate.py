#!/usr/bin/env python
"""Stratified 5-fold cross-validation over the entire dataset.

Why: the fixed train/val/test split only leaves a 68-image test set across
43 classes -- many classes have 0-2 test images, so its per-class metrics
are mostly noise. This trains one model per fold (each held-out fold is
never seen during that fold's training) and aggregates every fold's
held-out predictions into one report covering all ~1307 images, each
scored exactly once by a model that never trained on it -- a trustworthy
accuracy estimate that also uses 100% of the data instead of ~80%.

Usage:
    .venv/bin/python scripts/cross_validate.py
    .venv/bin/python scripts/cross_validate.py --config configs/train.yaml --set model_name=mobilenet_v3_large
"""

from __future__ import annotations

import argparse
import dataclasses
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from snake_classifier.config import TrainConfig
from snake_classifier.dataset import SnakeDataset, load_class_names
from snake_classifier.model import build_model
from snake_classifier.transforms import build_eval_transform
from snake_classifier.utils import save_json

from train import apply_overrides, run_training
from evaluate import evaluate_with_tta

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="key=value")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--cv-output-dir", default="outputs/cv")
    return parser.parse_args()


def ensure_folds(n_folds: int) -> None:
    fold0_train = DATA_DIR / "manifests" / "cv" / "fold0" / "train.csv"
    if fold0_train.exists():
        return
    print("Fold manifests not found, generating them first...")
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "make_cv_folds.py"), "--n-splits", str(n_folds)],
        check=True,
    )


def main() -> None:
    args = parse_args()
    base_config = TrainConfig.from_yaml(args.config)
    base_config = apply_overrides(base_config, args.overrides)

    ensure_folds(args.n_folds)

    class_names = load_class_names(base_config.data_dir)
    device = torch.device(base_config.device)

    cv_output_dir = Path(args.cv_output_dir)
    cv_output_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []
    all_preds: list[int] = []
    all_targets: list[int] = []

    for k in range(args.n_folds):
        print(f"\n{'=' * 20} Fold {k}/{args.n_folds - 1} {'=' * 20}")
        fold_manifests = DATA_DIR / "manifests" / "cv" / f"fold{k}"
        fold_config = dataclasses.replace(
            base_config,
            train_manifest=str(fold_manifests / "train.csv"),
            val_manifest=str(fold_manifests / "val.csv"),
            output_dir=str(cv_output_dir / f"fold{k}"),
        )
        result = run_training(fold_config)
        fold_results.append(
            {
                "fold": k,
                "best_epoch": result["best_epoch"],
                "best_metric": result["best_metric"],
                "best_val_acc": result["best_val_acc"],
            }
        )

        # Out-of-fold inference: this fold's held-out val images, scored by
        # the model that never trained on them.
        model = build_model(fold_config.model_name, len(class_names), pretrained=False)
        ckpt = torch.load(cv_output_dir / f"fold{k}" / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)

        val_ds = SnakeDataset(
            fold_config.data_dir,
            "val",
            build_eval_transform(fold_config.image_size),
            fold_config.bbox_padding,
            manifest_path=fold_config.val_manifest,
        )
        val_loader = DataLoader(val_ds, batch_size=fold_config.batch_size, shuffle=False)
        _, fold_acc, preds, targets = evaluate_with_tta(model, val_loader, device, desc=f"fold{k}-oof")
        print(f"Fold {k} out-of-fold accuracy: {fold_acc:.4f}")
        all_preds.extend(preds)
        all_targets.extend(targets)

    # One report over all ~1307 images, each scored by a model that never
    # trained on it.
    present_labels = sorted(set(all_targets) | set(all_preds))
    present_names = [class_names[i] for i in present_labels]
    report = classification_report(
        all_targets, all_preds, labels=present_labels, target_names=present_names,
        zero_division=0, output_dict=True,
    )
    print(f"\n{'=' * 20} Out-of-fold report (all data) {'=' * 20}")
    print(
        classification_report(
            all_targets, all_preds, labels=present_labels, target_names=present_names, zero_division=0
        )
    )
    cm = confusion_matrix(all_targets, all_preds, labels=present_labels).tolist()

    best_epochs = [r["best_epoch"] for r in fold_results]
    median_epoch = sorted(best_epochs)[len(best_epochs) // 2]

    summary = {
        "n_folds": args.n_folds,
        "fold_results": fold_results,
        "median_best_epoch": median_epoch,
        "oof_accuracy": sum(1 for p, t in zip(all_preds, all_targets) if p == t) / len(all_targets),
        "oof_classification_report": report,
        "oof_confusion_matrix": {"labels": present_names, "matrix": cm},
    }
    save_json(cv_output_dir / "oof_report.json", summary)
    print(f"\nOOF accuracy: {summary['oof_accuracy']:.4f}")
    print(f"Median best epoch across folds: {median_epoch} (use this for the final full-data retrain)")
    print(f"Saved {cv_output_dir / 'oof_report.json'}")


if __name__ == "__main__":
    main()
