#!/usr/bin/env python
"""Train the snake classifier on CPU.

Usage:
    .venv/bin/python scripts/train.py
    .venv/bin/python scripts/train.py --config configs/train.yaml --set epochs=50 --set batch_size=16

`run_training(config)` below is also imported directly by
scripts/cross_validate.py and scripts/train_final.py, so both the CLI and
those orchestrators share exactly one training loop implementation.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from snake_classifier.config import TrainConfig
from snake_classifier.dataset import SnakeDataset, load_class_names
from snake_classifier.engine import evaluate, train_one_epoch
from snake_classifier.model import (
    build_model,
    head_parameters,
    backbone_parameters,
    set_backbone_trainable,
)
from snake_classifier.transforms import build_eval_transform, build_train_transform
from snake_classifier.utils import build_weighted_sampler, save_checkpoint, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="key=value"
    )
    return parser.parse_args()


def apply_overrides(config: TrainConfig, overrides: list[str]) -> TrainConfig:
    field_types = {f.name: f.type for f in dataclasses.fields(config)}
    updates = {}
    for item in overrides:
        key, _, value = item.partition("=")
        ftype = field_types[key]
        if ftype == "bool":
            updates[key] = value.lower() in ("1", "true", "yes")
        elif ftype == "int":
            updates[key] = int(value)
        elif ftype == "float":
            updates[key] = float(value)
        else:
            updates[key] = value
    return dataclasses.replace(config, **updates)


def run_training(config: TrainConfig) -> dict:
    """Runs one full training job per `config` and returns a summary dict:
    {history, best_metric, best_val_acc, best_epoch, output_dir, class_names}.

    With `config.skip_validation=True` (no val set at all -- used for the
    final full-dataset retrain) there's no early stopping or "best"
    checkpoint selection: it just trains for exactly `config.epochs` and
    saves the final weights to both last.pt and best.pt, so downstream
    tooling (evaluate.py, export.py, predict.py) works unchanged either way.
    """
    set_seed(config.seed)
    device = torch.device(config.device)

    class_names = load_class_names(config.data_dir)
    num_classes = len(class_names)
    print(f"Classes: {num_classes}")

    train_ds = SnakeDataset(
        config.data_dir,
        "train",
        build_train_transform(config.image_size, config.random_erasing_p),
        config.bbox_padding,
        manifest_path=config.train_manifest,
    )
    print(f"Train samples: {len(train_ds)}", end="  ")

    if config.use_weighted_sampler:
        class_counts = train_ds.class_counts()
        row_labels = [int(r["class_id"]) for r in train_ds.rows]
        sampler = build_weighted_sampler(class_counts, row_labels)
        train_loader = DataLoader(
            train_ds, batch_size=config.batch_size, sampler=sampler, num_workers=config.num_workers
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers
        )

    val_loader = None
    if not config.skip_validation:
        val_ds = SnakeDataset(
            config.data_dir,
            "val",
            build_eval_transform(config.image_size),
            config.bbox_padding,
            manifest_path=config.val_manifest,
        )
        print(f"Val samples: {len(val_ds)}")
        val_loader = DataLoader(
            val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
        )
    else:
        print("Val samples: 0 (skip_validation=True)")

    model = build_model(config.model_name, num_classes, config.pretrained).to(device)
    if config.freeze_backbone_epochs > 0:
        set_backbone_trainable(model, False)
        print(f"Backbone frozen for the first {config.freeze_backbone_epochs} epoch(s)")

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = AdamW(
        [
            {"params": backbone_parameters(model), "lr": config.lr},
            {"params": head_parameters(model), "lr": config.head_lr},
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", dataclasses.asdict(config))

    assert config.early_stop_metric in ("val_loss", "val_acc")
    best_metric = float("inf") if config.early_stop_metric == "val_loss" else -float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(config.epochs):
        if epoch == config.freeze_backbone_epochs:
            set_backbone_trainable(model, True)
            print("Backbone unfrozen")

        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            mixup_alpha=config.mixup_alpha,
            desc=f"epoch {epoch + 1} train",
        )

        if val_loader is not None:
            val_loss, val_acc, _, _ = evaluate(
                model, val_loader, criterion, device, desc=f"epoch {epoch + 1} val"
            )
        else:
            val_loss, val_acc = float("nan"), float("nan")
        scheduler.step()
        elapsed = time.time() - start

        print(
            f"epoch {epoch + 1}/{config.epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
            f"({elapsed:.1f}s)"
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        save_json(output_dir / "history.json", history)

        save_checkpoint(output_dir / "last.pt", model, class_names, dataclasses.asdict(config))

        if val_loader is None:
            # No validation signal to select on -- keep training to the
            # fixed epoch count and treat the final epoch as "best".
            continue

        best_val_acc = max(best_val_acc, val_acc)
        current_metric = val_loss if config.early_stop_metric == "val_loss" else val_acc
        improved = current_metric < best_metric if config.early_stop_metric == "val_loss" else current_metric > best_metric

        if improved:
            best_metric = current_metric
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            save_checkpoint(output_dir / "best.pt", model, class_names, dataclasses.asdict(config))
            print(f"  new best {config.early_stop_metric}={best_metric:.4f}, checkpoint saved")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stop_patience:
                print(f"Early stopping: no {config.early_stop_metric} improvement in {config.early_stop_patience} epochs")
                break

    if config.skip_validation:
        # No val-based "best" checkpoint exists -- the final epoch's
        # weights are the only ones, so mirror them to best.pt too.
        save_checkpoint(output_dir / "best.pt", model, class_names, dataclasses.asdict(config))
        best_epoch = len(history)
        print(f"Done (no validation). Trained {len(history)} epochs. Checkpoints in {output_dir}")
    else:
        print(
            f"Done. Best {config.early_stop_metric}={best_metric:.4f} at epoch {best_epoch} "
            f"(best val_acc seen={best_val_acc:.4f}). Checkpoints in {output_dir}"
        )

    return {
        "history": history,
        "best_metric": best_metric,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "output_dir": str(output_dir),
        "class_names": class_names,
    }


def main() -> None:
    args = parse_args()
    config = TrainConfig.from_yaml(args.config)
    config = apply_overrides(config, args.overrides)
    run_training(config)


if __name__ == "__main__":
    main()
