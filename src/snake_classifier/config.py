from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    # Data
    data_dir: str = "data"
    image_size: int = 224
    bbox_padding: float = 0.15  # fraction of box size added as margin when cropping
    # Overrides for data/manifests/{train,val}.csv, e.g. for CV folds or a
    # full-dataset run. None = the standard fixed split.
    train_manifest: str | None = None
    val_manifest: str | None = None
    skip_validation: bool = False  # no val set at all: train for exactly `epochs`, no early stopping

    # Model
    model_name: str = "mobilenet_v3_small"  # or "mobilenet_v3_large"
    pretrained: bool = True
    freeze_backbone_epochs: int = 5  # train only the head for this many epochs first

    # Optimization
    epochs: int = 50
    batch_size: int = 32
    lr: float = 5e-5  # backbone LR once unfrozen -- kept low so pretrained features aren't wrecked by ~1k images
    head_lr: float = 7e-4
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    early_stop_patience: int = 12
    early_stop_metric: str = "val_loss"  # val_acc is noisy with only ~5 images/class in val

    # Regularization (small-dataset transfer learning)
    mixup_alpha: float = 0.1  # 0 disables mixup
    random_erasing_p: float = 0.15

    # Sampling
    use_weighted_sampler: bool = True

    # Runtime
    num_workers: int = 4
    seed: int = 42
    device: str = "cpu"

    # Output
    output_dir: str = "outputs/run"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**raw)
