from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_weighted_sampler(class_counts: dict[int, int], row_labels: list[int]) -> WeightedRandomSampler:
    """Inverse-frequency sampler so rare classes (as few as 6 images) are
    seen roughly as often as common ones (75 images) over an epoch."""
    class_weight = {cid: 1.0 / count for cid, count in class_counts.items()}
    sample_weights = [class_weight[label] for label in row_labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


class AverageMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def save_checkpoint(path: str | Path, model, class_names: list[str], config: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_names": class_names,
            "config": config,
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def save_json(path: str | Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
