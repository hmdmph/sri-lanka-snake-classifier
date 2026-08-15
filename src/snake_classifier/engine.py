from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from .utils import AverageMeter


def _mixup(images: torch.Tensor, labels: torch.Tensor, alpha: float):
    """Standard mixup (Zhang et al. 2018). Returns mixed images plus both
    label sets and the mixing weight, so the caller can blend the loss."""
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[perm]
    return mixed, labels, labels[perm], lam


def train_one_epoch(
    model, loader, optimizer, criterion, device, mixup_alpha: float = 0.0, desc: str = "train"
) -> tuple[float, float]:
    model.train()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images, labels = images.to(device), labels.to(device)
        mixed_images, labels_a, labels_b, lam = _mixup(images, labels, mixup_alpha)

        optimizer.zero_grad()
        outputs = model(mixed_images)
        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        loss.backward()
        optimizer.step()

        # Accuracy against the dominant label only -- an approximation
        # under mixup, good enough as a training-progress signal.
        preds = outputs.argmax(dim=1)
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update((preds == labels_a).float().mean().item(), images.size(0))
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc: str = "eval"):
    model.eval()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    all_preds: list[int] = []
    all_targets: list[int] = []
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update((preds == labels).float().mean().item(), images.size(0))
        all_preds.extend(preds.tolist())
        all_targets.extend(labels.tolist())
    return loss_meter.avg, acc_meter.avg, all_preds, all_targets
