from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


def load_class_names(data_dir: str | Path) -> list[str]:
    mapping = json.loads((Path(data_dir) / "class_mapping.json").read_text())
    return [name for name, _ in sorted(mapping.items(), key=lambda kv: kv[1])]


def _read_manifest(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


class SnakeDataset(Dataset):
    """Reads a manifest CSV built by scripts/prepare_data.py.

    Each row is one image (or one bounding box within an image, for images
    with multiple boxes). When a box is present the image is cropped to it
    with a configurable margin; otherwise the full image is used.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        transform=None,
        bbox_padding: float = 0.15,
        manifest_path: str | Path | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.bbox_padding = bbox_padding
        path = Path(manifest_path) if manifest_path is not None else self.data_dir / "manifests" / f"{split}.csv"
        self.rows = _read_manifest(path)

    def __len__(self) -> int:
        return len(self.rows)

    def _crop(self, image: Image.Image, row: dict) -> Image.Image:
        if row["x_center"] == "":
            return image
        w_img, h_img = image.size
        xc, yc, w, h = (
            float(row["x_center"]),
            float(row["y_center"]),
            float(row["width"]),
            float(row["height"]),
        )
        w *= 1.0 + self.bbox_padding
        h *= 1.0 + self.bbox_padding
        x1 = max(0.0, xc - w / 2) * w_img
        y1 = max(0.0, yc - h / 2) * h_img
        x2 = min(1.0, xc + w / 2) * w_img
        y2 = min(1.0, yc + h / 2) * h_img
        if x2 - x1 < 2 or y2 - y1 < 2:
            return image
        return image.crop((x1, y1, x2, y2))

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        with Image.open(self.data_dir / row["image_path"]) as im:
            image = im.convert("RGB")
            image = self._crop(image, row)
        label = int(row["class_id"])
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def class_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for row in self.rows:
            cid = int(row["class_id"])
            counts[cid] = counts.get(cid, 0) + 1
        return counts
