"""Shared dataset-scanning logic used by scripts/prepare_data.py and
scripts/make_cv_folds.py, so both build manifest rows the exact same way:
species trusted from the filename, bounding boxes kept for crop geometry
only, corrupt images dropped, multi-box images expanded into one row per
box, and label-file/filename class disagreements logged rather than trusted.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")


def class_name_from_filename(stem: str) -> str:
    return stem.rsplit("_", 1)[0]


def read_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        cid, xc, yc, w, h = line.split()
        boxes.append((int(cid), float(xc), float(yc), float(w), float(h)))
    return boxes


def scan_dataset(data_dir: str | Path) -> dict:
    """Walks data/images/{train,val,test} + data/labels/{train,val,test}
    and returns every classification row across the whole dataset, plus
    bookkeeping (mismatches, corrupt images, per-image-source split) needed
    by callers that want to report on or re-partition the data.

    Mismatch detection is self-consistent rather than depending on
    class_mapping.json (which this scan's own caller may rewrite): for each
    filename-derived class, the *majority* embedded box class id across all
    its label files is taken as that class's "expected" id, and any row
    whose embedded id disagrees with its own class's majority is flagged.
    This makes the scan idempotent -- re-running it after class_mapping.json
    has already been regenerated (with different, contiguous ids) still
    finds the same real annotation errors instead of flagging everything.

    Returns a dict with:
      rows: list of {image_path, class_name, x_center, y_center, width,
            height} (bbox fields are "" when the image had no label box)
      class_names: sorted list of every class found in filenames
      mismatches: label-file class id vs. filename class id disagreements
      corrupt_images: images that failed to open, skipped
      empty_label_count, multi_box_count: same bookkeeping prepare_data.py
            already reports
    """
    data_dir = Path(data_dir)

    rows: list[dict] = []
    class_names: set[str] = set()
    corrupt_images: list[str] = []
    empty_label_count = 0
    multi_box_count = 0
    # box_records holds (label_path, class_name, embedded_cid, xc, yc, w, h)
    # for a second pass once every class's majority embedded id is known.
    box_records: list[tuple[Path, str, int, float, float, float, float]] = []
    embedded_id_votes: dict[str, Counter] = defaultdict(Counter)

    for split in SPLITS:
        image_dir = data_dir / "images" / split
        label_dir = data_dir / "labels" / split
        if not image_dir.exists():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            stem = image_path.stem
            class_name = class_name_from_filename(stem)
            class_names.add(class_name)

            try:
                with Image.open(image_path) as im:
                    im.verify()
            except Exception as exc:  # noqa: BLE001 - report and skip
                corrupt_images.append(f"{image_path.relative_to(data_dir)}: {exc}")
                continue

            label_path = label_dir / f"{stem}.txt"
            boxes = read_boxes(label_path)

            if not boxes:
                empty_label_count += 1
                rows.append(
                    {
                        "image_path": str(image_path.relative_to(data_dir)),
                        "class_name": class_name,
                        "x_center": "",
                        "y_center": "",
                        "width": "",
                        "height": "",
                        "source": "yolo",
                        "license_code": "",
                    }
                )
                continue

            if len(boxes) > 1:
                multi_box_count += 1

            for cid, xc, yc, w, h in boxes:
                embedded_id_votes[class_name][cid] += 1
                box_records.append((label_path, class_name, cid, xc, yc, w, h))
                rows.append(
                    {
                        "image_path": str(image_path.relative_to(data_dir)),
                        "class_name": class_name,
                        "x_center": xc,
                        "y_center": yc,
                        "width": w,
                        "height": h,
                        "source": "yolo",
                        "license_code": "",
                    }
                )

    majority_id = {name: votes.most_common(1)[0][0] for name, votes in embedded_id_votes.items()}
    mismatches = [
        {
            "label_file": str(label_path.relative_to(data_dir)),
            "filename_class": class_name,
            "embedded_class_id": cid,
            "expected_class_id": majority_id[class_name],
        }
        for label_path, class_name, cid, _, _, _, _ in box_records
        if cid != majority_id[class_name]
    ]

    return {
        "rows": rows,
        "class_names": sorted(class_names),
        "mismatches": mismatches,
        "corrupt_images": corrupt_images,
        "empty_label_count": empty_label_count,
        "multi_box_count": multi_box_count,
    }


def scan_external(data_dir: str | Path) -> list[dict]:
    """Rows for images fetched by scripts/fetch_inat_images.py, from its
    attribution.csv (class + license + source URL already recorded there --
    no need to re-derive from directory layout). Full-image crop, no bbox,
    same as the fallback for YOLO images with no label box.

    Only classes present in data/species_mapping_draft.json at "high"
    confidence get fetched by that script in the first place, but this
    function doesn't re-check confidence -- it trusts whatever's already on
    disk under data/external/inaturalist/, since that's the point where a
    human is expected to have reviewed scripts/fetch_inat_images.py's output
    (see its module docstring) before this ever gets called.
    """
    import csv

    data_dir = Path(data_dir)
    attribution_path = data_dir / "external" / "inaturalist" / "attribution.csv"
    if not attribution_path.exists():
        return []

    with attribution_path.open(newline="") as f:
        attribution_rows = list(csv.DictReader(f))

    rows = []
    for a in attribution_rows:
        image_path = data_dir / a["file"]
        if not image_path.exists():
            continue
        try:
            with Image.open(image_path) as im:
                im.verify()
        except Exception:  # noqa: BLE001
            continue
        rows.append(
            {
                "image_path": a["file"],
                "class_name": a["class_name"],
                "x_center": "",
                "y_center": "",
                "width": "",
                "height": "",
                "source": "inaturalist",
                "license_code": a["license_code"],
            }
        )
    return rows
