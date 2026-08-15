#!/usr/bin/env python
"""Export a checkpoint to TFLite for the Flutter app.

Flutter's mature on-device ML ecosystem (tflite_flutter, live-camera
inference at 15-20fps on mid-range hardware) is built around TFLite/LiteRT,
not PyTorch Mobile -- whose own Lite Interpreter format is already
deprecated (see scripts/export.py). Path: PyTorch -> ONNX -> TensorFlow
SavedModel -> TFLite, via onnx2tf.

Requires the `tflite` extra: `uv sync --extra tflite` (pulls in
tensorflow-cpu + onnx2tf, a heavier install skipped by default).

Usage:
    .venv/bin/python scripts/export_tflite.py --checkpoint outputs/final/best.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from torchvision import transforms as T

from snake_classifier.dataset import SnakeDataset
from snake_classifier.model import build_model
from snake_classifier.transforms import IMAGENET_MEAN, IMAGENET_STD
from snake_classifier.utils import load_checkpoint

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/final/best.pt")
    parser.add_argument("--out-dir", default=None, help="Defaults to the checkpoint's directory")
    parser.add_argument(
        "--calibration-samples", type=int, default=100,
        help="Real training images used to calibrate int8 quantization (better than random data).",
    )
    return parser.parse_args()


def build_calibration_data(data_dir: str, image_size: int, bbox_padding: float, n: int) -> np.ndarray:
    # onnx2tf wants calibration images pre-normalized to [0, 1] only (no
    # mean/std) -- it applies mean/std itself (passed separately below) to
    # simulate the actual normalized range the traced ONNX graph expects,
    # without baking a normalization op into the graph itself.
    raw_transform = T.Compose([T.Resize(int(image_size * 1.14)), T.CenterCrop(image_size), T.ToTensor()])
    ds = SnakeDataset(data_dir, "train", raw_transform, bbox_padding)
    idxs = np.linspace(0, len(ds) - 1, num=min(n, len(ds)), dtype=int)
    # onnx2tf expects NHWC for its calibration generator
    batch = np.stack([ds[i][0].permute(1, 2, 0).numpy() for i in idxs])
    return batch.astype(np.float32)


def main() -> None:
    args = parse_args()
    ckpt = load_checkpoint(args.checkpoint)
    class_names = ckpt["class_names"]
    config = ckpt["config"]
    image_size = config["image_size"]

    model = build_model(config["model_name"], len(class_names), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent
    work_dir = out_dir / "_tflite_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    example = torch.rand(1, 3, image_size, image_size)
    onnx_path = work_dir / "model.onnx"
    torch.onnx.export(
        model, example, str(onnx_path),
        input_names=["image"], output_names=["logits"], opset_version=18,
    )
    print(f"Exported {onnx_path}")

    calibration_data = build_calibration_data(
        config["data_dir"], image_size, config["bbox_padding"], args.calibration_samples
    )
    calib_path = work_dir / "calibration.npy"
    np.save(calib_path, calibration_data)

    import onnx2tf

    # onnx2tf unconditionally fetches a reference test-image .npy from a
    # hardcoded GitHub release URL for an *optional* internal ONNX-vs-TF
    # output sanity check (which we don't request -- all the
    # check_onnx_tf_outputs_* flags are left at their False defaults). As
    # of this run both of its hardcoded URLs (GitHub release, Wasabi
    # fallback) are dead (404 / XML error), so the download always fails.
    # Since we never asked for the check the data feeds, stub the fetch
    # with correctly-shaped random data instead of depending on that
    # broken, unrequested network call.
    onnx2tf.onnx2tf.download_test_image_data = lambda: np.random.rand(20, 128, 128, 3).astype(np.float32)

    mean_4d = [[[list(IMAGENET_MEAN)]]]
    std_4d = [[[list(IMAGENET_STD)]]]
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(work_dir / "saved_model"),
        output_integer_quantized_tflite=True,
        custom_input_op_name_np_data_path=[["image", str(calib_path), mean_4d, std_4d]],
        non_verbose=True,
    )

    saved_model_dir = work_dir / "saved_model"
    tflite_files = sorted(saved_model_dir.glob("*.tflite"))
    print("Produced:", [f.name for f in tflite_files])

    fp32 = next((f for f in tflite_files if f.name == "model_float32.tflite"), None)
    int8 = next((f for f in tflite_files if "full_integer_quant" in f.name or "int8" in f.name), None)

    if fp32:
        shutil.copy(fp32, out_dir / "model.tflite")
        print(f"Saved {out_dir / 'model.tflite'} ({fp32.stat().st_size / 1e6:.2f} MB)")
    if int8:
        shutil.copy(int8, out_dir / "model_int8.tflite")
        print(f"Saved {out_dir / 'model_int8.tflite'} ({int8.stat().st_size / 1e6:.2f} MB)")

    (out_dir / "labels.json").write_text(json.dumps(class_names, indent=2) + "\n")
    shutil.rmtree(work_dir)
    print(f"Saved {out_dir / 'labels.json'}")


if __name__ == "__main__":
    main()
