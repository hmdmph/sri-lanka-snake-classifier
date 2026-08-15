#!/usr/bin/env python
"""Export a trained checkpoint for on-device (mobile) inference.

Produces, under the checkpoint's directory by default:
  - model.pt        TorchScript (fp32), mobile-optimized
  - model.ptl        Lite-interpreter build of the above, for PyTorch Mobile
                      (Android/iOS via the PyTorch Mobile / ExecuTorch runtime)
  - model_int8.ptl    Dynamically-quantized (int8 Linear layers) lite build,
                      smaller and faster on CPU; classifier head only, since
                      dynamic quantization does not touch conv weights
  - model.onnx        Optional, only if the `onnx` package is installed
                      (`uv sync --extra export`) -- convert this further to
                      TFLite/CoreML with onnx2tf / coremltools as needed
  - labels.json       class_names in output-index order, for the mobile app

Usage:
    .venv/bin/python scripts/export.py --checkpoint outputs/run/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from snake_classifier.model import build_model
from snake_classifier.utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/run/best.pt")
    parser.add_argument("--out-dir", default=None, help="Defaults to the checkpoint's directory")
    return parser.parse_args()


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
    out_dir.mkdir(parents=True, exist_ok=True)

    example = torch.rand(1, 3, image_size, image_size)

    def _mobile_optimize(traced_module):
        # optimize_for_mobile requires an XNNPACK-enabled torch build; the
        # plain pip CPU wheel doesn't have it, so fall back to the
        # unoptimized traced module (still runs fine, just skips the
        # mobile-specific op fusions).
        try:
            return optimize_for_mobile(traced_module)
        except RuntimeError as exc:
            print(f"optimize_for_mobile unavailable ({exc}); exporting without it")
            return traced_module

    traced = torch.jit.trace(model, example)
    mobile_model = _mobile_optimize(traced)
    torch.jit.save(mobile_model, out_dir / "model.pt")
    mobile_model._save_for_lite_interpreter(str(out_dir / "model.ptl"))
    print(f"Saved {out_dir / 'model.pt'} and model.ptl")

    quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    traced_q = torch.jit.trace(quantized, example)
    mobile_q = _mobile_optimize(traced_q)
    mobile_q._save_for_lite_interpreter(str(out_dir / "model_int8.ptl"))
    print(f"Saved {out_dir / 'model_int8.ptl'} (dynamic int8, classifier head only)")

    try:
        import onnx  # noqa: F401

        torch.onnx.export(
            model,
            example,
            str(out_dir / "model.onnx"),
            input_names=["image"],
            output_names=["logits"],
            opset_version=17,
        )
        print(f"Saved {out_dir / 'model.onnx'}")
    except ImportError:
        print("Skipping ONNX export ('onnx' not installed -- `uv sync --extra export` to enable)")

    (out_dir / "labels.json").write_text(json.dumps(class_names, indent=2) + "\n")

    for name in ("model.pt", "model.ptl", "model_int8.ptl"):
        path = out_dir / name
        if path.exists():
            print(f"{name}: {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
