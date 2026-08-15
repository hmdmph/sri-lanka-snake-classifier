# sri lankan snake clissifier

CPU-trained, mobile-friendly image classifier for Sri Lankan snake species —
43 classes, MobileNetV3, exported to both TorchScript and TFLite for
on-device (offline) inference. Trained from a small (~1,300 image)
hand-collected dataset plus a licensed iNaturalist expansion; see
`data/species_mapping_draft.json` for the (partially unverified — see its
`_readme`) Sinhala-name-to-scientific-name mapping and `data/dataset_info.json`
for per-class counts.

This repo ships a **pretrained checkpoint** (`outputs/final/`, 75.55%
out-of-fold accuracy — see [Cross-validation](#cross-validation-the-trustworthy-accuracy-number))
so you can run inference immediately without training anything yourself.
The raw training photos are **not** included (see
[Bring your own dataset](#bring-your-own-dataset) if you want to retrain or
extend it).

## Setup

The system Python (3.14) has no PyTorch wheels yet, so this project uses
[`uv`](https://docs.astral.sh/uv/) to manage a compatible interpreter and a
CPU-only PyTorch build (no CUDA download):

```bash
uv venv --python 3.12
uv sync
```

This creates `.venv/`. Every command below assumes `.venv/bin/python`
(or `source .venv/bin/activate` first).

Once `.venv/` exists, `make help` is the single entry point for every
command in this project (data prep, local training, AWS GPU training,
eval, export, cost reporting -- see the [AWS GPU training](#aws-gpu-training)
section below for that last one).

## Quickstart: run the pretrained model

No dataset needed for this — `outputs/final/` is committed to the repo:

```bash
.venv/bin/python scripts/predict.py --checkpoint outputs/final/best.pt path/to/photo.jpg
# or: make predict IMG=path/to/photo.jpg   (CHECKPOINT defaults to outputs/final/best.pt)
```

`outputs/final/` also has ready-to-use mobile exports: `model.pt` /
`model.ptl` / `model_int8.ptl` (PyTorch/TorchScript) and `model.tflite` /
`model_int8.tflite` (TFLite, for Android/iOS via `tflite_flutter` or
LiteRT). `labels.json` in the same directory gives the class-index order
both formats use.

## Bring your own dataset

The training photos themselves aren't in this repo (they're a private,
hand-collected + partially-licensed set — see `data/external/inaturalist/attribution.csv`
for the licensing on the non-hand-collected portion). To train or fine-tune
on your own data, lay it out the same way `scripts/prepare_data.py` expects
— a YOLO-style export (image + optional bounding-box label file), with the
species slug encoded in every filename:

```
data/
  images/{train,val,test}/<class-slug>_<index>.<jpg|jpeg|png>
  labels/{train,val,test}/<class-slug>_<index>.txt   # YOLO bbox format, optional
```

(Swap in wherever your images actually live — a local folder, an S3
bucket you `aws s3 sync s3://your-bucket/your-dataset/ data/` from first,
whatever. The paths above are just the expected on-disk layout, not real
data.)

`scripts/prepare_data.py` treats the filename as ground truth (a handful of
label files might have a bounding-box class id that disagrees with the
filename; logged to `data/prepare_report.json`, gitignored) and turns it
into classification manifests:

```bash
.venv/bin/python scripts/prepare_data.py
```

This is idempotent — safe to re-run any time you add/remove images. It:
- drops any class with zero example images (see `data/dataset_info.json`'s
  `class_counts`),
- writes `data/manifests/{train,val,test}.csv` (gitignored, regenerated) —
  one row per image, or per bounding box for images with more than one,
- falls back to the full (uncropped) image for any image with no bounding
  box,
- rewrites `data/class_mapping.json`, `data/dataset_info.json`, and
  `data/dataset.yaml` to match reality,
- flags classes with under 15 total images in `dataset_info.json`'s
  `low_data_classes` — per-class metrics for those will be noisy no matter
  what.

`scripts/fetch_inat_images.py` / `scripts/curate_external_images.py` are
the (optional) iNaturalist-expansion tools used to build the licensed
portion of the original dataset, if you want to grow a thin class the same
way.

## Train

```bash
.venv/bin/python scripts/train.py
.venv/bin/python scripts/train.py --set epochs=50 --set batch_size=16
```

Config lives in `configs/train.yaml`; any field can be overridden with
repeated `--set key=value`. Notable defaults:
- `model_name: mobilenet_v3_large` — ImageNet-pretrained; started this
  project with `mobilenet_v3_small` for its smaller mobile footprint, but
  CV showed `large` a clear accuracy win (63.6% -> 70.7% OOF, see below)
  worth the extra few MB. `mobilenet_v3_small` is still available via
  `--set model_name=mobilenet_v3_small` if footprint matters more than
  accuracy for a given deployment target. A `efficientnet_b0` alternative
  was also tried (72.46% OOF, at the point `large` was scoring 72.7%) and
  kept as a documented negative result on `experiment/efficientnet-b0`
  rather than adopted.
- `image_size: 320` — bumped from the original 224 after a CV ablation
  showed it was the single biggest lever tried: 72.7% -> 75.55% OOF, same
  architecture and every other hyperparameter unchanged.
- `freeze_backbone_epochs: 5` — trains only the classifier head first, then
  unfreezes the backbone at a low LR (`lr: 5e-5` vs. `head_lr: 7e-4`) so ~1k
  images can't wreck the pretrained ImageNet features.
- `weight_decay: 0.01`, `mixup_alpha: 0.1`, `random_erasing_p: 0.15` — a
  small-dataset regularization recipe, arrived at empirically over three
  runs (all kept under `outputs/run_v*` for comparison, see `history.json`
  and `test_report.json` in each):
  - **v1** (`weight_decay=1e-4`, no mixup/erasing, `lr=3e-4` post-unfreeze):
    hit 99.8% train accuracy by epoch 10 while val accuracy plateaued at
    51-55% — classic overfitting, the backbone moved too freely on too
    little data. Test accuracy 66.2%.
  - **v2** (`weight_decay=0.05`, `mixup_alpha=0.2`, `lr=2e-5`): fixed the
    overfit gap (train accuracy tracked val accuracy, ~45-53% both) but
    over-corrected into underfitting — test accuracy dropped to 57.3%.
  - **v3, current defaults**: same non-overfit train/val relationship as
    v2, but matches v1's 66.2% test accuracy (66.6% loss-wise it's better
    calibrated: test loss 1.37 vs. v1's 1.43) — same real-world performance
    without having memorized the training set, which should generalize
    better to photos outside this dataset than v1 does.
- `early_stop_metric: val_loss` — with only ~5 images/class in val (207
  total), val accuracy swings ±1-2% per epoch from noise alone; val loss is
  steadier and is what both checkpoint selection and early stopping key off.
- `use_weighted_sampler: true` — inverse-frequency sampling so rare classes
  (6 images) are seen about as often as common ones (75 images) per epoch.
- `device: cpu` — this is a CPU box; ~15-35s/epoch depending on load (32
  cores available, though PyTorch's default thread pool may not use all of
  them — see `torch.set_num_threads` if you want to tune it).

For evaluation, `scripts/evaluate.py --tta` averages each prediction with
its horizontal-flip view — a free accuracy bump at inference time that
needs no retraining.

Outputs go to `outputs/run/` (`best.pt`, `last.pt`, `history.json`,
`config.json`) — not committed to git (see `.gitignore`).

## Cross-validation (the trustworthy accuracy number)

The fixed train/val/test split only leaves a 68-image test set spread over
43 classes — many classes have 0-2 test images, so a single-split accuracy
number is mostly noise. `scripts/cross_validate.py` instead runs stratified
5-fold CV over the *entire* dataset (all ~1307 images, via
`scripts/make_cv_folds.py`) and scores every image exactly once, using only
the model from the fold that never trained on it:

```bash
.venv/bin/python scripts/cross_validate.py
```

**Result: 75.55% out-of-fold accuracy** (`mobilenet_v3_large`, 320px,
expanded iNaturalist-augmented dataset), per-fold 78.5% / 76.3% / 69.5% /
76.4% / 77.1% — this, not any single fixed-split test number, is the
trustworthy estimate of this recipe's real-world accuracy. Full per-class
breakdown in `outputs/cv/oof_report.json`. Progression to get here (all
measured the same OOF way, one variable changed at a time):

| Change | OOF accuracy |
| --- | --- |
| `mobilenet_v3_small` baseline | 58.7% |
| Switch to `mobilenet_v3_large` | 63.6% -> 70.7% |
| Expand dataset via iNaturalist (802 -> 1283 external images) | 72.7% |
| Try `efficientnet_b0` instead (rejected, didn't beat `large`) | 72.46% |
| Bump `image_size` 224px -> 320px | **75.55%** |

Fold 2 is persistently the weakest (69.5% here, similarly weak in the
efficientnet-b0 run) — worth a confusion-matrix look at whether specific
species pairs are being confused with each other, rather than assuming
it's purely a data-quantity problem.

CV also picks the epoch count for the deployment model: `median_best_epoch`
across the 5 folds (52 at this recipe) is what `scripts/train_final.py`
trains for.

## Final model (what ships)

```bash
.venv/bin/python scripts/train_final.py
```

Retrains on **100% of the data** (`data/manifests/all.csv`, no held-out
split — see `TrainConfig.skip_validation` in `config.py`) for the
CV-derived epoch count, using the same recipe validated above. This has no
val/test set of its own to self-report accuracy against; its expected
accuracy is the CV run's 75.55% OOF number, since it's the same recipe
trained the same way, just with ~25% more images than any single CV fold
saw. Output: `outputs/final/best.pt` — this is what `scripts/export.py` /
`scripts/export_tflite.py` should run against for the actual mobile
artifacts, not `outputs/run/`.

**This is exactly what's committed in `outputs/final/` in this repo**
(`mobilenet_v3_large`, 320px, 52 epochs, 2622 rows/2590 images including the
iNaturalist expansion, no val set), exported to both:
- `model.pt`/`model.ptl` (17.5MB) and `model_int8.ptl` (13.7MB) via
  `scripts/export.py` (PyTorch/TorchScript, dynamic int8), and
- `model.tflite` (17.0MB) and `model_int8.tflite` (4.8MB) via
  `scripts/export_tflite.py` (for the Flutter app — see "Export for
  mobile" below). TFLite output was spot-checked against the source
  PyTorch checkpoint on sample images and predictions matched exactly.

## Evaluate

```bash
.venv/bin/python scripts/evaluate.py --checkpoint outputs/run/best.pt --split test
```

Prints a per-class precision/recall/F1 report and saves it (plus a confusion
matrix) to `outputs/run/test_report.json`. Only meaningful against
`outputs/run/` (which has a genuine held-out test split) or a CV fold
checkpoint -- **not** `outputs/final/`, which trained on every image
including whatever would've been "test", so evaluating it on that split
would just be measuring memorization. For `outputs/final/`'s expected
accuracy, use the CV run's OOF number instead (see above).

## Predict

```bash
.venv/bin/python scripts/predict.py --checkpoint outputs/final/best.pt path/to/photo.jpg
```

## Export for mobile

```bash
.venv/bin/python scripts/export.py --checkpoint outputs/final/best.pt
```

Writes, next to the checkpoint:
- `model.pt` / `model.ptl` — TorchScript / PyTorch Mobile lite-interpreter
  build (fp32).
- `model_int8.ptl` — dynamically int8-quantized (classifier head only;
  dynamic quantization doesn't touch conv weights, so the size/speed win is
  modest — static or QAT quantization would do better if you need it).
- `model.onnx` — only if you `uv sync --extra export` first (adds the
  `onnx` package). Convert further to CoreML (`coremltools`) if targeting
  iOS directly from ONNX.
- `labels.json` — class names in output-index order, for the mobile app.

For the actual Flutter app target (TFLite/LiteRT, not PyTorch Mobile):

```bash
.venv/bin/python scripts/export_tflite.py --checkpoint outputs/final/best.pt
```

Requires the `tflite` extra (`uv sync --extra tflite`, pulls in
tensorflow-cpu + onnx2tf — a heavier install, skipped by default). Path:
PyTorch -> ONNX -> TensorFlow SavedModel -> TFLite. Writes `model.tflite`
(fp32) and `model_int8.tflite` (full integer quantized, ~4x smaller) next
to the checkpoint, plus its own `labels.json`. Verify the TFLite output
against the source PyTorch checkpoint's predictions on a few sample images
before trusting it in the app — the two should agree exactly (fp32 export,
no precision loss expected).

Two things worth knowing before shipping the `.ptl` files: the pip CPU build
of PyTorch used here doesn't have XNNPACK, so `optimize_for_mobile`'s op
fusions are skipped automatically (the export still works, just without that
extra optimization pass — rebuild PyTorch with XNNPACK, or use ONNX Runtime
Mobile instead, if that matters for latency). Separately, PyTorch's Lite
Interpreter format itself is now deprecated upstream in favor of
[ExecuTorch](https://docs.pytorch.org/executorch/stable/getting-started.html)
— `.ptl` still works today but ExecuTorch is the longer-term path if you're
starting mobile integration from scratch.

## AWS GPU training

CPU training works fine (`make train`, `make cv`), just slowly. This is an
optional path to run the same scripts on an on-demand `g4dn.xlarge` (T4
GPU) instead, entirely in *your own* AWS account.

**One-time setup**, in your own account:
1. An S3 bucket for run artifacts (code/data going up, logs/results coming
   back) — any name, any region.
2. An IAM role + instance profile the EC2 instance assumes, scoped to: read
   your bucket, write your bucket, and `ec2:TerminateInstances` restricted
   to instances tagged `Project=snake-train` (so the self-terminate step
   can't touch anything else in your account).
3. Export the env vars `launch_training.py` reads:
   ```bash
   export SNAKE_TRAIN_S3_BUCKET=your-bucket-name        # required
   export SNAKE_TRAIN_INSTANCE_PROFILE=your-profile-name # optional, defaults to snake-train-ec2-profile
   export SNAKE_TRAIN_AWS_REGION=us-east-1                # optional, defaults to us-east-1
   ```

Then:

```bash
make train-in-aws                                    # runs scripts/cross_validate.py on GPU
make train-in-aws AWS_TARGET=scripts/train_final.py   # or any other script
make aws-cost                                          # total spend so far
make aws-status                                        # any instances still running?
make aws-stop                                          # kill any stray ones
```

On-demand `g4dn.xlarge` pricing is looked up live via the AWS Pricing API
at launch time rather than hardcoded (check your own account's quotas —
particularly spot, which defaults to 0 in a lot of accounts/regions until
you request an increase — if you'd rather run spot instead of on-demand).
**Always prints a cost estimate and requires typed `yes` before launching
anything billable.** The instance is fully self-contained: its user-data
script pulls code + data from S3, trains, uploads results back to S3, and
**self-terminates** in a bash `trap ... EXIT` that fires on success,
failure, or the `AWS_MAX_HOURS` timeout alike — so a crash or hang doesn't
leave a bill-generating instance running.

Every run is logged locally to `aws/cost_log.json` (gitignored — exact,
since EC2 on-demand billing is per-second) and results land in
`outputs/aws/<run-id>/`. `make aws-cost` also cross-checks against AWS Cost
Explorer's actual billed EC2 total for your account.

## Project structure

```
configs/train.yaml          hyperparameters
data/
  class_mapping.json          43-class name -> id (committed, no images)
  dataset.yaml / dataset_info.json   committed metadata (counts, low-data classes)
  species_mapping_draft.json  Sinhala name -> scientific name, DRAFT/unverified -- read its _readme
  external/inaturalist/*.csv  attribution + curation records for the licensed portion (no images)
  images/, labels/             NOT included -- bring your own, see "Bring your own dataset"
  manifests/                   generated by prepare_data.py, gitignored
scripts/
  prepare_data.py            one-time dataset cleanup + fixed-split manifest builder
  make_cv_folds.py           stratified 5-fold manifest builder (data/manifests/cv/)
  train.py                   run_training(config) -- the one training loop, + its CLI
  cross_validate.py          trains all 5 folds, aggregates out-of-fold report
  train_final.py             retrains on 100% of data for deployment
  evaluate.py                test/val-set metrics (+ evaluate_with_tta)
  predict.py                 single-image inference
  export.py                  TorchScript / int8 / ONNX export for mobile
  export_tflite.py           TFLite export (PyTorch -> ONNX -> TF -> TFLite)
  aws/                       optional GPU-training-in-your-own-AWS-account tooling
src/snake_classifier/
  config.py                  TrainConfig dataclass (+ yaml loader)
  data_prep.py                shared filename/bbox scanning (prepare_data.py + make_cv_folds.py)
  dataset.py                 manifest-driven Dataset (crop-or-full-image)
  transforms.py               train/eval image transforms
  model.py                   MobileNetV3 factory + freeze/unfreeze helpers
  engine.py                  train_one_epoch / evaluate loops
  utils.py                   seeding, checkpoints, weighted sampler
outputs/                     checkpoints + reports -- gitignored EXCEPT outputs/final/
  cv/                         5 fold checkpoints + oof_report.json (regenerate via `make cv`)
  run/                        fixed-split experiment checkpoints (regenerate via `make train`)
  final/                      pretrained deployment model -- committed, see Quickstart
```
