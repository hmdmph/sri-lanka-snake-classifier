.DEFAULT_GOAL := help
.PHONY: help setup setup-aws setup-tflite setup-export \
        prepare-data fetch-data curate-data \
        train cv train-final \
        train-in-aws aws-cost aws-status aws-stop \
        eval eval-final predict export export-tflite \
        test clean clean-outputs

PYTHON := .venv/bin/python

# Override on the command line, e.g. `make predict IMG=photo.jpg`
# Defaults to outputs/final/best.pt, the pretrained checkpoint shipped in
# this repo -- override to outputs/run/best.pt (or a CV fold) once you've
# trained your own.
IMG ?=
CHECKPOINT ?= outputs/final/best.pt
AWS_TARGET ?= scripts/cross_validate.py
AWS_MAX_HOURS ?= 3

## help: Show this help (default target)
help:
	@echo "snake-train -- Sri Lankan snake species classifier"
	@echo ""
	@echo "Usage: make <target> [VAR=value ...]"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} /^## /{sub(/^## /, ""); print}' $(MAKEFILE_LIST) | \
		awk -F': ' '{printf "  %-16s %s\n", $$1, $$2}'
	@echo ""
	@echo "Common variables:"
	@echo "  CHECKPOINT=$(CHECKPOINT)        (used by eval, predict, export, export-tflite)"
	@echo "  IMG=path/to/photo.jpg           (used by predict)"
	@echo "  AWS_TARGET=$(AWS_TARGET)  (used by train-in-aws)"
	@echo "  AWS_MAX_HOURS=$(AWS_MAX_HOURS)                   (used by train-in-aws, safety cap)"

## setup: Create .venv and install base dependencies via uv
setup:
	uv venv --python 3.12
	uv sync

## setup-aws: Install AWS-training dependencies (boto3) on top of the base env
setup-aws:
	uv sync --extra aws

## setup-tflite: Install TFLite export dependencies (tensorflow, onnx2tf -- large download)
setup-tflite:
	uv sync --extra tflite

## setup-export: Install ONNX export dependencies
setup-export:
	uv sync --extra export

## prepare-data: Rebuild data/manifests/{train,val,test}.csv from the raw dataset (idempotent)
prepare-data:
	$(PYTHON) scripts/prepare_data.py

## fetch-data: Pull additional licensed photos from iNaturalist for high-confidence classes
fetch-data:
	$(PYTHON) scripts/fetch_inat_images.py

## curate-data: Flag newly-fetched images the current model disagrees with, for manual review
curate-data:
	$(PYTHON) scripts/curate_external_images.py --checkpoint $(CHECKPOINT)

## train: Train on the fixed 80/20 split (outputs/run/)
train:
	$(PYTHON) scripts/train.py

## cv: Stratified 5-fold cross-validation over the full dataset (outputs/cv/, ~1-2h on CPU)
cv:
	$(PYTHON) scripts/cross_validate.py

## train-final: Retrain on 100% of the data for the CV-derived epoch count (outputs/final/)
train-final:
	$(PYTHON) scripts/train_final.py

## train-in-aws: Run AWS_TARGET on a GPU EC2 instance (on-demand, ~$0.53/hr; ALWAYS confirms cost first)
train-in-aws: setup-aws
	$(PYTHON) scripts/aws/launch_training.py --target $(AWS_TARGET) --max-hours $(AWS_MAX_HOURS)

## aws-cost: Show total AWS EC2 spend for this project (tracked + AWS-billed)
aws-cost: setup-aws
	$(PYTHON) scripts/aws/report_cost.py

## aws-status: List any running/stopped snake-train EC2 instances
aws-status: setup-aws
	$(PYTHON) scripts/aws/stop_instances.py

## aws-stop: Terminate any stray snake-train EC2 instances (safety valve)
aws-stop: setup-aws
	$(PYTHON) scripts/aws/stop_instances.py --kill

## eval: Evaluate CHECKPOINT on the test split (per-class report + confusion matrix)
eval:
	$(PYTHON) scripts/evaluate.py --checkpoint $(CHECKPOINT) --split test --tta

## eval-final: Print the trustworthy out-of-fold accuracy for outputs/final/ (see outputs/cv/oof_report.json)
eval-final:
	@$(PYTHON) -c "import json; r = json.load(open('outputs/cv/oof_report.json')); print(f\"OOF accuracy: {r['oof_accuracy']:.4f}\")"

## predict: Run inference on IMG with CHECKPOINT (e.g. make predict IMG=photo.jpg)
predict:
	@test -n "$(IMG)" || (echo "Usage: make predict IMG=path/to/photo.jpg" && exit 1)
	$(PYTHON) scripts/predict.py --checkpoint $(CHECKPOINT) "$(IMG)"

## export: Export CHECKPOINT to TorchScript (.pt/.ptl) + int8 for mobile
export:
	$(PYTHON) scripts/export.py --checkpoint $(CHECKPOINT)

## export-tflite: Export CHECKPOINT to TFLite for the Flutter app (needs setup-tflite)
export-tflite: setup-tflite
	$(PYTHON) scripts/export_tflite.py --checkpoint $(CHECKPOINT)

## test: Quick smoke test -- 1-epoch training + predict, verifies the pipeline end-to-end
test:
	$(PYTHON) scripts/train.py --set epochs=1 --set num_workers=0 --set freeze_backbone_epochs=0 --set output_dir=outputs/_smoke_test
	$(PYTHON) scripts/predict.py --checkpoint outputs/_smoke_test/best.pt \
		"$$(find data/images/test -name '*.jpg' | head -1)"
	rm -rf outputs/_smoke_test
	@echo "OK: pipeline smoke test passed"

## clean: Remove Python cache files (safe, never touches outputs/ or data/)
clean:
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	rm -rf .pytest_cache

## clean-outputs: Remove ALL training outputs (checkpoints, logs, reports) -- asks first
clean-outputs:
	@echo "This deletes everything under outputs/ (checkpoints, CV results, reports)."
	@read -p "Type 'yes' to confirm: " reply; [ "$$reply" = "yes" ] || (echo "Aborted." && exit 1)
	rm -rf outputs
