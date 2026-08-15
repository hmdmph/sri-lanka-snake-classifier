#!/usr/bin/env python
"""Launch a GPU training run on EC2.

Safety model (do not weaken without re-reading this):
  - Always prints a cost estimate and requires typed "yes" confirmation
    before anything billable happens.
  - The instance self-terminates via its own IAM role at the end of the
    user-data script, whatever happens (success, failure, or timeout) --
    the whole script body runs inside a bash `trap ... EXIT`.
  - A --max-hours cap (default 3h) wraps the training step in `timeout`,
    so a hung job still gets killed and the instance still terminates.
  - The IAM role attached via SNAKE_TRAIN_INSTANCE_PROFILE should only be
    allowed to terminate instances tagged Project=snake-train -- scope it
    that narrowly in your own account, it shouldn't be able to touch
    anything else.
  - Every run is appended to aws/cost_log.json with actual wall-clock
    duration x the on-demand price in effect at launch (on-demand EC2
    billing is per-second, so this is an accurate estimate, not just a
    guess) -- see scripts/aws/report_cost.py.

Usage:
    .venv/bin/python scripts/aws/launch_training.py
    .venv/bin/python scripts/aws/launch_training.py --target scripts/train_final.py --max-hours 2
    .venv/bin/python scripts/aws/launch_training.py --yes   # skip the confirmation prompt
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tarfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
AWS_DIR = REPO_ROOT / "aws"
COST_LOG = AWS_DIR / "cost_log.json"

# Every value below is specific to *your* AWS account -- none of this is
# shared infrastructure. Set them via env vars (an .env-style export, or
# your shell profile) rather than hardcoding an account here.
REGION = os.environ.get("SNAKE_TRAIN_AWS_REGION", "us-east-1")
BUCKET = os.environ.get("SNAKE_TRAIN_S3_BUCKET", "")
INSTANCE_PROFILE = os.environ.get("SNAKE_TRAIN_INSTANCE_PROFILE", "snake-train-ec2-profile")
PROJECT_TAG = "snake-train"
DEFAULT_INSTANCE_TYPE = "g4dn.xlarge"

CODE_INCLUDE = ["src", "scripts", "configs", "Makefile", "pyproject.toml", "uv.lock"]
DATA_INCLUDE = ["data/images", "data/labels", "data/external", "data/class_mapping.json",
                "data/dataset.yaml", "data/species_mapping_draft.json"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="scripts/cross_validate.py",
                         help="Script to run remotely (relative to repo root).")
    parser.add_argument("--target-args", default="", help="Extra args passed through to --target.")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    parser.add_argument("--max-hours", type=float, default=3.0,
                         help="Hard cap: training is killed and the instance terminated after this long.")
    parser.add_argument("--yes", action="store_true", help="Skip the cost confirmation prompt.")
    return parser.parse_args()


def get_on_demand_price(pricing_client, instance_type: str, region: str) -> float:
    region_names = {
        "us-east-1": "US East (N. Virginia)",
        "us-east-2": "US East (Ohio)",
        "us-west-2": "US West (Oregon)",
    }
    resp = pricing_client.get_products(
        ServiceCode="AmazonEC2",
        Filters=[
            {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
            {"Type": "TERM_MATCH", "Field": "location", "Value": region_names[region]},
            {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
            {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
            {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
            {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
        ],
        MaxResults=1,
    )
    price_list = json.loads(resp["PriceList"][0])
    terms = price_list["terms"]["OnDemand"]
    (term,) = terms.values()
    (dimension,) = term["priceDimensions"].values()
    return float(dimension["pricePerUnit"]["USD"])


def find_latest_dlami(ec2_client) -> str:
    resp = ec2_client.describe_images(
        Owners=["amazon"],
        Filters=[
            {"Name": "name", "Values": ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Amazon Linux 2023) *"]},
            {"Name": "state", "Values": ["available"]},
        ],
    )
    images = sorted(resp["Images"], key=lambda i: i["CreationDate"], reverse=True)
    if not images:
        raise SystemExit("No Deep Learning AMI found -- AWS may have renamed it, check the console.")
    return images[0]["ImageId"]


def make_tarball(paths: list[str], out_path: Path) -> None:
    with tarfile.open(out_path, "w:gz") as tar:
        for rel in paths:
            full = REPO_ROOT / rel
            if full.exists():
                tar.add(full, arcname=rel)


def build_user_data(run_id: str, target: str, target_args: str, max_hours: float) -> str:
    max_seconds = int(max_hours * 3600)
    return f"""#!/bin/bash
set -x
exec > >(tee /var/log/snake-train.log) 2>&1

RUN_ID="{run_id}"
BUCKET="{BUCKET}"
S3_PREFIX="s3://$BUCKET/runs/$RUN_ID"

TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
REGION={REGION}

# Sync the log to S3 every 15s so the local poller can tail progress live.
( while true; do aws s3 cp /var/log/snake-train.log "$S3_PREFIX/log.txt" --region $REGION >/dev/null 2>&1; sleep 15; done ) &
LOG_SYNC_PID=$!

finish() {{
  EXIT_CODE=$?
  aws s3 cp /var/log/snake-train.log "$S3_PREFIX/log.txt" --region $REGION >/dev/null 2>&1
  if [ -d /home/ec2-user/snake-train/outputs ]; then
    tar -C /home/ec2-user/snake-train -czf /tmp/results.tar.gz outputs
    aws s3 cp /tmp/results.tar.gz "$S3_PREFIX/results.tar.gz" --region $REGION
  fi
  echo "$EXIT_CODE" > /tmp/exit_code
  aws s3 cp /tmp/exit_code "$S3_PREFIX/DONE" --region $REGION
  kill $LOG_SYNC_PID 2>/dev/null || true
  aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region $REGION
}}
trap finish EXIT

cd /home/ec2-user
mkdir -p snake-train/data
aws s3 cp "$S3_PREFIX/code.tar.gz" code.tar.gz --region $REGION
aws s3 cp "$S3_PREFIX/data.tar.gz" data.tar.gz --region $REGION
tar -xzf code.tar.gz -C snake-train
tar -xzf data.tar.gz -C snake-train
chown -R ec2-user:ec2-user snake-train
cd snake-train

# Find a python that already has a CUDA-enabled torch (DLAMI ships one,
# but the env name/mechanism varies by AMI generation -- don't assume a
# specific conda env name). Fall back to bootstrapping pip + torch on the
# system python if nothing pre-installed is found.
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
PYBIN=""
for candidate in /opt/conda/envs/*/bin/python3 /opt/conda/bin/python3 /usr/bin/python3; do
  if [ -x "$candidate" ] && "$candidate" -c "import torch" 2>/dev/null; then
    PYBIN="$candidate"
    break
  fi
done

if [ -z "$PYBIN" ]; then
  echo "No pre-installed torch found on this AMI, bootstrapping python3.12 from scratch"
  # The system /usr/bin/python3 on this AMI is 3.9 -- too old for current
  # torch wheels (need 3.10+) and for pypa's get-pip.py bootstrap script.
  # Install a modern interpreter via dnf and bootstrap pip with the
  # stdlib's own ensurepip (bundles its own wheel, no version-sensitive
  # external download like get-pip.py has).
  dnf install -y python3.12 >/dev/null 2>&1 || true
  if command -v python3.12 >/dev/null 2>&1; then
    PYBIN=python3.12
  else
    PYBIN=/usr/bin/python3
  fi
  "$PYBIN" -m ensurepip --upgrade >/dev/null 2>&1 || true
  "$PYBIN" -m pip install -q --upgrade pip
  "$PYBIN" -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi

echo "Using $PYBIN ($($PYBIN --version))"
"$PYBIN" -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())" || {{
  echo "FATAL: torch still not importable after setup, aborting before wasting GPU time."
  exit 1
}}

"$PYBIN" -m pip install -q scikit-learn pyyaml tqdm requests pillow 2>&1 | tail -5

"$PYBIN" scripts/prepare_data.py

timeout {max_seconds} "$PYBIN" {target} --set device=cuda {target_args}
"""


def main() -> None:
    if not BUCKET:
        raise SystemExit(
            "SNAKE_TRAIN_S3_BUCKET is not set. Create an S3 bucket in your own AWS "
            "account for run artifacts, then export SNAKE_TRAIN_S3_BUCKET=your-bucket-name "
            "(see the AWS GPU training section of README.md for the full one-time setup: "
            "bucket, IAM role/instance profile, etc.)."
        )
    args = parse_args()
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"

    session = boto3.Session(region_name=REGION)
    pricing = session.client("pricing", region_name="us-east-1")  # Pricing API only in us-east-1/ap-south-1
    ec2 = session.client("ec2")

    hourly_price = get_on_demand_price(pricing, args.instance_type, REGION)
    worst_case_cost = hourly_price * args.max_hours

    print(f"Run ID: {run_id}")
    print(f"Instance: {args.instance_type} in {REGION} (on-demand -- check your account's "
          f"spot quota for this instance family if you want to try spot instead)")
    print(f"Price: ${hourly_price:.4f}/hr")
    print(f"Hard cap: {args.max_hours}h -> worst-case cost ${worst_case_cost:.2f}")
    print(f"Target script: {args.target} {args.target_args}")
    print("(Actual cost will likely be much lower -- billing is per-second and training "
          "should finish well before the cap; see the cap as a runaway-cost safety net, not an estimate.)")

    if not args.yes:
        reply = input("\nType 'yes' to launch this billable instance: ").strip().lower()
        if reply != "yes":
            print("Aborted, nothing launched.")
            return

    AWS_DIR.mkdir(exist_ok=True)
    work_dir = AWS_DIR / "_work" / run_id
    work_dir.mkdir(parents=True)
    code_tar = work_dir / "code.tar.gz"
    data_tar = work_dir / "data.tar.gz"

    print("Packaging code...")
    make_tarball(CODE_INCLUDE, code_tar)
    print("Packaging data...")
    make_tarball(DATA_INCLUDE, data_tar)

    s3 = session.client("s3")
    prefix = f"runs/{run_id}"
    print(f"Uploading to s3://{BUCKET}/{prefix}/ ...")
    s3.upload_file(str(code_tar), BUCKET, f"{prefix}/code.tar.gz")
    s3.upload_file(str(data_tar), BUCKET, f"{prefix}/data.tar.gz")

    ami_id = find_latest_dlami(ec2)
    print(f"Using AMI {ami_id}")

    user_data = build_user_data(run_id, args.target, args.target_args, args.max_hours)

    launch_time = datetime.now(timezone.utc)
    resp = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=args.instance_type,
        MinCount=1,
        MaxCount=1,
        IamInstanceProfile={"Name": INSTANCE_PROFILE},
        UserData=user_data,
        InstanceInitiatedShutdownBehavior="terminate",
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Project", "Value": PROJECT_TAG},
                {"Key": "Name", "Value": f"snake-train-{run_id}"},
                {"Key": "RunId", "Value": run_id},
            ],
        }],
    )
    instance_id = resp["Instances"][0]["InstanceId"]
    print(f"Launched {instance_id}. Waiting for training to finish (polling S3 log)...")

    log_seen = 0
    result_state = None
    while True:
        time.sleep(15)
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=f"{prefix}/log.txt")
            content = obj["Body"].read().decode("utf-8", errors="replace")
            if len(content) > log_seen:
                sys.stdout.write(content[log_seen:])
                sys.stdout.flush()
                log_seen = len(content)
        except s3.exceptions.NoSuchKey:
            pass

        try:
            s3.head_object(Bucket=BUCKET, Key=f"{prefix}/DONE")
            result_state = "done"
            break
        except Exception:
            pass

        state = ec2.describe_instances(InstanceIds=[instance_id])["Reservations"][0]["Instances"][0]["State"]["Name"]
        if state in ("terminated", "shutting-down"):
            result_state = state
            break

    terminate_time = datetime.now(timezone.utc)
    duration_hours = (terminate_time - launch_time).total_seconds() / 3600
    actual_cost = duration_hours * hourly_price

    print(f"\nFinished ({result_state}). Duration: {duration_hours * 60:.1f} min. Cost: ${actual_cost:.4f}")

    out_dir = REPO_ROOT / "outputs" / "aws" / run_id
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        s3.download_file(BUCKET, f"{prefix}/results.tar.gz", str(out_dir / "results.tar.gz"))
        with tarfile.open(out_dir / "results.tar.gz") as tar:
            tar.extractall(out_dir)
        print(f"Results downloaded to {out_dir}")
    except Exception as exc:  # noqa: BLE001
        print(f"Could not download results (job may have failed before producing output): {exc}")

    log_entry = {
        "run_id": run_id,
        "instance_type": args.instance_type,
        "region": REGION,
        "target": args.target,
        "launch_time": launch_time.isoformat(),
        "terminate_time": terminate_time.isoformat(),
        "duration_hours": duration_hours,
        "hourly_price_usd": hourly_price,
        "estimated_cost_usd": actual_cost,
    }
    history = json.loads(COST_LOG.read_text()) if COST_LOG.exists() else []
    history.append(log_entry)
    COST_LOG.write_text(json.dumps(history, indent=2) + "\n")
    print(f"Logged to {COST_LOG}")


if __name__ == "__main__":
    main()
