#!/usr/bin/env python
"""Safety valve: list or terminate any running snake-train EC2 instances.

Every instance launch_training.py launches self-terminates on completion
or timeout, but if something goes really wrong (e.g. the instance loses
network connectivity and can't reach S3/EC2 API to signal completion or
terminate itself), this is the manual backstop. Only ever touches
instances tagged Project=snake-train.

Usage:
    .venv/bin/python scripts/aws/stop_instances.py          # list only
    .venv/bin/python scripts/aws/stop_instances.py --kill    # terminate them all
"""

from __future__ import annotations

import argparse
import os

import boto3

REGION = os.environ.get("SNAKE_TRAIN_AWS_REGION", "us-east-1")
PROJECT_TAG = "snake-train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kill", action="store_true", help="Terminate instead of just listing.")
    args = parser.parse_args()

    ec2 = boto3.client("ec2", region_name=REGION)
    resp = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Project", "Values": [PROJECT_TAG]},
            {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]},
        ]
    )
    instances = [i for r in resp["Reservations"] for i in r["Instances"]]

    if not instances:
        print("No running/stopped snake-train instances found.")
        return

    for inst in instances:
        name = next((t["Value"] for t in inst.get("Tags", []) if t["Key"] == "Name"), "?")
        print(f"{inst['InstanceId']}  {inst['InstanceType']}  {inst['State']['Name']}  {name}  "
              f"launched {inst['LaunchTime']}")

    if args.kill:
        ids = [i["InstanceId"] for i in instances]
        ec2.terminate_instances(InstanceIds=ids)
        print(f"\nTerminated: {ids}")
    else:
        print("\nRun with --kill to terminate these.")


if __name__ == "__main__":
    main()
