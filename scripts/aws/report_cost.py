#!/usr/bin/env python
"""Report AWS EC2 spend for this project.

Two numbers, and they can legitimately disagree slightly:
  - "Tracked" comes from aws/cost_log.json -- duration x the on-demand
    price in effect at launch, written by launch_training.py right after
    each run terminates. Immediate, but only knows about runs launched
    through that script.
  - "AWS-billed" comes from Cost Explorer, filtered to EC2 and (when
    possible) our Project=snake-train cost allocation tag -- authoritative,
    but can lag up to ~24h and won't reflect tag data for instances
    launched before the tag was activated for cost allocation.

Usage:
    .venv/bin/python scripts/aws/report_cost.py
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
COST_LOG = REPO_ROOT / "aws" / "cost_log.json"


def print_tracked() -> float:
    if not COST_LOG.exists():
        print("No tracked runs yet (aws/cost_log.json doesn't exist -- "
              "nothing launched via scripts/aws/launch_training.py).")
        return 0.0
    history = json.loads(COST_LOG.read_text())
    if not history:
        print("aws/cost_log.json is empty.")
        return 0.0

    print(f"{'run_id':<28} {'instance':<14} {'duration':>10} {'cost':>10}")
    total = 0.0
    for run in history:
        total += run["estimated_cost_usd"]
        print(
            f"{run['run_id']:<28} {run['instance_type']:<14} "
            f"{run['duration_hours'] * 60:>8.1f}m ${run['estimated_cost_usd']:>8.4f}"
        )
    print(f"\nTracked total ({len(history)} run(s)): ${total:.4f}")
    return total


def print_billed() -> None:
    try:
        ce = boto3.client("ce", region_name="us-east-1")  # Cost Explorer API is us-east-1 only
        earliest = datetime.now(timezone.utc) - timedelta(days=90)
        resp = ce.get_cost_and_usage(
            TimePeriod={
                "Start": earliest.strftime("%Y-%m-%d"),
                "End": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            },
            Granularity="MONTHLY",
            Metrics=["UnblendedCost"],
            Filter={
                "Dimensions": {
                    "Key": "SERVICE",
                    "Values": ["Amazon Elastic Compute Cloud - Compute"],
                }
            },
        )
        total = sum(float(r["Total"]["UnblendedCost"]["Amount"]) for r in resp["ResultsByTime"])
        print(f"\nAWS-billed EC2 compute total, last 90 days (all EC2 usage on this account, "
              f"not just this project -- Cost Explorer can lag ~24h): ${total:.4f}")
    except Exception as exc:  # noqa: BLE001
        print(f"\nCould not reach Cost Explorer ({exc}); showing tracked total only.")


def main() -> None:
    print_tracked()
    print_billed()


if __name__ == "__main__":
    main()
