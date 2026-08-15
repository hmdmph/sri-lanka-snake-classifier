#!/usr/bin/env python
"""Fetch legitimately-licensed Sri Lankan snake photos from iNaturalist to
expand the training set beyond the original 1307 hand-collected images.

SAFETY NOTE: this only fetches for classes whose scientific-name mapping in
data/species_mapping_draft.json meets --min-confidence (default: "high"
only). That file is a Claude-drafted, NOT expert-verified mapping -- fetching
under a wrong or ambiguous mapping would silently pollute the training set
with photos of the wrong species. See that file's _readme for the known
"mapila" naming-ambiguity issue before raising --min-confidence to
"moderate" or lower.

Downloads go to data/external/inaturalist/<class-slug>/, kept separate from
the original hand-collected data/images/{train,val,test}/ so they can be
reviewed, filtered, or removed independently. Attribution (observer,
license, source URL) for every downloaded image is recorded in
data/external/inaturalist/attribution.csv -- required for CC-BY compliance
if any of this data reaches a public app.

Usage:
    .venv/bin/python scripts/fetch_inat_images.py
    .venv/bin/python scripts/fetch_inat_images.py --max-per-class 100 --min-confidence moderate
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
API_BASE = "https://api.inaturalist.org/v1"
USER_AGENT = "snake-train-dataset-builder/0.1 (non-commercial research/education dataset for a Sri Lankan snake ID app)"
CONFIDENCE_ORDER = {"unresolved": 0, "moderate": 1, "high": 2}
LICENSES = ["cc0", "cc-by", "cc-by-nc"]  # matches GBIF's accepted set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", default="data/species_mapping_draft.json")
    parser.add_argument("--max-per-class", type=int, default=80)
    parser.add_argument(
        "--min-confidence", choices=["high", "moderate", "unresolved"], default="high",
        help="Only fetch classes at or above this confidence level.",
    )
    parser.add_argument("--place", default="Sri Lanka")
    parser.add_argument("--request-delay", type=float, default=1.0, help="Seconds between API calls (rate-limit courtesy)")
    return parser.parse_args()


def get_place_id(place_name: str) -> int:
    resp = requests.get(
        f"{API_BASE}/places/autocomplete",
        params={"q": place_name},
        headers={"User-Agent": USER_AGENT},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json()["results"]
    for r in results:
        if r["name"].lower() == place_name.lower() and r.get("admin_level") == 0:
            return r["id"]
    return results[0]["id"]


def resolve_current_name(scientific_name: str) -> str:
    """iNaturalist's `taxon_name` observation filter matches the currently
    accepted name, not older synonyms (e.g. Xenochrophis piscator is now
    Fowlea piscator) -- resolve via taxon search first so a stale name in
    the mapping doesn't silently return zero results.

    SAFETY-CRITICAL: the /v1/taxa search is a loose text match, not a
    synonym lookup -- it happily returns unrelated species (a "Naja naja"
    query once matched "Erythromma najas", a damselfly, ahead of Naja naja
    itself). Only ever treat this as a redirect when the exact queried name
    is ABSENT from the results (i.e. truly no longer a valid taxon) *and*
    exactly one candidate comes back whose matched_term contains the
    original name as a substring (a genuine subspecies/synonym match, like
    "Trimeresurus trigonocephalus" -> "Craspedocephalus trigonocephalus").
    Anything more ambiguous than that is left unresolved rather than guessed.
    """
    resp = requests.get(
        f"{API_BASE}/taxa",
        params={"q": scientific_name, "rank": "species"},
        headers={"User-Agent": USER_AGENT},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json()["results"]

    for t in results:
        if t["name"].lower() == scientific_name.lower() and t.get("is_active"):
            return scientific_name  # exact name is still current -- no redirect needed

    candidates = [
        t for t in results
        if t.get("is_active") and (t.get("matched_term") or "").lower() == scientific_name.lower()
    ]
    if len(candidates) == 1:
        print(f"  note: {scientific_name!r} -> current name {candidates[0]['name']!r} (synonym match)")
        return candidates[0]["name"]

    print(f"  WARNING: {scientific_name!r} not found and no unambiguous synonym redirect "
          f"({len(candidates)} candidates) -- leaving name as-is, likely 0 results")
    return scientific_name


def fetch_observations(scientific_name: str, place_id: int, max_images: int, delay: float) -> list[dict]:
    observations: list[dict] = []
    page = 1
    while len(observations) < max_images:
        resp = requests.get(
            f"{API_BASE}/observations",
            params={
                "taxon_name": scientific_name,
                "place_id": place_id,
                "quality_grade": "research",
                "photos": "true",
                "photo_license": ",".join(LICENSES),
                "per_page": 200,
                "page": page,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json()["results"]
        if not results:
            break
        observations.extend(results)
        if len(results) < 200:
            break
        page += 1
        time.sleep(delay)
    return observations


def main() -> None:
    args = parse_args()
    mapping = json.loads((REPO_ROOT / args.mapping).read_text())
    min_rank = CONFIDENCE_ORDER[args.min_confidence]

    targets = {
        name: info
        for name, info in mapping.items()
        if name != "_readme"
        and info.get("scientific_name")
        and CONFIDENCE_ORDER.get(info["confidence"], -1) >= min_rank
    }
    print(f"Fetching for {len(targets)} classes (min-confidence={args.min_confidence}): {sorted(targets)}")

    place_id = get_place_id(args.place)
    print(f"Resolved place_id={place_id} for {args.place!r}")

    external_dir = DATA_DIR / "external" / "inaturalist"
    external_dir.mkdir(parents=True, exist_ok=True)
    attribution_path = external_dir / "attribution.csv"
    existing_ids: set[str] = set()
    rows: list[dict] = []
    if attribution_path.exists():
        with attribution_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
            existing_ids = {r["file"] for r in rows}

    fieldnames = ["class_name", "scientific_name", "file", "observation_id", "observation_url", "observer", "license_code"]

    for class_name, info in sorted(targets.items()):
        sci = info["scientific_name"]
        print(f"\n{class_name} ({sci})...")
        try:
            sci = resolve_current_name(sci)
            observations = fetch_observations(sci, place_id, args.max_per_class, args.request_delay)
        except requests.RequestException as exc:
            print(f"  API error, skipping: {exc}")
            continue
        print(f"  {len(observations)} observations found")

        out_dir = external_dir / class_name
        out_dir.mkdir(exist_ok=True)
        count = 0
        for obs in observations:
            if count >= args.max_per_class:
                break
            for photo in obs.get("photos", []):
                if count >= args.max_per_class:
                    break
                license_code = photo.get("license_code")
                if license_code not in LICENSES:
                    continue
                rel_path = f"external/inaturalist/{class_name}/{obs['id']}_{photo['id']}.jpg"
                if rel_path in existing_ids:
                    count += 1
                    continue
                url = photo["url"].replace("square.jpg", "medium.jpg")
                try:
                    img_resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
                    img_resp.raise_for_status()
                except requests.RequestException as exc:
                    print(f"  failed {url}: {exc}")
                    continue
                (DATA_DIR / rel_path).write_bytes(img_resp.content)
                rows.append(
                    {
                        "class_name": class_name,
                        "scientific_name": sci,
                        "file": rel_path,
                        "observation_id": obs["id"],
                        "observation_url": f"https://www.inaturalist.org/observations/{obs['id']}",
                        "observer": obs.get("user", {}).get("login", ""),
                        "license_code": license_code,
                    }
                )
                existing_ids.add(rel_path)
                count += 1
                time.sleep(0.2)
        print(f"  downloaded {count} images")

        # Write after every class, not just at the end -- a crash partway
        # through shouldn't lose attribution for images already saved.
        with attribution_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nWrote {attribution_path} ({len(rows)} attributed images total)")


if __name__ == "__main__":
    main()
