#!/usr/bin/env python3
"""Download a sample of real ICSR case data from openFDA's FAERS API.

No account/CAPTCHA needed (unlike VAERS's bulk download, which is human-gated).
This gives structured fields (drug, reaction, demographics, seriousness) but
weak/no free-text narrative -- see docs/DATA_SOURCES.md section A vs B. Use this
for schema-field grounding and as a MedDRA-coded reaction source, not as the
primary narrative-text training source.

Usage:
    python scripts/download_faers_sample.py --count 1000 --out data/raw/faers_sample.jsonl
"""
import argparse
import json
import time
from pathlib import Path

import requests

OPENFDA_EVENT_URL = "https://api.fda.gov/drug/event.json"
PAGE_SIZE = 100  # openFDA's per-request max for standard (non-count) queries


def fetch_faers_records(count: int, sleep_between: float = 0.3) -> list[dict]:
    records: list[dict] = []
    skip = 0
    while len(records) < count:
        params = {"limit": min(PAGE_SIZE, count - len(records)), "skip": skip}
        resp = requests.get(OPENFDA_EVENT_URL, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json().get("results", [])
        if not batch:
            break
        records.extend(batch)
        skip += PAGE_SIZE
        time.sleep(sleep_between)  # openFDA's unauthenticated rate limit is modest, be polite
    return records[:count]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--out", default="data/raw/faers_sample.jsonl")
    args = ap.parse_args()

    records = fetch_faers_records(args.count)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"downloaded {len(records)} FAERS records -> {out_path}")


if __name__ == "__main__":
    main()
