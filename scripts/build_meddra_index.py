#!/usr/bin/env python3
"""Build an embedding index of MedDRA-adjacent reaction terms for retrieval-based coding.

Per docs/DATA_SOURCES.md section D: the full official MedDRA term list is licensed by
MSSO and not freely redistributable in bulk. This script defaults to building the index
from openFDA FAERS data instead — FAERS ships reactions already MedDRA-coded under FDA's
public-domain data terms, which sidesteps the licensing question for a bootstrap index.

If your org has an active MedDRA subscription, swap --source to a local export file
(one PT per line, or a CSV with pt,soc columns) instead of the API.

Usage:
    python scripts/build_meddra_index.py --source api --limit 2000 --out data/meddra_index
    python scripts/build_meddra_index.py --source file --terms-file data/my_meddra_terms.csv --out data/meddra_index
    python scripts/build_meddra_index.py --query "severe headache and vomiting" --index data/meddra_index
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

OPENFDA_EVENT_URL = "https://api.fda.gov/drug/event.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def fetch_terms_from_openfda(limit: int) -> list[str]:
    """Pull distinct reaction.meddrapt values from openFDA's public FAERS endpoint."""
    terms: set[str] = set()
    skip = 0
    page_size = 100
    while len(terms) < limit:
        params = {
            "search": "patient.reaction.reactionmeddrapt:*",
            "count": "patient.reaction.reactionmeddrapt.exact",
            "limit": page_size,
            "skip": skip,
        }
        resp = requests.get(OPENFDA_EVENT_URL, params=params, timeout=30)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            break
        for r in results:
            terms.add(r["term"])
        skip += page_size
        if len(results) < page_size:
            break
    return sorted(terms)[:limit]


def load_terms_from_file(path: str) -> list[tuple[str, str | None]]:
    """Load (term, soc) pairs from a local file — plain text (one term per line) or CSV with pt,soc columns."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        with p.open() as f:
            reader = csv.DictReader(f)
            return [(row["pt"], row.get("soc")) for row in reader]
    return [(line.strip(), None) for line in p.read_text().splitlines() if line.strip()]


def build_index(terms: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(terms, show_progress_bar=True, normalize_embeddings=True)
    np.save(out_dir / "embeddings.npy", embeddings)
    (out_dir / "terms.json").write_text(json.dumps(terms, indent=2))
    print(f"indexed {len(terms)} terms -> {out_dir}")


def query_index(text: str, index_dir: Path, top_k: int = 5) -> list[tuple[str, float]]:
    embeddings = np.load(index_dir / "embeddings.npy")
    terms = json.loads((index_dir / "terms.json").read_text())
    model = SentenceTransformer(MODEL_NAME)
    query_vec = model.encode([text], normalize_embeddings=True)[0]
    scores = embeddings @ query_vec  # cosine similarity, since both sides are normalized
    top_idx = np.argsort(-scores)[:top_k]
    return [(terms[i], float(scores[i])) for i in top_idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["api", "file"], default="api")
    ap.add_argument("--terms-file", default=None, help="required if --source file")
    ap.add_argument("--limit", type=int, default=2000, help="max terms to pull from openFDA")
    ap.add_argument("--out", default="data/meddra_index", help="output index directory")
    ap.add_argument("--query", default=None, help="skip building, just query an existing index")
    ap.add_argument("--index", default="data/meddra_index", help="index dir to query against")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    if args.query:
        results = query_index(args.query, Path(args.index), args.top_k)
        for term, score in results:
            print(f"{score:.4f}  {term}")
        return

    if args.source == "api":
        terms = fetch_terms_from_openfda(args.limit)
    else:
        if not args.terms_file:
            raise SystemExit("--terms-file required when --source file")
        terms = [t for t, _ in load_terms_from_file(args.terms_file)]

    build_index(terms, Path(args.out))


if __name__ == "__main__":
    main()
