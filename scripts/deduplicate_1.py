"""
Find near-duplicate text files using Trafilatura SimHash + simple banded LSH.

Usage:
  pip install trafilatura tqdm
  python simhash_dedup.py /path/to/txt/root --threshold 0.90 --out similar_pairs.tsv
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import argparse
import csv
import hashlib
import json
import pickle
import re

from collections import defaultdict
from collections import Counter
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Iterable

from tqdm import tqdm
from trafilatura.deduplication import content_fingerprint


# TODO: use argparse
data_path: str = "/data/data"
output_dir = "/data/artifacts/dedup"
max_bucket_size = 2_500
min_chars: int = 200
threshold: float = 0.85
encoding: str = "utf-8"
bands: int = 4


def normalize_text(text: str) -> str:
    # lowercase, strip out non-alphanumeric, remove stopwords
    text = text.replace("\n", " ")
    text = re.sub(r"[^\w\,\.\?\!\-\&\s]+", "", text)
    text = re.sub(r"[\s]+", " ", text)
    text = text.lower()
    return text.strip()


def iter_txt_files(root: str | Path) -> Iterable[Path]:
    if type(root) is str:
        root = Path(root)
    yield from root.rglob("*.txt")


def simhash64(text: str) -> int:
    # Trafilatura returns a hex string like 'd2ff47ba297cc254'
    fp = content_fingerprint(text)
    return int(fp, 16)


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def simhash_similarity(a: int, b: int, bits: int = 64) -> float:
    return 1.0 - (hamming_distance(a, b) / bits)


def exact_content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def make_band_keys(h: int, bands: int) -> list[tuple[int, int]]:
    """
    Split the 64-bit hash into `bands` chunks. Return (band_index, band_value).

    For threshold T, max_hamming = floor((1 - T) * 64).
    Using max_hamming + 1 bands gives the pigeonhole guarantee:
    if two hashes differ in <= max_hamming bits, at least one band is identical.
    """
    bits = 64
    base_width = bits // bands
    remainder = bits % bands

    keys = []
    shift = 0

    for band_idx in range(bands):
        width = base_width + (1 if band_idx < remainder else 0)
        mask = (1 << width) - 1
        band_value = (h >> shift) & mask
        keys.append((band_idx, band_value))
        shift += width

    return keys


def process_file(path_str: str, encoding: str, min_chars: int):
    path = Path(path_str)

    try:
        raw = path.read_text(encoding=encoding, errors="ignore")
    except Exception as e:
        return {
            "ok": False,
            "path": path_str,
            "error": str(e),
        }

    text = normalize_text(raw)

    if len(text) < min_chars:
        return {
            "ok": False,
            "path": path_str,
            "error": "too_short",
        }

    h = simhash64(text)
    sha1 = exact_content_hash(text)

    return {
        "ok": True,
        "path": path_str,
        "chars": len(text),
        "simhash": h,
        "sha1": sha1,
    }


# dump the processed data
def dump_hash_outputs(
    output_dir: Path,
    docs: list[dict],
    exact_groups: dict,
    buckets: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Human-inspectable metadata.
    docs_jsonl = output_dir / "docs.jsonl"
    with docs_jsonl.open("w", encoding="utf-8") as f:
        for doc_id, doc in enumerate(docs):
            row = {
                "doc_id": doc_id,
                "path": doc["path"],
                "chars": doc["chars"],
                "simhash": f"{doc['simhash']:016x}",
                "sha1": doc["sha1"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Exact duplicate groups only, excluding singleton groups.
    exact_jsonl = output_dir / "exact_groups.jsonl"
    with exact_jsonl.open("w", encoding="utf-8") as f:
        for sha1, doc_ids in exact_groups.items():
            if len(doc_ids) > 1:
                row = {
                    "sha1": sha1,
                    "doc_ids": doc_ids,
                    "paths": [docs[i]["path"] for i in doc_ids],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Bucket dump can be large, so pickle is better than JSON.
    # Keys are tuples like: (band_index, band_value)
    with (output_dir / "buckets.pkl").open("wb") as f:
        pickle.dump(dict(buckets), f, protocol=pickle.HIGHEST_PROTOCOL)

    # Optional compact manifest.
    manifest = {
        "num_docs": len(docs),
        "num_exact_groups_total": len(exact_groups),
        "num_exact_duplicate_groups": sum(
            1 for group in exact_groups.values() if len(group) > 1
        ),
        "num_buckets": len(buckets),
        "files": {
            "docs": str(docs_jsonl),
            "exact_groups": str(exact_jsonl),
            "buckets": str(output_dir / "buckets.pkl"),
        },
    }

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def load_hash_outputs_reconstruct_exact_groups(output_dir: Path):
    output_dir = Path(output_dir)

    docs_path = output_dir / "docs.jsonl"
    buckets_path = output_dir / "buckets.pkl"

    if not docs_path.exists():
        raise FileNotFoundError(f"Missing docs file: {docs_path}")

    if not buckets_path.exists():
        raise FileNotFoundError(f"Missing buckets file: {buckets_path}")

    docs = []
    exact_groups = defaultdict(list)

    with docs_path.open("r", encoding="utf-8") as f:
        for doc_id, line in enumerate(f):
            row = json.loads(line)

            doc = {
                "path": row["path"],
                "chars": row["chars"],
                "simhash": int(row["simhash"], 16),
                "sha1": row["sha1"],
            }

            docs.append(doc)
            exact_groups[doc["sha1"]].append(doc_id)

    with buckets_path.open("rb") as f:
        buckets = pickle.load(f)

    return docs, exact_groups, buckets


def pack_pair(a: int, b: int, n_docs: int) -> int:
    if a > b:
        a, b = b, a
    return a * n_docs + b


def check_matches_streaming(
    docs: list[dict],
    buckets: dict,
    max_hamming: int,
    max_bucket_size: int = 2_000,
):
    hashes = [doc["simhash"] for doc in docs]
    n_docs = len(docs)

    seen_pairs = set()
    matches = []

    raw_pair_occurrences = 0
    unique_pairs_checked = 0
    skipped_buckets = 0

    bucket_doc_lists = sorted(buckets.values(), key=len)

    for bucket_docs in tqdm(bucket_doc_lists, desc="Checking candidates"):
        k = len(bucket_docs)

        if k < 2:
            continue

        if k > max_bucket_size:
            skipped_buckets += 1
            continue

        for a, b in combinations(bucket_docs, 2):
            raw_pair_occurrences += 1

            pair_key = pack_pair(a, b, n_docs)

            if pair_key in seen_pairs:
                continue

            seen_pairs.add(pair_key)
            unique_pairs_checked += 1

            dist = (hashes[a] ^ hashes[b]).bit_count()

            if dist <= max_hamming:
                sim = 1.0 - (dist / 64)
                matches.append((sim, dist, docs[a], docs[b]))

    print(f"raw pair occurrences: {raw_pair_occurrences:,}")
    print(f"unique pairs checked: {unique_pairs_checked:,}")
    print(f"seen pair keys stored: {len(seen_pairs):,}")
    print(f"skipped buckets: {skipped_buckets:,}")
    print(f"matches: {len(matches):,}")

    return matches


if not (0.0 < threshold <= 1.0):
    raise ValueError("--threshold must be in (0, 1].")

max_hamming = int((1.0 - threshold) * 64)

## NO! memory explosion
## At least 1 band; for 0.90 threshold, max_hamming ~= 6, so bands = 7.
# bands = max(1, max_hamming + 1)

files = list(iter_txt_files(data_path))
print(f"Found {len(files):,} .txt files")
print(f"Threshold: {threshold:.3f}")
print(f"Max Hamming distance: {max_hamming}")
print(f"Bands: {bands}")

output_dir = Path(output_dir)
already_proc: bool = False
if (output_dir / "buckets.pkl").is_file() \
    and (output_dir / "docs.jsonl").is_file() \
    and (output_dir / "exact_groups.jsonl").is_file() \
    and (output_dir / "manifest.json").is_file():
    already_proc = True
    docs, exact_groups, buckets = load_hash_outputs_reconstruct_exact_groups(output_dir)
    print("loaded data!")

if already_proc:
    print("skipping, data loaded")
else:
    docs = []
    exact_groups = defaultdict(list)
    buckets = defaultdict(list)

    workers = max(1, os.cpu_count() - 1)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(
                process_file,
                str(path),
                encoding,
                min_chars,
            )
            for path in files
        ]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Reading + hashing"):
            result = fut.result()

            if not result["ok"]:
                if result["error"] not in {"too_short"}:
                    print(f"Skipping file: {result['path']} ({result['error']})")
                continue

            doc_id = len(docs)

            docs.append(
                {
                    "path": result["path"],
                    "chars": result["chars"],
                    "simhash": result["simhash"],
                    "sha1": result["sha1"],
                }
            )

            exact_groups[result["sha1"]].append(doc_id)

            for key in make_band_keys(result["simhash"], bands):
                buckets[key].append(doc_id)


    dump_hash_outputs(output_dir, docs, exact_groups, buckets)
    print(f"Kept {len(docs):,} docs after min length filter")
    print(f"Built {len(buckets):,} non-empty buckets")




bucket_sizes = [len(v) for v in buckets.values()]
size_counts = Counter(bucket_sizes)

print(f"num buckets: {len(bucket_sizes):,}")
print(f"max bucket size: {max(bucket_sizes):,}")
print(f"buckets >= 100: {sum(1 for s in bucket_sizes if s >= 100):,}")
print(f"buckets >= 1_000: {sum(1 for s in bucket_sizes if s >= 1_000):,}")
print(f"buckets >= 10_000: {sum(1 for s in bucket_sizes if s >= 10_000):,}")

largest = sorted(bucket_sizes, reverse=True)[:20]
print("largest bucket sizes:", largest)

estimated_pairs = sum(s * (s - 1) // 2 for s in bucket_sizes if s <= 10_000)
print(f"estimated raw candidate pairs from buckets <= 10k: {estimated_pairs:,}")



bucket_doc_lists = [
    v for v in buckets.values()
    if 2 <= len(v) <= max_bucket_size
]

estimated_pairs = sum(
    len(v) * (len(v) - 1) // 2
    for v in bucket_doc_lists
)

print(f"candidate buckets: {len(bucket_doc_lists):,}")
print(f"estimated raw candidate pairs: {estimated_pairs:,}")

matches = check_matches_streaming(
    docs=docs,
    buckets=buckets,
    max_hamming=max_hamming,
    max_bucket_size=max_bucket_size,
)
matches = sorted(matches, key=lambda x: x[0], reverse=True)

jsonl_matches = [json.dumps(m) for m in matches]
out_file = output_dir / "matches.jsonl"
out_file.write_text('\n'.join(jsonl_matches))

print("continue to part 2")