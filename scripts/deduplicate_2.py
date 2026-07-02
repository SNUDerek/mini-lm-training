"""
Find near-duplicate text files using Trafilatura SimHash + simple banded LSH.

Usage:
  pip install trafilatura tqdm
  python simhash_dedup.py /path/to/txt/root --threshold 0.90 --out similar_pairs.tsv
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import os

import argparse
import csv
import hashlib
import json
import pickle
import re
import time

from collections import deque
from functools import lru_cache
from pathlib import Path
from threading import local

import lmdb

from tqdm import tqdm


# TODO: later, use argparse
output_dir = "/data/artifacts/dedup"
shingle_size = 7
mode = "either"
min_sim_score = 0.90
min_shingles = 10
min_jaccard = 0.90
min_containment = 0.95
lmdb_data_is_loaded: bool = True
# shrink if causing OOMs.
# for me, this is using about <50 GB total
LRU_CACHE_SIZE = 1_600


output_dir = Path(output_dir)
match_file = output_dir / "matches.jsonl"
match_text = match_file.read_text(encoding='utf-8')
matches = [json.loads(s) for s in match_text.split("\n")]
matches = sorted(matches, key=lambda x: x[0], reverse=True)
print(f"loaded {len(matches)} matches")


high_matches = [m for m in matches if m[0] >= min_sim_score]
print(f"checking {len(high_matches)} with score >= {min_sim_score}")

unique_files = set()
for sim_score, _, file1_dict, file2_dict in tqdm(matches):
    path_a = file1_dict["path"]
    path_b = file2_dict["path"]
    unique_files.add(str(path_a))
    unique_files.add(str(path_b))
print(f"targeting {len(unique_files)} files")


def normalize_text(text: str) -> str:
    # lowercase, strip out non-alphanumeric, remove stopwords
    text = text.replace("\n", " ")
    text = re.sub(r"[^\w\,\.\?\!\-\&\s]+", "", text)
    text = re.sub(r"[\s]+", " ", text)
    text = text.lower()
    return text.strip()
    

def word_shingles(tokens: list[str], n: int = 7) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    window = deque(tokens[:n], maxlen=n)
    shingles = {tuple(window)}
    for token in tokens[n:]:
        window.append(token)
        shingles.add(tuple(window))
    return shingles


def load_shingles(path_str: str, n: int) -> frozenset[tuple[str, ...]]:
    path = Path(path_str)
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = normalize_text(text)
    tokens = text.split()
    return frozenset(word_shingles(tokens, n=n))


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0

    inter = len(a & b)
    union = len(a) + len(b) - inter

    if union == 0:
        return 0.0

    return inter / union


def containment(a: set, b: set) -> float:
    """
    Symmetric-ish containment score:
    how much the smaller shingle set is contained in the larger one.
    """
    if not a or not b:
        return 0.0

    inter = len(a & b)
    return inter / min(len(a), len(b))


lmdb_file = (output_dir / "shingles.lmdb")
env = lmdb.open(str(lmdb_file), map_size=1_000 * 1024**3, )  # 1TB ceiling


if lmdb_data_is_loaded:
    print("skipping, data is loaded")
else:
    temp_dict = {}
    for idx, file in enumerate(tqdm(unique_files)):
        shingles: frozenset = load_shingles(file, n=shingle_size)
        fpath = str(file)
        temp_dict[fpath.encode('utf-8')] = pickle.dumps(shingles)
        if idx > 0 and idx % 1000 == 0:
            with env.begin(write=True) as txn:
                for fp, shin in temp_dict.items():
                    txn.put(fp, shin)
            env.sync()
            del temp_dict
            temp_dict = {}
            time.sleep(0.1)

if lmdb_data_is_loaded:
    print("skipping, data is loaded")
else:
    reprocess_files = []

    with env.begin() as txn:

        for fpath in tqdm(unique_files):
            if not txn.get(fpath.encode('utf-8')):
                reprocess_files.append(fpath)

    print(len(set(reprocess_files)))

    with env.begin(write=True) as txn:
        for fpath in tqdm(set(reprocess_files)):
            shingles: frozenset = load_shingles(fpath, n=shingle_size)
            txn.put(fpath.encode('utf-8'), pickle.dumps(shingles))
    env.sync()


_thread_state = local()


def get_thread_txn(env):
    """
    One read transaction per worker thread.

    Do not share a single LMDB transaction object across threads.
    """
    txn = getattr(_thread_state, "txn", None)

    if txn is None:
        txn = env.begin(write=False)
        _thread_state.txn = txn

    return txn


@lru_cache(LRU_CACHE_SIZE)
def load_lmdb_shingles(txn, path: str):
    raw = txn.get(path.encode("utf-8"))

    if raw is None:
        raise KeyError(f"Missing shingles in LMDB for path: {path}")

    return pickle.loads(raw)


def accept_match_record(match):
    sim_score, _, file1_dict, file2_dict = match

    if sim_score < min_sim_score:
        return None

    txn = get_thread_txn(env)

    path_a = file1_dict["path"]
    path_b = file2_dict["path"]

    shingles_a = load_lmdb_shingles(txn, path_a)
    shingles_b = load_lmdb_shingles(txn, path_b)

    # Optional: skip tiny docs, if useful.
    if min(len(shingles_a), len(shingles_b)) < min_shingles:
        return None

    jac = jaccard(shingles_a, shingles_b)
    cont = containment(shingles_a, shingles_b)

    if mode == "jaccard":
        accept = jac >= min_jaccard
    elif mode == "containment":
        accept = cont >= min_containment
    elif mode == "both":
        accept = jac >= min_jaccard and cont >= min_containment
    else:
        accept = jac >= min_jaccard or cont >= min_containment

    if not accept:
        return None

    metadict = {}

    metadict.update({
        f"{key}_a": value
        for key, value in file1_dict.items()
    })

    metadict.update({
        f"{key}_b": value
        for key, value in file2_dict.items()
    })

    metadict.update({
        "simhash_similarity": sim_score,
        "word_shingle_jaccard": jac,
        "word_shingle_containment": cont,
        "len_shingles_a": len(shingles_a),
        "len_shingles_b": len(shingles_b),
    })

    return metadict


matches.sort(key=lambda x: (x[2]['path'], x[3]['path']))


accepted_matches = []
kept = 0
total = 0

for match in tqdm(matches, desc="Scoring matches"):
    total += 1
    result = accept_match_record(match)
    if result is None:
        continue
    accepted_matches.append(result)
    kept += 1
    if total % 2_000 == 0:
        print(f"processed={total:,}, kept={kept:,}")

print(f"processed: {total:,}")
print(f"kept: {kept:,}")


result_jsonl = [json.dumps(r) for r in accepted_matches]
(output_dir / "accepted_matches.jsonl").write_text("\n".join(result_jsonl))

env.close()