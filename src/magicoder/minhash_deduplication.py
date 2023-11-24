"""Migrated from: https://github.com/bigcode-project/bigcode-dataset. License: Apache 2.0"""

from __future__ import annotations

import gc
import hashlib
import logging
import multiprocessing as mp
import os
import random
import re
import struct
import time
import warnings

warnings.warn(
    "This deduplication strategy is not verified to work. We did not use this in our experiments."
)
choice = input("Are you sure you want to continue? [y/n]")
if choice.lower().strip() != "y":
    exit()

from collections import defaultdict
from itertools import tee
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, cast

from magicoder.utils import write_jsonl

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from dataclasses import dataclass, field

    import datasets
    import numpy as np
    from datasets import load_dataset
    from scipy.integrate import quad as integrate
    from tqdm import tqdm
    from transformers import HfArgumentParser


SEED = 42
NON_ALPHA = re.compile("[^A-Za-z_0-9]")
RNG = np.random.RandomState(SEED)
MAX_HASH = np.uint64((1 << 32) - 1)
MERSENNE_PRIME = np.uint64((1 << 61) - 1)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
datasets.logging.set_verbosity_error()


def ngrams(sequence: List[str], n: int, min_ngram_size: int) -> Iterable:
    """
    Directly taken from nltk package to avoid dependency.

    Parameters
    ----------
    sequence : list
        The sequence of items to be n-grammed.
    n : int
        The order of the n-grams to be extracted.
    min_ngram_size : int
        The minimum size of n-grams.

    Returns
    -------
    Iterable
        The n-grams generated from the sequence.
    """
    if len(sequence) < min_ngram_size:
        return []
    iterables = tee(sequence, n)
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)


def sha1_hash32(data):
    """
    Directly taken from datasketch package to avoid dependency.

    Parameters
    ----------
    data : bytes

    Returns
    -------
    int
    """
    return struct.unpack("<I", hashlib.sha1(data).digest()[:4])[0]


def embed_func(
    data: dict,
    idx: int,
    *,
    num_perm: int,
    columns: list[str],
    ngram_size: int,
    hashranges: List[Tuple[int, int]],
    permutations: np.ndarray,
    min_ngram_size: int = 5,
) -> Dict[str, Any]:
    """
    Combined with some datasketch code to better parallelize computation.

    Parameters
    ----------
    content : str
        The content to be embedded.
    idx : int
        The index of the content.
    num_perm : int
        The number of permutations.
    ngram_size : int
        The size of n-grams.
    hashranges : List[Tuple[int, int]]
        The ranges of hash values.
    permutations : np.ndarray
        The permutations for the minhash.
    min_ngram_size : int
        The minimum size of n-grams.

    Returns
    -------
    Dict[str, Any]
        The hash values in each range and the index.
    """
    content = "\n\n".join(data[column] for column in columns)
    hashvalues = np.ones(num_perm, dtype=np.uint64) * MAX_HASH
    tokens = {
        " ".join(t)
        for t in ngrams(NON_ALPHA.split(content), ngram_size, min_ngram_size)
    }
    hv = np.array(
        [sha1_hash32(token.encode("utf-8")) for token in tokens], dtype=np.uint64
    )  # noqa: E501
    a, b = permutations
    phv = np.bitwise_and(
        ((hv * np.tile(a, (len(hv), 1)).T).T + b) % MERSENNE_PRIME, MAX_HASH
    )  # noqa: E501
    hashvalues = np.vstack([phv, hashvalues]).min(axis=0)
    Hs = [bytes(hashvalues[start:end].byteswap().data) for start, end in hashranges]
    return {"__signatures__": Hs, "__id__": idx}


def optimal_param(
    threshold: float,
    num_perm: int,
    false_positive_weight: float = 0.5,
    false_negative_weight: float = 0.5,
):
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative, taken from datasketch.

    Parameters
    ----------
    threshold : float
        The threshold for similarity.
    num_perm : int
        The number of permutations.
    false_positive_weight : float
        The weight of false positive.
    false_negative_weight : float
        The weight of false negative.

    Returns
    -------
    Tuple[int, int]
        The optimal `b` and `r` parameters.
        The number of bands, and the number of rows per band respectively.
    """

    def false_positive_probability(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - s ** float(r)) ** float(b)

        a, _ = integrate(proba, 0.0, threshold)
        return a

    def false_negative_probability(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - (1 - s ** float(r)) ** float(b))

        a, _ = integrate(proba, threshold, 1.0)
        return a

    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = false_positive_probability(threshold, b, r)
            fn = false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


class UnionFind:
    def __init__(self):
        self.parent: Dict[int, int] = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        self.parent[px] = self.parent[py] = min(px, py)


@dataclass(frozen=True)
class Args:
    dataset: str = field(default="json", metadata={"help": "The dataset to use"})
    config: str = field(default="default", metadata={"help": "Dataset config"})
    split: str = field(default="train", metadata={"help": "Dataset split"})
    data_files: list[str] | None = field(
        default=None, metadata={"help": "Dataset data files (e.g., jsonl files)"}
    )  # noqa: E501
    output_file: str | None = field(default=None, metadata={"help": "Output file"})
    data_dir: str | None = field(
        default=None, metadata={"help": "Dataset data directory"}
    )
    revision: str = field(default="main", metadata={"help": "Dataset revision"})
    columns: list[str] = field(
        default_factory=list, metadata={"help": "Dataset columns"}
    )
    cache_dir: str = field(default=".cache", metadata={"help": "Cache directory"})
    ngram_size: int = field(
        default=5, metadata={"help": "The ngram size to use for MinHash"}
    )
    num_perm: int = field(default=256, metadata={"help": "Number of permutations"})
    threshold: float = field(default=0.7, metadata={"help": "Minhash threshold"})
    min_ngram_size: int = field(
        default=10, metadata={"help": "Shorter documents will be removed"}
    )
    output: str | None = field(
        default=None, metadata={"help": "Store the deduplicated dataset"}
    )


if __name__ == "__main__":
    args = cast(Args, HfArgumentParser(Args).parse_args_into_dataclasses()[0])

    global uf
    mp.set_start_method("fork", force=True)
    uf = UnionFind()

    OUTPUT_BASE = Path(args.output or "output")
    if args.output_file is not None:
        OUTPUT_BASE.mkdir(exist_ok=True, parents=True)
    output = OUTPUT_BASE / "deduplicated"

    logging.basicConfig(level=logging.INFO)

    time_measures = {}
    start_time = time.time()

    B, R = optimal_param(args.threshold, args.num_perm)
    HASH_RANGES = [(i * R, (i + 1) * R) for i in range(B)]
    HASH_TABLES: list = [defaultdict(set) for _ in range(B)]

    time_measures["load_dataset"] = time.time()
    ds = load_dataset(
        "json",
        args.config,
        data_dir=args.data_dir,
        data_files=args.data_files,
        split=args.split,
        # use_auth_token=True,
        cache_dir=args.cache_dir,
        revision=args.revision,
        num_proc=os.cpu_count(),
    )
    time_measures["load_dataset"] = time.time() - time_measures["load_dataset"]
    DATA_SIZE = len(ds)
    PERMUTATIONS = np.array(
        [
            (
                RNG.randint(1, MERSENNE_PRIME, dtype=np.uint64),
                RNG.randint(0, MERSENNE_PRIME, dtype=np.uint64),
            )
            for _ in range(args.num_perm)
        ],
        dtype=np.uint64,
    ).T

    time_measures["minhash"] = time.time()
    embedded = ds.map(
        function=embed_func,
        fn_kwargs={
            "num_perm": args.num_perm,
            "hashranges": HASH_RANGES,
            "ngram_size": args.ngram_size,
            "permutations": PERMUTATIONS,
            "min_ngram_size": args.min_ngram_size,
            "columns": args.columns,
        },
        # input_columns=args.columns,
        remove_columns=ds.column_names,
        num_proc=os.cpu_count(),
        with_indices=True,
        desc="Fingerprinting...",
    )
    time_measures["minhash"] = time.time() - time_measures["minhash"]

    time_measures["clustering"] = time.time()
    batch_size: int = 10000
    for i in tqdm(
        range(0, len(embedded), batch_size),
        dynamic_ncols=True,
        desc="Iterating MinHashes...",  # noqa: E501
    ):
        batch = embedded[i : i + batch_size]
        for key, Hs in zip(batch["__id__"], batch["__signatures__"]):
            for H, hashtable in zip(Hs, HASH_TABLES):
                hashtable[H].add(key)
    for table in tqdm(HASH_TABLES, dynamic_ncols=True, desc="Clustering..."):
        for cluster in table.values():
            if len(cluster) <= 1:
                continue
            idx = min(cluster)
            for x in cluster:
                uf.union(x, idx)
    time_measures["clustering"] = time.time() - time_measures["clustering"]

    time_measures["filtering"] = time.time()
    gc.freeze()
    gc.disable()
    ds = ds.map(
        function=lambda _, idx: {"__cluster__": uf.find(idx)},
        with_indices=True,
        num_proc=os.cpu_count(),
        new_fingerprint=str(random.getrandbits(128)),
        desc="Finding clusters...",
    )
    gc.enable()
    gc.collect()
    # This is where the deduplication happens
    # Since there is no easy groupby in datasets
    # I will use this simple filter for now
    final_data = ds.filter(
        function=lambda record, idx: record["__cluster__"] == idx,
        with_indices=True,
        num_proc=os.cpu_count(),
        desc="Filtering clusters...",
    )
    time_measures["filtering"] = time.time() - time_measures["filtering"]

    time_measures["save"] = time.time()
    final_data = final_data.remove_columns(["__cluster__"])
    if args.output_file is not None:
        write_jsonl(Path(args.output_file), final_data)
    else:
        final_data.save_to_disk(output)

    time_measures["save"] = time.time() - time_measures["save"]

    FINAL_DATA_SIZE = len(final_data)
    DUP_SIZE = DATA_SIZE - FINAL_DATA_SIZE
    PAD = 32

    for key, value in time_measures.items():
        logger.info(f"{key:<{PAD}}: {value:.2f} seconds")
    logger.info(f"{'Data Number (before)':<{PAD}}: {DATA_SIZE}")
    logger.info(
        f"{'Data Number (after)':<{PAD}}: {FINAL_DATA_SIZE} ({FINAL_DATA_SIZE / DATA_SIZE:.2%})"  # noqa: E501
    )
    logger.info(
        f"{'Duplicate Number':<{PAD}}: {DUP_SIZE} ({DUP_SIZE / DATA_SIZE:.2%})"
    )  # noqa: E501
    logger.info(f"{'Total Time':<{PAD}}: {time.time() - start_time:.2f} seconds")
    logger.info(
        f"{'Deduplicated Dataset':<{PAD}}: {output if args.output_file is None else args.output_file}"
    )
    logger.info("🤗 Happy Deduplicating 🤗")
