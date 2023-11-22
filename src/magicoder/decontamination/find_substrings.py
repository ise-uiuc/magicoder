# type: ignore
"""Migrated from: https://github.com/bigcode-project/bigcode-dataset. License: Apache 2.0"""

import argparse
import json
import os
import shutil
from copy import deepcopy
from glob import glob
from pathlib import Path

from datasets import load_dataset

from magicoder.utils import write_jsonl

from .benchmark_data import FILTER_OUT
from .utils import add_dict, shard_dataset

SHARD_SIZE = 1000 << 20  # 1GB
LANGUAGE_COL = "lang"
# LANGUAGES = ["Python", "Java", "JavaScript"]


def dump_benchmarks(file_path: str):
    """
    Dump the dictionary of benchmark samples that are filtered out
    """
    with open(file_path, "w") as f:
        json.dump(FILTER_OUT, f, indent=2)


def filter_reason_to_benchmark_name(filter_reason: str):
    assert filter_reason.endswith("_match")
    return filter_reason[:-6]


def benchmark_name_to_filter_reason(benchmark_name: str):
    return f"{benchmark_name}_match"


def update_benchmark_dict(
    filter_out: dict, benchmark_cache: str, excluded_data_cache: str
):
    """
    Iterates on current benchmark-samples. If a sample is found in the cached benchmark-samples, it is removed (it does not need to be searched),
    and the corresponding data-samples from the cache are added to `exclude_data`

    Returns:
    - `updated`: an updated benchmark dict where samples from the cache are removed (they do not need to be searched anymore)
    - `exclude_data`: a list of files to remove from the dataset
    """
    updated = deepcopy(filter_out)
    exclude_data = []
    with open(benchmark_cache) as f:
        benchmark_cache = json.load(f)
    with open(excluded_data_cache) as f:
        excluded_data_cache = json.load(f)

    for bench, samples in filter_out.items():
        for bench_sample in samples:
            # Benchmark-sample was found in cache
            if bench in benchmark_cache and bench_sample in benchmark_cache[bench]:
                # No need to search for this sample in the dataset
                updated[bench].remove(bench_sample)
                # Corresponding data-samples will be excluded from the dataset.
                exclude_data += [
                    data_sample
                    for data_sample in excluded_data_cache
                    if data_sample["filter_reason"]
                    == benchmark_name_to_filter_reason(bench)
                    and data_sample["matched_substring"] == bench_sample
                ]

    print("After loading cache, will search for:")
    for benchmark, values in updated.items():
        print(f"  num strings from {benchmark}: {len(values)}")
    # Remove empty benchmarks
    updated = {key: value for key, value in updated.items() if len(value) > 0}
    return updated, exclude_data


def find_substrings(data, columns, filter_out, return_matched=False):
    """
    filter_out: Dict[str, List[str]] mapping from benchmark name to list of strings that need to be
    filtered-out.
    Return True, None if the file should be included in the dataset.
    Otherwise return False and some metadata about the file excluded
    """
    content = "\n\n".join([data[col].lower() for col in columns])
    # For each substring, try to find it in the file (case insensitive)
    for benchmark, substrings in filter_out.items():
        for substring in substrings:
            if substring.lower() in content:
                if return_matched:
                    return False, benchmark_name_to_filter_reason(benchmark), substring
                else:
                    return False, benchmark_name_to_filter_reason(benchmark)

    # Return True, None if none of the substrings was found
    if return_matched:
        return True, None, None
    else:
        return True, None


def aggregate_meta(tmp_meta_dir: str):
    res = {}
    for file in glob(f"{tmp_meta_dir}/*-meta.json"):
        with open(file, "r") as f:
            meta = json.load(f)
        add_dict(res, meta)
    return res


def concatenate_meta(tmp_meta_dir: str):
    res = []
    for file in glob(f"{tmp_meta_dir}/*-excluded-data.json"):
        with open(file, "r") as f:
            meta = json.load(f)
        res += meta
    return res


class Meta:
    def __init__(self) -> None:
        self.meta_dict = dict()

    def update(self, lang: str, filter_reason: str):
        if lang not in self.meta_dict:
            self.meta_dict[lang] = {}
        if filter_reason not in self.meta_dict[lang]:
            self.meta_dict[lang][filter_reason] = 0
        self.meta_dict[lang][filter_reason] += 1


class SubstringFilterer(object):
    def __init__(
        self,
        output_dir: str,
        output_file: str,
        cached_decontamination_dir: str,
        split_languages: bool,
        cache_retrieval_key: str,
        columns: list[str],
        tmp_meta_dir=None,
        data_dir=None,
    ) -> None:
        self.output_dir = output_dir
        self.output_file = output_file
        self.split_languages = split_languages
        self.cache_retrieval_key = cache_retrieval_key
        self.columns = columns
        self.tmp_meta_dir = (
            tmp_meta_dir if tmp_meta_dir is not None else f"{output_dir}/tmp/meta"
        )
        self.data_dir = data_dir if data_dir is not None else f"{output_dir}/data"
        os.makedirs(self.tmp_meta_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        # Save benchmark data
        self.excluded_data_cache = os.path.join(self.output_dir, "excluded-data.json")
        self.benchmarks_cache = os.path.join(output_dir, "benchmarks.json")
        dump_benchmarks(self.benchmarks_cache)

        if cached_decontamination_dir is not None:
            # Load cache
            self.filter_out, self.exclude_data = update_benchmark_dict(
                FILTER_OUT,
                os.path.join(cached_decontamination_dir, "benchmarks.json"),
                os.path.join(cached_decontamination_dir, "excluded-data.json"),
            )
            # All hashes should be unique
            hash_list = [
                data_sample["data"][self.cache_retrieval_key]
                for data_sample in self.exclude_data
            ]
            assert len(hash_list) == len(set(hash_list))
            # dict: retrieval-key (hash/content) -> data-sample
            self.exclude_data_index = {
                data_sample["data"][self.cache_retrieval_key]: data_sample
                for data_sample in self.exclude_data
            }
            self.use_cached_decontamination = True
        else:
            self.filter_out = FILTER_OUT
            self.exclude_data = None
            self.exclude_data_index = {}
            self.use_cached_decontamination = False

    def _filter_file(self, sample):
        should_include, filter_reason, matched_substring = True, None, None
        if self.use_cached_decontamination:
            # According to cache, this data sample should be excluded
            if sample[self.cache_retrieval_key] in self.exclude_data_index:
                should_include = False
                filter_reason = self.exclude_data_index[
                    sample[self.cache_retrieval_key]
                ]["filter_reason"]
                matched_substring = self.exclude_data_index[
                    sample[self.cache_retrieval_key]
                ]["matched_substring"]
        # If sample has passed the cache, check the other substrings
        if should_include:
            should_include, filter_reason, matched_substring = find_substrings(
                sample, self.columns, self.filter_out, return_matched=True
            )
        return should_include, filter_reason, matched_substring

    def _filter(self, batch: dict, idx):
        meta = Meta()
        excluded_data = []
        features = batch.keys()
        res = {k: [] for k in features}
        for sample in zip(*[batch[k] for k in features]):
            sample = {k: v for k, v in zip(features, sample)}
            should_include, filter_reason, matched_substring = self._filter_file(sample)
            if not should_include:
                meta.update(sample.get(LANGUAGE_COL, "unknown"), filter_reason)
                excluded_data.append(
                    {
                        "data": sample,
                        "filter_reason": filter_reason,
                        "matched_substring": matched_substring,
                    }
                )
            else:
                # Add to output
                for k in features:
                    res[k].append(sample[k])

        # Record Meta
        with open(
            os.path.join(self.tmp_meta_dir, f"{idx[0]}-{idx[-1]}-meta.json"), "w"
        ) as f:
            json.dump(meta.meta_dict, f)
        with open(
            os.path.join(self.tmp_meta_dir, f"{idx[0]}-{idx[-1]}-excluded-data.json"),
            "w",
        ) as f:
            json.dump(excluded_data, f, indent=2)
        return res

    def filter_dataset(self, ds, num_proc, batch_size):
        filtered = ds.map(
            self._filter,
            batched=True,
            batch_size=batch_size,
            with_indices=True,
            num_proc=num_proc,
            load_from_cache_file=False,
        )
        print("Number of samples in the new dataset: ", len(filtered))
        return filtered

    def finalize(self):
        # Dump meta
        meta = aggregate_meta(self.tmp_meta_dir)
        print(meta)
        with open(os.path.join(self.output_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        # Dump excluded-data.json
        meta = concatenate_meta(self.tmp_meta_dir)
        print("Number of excluded examples: ", len(meta))
        with open(self.excluded_data_cache, "w") as f:
            json.dump(meta, f, indent=2)
        # delete temporary meta data
        shutil.rmtree(self.tmp_meta_dir)

    # def save(self, filtered, num_proc):
    #     # Save shards
    #     if self.split_languages:
    #         for lang in LANGUAGES:
    #             print(f"Sharding subset: {lang}")
    #             target_dir = os.path.join(self.data_dir, lang.lower())
    #             os.makedirs(target_dir, exist_ok=True)
    #             subset = filtered.filter(lambda example: example[LANGUAGE_COL] == lang, num_proc=num_proc)
    #             shard_dataset(subset, SHARD_SIZE, target_dir, num_proc=16)
    #     else:
    #         shard_dataset(filtered, SHARD_SIZE, self.data_dir, num_proc=16)

    def run(self, dataset, num_proc, batch_size):
        filtered = self.filter_dataset(dataset, num_proc, batch_size)
        write_jsonl(Path(self.output_file), filtered)
        # Finalize meta-data
        self.finalize()
        # Save filtered dataset.
        # NOTE: we save to jsonl so this is not needed
        # self.save(filtered, num_proc)
        return filtered


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        default="json",
        type=str,
        help="Name or path of the HF dataset to decontaminate",
    )
    parser.add_argument("--data_files", nargs="+", default=None, help="Data files")
    parser.add_argument(
        "--columns",
        nargs="+",
        required=True,
        help="Columns to form the text to search for",
    )
    parser.add_argument(
        "--output_file", required=True, type=str, help="Path to save output jsonl data"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Path to save output data and metadata",
    )
    parser.add_argument("--num_proc", type=int, default=200, help="Number of processes")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Size of batches passed to Dataset.map",
    )
    parser.add_argument(
        "--cached_decontamination_dir",
        type=str,
        default=None,
        help="Directory containing a `benchmarks.json` and `excluded_data.json` files from a previous decontamination run."
        "Will use this data to avoid searching again for strings that were previously decontaminated."
        "It's up to the user to ensure that the dataset being decontaminated is a subset of the one from the cached decontamination run"
        "(Otherwise not all the benchmark samples will be checked against new data samples)",
    )
    parser.add_argument(
        "--cache_retrieval_key",
        type=str,
        default="hexsha",
        help="Key used to retrieve examples from the cache. Ideally `hexsha`. Otherwise, another unique feature in case the hash is not present, like `content`)",
    )
    parser.add_argument(
        "--split_languages",
        action="store_true",
        help="If True, will create one subfolder per language for the output dataset.",
    )
    return parser.parse_args()


def main():
    args = arguments()

    filterer = SubstringFilterer(
        output_dir=args.output_dir,
        output_file=args.output_file,
        columns=args.columns,
        cached_decontamination_dir=args.cached_decontamination_dir,
        split_languages=args.split_languages,
        cache_retrieval_key=args.cache_retrieval_key,
    )

    ds = load_dataset(
        args.dataset_name,
        split="train",
        data_files=args.data_files,
        # chunksize=40 << 20
    )

    filterer.run(ds, args.num_proc, args.batch_size)


if __name__ == "__main__":
    main()
