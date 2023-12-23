import json
import random
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import HfArgumentParser

import magicoder


@dataclass(frozen=True)
class Args:
    max_seeds_to_collect: int = field(default=50000)
    continue_from: str | None = field(default=None)

    # Keep the following arguments unchanged for reproducibility
    seed: int = field(default=976)

    min_lines: int = field(default=1)
    max_lines: int = field(default=15)
    max_fragments: int = field(default=3)
    chunk_size: int = field(default=1000)

    dataset_name: str = field(default="bigcode/starcoderdata")
    data_dir: str | None = field(default="python")
    max_considered_data: int | None = field(default=150000)

    tag: str = field(
        default="",
        metadata={
            "help": "Custom tag as part of the output filename, not affecting the fingerprint"
        },
    )
    text_wrap: int | None = field(default=None)

    def fingerprint(self) -> str:
        # The combination of arguments can uniquely determine the generation process
        args = [
            self.seed,
            self.min_lines,
            self.max_lines,
            self.max_fragments,
            self.chunk_size,
            self.dataset_name,
            self.data_dir,
            self.max_considered_data,
        ]
        # for backward compatibility, only add if needed
        if self.text_wrap is not None:
            args.append(self.text_wrap)
        return magicoder.utils.compute_fingerprint(*args, hash_length=5)


def fragments_to_text(fragments: list[str]) -> str:
    return "...\n".join(fragments)


def map_dataset(examples: dict, indices: list[int], args: Args) -> dict:
    random.seed(args.seed + indices[0])
    seed_fragments = [
        extract_fragments(args, content) for content in examples["content"]
    ]
    seed = [
        (fragments_to_text(fragments) if fragments is not None else None)
        for fragments in seed_fragments
    ]
    return {
        "seed": seed,
        "raw_index": indices,
    }


def uniform_partition(n: int, k: int) -> list[int]:
    """Partition n into k non-negative integers. Stars and bars method.
    x1 + x2 + ... + xk = n; xi >= 0. Can be transformed to positive case:
    y1 + y2 + ... + yk = n - k; yi = xi + 1 > 0"""
    assert n >= 0, "n should be >= 0"
    assert k > 0, " should be > 0"
    random_numbers = [random.randint(0, n) for _ in range(k - 1)]
    values = [0] + sorted(random_numbers) + [n]
    intervals = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    assert sum(intervals) == n
    assert len(intervals) == k
    return intervals


def uniform_partition_positive(n: int, k: int) -> list[int]:
    return [x + 1 for x in uniform_partition(n - k, k)]


def is_en(content: str, seed: int) -> bool:
    import langdetect

    langdetect.DetectorFactory.seed = seed
    try:
        return langdetect.detect(content) == "en"
    except langdetect.lang_detect_exception.LangDetectException:
        return False


def place_blocks(n: int, sizes: list[int]) -> list[int]:
    """Randomly place k blocks of sizes `sizes` in a line of length n. Return the starting positions."""
    assert n >= 0, "n should be >= 0"
    k = len(sizes)
    assert k > 0, "k should be > 0"
    assert sum(sizes) <= n, "sum(sizes) should be <= n"
    if k == 1:
        return [random.randint(0, n - sizes[0])]
    all_but_last_pos = place_blocks(n - sizes[-1], sizes[:-1])
    last_pos = random.randint(all_but_last_pos[-1] + sizes[-2], n - sizes[-1])
    result = all_but_last_pos + [last_pos]
    assert len(result) == k
    for i in range(k - 1):
        assert result[i] + sizes[i] <= result[i + 1]
    return result


def extract_fragments(args: Args, document: str) -> list[str] | None:
    if args.text_wrap is not None:
        document = textwrap.fill(
            document,
            width=args.text_wrap,
            replace_whitespace=False,
            expand_tabs=False,
            break_on_hyphens=False,
            drop_whitespace=False,
            break_long_words=False,
        )
    if args.data_dir == "markdown" and not is_en(document, args.seed):
        return None
    lines = document.splitlines(keepends=True)

    # special rules
    if args.data_dir == "jupyter-scripts-dedup-filtered":
        lines = [
            line
            for line in lines
            if "jupyter" not in line.lower() and "jupytext" not in line.lower()
        ]
    elif args.data_dir == "markdown":
        lines = [
            line
            for line in lines
            if "http:" not in line and "https:" not in line and "www." not in line
        ]

    lines = [line for line in lines if line.strip() != ""]

    if len(lines) < args.min_lines or len(lines) == 0:
        return None
    max_lines = min(args.max_lines, len(lines))
    assert max_lines >= args.min_lines
    n_lines_to_consider = random.randint(args.min_lines, max_lines)
    max_fragments = min(n_lines_to_consider, args.max_fragments)
    n_fragments = random.randint(1, max_fragments)
    fragment_sizes = uniform_partition_positive(n_lines_to_consider, n_fragments)
    fragment_indices = place_blocks(len(lines), fragment_sizes)
    fragments = [
        "".join(lines[i : i + size])
        for i, size in zip(fragment_indices, fragment_sizes)
    ]
    # random.shuffle(fragments)
    return fragments


def main():
    args = cast(Args, HfArgumentParser(Args).parse_args_into_dataclasses()[0])
    split = (
        f"train[:{args.max_considered_data}]"
        if args.max_considered_data is not None
        else "train"
    )
    dataset: Dataset = load_dataset(
        args.dataset_name,
        data_dir=args.data_dir,
        split=split,
        num_proc=magicoder.utils.N_CORES,
    )
    random.seed(args.seed)
    # map_fn = get_map_dataset(args)
    num_proc = magicoder.utils.N_CORES if args.data_dir == "markdown" else None
    dataset = dataset.map(
        function=map_dataset,
        fn_kwargs=dict(args=args),
        with_indices=True,
        batched=True,
        num_proc=num_proc,
        batch_size=args.chunk_size,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    )
    dataset = dataset.shuffle(seed=args.seed)

    # Every run should produce the same data as long as the default params are not changed
    data_fingerprint = args.fingerprint()
    timestamp = magicoder.utils.timestamp()
    tag = "" if args.tag == "" else f"-{args.tag}"
    path = Path(f"data-seed{tag}-{data_fingerprint}-{timestamp}.jsonl")
    assert not path.exists()
    f_out = path.open("w")
    print("Saving to", path)

    n_success = 0
    all_seed_texts = set[str]()

    def get_seed_text(seed: str) -> str:
        lines = seed.splitlines()
        lines = [line for line in lines if not line.startswith("<fragment")]
        text = "".join("".join(lines).split())
        return text

    pbar = tqdm(total=args.max_seeds_to_collect)
    for example in dataset:
        if n_success >= args.max_seeds_to_collect:
            break
        if example["seed"] is None:
            continue
        seed_text = get_seed_text(example["seed"])
        # remove those with only symbols
        if all(not c.isalpha() for c in seed_text):
            # print("[filter(symbols Only)]", example["seed"], sep="\n")
            continue
        if seed_text in all_seed_texts:
            # print("[filter(duplicate)]", example["seed"], sep="\n")
            continue
        all_seed_texts.add(seed_text)
        data = dict(
            raw_index=example["raw_index"],
            seed=example["seed"],
        )
        # print("[Seed]", example["seed"], sep="\n")
        f_out.write(json.dumps(data) + "\n")
        n_success += 1
        pbar.update(1)


if __name__ == "__main__":
    main()
