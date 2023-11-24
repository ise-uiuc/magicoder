import itertools
import json
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from tqdm.auto import tqdm
from transformers import HfArgumentParser

from magicoder.utils import read_jsonl, write_jsonl


@dataclass(frozen=True)
class Args:
    data_files: list[str]
    output_file: str
    analysis_dir: str | None = field(
        default=None,
        metadata={
            "help": "The path to the directory containing the analysis of the filtering process. If not provided, no analysis will be performed."
        },
    )
    no_filter: bool = field(
        default=False,
        metadata={
            "help": "Do not filter the data, but randomize the order of the data in the same way as the filtering process."
        },
    )
    seed: int = field(default=666)


def filter_same_seed_problem_solution(
    raw_data: list[dict],
) -> tuple[list[dict], list[dict]]:
    chosen_data: list[dict] = []
    seeds: set[str] = set()
    problems: set[str] = set()
    solutions: set[str] = set()
    rejected_data: list[dict] = []
    for d in tqdm(raw_data, desc="Filtering same seed, problem, and solution"):
        seed = remove_all_whitespaces(d["seed"])
        problem = remove_all_whitespaces(d["problem"])
        solution = remove_all_whitespaces(d["solution"])
        if seed not in seeds and problem not in problems and solution not in solutions:
            chosen_data.append(d)
            seeds.add(seed)
            problems.add(problem)
            solutions.add(solution)
        else:
            reason = (
                "duplicate seeds"
                if seed in seeds
                else "duplicate problems"
                if problem in problems
                else "duplicate solutions"
            )
            rejected_data.append(dict(reason=reason, **d))
    return chosen_data, rejected_data


def remove_all_whitespaces(text: str) -> str:
    return "".join(text.split())


def detect_codeblocks(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    codeblocks: list[str] = []
    start_index: int | None = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("```"):
            if start_index is None:
                start_index = idx
            else:
                codeblocks.append("".join(lines[start_index + 1 : idx]))
                start_index = None
    return codeblocks


def filter_same_codeblocks(raw_data: list[dict]) -> tuple[list[dict], list[dict]]:
    """Filter out data whose solution just copies the problem."""
    chosen_data: list[dict] = []
    rejected_data: list[dict] = []
    for d in tqdm(raw_data, desc="Filtering same codeblocks"):
        problem_codeblocks = list(
            map(remove_all_whitespaces, detect_codeblocks(d["problem"]))
        )
        solution_codeblocks = list(
            map(remove_all_whitespaces, detect_codeblocks(d["solution"]))
        )
        iter = itertools.product(problem_codeblocks, solution_codeblocks)
        if any(p == s for p, s in iter):
            rejected_data.append(dict(reason="Solution copies the problem", **d))
            continue
        chosen_data.append(d)
    return chosen_data, rejected_data


ALL_LANGS = [
    "python",
    "typescript",
    "csharp",
    "rust",
    "swift",
    "php",
    "java",
    "cpp",
    "shell",
]


def save_analysis(chosen_data: list[dict], rejected_data: list[dict], output_dir: Path):
    """Save to `output_dir` the analysis of the filtering process:
    - How many data are filtered out for each language?
    - How many data are filtered out for each reason?
    - Examples of filtered data for each reason in each language
    - Data that are filtered"""
    # Data that are filtered
    rejected_data = sorted(rejected_data, key=lambda x: x["reason"])
    write_jsonl(output_dir / "rejected_data.jsonl", rejected_data)
    chosen_data_dict = dict[str, list[dict]]()
    rejected_data_dict = dict[str, list[dict]]()
    for d in chosen_data:
        chosen_data_dict.setdefault(d["lang"], []).append(d)
    for d in rejected_data:
        rejected_data_dict.setdefault(d["lang"], []).append(d)
    all_langs = set(chosen_data_dict.keys()) | set(rejected_data_dict.keys())
    all_reasons = set(d["reason"] for d in rejected_data)
    # - How many data are filtered out for each language?
    # - How many data are filtered out for each reason?
    analysis_dict = {
        "overall": {
            "total": len(chosen_data) + len(rejected_data),
            "chosen": len(chosen_data),
            "rejected": len(rejected_data),
            "chosen_ratio": f"{len(chosen_data) / (len(chosen_data) + len(rejected_data)):.2f}",
        },
        "lang": {
            lang: dict(
                total=(chosen_len := len(chosen_data_dict.get(lang, [])))
                + (rejected_len := len(rejected_data_dict.get(lang, []))),
                chosen=chosen_len,
                rejected=rejected_len,
                chosen_ratio=f"{chosen_len / (chosen_len + rejected_len):.2f}",
            )
            for lang in all_langs
        },
        "reason": {
            reason: sum(1 for d in rejected_data if d["reason"] == reason)
            for reason in set(all_reasons)
        },
    }
    (output_dir / "analysis.json").write_text(json.dumps(analysis_dict, indent=2))
    # Examples of filtered data for each reason in each language
    max_examples_per_reason = 5
    examples_dir = output_dir / "examples"
    examples_dir.mkdir()
    for lang in all_langs:
        for reason in all_reasons:
            examples = [
                f"[Seed]\n{d['seed']}\n\n[Prompt]\n\n[Problem]\n{d['problem']}\n\n[Solution]\n{d['solution']}"
                for d in rejected_data_dict.get(lang, [])
                if d["reason"] == reason
            ]
            examples = examples[:max_examples_per_reason]
            reason_str = reason.replace(" ", "_")
            for i, example in enumerate(examples):
                (examples_dir / f"{lang}-{reason_str}-{i}.txt").write_text(example)


def main():
    args = cast(Args, HfArgumentParser(Args).parse_args_into_dataclasses()[0])
    if args.analysis_dir is not None and not args.no_filter:
        Path(args.analysis_dir).mkdir(exist_ok=False, parents=False)
    raw_data: list[dict] = []
    for data_file in args.data_files:
        data = read_jsonl(Path(data_file))
        language = data_file.split("-")[1]
        assert language in ALL_LANGS, f"Unknown language {language}"
        raw_data.extend(dict(lang=language, **d) for d in data)
    random.seed(args.seed)
    random.shuffle(raw_data)

    if args.no_filter:
        print("No filtering, just randomizing the order of the data..")
        write_jsonl(Path(args.output_file), raw_data)
        return

    chosen_data = raw_data
    chosen_data, rejected_data_1 = filter_same_seed_problem_solution(chosen_data)
    print(f"After filtering: {len(raw_data)} -> {(n_last := len(chosen_data))}")

    warnings.warn(
        "In practice, filtering data whose solution copies the problem does not help much."
        "So we disabled it. But this conclusion remains to be verified."
    )
    # chosen_data, rejected_data_2 = filter_same_codeblocks(chosen_data)
    # print(f"After filtering: {n_last} -> {(n_last := len(chosen_data))}")
    write_jsonl(Path(args.output_file), chosen_data)
    if args.analysis_dir is not None:
        print("Saving analysis..")
        save_analysis(
            chosen_data,
            rejected_data_1,  # + rejected_data_2,
            Path(args.analysis_dir),
        )


if __name__ == "__main__":
    main()
