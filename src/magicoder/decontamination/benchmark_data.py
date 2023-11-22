"""Migrated from: https://github.com/bigcode-project/bigcode-dataset. License: Apache 2.0"""

"""data to filter out of the dataset"""
import itertools
import json
import os
from pathlib import Path

from datasets import load_dataset

# HumanEval solutions that are considered simple/generic enough to be kept in the training dataset
HUMAN_EVAL_STRINGS_OK = [
    "return x + y",
    "return len(string)",
    "return n**2",
    "return " ".join(strings)",
]


def extract_ds_1000_prompt(prompt: str):
    if "SOLUTION START" in prompt:
        assert prompt.count("SOLUTION START") == 1
        return prompt.split("SOLUTION START")[0]
    elif "BEGIN SOLUTION" in prompt:
        assert prompt.count("BEGIN SOLUTION") == 1
        return prompt.split("BEGIN SOLUTION")[0]
    else:
        raise ValueError()


def load_ds_1000():
    DS1000_PATH_NAME = os.getenv("DS1000_PATH", None)
    assert (
        DS1000_PATH_NAME is not None
    ), "Please set the environment variable DS1000_PATH to the path of `ds1000_data`"
    DS1000_PATH = Path(DS1000_PATH_NAME)  # type: ignore
    data = []
    for prompt_file in DS1000_PATH.glob("*/Insertion/q*/prompt.txt"):
        with open(prompt_file) as f:
            data.append(extract_ds_1000_prompt(f.read()))
    return data


def load_mbpp():
    MBPP_PATH_NAME = os.getenv("MBPP_PATH", None)
    assert (
        MBPP_PATH_NAME is not None
    ), "Please set the environment variable MBPP_PATH to the path of `mbpp.jsonl`"
    MBPP_PATH = Path(MBPP_PATH_NAME)
    TEST_IDS = list(range(11, 511))
    data = []
    with open(MBPP_PATH) as f:
        for line in f:
            data.append(json.loads(line))

    data = [sample for sample in data if sample["task_id"] in TEST_IDS]

    assert len(data) == 500

    # Checksum / version issues here
    # dataset = load_dataset("mbpp", split="test")
    return data


def mbpp_docstrings():
    data = load_mbpp()
    return [sample["text"] for sample in data]


def mbpp_solutions():
    data = load_mbpp()
    return [sample["code"] for sample in data]


def extract_docstring(prompt: str) -> str:
    if '"""' in prompt:
        if prompt.count('"""') == 2:
            return prompt.split('"""')[1].strip()
        elif prompt.count('"""') == 4:
            return prompt.split('"""')[3].strip()
        else:
            raise ValueError()
    elif "'''" in prompt:
        assert prompt.count("'''") == 2
        return prompt.split("'''")[1].strip()
    else:
        raise ValueError()


def human_eval_docstrings():
    ds = load_dataset("openai_humaneval", split="test")
    docstrings = [extract_docstring(v["prompt"]) for v in ds]
    return docstrings


def apps_solutions():
    """
    Solutions column contains a list of strings
    """
    ds = load_dataset("codeparrot/apps", split="test")
    solutions = [sample["solutions"] for sample in ds if len(sample["solutions"]) > 0]
    res = itertools.chain.from_iterable(json.loads(sample) for sample in solutions)
    return list(res)


def multipl_e_docstrings():
    languages = [
        "cpp",
        "cs",
        "d",
        "go",
        "java",
        "jl",
        "js",
        "lua",
        "php",
        "pl",
        "py",
        "r",
        "rb",
        "rkt",
        "rs",
        "scala",
        "sh",
        "swift",
        "ts",
    ]
    # languages = ["py", "java", "js"]
    src_datas = ["humaneval", "mbpp"]
    variations = ["", "-remove"]
    data = []
    for lang in languages:
        for src_data in src_datas:
            for variation in variations:
                if src_data == "mbpp" and variation == "-remove":
                    continue
                ds = load_dataset(
                    "nuprl/MultiPL-E", f"{src_data}-{lang}{variation}", split="test"
                )
                data += [sample["prompt"].strip() for sample in ds]
    return data


def load_dataset_column(dataset: str, column: str, split: str, name=None):
    ds = load_dataset(dataset, split=split, name=name)
    res = [sample[column].strip() for sample in ds]
    # Only return non-empty strings
    return [sample for sample in res if len(sample) > 0]


LAZY_FILTER_OUT = {
    "mbpp_docstrings": lambda: mbpp_docstrings(),
    "mbpp_solutions": lambda: mbpp_solutions(),
    "human_eval_docstrings": lambda: human_eval_docstrings(),
    "human_eval_solutions": lambda: [
        s
        for s in load_dataset_column("openai_humaneval", "canonical_solution", "test")
        if s not in HUMAN_EVAL_STRINGS_OK
    ],
    "apps_docstrings": lambda: load_dataset_column(
        "codeparrot/apps", "question", "test"
    ),
    # 115212 examples to filter-out in apps-solutions, which would take way too much time without any hashing trick
    # "apps_solutions": apps_solutions(),
    # MultiPL-E samples are from HumanEval and MBPP: we are already looking for them
    # "multipl-e_docstrings": multipl_e_docstrings(),
    # There is no solution provided with multipl-e
    "gsm8k_questions": lambda: load_dataset_column("gsm8k", "question", "test", "main"),
    "ds_1000_prompts": lambda: load_ds_1000(),
}

IGNORED = os.getenv("IGNORED", "").split(":")
print("Ignoring:", IGNORED)
for ignored in IGNORED:
    if ignored != "" and ignored in LAZY_FILTER_OUT:
        del LAZY_FILTER_OUT[ignored]
FILTER_OUT = {k: v() for k, v in LAZY_FILTER_OUT.items()}


for benchmark, values in FILTER_OUT.items():
    print(f"num strings from {benchmark}: {len(values)}")
