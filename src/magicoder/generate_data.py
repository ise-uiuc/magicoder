import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import HfArgumentParser

import magicoder

# DO NOT CHANGE THE FOLLOWING
SYSTEM = "You are exceptionally skilled at crafting high-quality programming problems and offering precise solutions."
ERROR_MARGIN = 10


@dataclass(frozen=True)
class Args:
    seed_code_start_index: int
    # `seed_code_start_index` + `max_new_data` is the last-to-end seed code index
    max_new_data: int
    continue_from: str | None = field(default=None)

    # Keep the following arguments unchanged for reproducibility
    seed: int = field(default=976)

    temperature: float = field(default=0.0)
    model: str = field(default="gpt-3.5-turbo-1106")
    model_max_tokens: int = field(default=8192)
    max_new_tokens: int = field(default=2500)

    min_lines: int = field(default=1)
    max_lines: int = field(default=15)
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

    def fingerprint(self, prompt_template: str) -> str:
        # The combination of arguments can uniquely determine the generation process
        args = (
            self.seed,
            self.temperature,
            self.model,
            self.model_max_tokens,
            self.min_lines,
            self.max_lines,
            self.chunk_size,
            self.dataset_name,
            self.data_dir,
            self.max_considered_data,
            prompt_template,
            SYSTEM,
            ERROR_MARGIN,
        )
        return magicoder.utils.compute_fingerprint(*args, hash_length=5)


def map_dataset(examples: dict, indices: list[int], args: Args) -> dict:
    random.seed(args.seed + indices[0])
    seed_snippets = [
        extract_seed_code(args, content) for content in examples["content"]
    ]
    return {
        "seed": seed_snippets,
        "raw_index": indices,
    }


def extract_seed_code(args: Args, document: str) -> str:
    lines = document.splitlines(keepends=True)
    start_index = random.choice(range(len(lines)))
    n_lines_to_consider = random.randint(args.min_lines, args.max_lines)
    code = "".join(lines[start_index : start_index + n_lines_to_consider])
    return code


def parse_problem_solution(response_text: str) -> tuple[str, str] | None:
    lines = response_text.splitlines(keepends=True)
    problem_start_index: int | None = None
    solution_start_index: int | None = None
    for idx, line in enumerate(lines):
        if "[problem description]" in line.lower() and problem_start_index is None:
            problem_start_index = idx
        if "[solution]" in line.lower() and solution_start_index is None:
            solution_start_index = idx
    if problem_start_index is None or solution_start_index is None:
        return None
    if problem_start_index >= solution_start_index:
        return None
    problem = "".join(lines[problem_start_index + 1 : solution_start_index]).strip()
    solution = "".join(lines[solution_start_index + 1 :]).strip()
    return problem, solution


def main():
    args, *_ = cast(
        tuple[Args, ...], HfArgumentParser(Args).parse_args_into_dataclasses()
    )
    split = (
        f"train[:{args.max_considered_data}]"
        if args.max_considered_data is not None
        else "train"
    )
    assert magicoder.utils.OPENAI_CLIENT is not None
    dataset: Dataset = load_dataset(
        args.dataset_name,
        data_dir=args.data_dir,
        split=split,
        num_proc=magicoder.utils.N_CORES,
    )
    random.seed(args.seed)
    # map_fn = get_map_dataset(args)
    dataset = dataset.map(
        function=map_dataset,
        fn_kwargs=dict(args=args),
        with_indices=True,
        batched=True,
        batch_size=args.chunk_size,
    )
    dataset = dataset.shuffle(seed=args.seed)
    dataset = dataset.map(lambda _, index: {"index": index}, with_indices=True)

    # Every run should produce the same data as long as the default params are not changed
    start_index = args.seed_code_start_index
    end_index = min(start_index + args.max_new_data, len(dataset))
    dataset = dataset.select(range(start_index, end_index))

    prompt_template = Path("data/prompt.txt").read_text()
    timestamp = magicoder.utils.timestamp()
    data_fingerprint = args.fingerprint(prompt_template)
    if args.continue_from is not None:
        assert data_fingerprint in args.continue_from, "Fingerprint mismatch"
        assert f"{start_index}_{end_index}" in args.continue_from, "Index mismatch"
        old_path = Path(args.continue_from)
        assert old_path.exists()
        old_data = magicoder.utils.read_jsonl(old_path)
        assert len(old_data) > 0
        last_index = old_data[-1]["index"]
        n_skipped = last_index - start_index + 1
        print("Continuing from", old_path)
        f_out = old_path.open("a")
    else:
        tag = "" if args.tag == "" else f"-{args.tag}"
        path = Path(
            f"data{tag}-{data_fingerprint}-{start_index}_{end_index}-{timestamp}.jsonl"
        )
        assert not path.exists()
        f_out = path.open("w")
        print("Saving to", path)
        n_skipped = 0
    for index, example in enumerate(tqdm(dataset)):
        if index < n_skipped:
            continue
        assert index + start_index == example["index"]
        prompt = prompt_template.format(code=example["seed"])
        # Make sure the generation is within the context size of the model
        max_new_tokens = min(
            args.max_new_tokens,
            args.model_max_tokens
            - magicoder.utils.num_tokens_from_string(prompt, args.model)
            # error margin (e.g., due to conversation tokens)
            - ERROR_MARGIN,
        )
        if max_new_tokens <= 0:
            continue
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ]
        openai_seed = args.seed + example["index"]
        response = magicoder.utils.chat_completions_with_backoff(
            model=args.model,
            messages=messages,
            max_tokens=max_new_tokens,
            n=1,
            temperature=args.temperature,
            seed=openai_seed,
        )
        print(openai_seed)
        choice = response.choices[0]
        if choice.finish_reason != "stop":
            continue
        parsing_result = parse_problem_solution(choice.message.content)
        if parsing_result is None:
            continue
        problem, solution = parsing_result
        if len(problem) == 0 or len(solution) == 0:
            continue
        fingerprint = response.system_fingerprint
        assert fingerprint is not None
        # In this dict seed means "seed code snippet" instead of "random seed"
        data = dict(
            raw_index=example["raw_index"],
            index=example["index"],
            seed=example["seed"],
            openai_fingerprint=fingerprint,
            problem=problem,
            solution=solution,
        )

        print("[Problem Description]", problem, sep="\n", end="\n\n")
        print("[Solution]", solution, sep="\n")

        f_out.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    main()
