import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast

from evalplus.data import get_human_eval_plus, write_jsonl
from tqdm.auto import tqdm
from transformers import HfArgumentParser

from experiments.utils import wget
from magicoder.llm_wrapper import GenerationConfig, get_model_context
from magicoder.prompt_template import MAGICODER_PROMPT
from magicoder.utils import chunked, read_jsonl


class Text2CodeProblem(TypedDict):
    id: str
    instruction: str
    response_prefix: str


MBPP_INSTRUCTION = """{nl_description} Your code should satisfy the following assertion:
```python
{assertions}
```
Enclose your solution in ```python and ```"""


def get_mbpp_raw_problems(is_testset: bool = True) -> list[dict]:
    testset_range = range(11, 511)

    def is_considered(id: int) -> bool:
        return id in testset_range if is_testset else id not in testset_range

    path = wget(
        "https://github.com/google-research/google-research/raw/master/mbpp/mbpp.jsonl"
    )
    mbpp_problems: list[dict] = read_jsonl(path)

    problems = [p for p in mbpp_problems if is_considered(p["task_id"])]
    return problems


def get_humaneval_raw_problems() -> list[dict]:
    problems = get_human_eval_plus()
    return list(problems.values())


def map_mbpp_problem(p: dict) -> Text2CodeProblem:
    id = p["task_id"]
    nl_description = p["content"]
    if not nl_description.endswith("."):
        nl_description += "."
    nl_description = nl_description.strip()
    assertion = p["test_list"][0].strip()
    instruction = MBPP_INSTRUCTION.format(
        nl_description=nl_description, assertions=assertion
    )
    return Text2CodeProblem(
        id=str(id), instruction=instruction, response_prefix=assertion
    )


def map_humaneval_problem(p: dict) -> Text2CodeProblem:
    id = p["task_id"]
    prompt = p["prompt"]
    prompt = prompt.strip()
    # try:
    #     docstring_index = prompt.index('"""')
    # except ValueError:
    #     docstring_index = prompt.index("'''")
    # signature = prompt[:docstring_index].strip()
    # Instruction
    instruction = f"""Write a solution to the following problem:
```python
{prompt}
```"""
    response_prefix = f"""```python
{prompt}"""
    return Text2CodeProblem(
        id=id, instruction=instruction, response_prefix=response_prefix
    )


@dataclass(frozen=True)
class Args:
    model_key: str
    dataset: Literal["humaneval", "mbpp"]
    save_path: str

    n_batches: int
    n_problems_per_batch: int
    n_samples_per_problem: int
    prompted: bool

    model_name_or_path: str | None = None


def main():
    parser = HfArgumentParser((Args, GenerationConfig))
    args, generation_config = cast(
        tuple[Args, GenerationConfig],
        parser.parse_args_into_dataclasses(),
    )
    raw_problem_fn, map_problem_fn = (
        (get_humaneval_raw_problems, map_humaneval_problem)
        if args.dataset == "humaneval"
        else (lambda: get_mbpp_raw_problems(is_testset=True), map_mbpp_problem)
    )
    raw_problems = raw_problem_fn()
    problems = list(map(map_problem_fn, raw_problems))

    state = get_model_context(args.model_key, args.model_name_or_path)

    problems_chunked = list(chunked(list(problems), args.n_problems_per_batch))
    iter = itertools.product(problems_chunked, range(args.n_batches))
    n_total = len(problems_chunked) * args.n_batches

    Path(args.save_path).write_text("")
    for problems, batch_idx in tqdm(iter, total=n_total):
        task_ids = [problem["id"] for problem in problems]
        prompts = [
            # TODO: make it generic for all models
            MAGICODER_PROMPT.format(
                instruction=problem["instruction"], response=problem["response_prefix"]
            )
            for problem in problems
        ]
        print("PROMPT")
        print(prompts[-1])
        all_prompts = prompts * args.n_samples_per_problem
        all_task_ids = task_ids * args.n_samples_per_problem
        response = state.complete(generation_config, all_prompts)
        completions = response.decoded_outputs
        assert len(problems) <= args.n_problems_per_batch
        assert len(completions) == len(problems) * args.n_samples_per_problem
        print("COMPLETION")
        print(completions[-1])
        samples = [
            dict(
                task_id=task_id,
                completion=completion[
                    : index
                    if (index := completion.find("```")) != -1
                    else len(completion)
                ],
            )
            for task_id, completion in zip(all_task_ids, completions)
        ]
        write_jsonl(args.save_path, samples, append=True)


if __name__ == "__main__":
    main()
