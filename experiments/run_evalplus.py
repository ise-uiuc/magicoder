import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from evalplus.data import get_human_eval_plus, write_jsonl
from tqdm.auto import tqdm
from transformers import HfArgumentParser

from magicoder.llm_wrapper import GenerationConfig, get_model_context
from magicoder.prompt_template import MAGICODER_PROMPT
from magicoder.utils import chunked


@dataclass(frozen=True)
class Args:
    model_key: str
    save_path: str

    n_batches: int
    n_problems_per_batch: int
    n_samples_per_problem: int
    prompted: bool

    model_name_or_path: str | None = None

    # @property
    # def n_samples(self) -> int:
    #     return self.n_batches * self.batch_size


PROMPT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def form_prompt(prompt: str) -> str:
    # return (
    #     "Below is an instruction that describes a task. "
    #     "Write a response that appropriately completes the request.\n\n"
    #     f"### Instruction:\nCreate a Python script for this problem:\n```python\n{prompt}```\n\n### Response:\n```python\n"
    # )
    #     return f"""Write an appropriate response to the following instruction.

    # # Instruction
    instruction = f"""Continue the implementation of the following code:
```python
{prompt}
```"""
    response = f"""```python
{prompt.strip()}"""
    return MAGICODER_PROMPT.format(instruction=instruction, response=response)


def main():
    parser = HfArgumentParser((Args, GenerationConfig))
    args, generation_config = cast(
        tuple[Args, GenerationConfig],
        parser.parse_args_into_dataclasses(),
    )
    problems = get_human_eval_plus()
    state = get_model_context(args.model_key, args.model_name_or_path)

    problems_chunked = list(chunked(list(problems.items()), args.n_problems_per_batch))
    iter = itertools.product(problems_chunked, range(args.n_batches))
    n_total = len(problems_chunked) * args.n_batches

    Path(args.save_path).write_text("")
    for task_id_and_problems, batch_idx in tqdm(iter, total=n_total):
        task_ids, problems = zip(*task_id_and_problems)
        prompts = [problem["prompt"] for problem in problems]
        if args.prompted:
            prompts = [form_prompt(prompt) for prompt in prompts]
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
