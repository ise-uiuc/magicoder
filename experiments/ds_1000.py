import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, cast

from ds1000 import DS1000Dataset, DS1000Problem
from tqdm.auto import tqdm
from transformers import HfArgumentParser

from magicoder.llm_wrapper import (
    GenerationConfig,
    ModelContext,
    create_infilling_prompt,
    get_model_context,
)
from magicoder.prompt_template import MAGICODER_PROMPT

WIZARDCODER_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}

### Response:
{response}"""

PROMPT = MAGICODER_PROMPT if os.getenv("WIZARDCODER") is None else WIZARDCODER_PROMPT
print("Using prompt:")
print(PROMPT)


@dataclass
class Args:
    dataset_path: str
    model_key: str
    model_name_or_path: str
    mode: Literal["Insertion", "Completion"]
    output_dir: str

    temperature: float = field(default=0.2)
    top_p: float = field(default=0.5)
    max_length: int = field(default=1024)
    n_samples_per_batch: int = field(default=10)
    n_batches: int = field(default=4)

    def to_generation_config(self) -> GenerationConfig:
        return GenerationConfig(
            # Use max_length to control
            max_new_tokens=9999999999999,
            top_p=self.top_p,
            temperature=self.temperature,
            max_length=self.max_length,
        )


def postprocess(text: str) -> str:
    return text.split("```")[0]


def create_prompt(
    args: Args, model_context: ModelContext, problem: DS1000Problem
) -> str:
    prompt = problem["prompt"]
    if args.mode == "Insertion":
        prompt = preprocess_insertion_prompt(prompt)
        assert prompt.count("[insert]") == 1
        prefix, suffix = prompt.split("[insert]")
        prompt = create_infilling_prompt(
            model_key=args.model_key,
            prefix=prefix,
            suffix=suffix,
            tokenizer=model_context.tokenization_context.tokenizer,
        )
    else:
        assert args.mode == "Completion"
        instruction, response_prefix = preprocess_completion_prompt(problem["prompt"])
        prompt = PROMPT.format(
            instruction=instruction,
            response=response_prefix,
        )
    return prompt


def generate(
    args: Args,
    model_context: ModelContext,
    problem: DS1000Problem,
):
    lib: str = problem["lib"]
    model_key = args.model_key.replace("/", "-")
    problem_id: str = f"q{problem.problem_id}"
    path = Path(args.output_dir) / model_key / lib / args.mode / problem_id
    finishing_signal = path / "FINISHED"
    if finishing_signal.exists():
        print("Skipping:", path)
        return
    if not path.exists():
        print("Making directory:", path)
        path.mkdir(parents=True, exist_ok=True)
    config = args.to_generation_config()
    prompt = create_prompt(args, model_context, problem)
    print("========PROMPT=======")
    print(prompt)
    print("========PROMPT=======")
    for batch_idx in range(args.n_batches):
        print(f"Generating batch {batch_idx} of {args.n_batches}")
        response = model_context.complete(
            config=config,
            prompts=[prompt] * args.n_samples_per_batch,
        )
        print("=======RESPOSE[-1]=======")
        # postprocess_fn: Callable[[str], str] = (
        #     (lambda x: x) if args.mode == "Insertion" else postprocess
        # )
        postprocess_fn = postprocess
        print(postprocess_fn(response.decoded_outputs[-1]))
        print("=======RESPOSE[-1]=======")
        for idx, sample in enumerate(response.decoded_outputs):
            sample = postprocess_fn(sample)
            global_index = batch_idx * args.n_samples_per_batch + idx
            output_file = path / f"{global_index}.py"
            output_file.write_text(sample)
    finishing_signal.touch()


def preprocess_completion_prompt(prompt: str) -> tuple[str, str]:
    """Preprocess the DS-1000 prompt (Completion mode) into instruction and response prefix"""
    # hit = False
    if not "SOLUTION START" in prompt:
        answer_index = prompt.rindex("A:")
        answer = prompt[answer_index + 2 :].strip()
        instruction: str = prompt[:answer_index].strip()
        if instruction.startswith("Problem:"):
            instruction = instruction[len("Problem:") :].strip()
        if "### BEGIN SOLUTION" in prompt:
            assert prompt.count("<code>") == 1
            assert prompt.count("</code>") == 0
            lines = answer.splitlines(keepends=True)
            return_line, result_line, begin_line = lines[-3:]
            assert return_line.strip().startswith("# return")
            assert result_line.strip().startswith("# ")
            assert begin_line.strip() == "### BEGIN SOLUTION"
            response = "".join(lines[:-3]).strip()
            hint = begin_line.replace("###", "#").replace("BEGIN SOLUTION", "Solution")
            response += f"\n{hint}\n"
        else:
            assert "BEGIN SOLUTION" in prompt
            assert prompt.count("<code>") == 2
            assert prompt.count("</code>") == 1
            first_block_start = prompt.index("<code>")
            first_block_end = prompt.index("</code>")
            second_block_start = prompt.index("<code>", first_block_start + 1)
            assert first_block_end < second_block_start
            lines = answer.splitlines(keepends=True)
            block_end, instruction_line, begin_line, block_start = lines[-4:]
            assert begin_line.strip() == "BEGIN SOLUTION"
            assert block_start.strip() == "<code>"
            if not block_end.strip() == "</code>":
                if lines[-6].strip() == "</code>":
                    response_prefix = lines[:-6]
                    starting_lines = lines[-5:-2]
                else:
                    assert instruction_line.strip() == "</code>"
                    response_prefix = lines[:-3]
                    starting_lines = lines[-2:-2]
            else:
                response_prefix = lines[:-4]
                starting_lines = lines[-3:-2]
            starting_lines = [f"# {line.lstrip()}" for line in starting_lines]
            response = "".join([*response_prefix, *starting_lines]).strip()
            response += "\n# Solution\n"
    else:
        # hit = True
        assert prompt.count("<code>") == 0
        assert prompt.count("</code>") == 0
        assert prompt.strip().endswith("# SOLUTION START")
        code_prefix = prompt[: prompt.rindex("# SOLUTION START")].strip()
        instruction = f"""Write a solution to the following problem:
```python
{code_prefix}
```"""
        response = f"```python\n{code_prefix}\n# Solution\n"
    instruction = instruction.replace("<code>", "```python").replace("</code>", "```")
    response = response.replace("<code>", "```python").replace("</code>", "```")
    # if hit:
    #     print("[Instruction]")
    #     print(instruction)
    #     print("[Response]")
    #     print(response)
    #     breakpoint()
    return instruction, response


def preprocess_insertion_prompt(prompt: str) -> str:
    pattern = """</code>
BEGIN SOLUTION
<code>
[insert]
</code>
END SOLUTION"""
    pattern_index = prompt.index(pattern)
    # pattern_block = prompt[pattern_index:]
    prefix = prompt[:pattern_index]
    # hit = False
    if pattern + "\n<code>" in prompt:
        index = prompt.index("<code>", pattern_index + len(pattern))
        suffix = prompt[index + len("<code>") :]
    else:
        # hit = True
        assert pattern in prompt
        suffix = ""
    final_prompt = prefix.strip() + "\n[insert]\n" + suffix.strip()
    final_prompt = final_prompt.replace("<code>", "```python").replace("</code>", "```")
    # if hit:
    #     print(final_prompt)
    #     breakpoint()
    return final_prompt


def main():
    args = cast(Args, HfArgumentParser(Args).parse_args_into_dataclasses()[0])
    dataset = DS1000Dataset(args.dataset_path, mode=args.mode)

    all_problems = [
        problem
        for problems in dataset.data.values()
        for problem in problems
        if args.mode == "Completion" or problem["lib"] != "Matplotlib"
    ]
    model_context = get_model_context(
        model_key=args.model_key,
        model_name_or_path=args.model_name_or_path,
    )
    for problem in tqdm(all_problems):
        generate(args, model_context, problem)


if __name__ == "__main__":
    main()
