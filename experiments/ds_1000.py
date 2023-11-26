from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

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
    n_samples_per_batch: int = field(default=20)
    n_batches: int = field(default=2)

    def to_generation_config(self) -> GenerationConfig:
        return GenerationConfig(
            # Use max_length to control
            max_new_tokens=9999999999999,
            top_p=self.top_p,
            temperature=self.temperature,
            max_length=self.max_length,
        )


def postprocess(code: str) -> str:
    return code.split("END SOLUTION")[0].split("```")[0]


def create_prompt(args: Args, model_context: ModelContext, given_prompt: str) -> str:
    prompt = given_prompt.replace("<code>", "```python").replace("</code>", "```")
    if args.mode == "Insertion":
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
        prompt = MAGICODER_PROMPT.format(
            instruction=f'{prompt}\n\nComplete the Python code in "..."',
            response="```python",
        )
    return prompt


def generate(args: Args, model_context: ModelContext, problem: DS1000Problem):
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
    prompt = create_prompt(args, model_context, problem["prompt"])
    print("PROMPT")
    print(prompt)
    for batch_idx in range(args.n_batches):
        print(f"Generating batch {batch_idx} of {args.n_batches}")
        response = model_context.complete(
            config=config,
            prompts=[prompt] * args.n_samples_per_batch,
        )
        print("RESPOSE[-1]")
        print(postprocess(response.decoded_outputs[-1]))
        for idx, sample in enumerate(response.decoded_outputs):
            sample = postprocess(sample)
            global_index = batch_idx * args.n_samples_per_batch + idx
            output_file = path / f"{global_index}.py"
            output_file.write_text(sample)
    finishing_signal.touch()


def preprocess_prompt(prompt: str) -> str:
    if not "SOLUTION START" in prompt:
        answer_index = prompt.rindex("A:")
        answer = prompt[answer_index:]
        if "### BEGIN SOLUTION" in prompt:
            assert prompt.count("<code>") == 1
            assert prompt.count("</code>") == 0
            lines = answer.splitlines()
            return_line, result_line, begin_line = lines[-3:]
            assert return_line.strip().startswith("# return")
            assert result_line.strip().startswith("# ")
            assert begin_line.strip() == "### BEGIN SOLUTION"
        else:
            assert "BEGIN SOLUTION" in prompt
            assert prompt.count("<code>") == 2
            assert prompt.count("</code>") == 1
            lines = answer.splitlines()
            block_end, instruction_line, begin_line, block_start = lines[-4:]
            if not block_end.strip() == "</code>":
                if lines[-6].strip() == "</code>":
                    pass
                else:
                    assert instruction_line.strip() == "</code>"
            # print(instruction_line.strip())
            assert begin_line.strip() == "BEGIN SOLUTION"
            assert block_start.strip() == "<code>"
    else:
        assert prompt.count("<code>") == 0
        assert prompt.count("</code>") == 0
        assert prompt.strip().endswith("# SOLUTION START")


def main():
    args = cast(Args, HfArgumentParser(Args).parse_args_into_dataclasses()[0])
    dataset = DS1000Dataset(args.dataset_path, mode=args.mode)

    all_problems = [
        problem
        for problems in dataset.data.values()
        for problem in problems
        if args.mode == "Completion" or problem["lib"] != "Matplotlib"
    ]
    # model_context = get_model_context(
    #     model_key=args.model_key,
    #     model_name_or_path=args.model_name_or_path,
    # )
    for problem in tqdm(all_problems):
        prompt: str = problem["prompt"]
        preprocess_prompt(prompt)
        # generate(args, model_context, problem)

    # generated_code = api_call(ds_data["Pandas"][0]["prompt"])

    # # look at reference_code to understand the question
    # reference_code = ds_data["Pandas"][0]["reference_code"]

    # # run official test on the generated code to
    # is_correct = ds_data["Pandas"][0].test(generated_code)

    # # checkout other available information in each question
    # # we will go over them in the next section
    # print(ds_data["Pandas"][0].keys())


if __name__ == "__main__":
    main()
