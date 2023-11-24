from dataclasses import dataclass, field
from typing import Literal, cast

from datasets import load_dataset
from transformers import HfArgumentParser

from magicoder.prompt_template import SRC_INSTRUCT_INSTRUCTION_PROMPT
from magicoder.utils import N_CORES, read_jsonl, write_jsonl

DatasetKey = Literal["evol-instruct", "codealpaca", "src-instruct", "combine"]


@dataclass(frozen=True)
class Args:
    dataset_path: str
    key: DatasetKey
    output_file: str
    data_files: list[str] | None = field(default=None)
    split: str = field(default="train")


def map_evol_instruct(example: dict) -> dict:
    instruction = example["instruction"]
    response = example["output"]
    return dict(
        instruction=instruction,
        response=response,
    )


def form_codealpaca_instruction(instruction: str, input: str) -> str:
    if input.strip() == "":
        return instruction
    return f"{instruction}\nInput: {input}"


def map_codealpaca(example: dict) -> dict:
    instruction = [
        form_codealpaca_instruction(instruction, input)
        for instruction, input in zip(example["instruction"], example["input"])
    ]
    response = example["output"]
    return dict(
        instruction=instruction,
        response=response,
    )


def map_src_instruct(example: dict) -> dict:
    instructions = [
        SRC_INSTRUCT_INSTRUCTION_PROMPT.format(problem=problem)
        for problem in example["problem"]
    ]
    keys = [key for key in example.keys() if key not in ["problem", "solution"]]
    kwargs = {key: example[key] for key in keys}
    return dict(instruction=instructions, response=example["solution"], **kwargs)


def map_fn(example: dict, key: DatasetKey) -> dict:
    if key == "evol-instruct":
        return map_evol_instruct(example)
    elif key == "codealpaca":
        return map_codealpaca(example)
    elif key == "src-instruct":
        return map_src_instruct(example)
    else:
        raise ValueError(f"Unknown key: {key}")


def main():
    args = cast(Args, HfArgumentParser(Args).parse_args_into_dataclasses()[0])
    if args.key == "combine":
        assert args.dataset_path == "json" and args.data_files is not None
        all_data: list[dict] = []
        for data_file in args.data_files:
            data = read_jsonl(data_file)
            all_data.extend(
                dict(instruction=item["instruction"], response=item["response"])
                for item in data
            )
        write_jsonl(args.output_file, all_data)
    else:
        dataset = load_dataset(
            args.dataset_path,
            data_files=args.data_files,
            split=args.split,
            num_proc=N_CORES,
        )
        dataset = dataset.map(
            map_fn,
            fn_kwargs=dict(key=args.key),
            batched=True,
            num_proc=N_CORES,
            remove_columns=dataset.column_names,
        )
        write_jsonl(args.output_file, dataset)


if __name__ == "__main__":
    main()
