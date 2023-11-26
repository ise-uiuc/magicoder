import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from transformers import HfArgumentParser

from magicoder.utils import read_jsonl, write_jsonl


@dataclass(frozen=True)
class Args:
    data_files: list[str]
    seed: int = field(default=1)
    n_datapoints: int = field(default=30000)


def main():
    parser = HfArgumentParser((Args,))
    args = cast(Args, parser.parse_args_into_dataclasses()[0])
    all_data: list[dict] = []
    for data_file in args.data_files:
        all_data.extend(read_jsonl(data_file))
    python_data: list[dict] = []
    other_data: list[dict] = []
    for data in all_data:
        if (
            data["lang"] == "python"
            or "python" in data["instruction"] + data["response"]
        ):
            python_data.append(data)
        else:
            other_data.append(data)
    random.seed(args.seed)
    python_data = random.sample(python_data, k=args.n_datapoints)
    other_data = random.sample(other_data, k=args.n_datapoints)

    output_path_python = Path("data-ablation-python.jsonl")
    output_path_others = Path("data-ablation-non_python.jsonl")

    def ask_and_write(path: Path, data: list[dict]):
        write_data = True
        if path.exists():
            option = input(f"{path} already exists. Overwrite? [y/n] ")
            write_data = option.lower() == "y"
        if write_data:
            print("Writing data to", path)
            write_jsonl(path, data)

    ask_and_write(output_path_python, python_data)
    ask_and_write(output_path_others, other_data)


if __name__ == "__main__":
    main()
