import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import HfArgumentParser

from magicoder.utils import read_jsonl


@dataclass(frozen=True)
class Args:
    data_file: str
    output_path: str
    max_considered_data: int | None = field(default=150000)


def get_dataset(args: Args, lang: str) -> Dataset:
    name = "bigcode/starcoderdata" if lang != "swift" else "bigcode/the-stack"
    if lang == "csharp":
        lang = "c-sharp"
    data_dir = lang if lang != "swift" else "data/swift"
    return load_dataset(
        name,
        data_dir=data_dir,
        split=f"train[:{args.max_considered_data}]",
    )


if __name__ == "__main__":
    args = cast(Args, HfArgumentParser(Args).parse_args_into_dataclasses()[0])
    datasets: dict[str, Dataset] = {}
    all_data = read_jsonl(args.data_file)
    out_file = Path(args.output_path).open("w")
    for d in tqdm(all_data):
        if d["lang"] not in datasets:
            datasets[d["lang"]] = get_dataset(args, d["lang"])
        dataset = datasets[d["lang"]]
        index = d["raw_index"]
        raw_data = dataset[index]
        assert d["seed"] in raw_data["content"]
        content = raw_data["content"]
        if content.startswith("<reponame>") or content.startswith("<filename>"):
            # remove the first line
            raw_data["content"] = content[content.index("\n") + 1 :]
        raw_data["seed"] = d["seed"]
        raw_data["lang"] = d["lang"]
        out_file.write(json.dumps(raw_data) + "\n")
