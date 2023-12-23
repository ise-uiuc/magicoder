import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import HfArgumentParser

import magicoder

# DO NOT CHANGE THE FOLLOWING
ERROR_MARGIN = 10
SYSTEM_PROMPT = "You are exceptionally skilled at coding. You consistently deliver accurate and reliable responses to user instructions"


@dataclass(frozen=True)
class Args:
    data_files: list[str] = field(metadata={"help": "Path to the seed code snippets"})
    max_new_data: int
    start_index: int = field(default=0)
    continue_from: str | None = field(default=None)

    # Keep the following arguments unchanged for reproducibility
    seed: int = field(default=977)

    temperature: float = field(default=0.0)
    model: str = field(default="gpt-3.5-turbo-1106")
    model_max_tokens: int = field(default=8192)
    max_new_tokens: int = field(default=2500)

    rpm: int = field(
        default=1, metadata={"help": "Requests per minute to the model API"}
    )
    delay: int | None = field(
        default=None, metadata={"help": "Delay between batched requests in seconds"}
    )

    tag: str = field(
        default="",
        metadata={
            "help": "Custom tag as part of the output filename, not affecting the fingerprint"
        },
    )

    def fingerprint(self) -> str:
        # The combination of arguments can uniquely determine the generation process
        args = (
            self.data_files,
            self.seed,
            self.temperature,
            self.model,
            self.model_max_tokens,
            ERROR_MARGIN,
            SYSTEM_PROMPT,
        )
        return magicoder.utils.compute_fingerprint(*args, hash_length=5)


async def main():
    args = cast(Args, HfArgumentParser(Args).parse_args_into_dataclasses()[0])
    openai_client = magicoder.utils.OpenAIClient()
    raw_dataset: Dataset = load_dataset(
        "json",
        data_files=args.data_files,
        split="train",
        num_proc=magicoder.utils.N_CORES,
    )

    # Every run should produce the same data as long as the default params are not changed
    start_index = args.start_index
    end_index = min(start_index + args.max_new_data, len(raw_dataset))
    raw_dataset = raw_dataset.select(range(start_index, end_index))
    dataset = raw_dataset.to_list()

    timestamp = magicoder.utils.timestamp()
    data_fingerprint = args.fingerprint()
    if args.continue_from is not None:
        assert (
            data_fingerprint in args.continue_from
        ), f"Fingerprint mismatch: {data_fingerprint}"
        assert f"-{start_index}-" in args.continue_from, "Index mismatch"
        old_path = Path(args.continue_from)
        assert old_path.exists()
        old_data = magicoder.utils.read_jsonl(old_path)
        assert len(old_data) > 0
        last_seed = old_data[-1]["seed"]
        seed_index = next(
            idx for idx, d in enumerate(dataset) if d["seed"] == last_seed
        )
        n_skipped = seed_index + 1
        # n_skipped = last_index - start_index + 1
        print(f"Continuing from {old_path} with {n_skipped} seed snippets skipped")
        f_out = old_path.open("a")
    else:
        tag = "" if args.tag == "" else f"-{args.tag}"
        path = Path(f"data{tag}-{data_fingerprint}-{start_index}-{timestamp}.jsonl")
        assert not path.exists()
        f_out = path.open("w")
        print("Saving to", path)
        n_skipped = 0
    dataset = dataset[n_skipped:]
    chunked_dataset = list(magicoder.utils.chunked(dataset, n=args.rpm))
    pbar = tqdm(chunked_dataset)
    n_succeeded = 0
    for chunk_index, examples in enumerate(pbar):
        # map to the index in the original seed snippets
        effective_index = chunk_index * args.rpm + start_index + n_skipped
        if chunk_index > 0 and args.rpm != 1:
            print("Sleeping for 60 seconds...")
            time.sleep(60)
        # assert index + start_index == example["index"]
        request_params = list[dict[str, Any]]()
        for index, example in enumerate(examples):
            seed = args.seed + effective_index + index
            instruction = example["instruction"]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
            ]
            # Make sure the generation is within the context size of the model
            max_new_tokens = min(
                args.max_new_tokens,
                args.model_max_tokens
                - magicoder.utils.num_tokens_from_string(
                    f"{SYSTEM_PROMPT}\n{instruction}", args.model
                )
                # error margin (e.g., due to conversation tokens)
                - ERROR_MARGIN,
            )
            # if max_new_tokens <= 0:
            #     continue
            params = dict(
                model=args.model,
                messages=messages,
                max_tokens=max_new_tokens,
                n=1,
                temperature=args.temperature,
                seed=seed,
            )
            request_params.append(params)
        assert len(request_params) == len(examples)
        print(f"Ready to make {len(request_params)} requests")
        responses = await openai_client.dispatch_chat_completions(
            request_params, delay=args.delay
        )
        assert len(examples) == len(responses)
        for example, response in zip(examples, responses):
            if isinstance(response, BaseException):
                print("Exception when generating response:", response)
                continue
            choice = response.choices[0]
            if choice.finish_reason != "stop":
                print("Failed to generate a complete response")
                continue
            output = choice.message.content
            if output is None:
                continue
            fingerprint = response.system_fingerprint
            assert fingerprint is not None
            data = dict(
                response=output,
                **example,
            )

            print(
                "[Instruction]",
                example["instruction"],
                "[Response]",
                output,
                sep="\n",
                end="\n\n",
            )
            n_succeeded += 1
            f_out.write(json.dumps(data) + "\n")
            f_out.flush()
        total_requests = chunk_index * args.rpm + len(examples)
        pbar.set_description(f"Success ratio: {n_succeeded} / {total_requests}")


if __name__ == "__main__":
    asyncio.run(main())
