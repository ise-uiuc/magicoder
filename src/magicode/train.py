import random
from dataclasses import dataclass, field
from typing import TypedDict, cast

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import HfArgumentParser, Trainer, TrainingArguments

from magicode.llm_wrapper import (  # ChatPiece,; ChatTokenizationContext,
    DecodingConfig,
    EncodingConfig,
    ModelContext,
    TokenizationContext,
    get_model_context,
    pad_sequences,
)
from magicode.utils import N_CORES

# from evalplus.data import get_human_eval
# from trl import DPOTrainer

# from peft import LoraConfig, TaskType, get_peft_model

# DEVICE_MAP = {
#     "model.embed_tokens": 0,
#     "model.layers.0": 1,
#     "model.layers.1": 1,
#     "model.layers.2": 1,
#     "model.layers.3": 1,
#     "model.layers.4": 1,
#     "model.layers.5": 1,
#     "model.layers.6": 1,
#     "model.layers.7": 1,
#     "model.layers.8": 1,
#     "model.layers.9": 1,
#     "model.layers.10": 1,
#     "model.layers.11": 1,
#     "model.layers.12": 1,
#     "model.layers.13": 1,
#     "model.layers.14": 1,
#     "model.layers.15": 1,
#     "model.layers.16": 1,
#     "model.layers.17": 1,
#     "model.layers.18": 1,
#     "model.layers.19": 1,
#     "model.layers.20": 1,
#     "model.layers.21": 1,
#     "model.layers.22": 1,
#     "model.layers.23": 1,
#     "model.layers.24": 1,
#     "model.layers.25": 1,
#     "model.layers.26": 1,
#     "model.layers.27": 1,
#     "model.layers.28": 1,
#     "model.layers.29": 1,
#     "model.layers.30": 1,
#     "model.layers.31": 1,
#     "model.norm": 1,
#     "lm_head": 1,
# }


@dataclass(frozen=True)
class ModelArguments:
    model_key: str
    model_name_or_path: str | None = None


@dataclass(frozen=True)
class DataArguments:
    data_path: str


PROMPT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}


# Ignored index in CrossEntropyLoss
IGNORED_INDEX = -100


MAX_TRAINING_SEQ_LENGTH = 1180
PAD_TO_MAX_LENGTH = False


def get_map_raw_dataset(context: TokenizationContext):
    def map_raw_dataset(examples: dict[str, list[str]]) -> dict:
        prompts = examples["prompt"]
        completions = examples["completion"]
        assert len(prompts) == len(completions)
        prompt_config = EncodingConfig(add_bos=True, add_eos=False)
        completion_config = EncodingConfig(add_bos=False, add_eos=True)
        prompt_id_batches = context.encode(prompt_config, prompts)
        completion_id_batches = context.encode(completion_config, completions)
        # prompt_id_batches = context.tokenization_context.encode(prompt_config, prompts)
        # completion_id_batches = context.tokenization_context.encode(
        #     completion_config, completions
        # )
        assert len(prompt_id_batches) == len(completion_id_batches)
        untruncated_input_ids = [
            (instruction_ids + response_ids)
            for instruction_ids, response_ids in zip(
                prompt_id_batches, completion_id_batches
            )
        ]
        exceeding_length = [
            len(input_id) > MAX_TRAINING_SEQ_LENGTH
            for input_id in untruncated_input_ids
        ]
        input_ids = [
            input_id[:MAX_TRAINING_SEQ_LENGTH] for input_id in untruncated_input_ids
        ]
        # NOTE: no need to set EOF to IGNORED_INDEX as it is *implicitly* ignored inside
        # the model.forward that shifts the logits left by 1
        labels = [
            (list(map(lambda _: IGNORED_INDEX, instruction_ids)) + response_ids)[
                :MAX_TRAINING_SEQ_LENGTH
            ]
            for instruction_ids, response_ids in zip(
                prompt_id_batches, completion_id_batches
            )
        ]
        # `len` of each returned value must be the same, which is required by `tokenizer.map`
        # After `map`, they are treated as individual pieces of data, not as a batch.
        assert len(input_ids) == len(labels)
        for input_id_batch, label_batch in zip(input_ids, labels):
            assert len(input_id_batch) == len(label_batch)
        print(context.decode(DecodingConfig.default(), input_ids[0:])[0])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "exceeding_length": exceeding_length,
        }

    return map_raw_dataset


def get_data_collator(pad_token_id: int):
    """Pad input_ids to the right, create labels by setting the padding tokens to -100, and
    create attention_mask to ignore the padding tokens"""

    def collate(examples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        input_ids_unpadded = [example["input_ids"] for example in examples]
        labels_unpadded = [example["labels"] for example in examples]
        padding_length = MAX_TRAINING_SEQ_LENGTH if PAD_TO_MAX_LENGTH else None
        input_ids = pad_sequences(
            input_ids_unpadded, pad_token_id, "right", padding_length=padding_length
        )
        labels = pad_sequences(
            labels_unpadded, IGNORED_INDEX, "right", padding_length=padding_length
        )

        assert input_ids.shape == labels.shape
        assert len(input_ids) == len(examples)
        # Enforced in `map_raw_dataset`
        assert input_ids.shape[-1] <= MAX_TRAINING_SEQ_LENGTH
        if PAD_TO_MAX_LENGTH:
            assert input_ids.shape[-1] == MAX_TRAINING_SEQ_LENGTH

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(pad_token_id),
        }

    return collate


# LORA_CONFIG = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     inference_mode=False,
#     r=8,
#     lora_alpha=32,
#     lora_dropout=0.1,
# )

# N_CORES = 64


@dataclass(frozen=True)
class DatasetConfig:
    dataset_name: str
    dpo_jsonl_path: str | None = field(default=None)
    dpo_sft: bool = field(default=False)
    sub_name: str | None = field(default=None)
    n_samples: int = field(default=2000)


def split_python_function_body(code: str) -> tuple[str, str] | None:
    """Split the code into the function signature + docstring and the function body."""
    docstring_start = code.find('"""')
    docstring_end = code.find('"""\n', docstring_start + 3)
    if docstring_start == -1 or docstring_end == -1:
        return None
    docstring_end += 4
    header, body = code[:docstring_end], code[docstring_end:]
    return header, body


# def map_code_search_net(examples: dict[str, list[str]]) -> dict[str, list[str]]:
#     func_codes = examples["func_code_string"]
#     func_codes = [
#         new_code
#         for code in func_codes
#         if (new_code := utils.reformat_python(code)) is not None
#     ]
#     parts = filter(lambda x: x is not None, map(split_python_function_body, func_codes))
#     try:
#         signatures_and_docstrings, bodies = list(zip(*parts))  # type: ignore # fmt: off
#     except ValueError:
#         return {"prompt": [], "completion": []}
#     # # if len(result) != 2:
#     # #     breakpoint()
#     # signatures_and_docstrings, bodies = result
#     assert len(signatures_and_docstrings) == len(bodies)
#     return {
#         "prompt": list[str](signatures_and_docstrings),
#         "completion": list[str](bodies),
#     }


def map_wizardcoder_reproduce(examples: dict[str, list[str]]) -> dict[str, list[str]]:
    prompts = [
        PROMPT.format(instruction=instruction.rstrip())
        for instruction in examples["instruction"]
    ]
    completions = [output.lstrip() for output in examples["output"]]
    return {"prompt": prompts, "completion": completions}


def map_code_alpaca(examples: dict[str, list[str]]) -> dict[str, list[str]]:
    prompts = [
        PROMPT_DICT["prompt_no_input"].format(instruction=instruction)
        if input.strip() == ""
        else PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input)
        for instruction, input in zip(examples["instruction"], examples["input"])
    ]
    completions = [output.lstrip() for output in examples["output"]]
    return {"prompt": prompts, "completion": completions}


def map_gpt4_teacher(examples: dict[str, list[str]]) -> dict[str, list[str]]:
    prompts = [
        PROMPT.format(instruction=instruction)
        # PROMPT_DICT["prompt_no_input"].format(instruction=instruction)
        # if input.strip() == ""
        # else PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input)
        for instruction, input in zip(examples["instruction"], examples["input"])
    ]
    completions = [
        f"```python\n{output.strip()}\n```" for output in examples["response"]
    ]
    return {"prompt": prompts, "completion": completions}


def get_gpt4_teacher(config: DatasetConfig) -> Dataset:
    dataset = load_dataset(
        "json",
        num_proc=N_CORES,
        data_files=["gpt4-code-instruct-python.jsonl"],
        split="train",
    )
    indices = random.sample(range(len(dataset)), k=config.n_samples)
    dataset = dataset.select(indices)
    dataset = dataset.map(
        map_gpt4_teacher,
        batched=True,
        num_proc=N_CORES,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    )
    return dataset


def get_wizardcoder_reproduce(config: DatasetConfig) -> Dataset:
    # mlabonne/Evol-Instruct-Python-26k
    dataset = load_dataset(
        "theblackcat102/evol-codealpaca-v1", split="train", num_proc=N_CORES
    )
    if config.dpo_jsonl_path is not None:
        pref_dataset = load_dataset(
            "json", data_files=[config.dpo_jsonl_path], split="train"
        )

        def map_pref_data(examples: dict[str, list[str]]) -> dict[str, list[str]]:
            prompts: list[str] = []
            outputs: list[str] = []
            for index, prompt in zip(examples["index"], examples["prompt"]):
                assert dataset[index]["instruction"] == prompt
                prompts.append(PROMPT.format(instruction=prompt))
                outputs.append(dataset[index]["output"])
            if config.dpo_sft:
                return {
                    "prompt": prompts,
                    "completion": examples["rejected"],
                }
            else:
                return {
                    "prompt": prompts,
                    "chosen": examples["chosen"],
                    # "chosen": examples["chosen"],
                    "rejected": examples["rejected"],
                }

        pref_dataset = pref_dataset.map(map_pref_data, batched=True, num_proc=N_CORES)
        return pref_dataset
    # indices = random.sample(range(len(dataset)), k=config.n_samples)
    # dataset = dataset.select(indices)
    dataset = dataset.map(
        map_wizardcoder_reproduce,
        batched=True,
        num_proc=N_CORES,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    )
    return dataset


def get_code_alpaca(config: DatasetConfig) -> Dataset:
    dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train", num_proc=N_CORES)
    # indices = random.sample(range(len(dataset)), k=config.n_samples)
    # dataset = dataset.select(indices)
    dataset = dataset.map(
        map_code_alpaca,
        batched=True,
        num_proc=N_CORES,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    )
    return dataset


def map_instructional_prompt(examples: dict[str, list[str]]) -> dict[str, list[str]]:
    prompts = [
        PROMPT.format(instruction=instruction.rstrip())
        for instruction in examples["prompt"]
    ]
    completions = [output.lstrip() for output in examples["completion"]]
    return {"prompt": prompts, "completion": completions}


# def get_code_search_net(config: DatasetConfig) -> Dataset:
#     dataset = load_dataset(
#         "code_search_net", config.sub_name, split="train", num_proc=N_CORES
#     )
#     print("Size of code_search_net:", len(dataset))
#     dataset = dataset.select_columns(["func_code_string"])
#     dataset = dataset.map(
#         # map_code_search_net,
#         batched=True,
#         num_proc=N_CORES,
#         remove_columns=dataset.column_names,
#         load_from_cache_file=True,
#     )
#     indices = random.sample(range(len(dataset)), k=config.n_samples)
#     dataset = dataset.select(indices)
#     return dataset


# def get_humaneval(config: DatasetConfig) -> Dataset:
#     data = get_human_eval()
#     prompts = [value["prompt"] for value in data.values()]
#     completions = [value["canonical_solution"] for value in data.values()]
#     dataset = Dataset.from_dict({"prompt": prompts, "completion": completions})
#     return dataset


def get_evol_code(config: DatasetConfig) -> Dataset:
    dataset = load_dataset(
        "json", data_files=["evol_code_new_responded.jsonl"], split="train"
    )

    # dataset = dataset.remove_columns(["seed"])
    # dataset_old = load_dataset(
    #     "json", data_files=["evol_code_responded.jsonl"], split="train"
    # )
    # dataset = concatenate_datasets([dataset, dataset_old])

    dataset = dataset.rename_columns(
        {
            "instruction": "prompt",
            "response": "completion",
        }
    )
    dataset = dataset.map(map_instructional_prompt, batched=True, num_proc=N_CORES)
    return dataset


def get_complex_instructions(config: DatasetConfig) -> Dataset:
    dataset = load_dataset(
        "json",
        data_files=["complex-instructions.jsonl", "complex-instructions-1108.jsonl"],
        split="train",
    )
    dataset = dataset.rename_columns(
        {
            "instruction": "prompt",
            "response": "completion",
        }
    )
    dataset = dataset.map(map_instructional_prompt, batched=True, num_proc=N_CORES)
    return dataset


def train():
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DatasetConfig))
    model_args, training_args, dataset_config = cast(
        tuple[ModelArguments, TrainingArguments, DatasetConfig],
        parser.parse_args_into_dataclasses(),
    )
    if dataset_config.dataset_name == "sahil2801/CodeAlpaca-20k":
        dataset = get_code_alpaca(dataset_config)
    elif dataset_config.dataset_name == "gpt4-code-instruct-python.jsonl":
        dataset = get_gpt4_teacher(dataset_config)
    elif dataset_config.dataset_name == "evol-instruct":
        dataset = get_wizardcoder_reproduce(dataset_config)
    elif dataset_config.dataset_name == "complex-instructions":
        dataset = get_complex_instructions(dataset_config)
    else:
        dataset = get_evol_code(dataset_config)

    model_key = model_args.model_key
    if (model_name_or_path := model_args.model_name_or_path) is None:
        model_name_or_path = model_key

    tokenization_context = TokenizationContext.from_model_key(
        model_key, model_name_or_path
    )
    if dataset_config.dpo_jsonl_path is None or dataset_config.dpo_sft:
        map_raw_dataset = get_map_raw_dataset(tokenization_context)
        train_dataset = dataset.map(
            map_raw_dataset,
            batched=True,
            num_proc=N_CORES,
            remove_columns=dataset.column_names,
            load_from_cache_file=False,  # not args.overwrite_cache
            desc="Running tokenizer on train dataset",
        )
        msg = f"#Examples truncated: {sum(train_dataset['exceeding_length'])} / {len(dataset)}"
        print(msg)
    else:
        train_dataset = dataset

    # Shuffling
    if training_args.eval_steps is None and training_args.evaluation_strategy == "no":
        train_dataset = train_dataset.shuffle(seed=0)
        eval_dataset = None
    else:
        print("Splitting dataset")
        split_dataset = train_dataset.train_test_split(
            test_size=0.1, shuffle=True, seed=0
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

    state = get_model_context(
        model_key,
        model_name_or_path,
        tokenization_context,
        inference_mode=False,
    )
    # Use LoRA when doing DPO
    # if dataset_config.dpo_jsonl_path is not None and not dataset_config.dpo_sft:
    #     print("Getting peft model")
    #     state.model = get_peft_model(state.model, LORA_CONFIG)

    print("Parallel mode:", training_args.parallel_mode)
    data_collator = get_data_collator(state.tokenization_context.pad_token_id)

    # if dataset_config.dpo_jsonl_path is not None and not dataset_config.dpo_sft:
    #     print("Ready to do DPO")
    #     training_args.remove_unused_columns = False
    #     pad_token_id = state.tokenization_context.pad_token_id
    #     state.tokenization_context.tokenizer.pad_token_id = pad_token_id
    #     # state.model = state.model.to("cuda:0")
    #     # ref_model = state.model.to("cuda:0").eval()
    #     # from transformers import AutoModelForCausalLM

    #     # print("Loading ref model:", state.model.name_or_path)

    trainer = Trainer(
        model=state.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        # eval_dataset=small_eval_dataset,
        # compute_metrics=compute_metrics,
    )

    # NOTE: the checkpoint will override the initialized model
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()
