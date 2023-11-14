import random
from dataclasses import dataclass, field
from typing import TypedDict, cast

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import HfArgumentParser, Trainer, TrainingArguments

from magicoder.llm_wrapper import (  # ChatPiece,; ChatTokenizationContext,
    DecodingConfig,
    EncodingConfig,
    ModelContext,
    TokenizationContext,
    get_model_context,
    pad_sequences,
)
from magicoder.utils import N_CORES


@dataclass(frozen=True)
class ModelArguments:
    model_key: str
    model_name_or_path: str | None = None


# PROMPT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

# ### Instruction:
# {instruction}

# ### Response:
# """

MAGICODER_PROMPT = """Write a solution to the following programming problem.

[Problem]
{problem}

[Solution]
"""

# PROMPT_DICT = {
#     "prompt_input": (
#         "Below is an instruction that describes a task, paired with an input that provides further context. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
#     ),
#     "prompt_no_input": (
#         "Below is an instruction that describes a task. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Response:\n"
#     ),
# }


# Ignored index in CrossEntropyLoss
IGNORED_INDEX = -100


def get_map_raw_dataset(args: "Args", context: TokenizationContext):
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
            len(input_id) > args.max_training_seq_length
            for input_id in untruncated_input_ids
        ]
        input_ids = [
            input_id[: args.max_training_seq_length]
            for input_id in untruncated_input_ids
        ]
        # NOTE: no need to set EOF to IGNORED_INDEX as it is *implicitly* ignored inside
        # the model.forward that shifts the logits left by 1
        labels = [
            (list(map(lambda _: IGNORED_INDEX, instruction_ids)) + response_ids)[
                : args.max_training_seq_length
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


def get_data_collator(args: "Args", pad_token_id: int):
    """Pad input_ids to the right, create labels by setting the padding tokens to -100, and
    create attention_mask to ignore the padding tokens"""

    def collate(examples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        input_ids_unpadded = [example["input_ids"] for example in examples]
        labels_unpadded = [example["labels"] for example in examples]
        padding_length = (
            args.max_training_seq_length if args.pad_to_max_length else None
        )
        input_ids = pad_sequences(
            input_ids_unpadded, pad_token_id, "right", padding_length=padding_length
        )
        labels = pad_sequences(
            labels_unpadded, IGNORED_INDEX, "right", padding_length=padding_length
        )

        assert input_ids.shape == labels.shape
        assert len(input_ids) == len(examples)
        # Enforced in `map_raw_dataset`
        assert input_ids.shape[-1] <= args.max_training_seq_length
        if args.pad_to_max_length:
            assert input_ids.shape[-1] == args.max_training_seq_length

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
class Args:
    datafile_paths: list[str] = field(default_factory=list)
    max_training_seq_length: int = field(default=1180)
    pad_to_max_length: bool = field(default=False)
    eval_dataset_size: float = field(
        default=0.05, metadata={"help": "0--1 means ratio, >1 means number of examples"}
    )


# def map_wizardcoder_reproduce(examples: dict[str, list[str]]) -> dict[str, list[str]]:
#     prompts = [
#         PROMPT.format(instruction=instruction.rstrip())
#         for instruction in examples["instruction"]
#     ]
#     completions = [output.lstrip() for output in examples["output"]]
#     return {"prompt": prompts, "completion": completions}


# def map_code_alpaca(examples: dict[str, list[str]]) -> dict[str, list[str]]:
#     prompts = [
#         PROMPT_DICT["prompt_no_input"].format(instruction=instruction)
#         if input.strip() == ""
#         else PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input)
#         for instruction, input in zip(examples["instruction"], examples["input"])
#     ]
#     completions = [output.lstrip() for output in examples["output"]]
#     return {"prompt": prompts, "completion": completions}


# def get_wizardcoder_reproduce(config: DatasetConfig) -> Dataset:
#     # mlabonne/Evol-Instruct-Python-26k
#     dataset = load_dataset(
#         "theblackcat102/evol-codealpaca-v1", split="train", num_proc=N_CORES
#     )
#     dataset = dataset.map(
#         map_wizardcoder_reproduce,
#         batched=True,
#         num_proc=N_CORES,
#         remove_columns=dataset.column_names,
#         load_from_cache_file=False,
#     )
#     return dataset


# def get_code_alpaca(config: DatasetConfig) -> Dataset:
#     dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train", num_proc=N_CORES)
#     # indices = random.sample(range(len(dataset)), k=config.n_samples)
#     # dataset = dataset.select(indices)
#     dataset = dataset.map(
#         map_code_alpaca,
#         batched=True,
#         num_proc=N_CORES,
#         remove_columns=dataset.column_names,
#         load_from_cache_file=False,
#     )
#     return dataset


def get_dataset(args: Args) -> Dataset:
    dataset = load_dataset("json", data_files=args.datafile_paths, split="train")

    # dataset = dataset.remove_columns(["seed"])
    # dataset_old = load_dataset(
    #     "json", data_files=["evol_code_responded.jsonl"], split="train"
    # )
    # dataset = concatenate_datasets([dataset, dataset_old])
    def map_fn(examples: dict[str, list[str]]) -> dict[str, list[str]]:
        prompts = [
            MAGICODER_PROMPT.format(problem=problem) for problem in examples["problem"]
        ]
        completions = [output.lstrip() for output in examples["solution"]]
        return {"prompt": prompts, "completion": completions}

    dataset = dataset.map(
        map_fn,
        batched=True,
        num_proc=N_CORES,
        remove_columns=dataset.column_names,
    )
    return dataset


def train():
    parser = HfArgumentParser((ModelArguments, TrainingArguments, Args))
    model_args, training_args, args = cast(
        tuple[ModelArguments, TrainingArguments, Args],
        parser.parse_args_into_dataclasses(),
    )
    giga = 70
    print("creating tensors")
    x = torch.zeros((giga * 1024**3,), dtype=torch.int8, device="cuda")
    del x
    dataset = get_dataset(args)

    model_key = model_args.model_key
    if (model_name_or_path := model_args.model_name_or_path) is None:
        model_name_or_path = model_key

    tokenization_context = TokenizationContext.from_model_key(
        model_key, model_name_or_path
    )
    # if dataset_config.dpo_jsonl_path is None or dataset_config.dpo_sft:
    map_raw_dataset = get_map_raw_dataset(args, tokenization_context)
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
    # else:
    #     train_dataset = dataset

    # Shuffling
    if training_args.eval_steps is None and training_args.evaluation_strategy == "no":
        train_dataset = train_dataset.shuffle(seed=training_args.seed)
        eval_dataset = None
    else:
        print("Splitting dataset")
        split_dataset = train_dataset.train_test_split(
            test_size=args.eval_dataset_size,
            shuffle=True,
            seed=training_args.seed,
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
    data_collator = get_data_collator(args, state.tokenization_context.pad_token_id)

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
