import random
from dataclasses import dataclass, field
from typing import cast

import torch
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import HfArgumentParser, Trainer, TrainingArguments

from magicoder.llm_wrapper import (
    DecodingConfig,
    EncodingConfig,
    TokenizationContext,
    get_model_context,
    pad_sequences,
)
from magicoder.prompt_template import MAGICODER_PROMPT
from magicoder.utils import N_CORES, read_jsonl, write_jsonl


@dataclass(frozen=True)
class ModelArguments:
    model_key: str
    model_name_or_path: str | None = None


# Ignored index in CrossEntropyLoss
IGNORED_INDEX = -100

# "python",
# "typescript",
# "csharp",
# "rust",
# "swift",
# "php",
# "java",
# "cpp",
# "shell",


def process_data(data: dict, lang: str) -> dict | None:
    docstring = data["docstring"].strip()
    if len(docstring) == 0:
        return None
    if docstring.count(docstring[0]) == len(docstring):
        return None
    code = data["function"].strip()
    if lang == "python":
        double_quote_doc = code.count('"""') >= 2
        single_quote_doc = code.count("'''") >= 2
        if double_quote_doc:
            start_index = code.index('"""')
            end_index = code.index('"""', start_index + 3)
        elif single_quote_doc:
            start_index = code.index("'''")
            end_index = code.index("'''", start_index + 3)
        else:
            # print("None..")
            # print(code)
            # print("None...")
            return None
        prompt = code[: end_index + 3]
        completion = code[end_index + 3 :]
    else:
        prefix = "/// " if lang != "shell" else "### "
        prompt = "".join(prefix + line for line in docstring.splitlines(keepends=True))
        codelines = code.splitlines(keepends=True)
        prompt += "\n" + "".join(codelines[:1])
        completion = "".join(codelines[1:])
    # print("[Prompt]", prompt, "[Completion]", completion, sep="\n")
    # breakpoint()
    return {"prompt": prompt, "completion": completion}


def preprocess(data: list[dict]) -> list[dict]:
    pairs: list[dict] = []
    for d in data:
        functions = d["function"]
        for function in functions:
            result = process_data(function, d["lang"])
            if result is not None:
                pairs.append(result)
    return pairs


def map_dataset(
    examples: dict[str, list[str]],
    args: "Args",
    context: TokenizationContext,
) -> dict:
    prompts = examples["prompt"]
    completions = examples["completion"]
    print("[Prompt]", prompts[0], "[Completion]", completions[0], sep="\n")

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
        input_id[: args.max_training_seq_length] for input_id in untruncated_input_ids
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


@dataclass(frozen=True)
class Args:
    covered_pairs_path: str
    all_pairs_path: str
    n_samples: int = field(default=75197)
    max_training_seq_length: int = field(default=1024)
    pad_to_max_length: bool = field(default=False)
    use_flash_attention: bool = field(default=False)


def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments, Args))
    model_args, training_args, args = cast(
        tuple[ModelArguments, TrainingArguments, Args],
        parser.parse_args_into_dataclasses(),
    )
    data_covered = read_jsonl(args.covered_pairs_path)
    preprocessed_covered = preprocess(data_covered)
    print("Length (covered):", len(preprocessed_covered))
    data_all_except_covered = read_jsonl(args.all_pairs_path)
    preprocessed_all = preprocess(data_all_except_covered)
    print("Length (all except coverred):", len(preprocessed_all))
    random.seed(training_args.seed)
    preprocessed_all = random.sample(
        preprocessed_all, k=args.n_samples - len(preprocessed_covered)
    )
    all = preprocessed_covered + preprocessed_all
    assert len(all) == args.n_samples
    random.shuffle(all)
    dataset = Dataset.from_list(all, split="train")
    assert len(dataset) == args.n_samples
    model_key = model_args.model_key
    if (model_name_or_path := model_args.model_name_or_path) is None:
        model_name_or_path = model_key

    tokenization_context = TokenizationContext.from_model_key(
        model_key, model_name_or_path
    )
    # if dataset_config.dpo_jsonl_path is None or dataset_config.dpo_sft:
    train_dataset = dataset.map(
        function=map_dataset,
        fn_kwargs=dict(args=args, context=tokenization_context),
        batched=True,
        num_proc=N_CORES,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,  # not args.overwrite_cache
        desc="Running tokenizer on train dataset",
    )
    msg = f"#Examples truncated: {sum(train_dataset['exceeding_length'])} / {len(train_dataset)}"
    print(msg)

    train_dataset = train_dataset.shuffle(seed=training_args.seed)

    state = get_model_context(
        model_key,
        model_name_or_path,
        tokenization_context,
        inference_mode=False,
        use_flash_attention=args.use_flash_attention,
    )

    print("Parallel mode:", training_args.parallel_mode)
    data_collator = get_data_collator(args, state.tokenization_context.pad_token_id)

    # neftune_noise_alpha
    trainer = Trainer(
        model=state.model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        data_collator=data_collator,
        # eval_dataset=small_eval_dataset,
        # compute_metrics=compute_metrics,
    )

    # NOTE: the checkpoint will override the initialized model
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    state.tokenization_context.tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
