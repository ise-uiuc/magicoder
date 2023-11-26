from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig as TransformersGenerationConfig
from transformers import PreTrainedModel, PreTrainedTokenizer

# from peft import PeftModel, PeftConfig

# Tokenization side modeling
PaddingSide = Literal["left", "right"]
# Input: a batch of chat pieces; Output: a batch of instructions and responses
# The instances should encode in a way that the model can predict response from instruction
InputIds = list[int]


@dataclass(frozen=True)
class DecodingConfig:
    skip_special_tokens: bool

    @staticmethod
    def default() -> "DecodingConfig":
        return DecodingConfig(skip_special_tokens=True)


# TransformChatPieceFunc = Callable[[ChatPiece], tuple[str, str]]


@dataclass(frozen=True)
class EncodingConfig:
    add_bos: bool
    add_eos: bool
    truncation: int | None = field(default=None)

    @staticmethod
    def default() -> "EncodingConfig":
        return EncodingConfig(add_bos=False, add_eos=False)


@dataclass(frozen=True)
class TokenizationContext:
    tokenizer: PreTrainedTokenizer
    pad_token_id: int
    bos_token: str
    eos_token: str

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @staticmethod
    def from_model_key(
        model_key: str, model_name_or_path: str | None = None
    ) -> "TokenizationContext":
        # use_fast = model_key not in SupportedModelKeys.codellama_models()
        use_fast = True
        # if model_name_or_path is None:
        #     model_name_or_path = model_key
        # TODO: check if tokenizers cannot be loaded with path
        model_name_or_path = model_key
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast)
        tokenization_context = TokenizationContext.from_tokenizer(tokenizer)
        return tokenization_context

    @staticmethod
    def from_tokenizer(tokenizer: PreTrainedTokenizer) -> "TokenizationContext":
        if (pad_token_id := tokenizer.pad_token_id) is None:
            pad_token_id = tokenizer.eos_token_id
        assert pad_token_id is not None
        bos_token = tokenizer.bos_token
        eos_token = tokenizer.eos_token
        return TokenizationContext(
            tokenizer=tokenizer,
            pad_token_id=pad_token_id,
            bos_token=bos_token,
            eos_token=eos_token,
        )

    def encode(self, config: EncodingConfig, text_list: list[str]) -> list[list[int]]:
        # eos_token = self.eos_token if config.add_eos else ""
        # bos_token = self.bos_token if config.add_bos else ""
        # if eos_token != "" or bos_token != "":
        #     text_list = [f"{bos_token}{text}{eos_token}" for text in text_list]
        # The string concatenation above may not always work for all tokenizers (strange).
        # e.g., when codellama's tokenizer is used with "<s>[INST]".
        if config.truncation is not None:
            extra_args = dict(truncation=True, max_length=config.truncation)
        else:
            extra_args = {}
        input_ids = self.tokenizer(
            text_list,
            add_special_tokens=False,
            **extra_args,
        )["input_ids"]
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        bos_token_ids = (
            [bos_token_id] if config.add_bos and bos_token_id is not None else []
        )
        eos_token_ids = (
            [eos_token_id] if config.add_eos and eos_token_id is not None else []
        )
        if len(bos_token_ids) > 0 or len(eos_token_ids) > 0:
            input_ids = [
                bos_token_ids + input_id + eos_token_ids for input_id in input_ids
            ]
        return input_ids

    def decode(
        self, config: DecodingConfig, input_ids: list[InputIds] | torch.Tensor
    ) -> list[str]:
        return self.tokenizer.batch_decode(
            input_ids, skip_special_tokens=config.skip_special_tokens
        )

    def encode_with_padding(
        self, padding_side: PaddingSide, config: EncodingConfig, text_list: list[str]
    ) -> torch.Tensor:
        input_ids_unpadded = self.encode(config, text_list)
        return pad_sequences(
            sequences=input_ids_unpadded,
            pad_value=self.pad_token_id,
            padding_side=padding_side,
        )


def pad_sequences(
    sequences: list[list[int]],
    pad_value: int,
    padding_side: Literal["left", "right"],
    dtype: torch.dtype = torch.long,
    padding_length: int | None = None,
) -> torch.Tensor:
    tensors = [torch.tensor(sequence, dtype=dtype) for sequence in sequences]
    max_len = max(len(sequence) for sequence in sequences)
    if padding_length is not None:
        assert padding_length >= max_len, "padding_length must be >= max_len"
        max_len = padding_length
    if padding_side == "right":
        result = torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=pad_value
        )
        remaining_length = max_len - result.shape[-1]
        # padding matrix of (batch_size * remaining_length)
        shape = result.shape[:-1] + (remaining_length,)
        padding_matrix = torch.full(shape, pad_value, dtype=dtype)
        result = torch.cat([result, padding_matrix], dim=-1)
    else:
        padded_tensors: list[torch.Tensor] = []
        for tensor in tensors:
            n_pad_values = max_len - len(tensor)
            padded_values = torch.full((n_pad_values,), pad_value, dtype=dtype)
            padded_tensor = torch.cat([padded_values, tensor], dim=0)
            assert len(padded_tensor) == max_len
            padded_tensors.append(padded_tensor)
        result = torch.stack(padded_tensors, dim=0)
    assert result.shape == torch.Size([len(sequences), max_len])
    return result


# @dataclass(frozen=True)
# class ChatTokenizationContext:
#     # TODO(refactor): maybe make transform_chatpieces to be
#     # Callable[[TokenizationContext], TransformChatPieceFunc]
#     # and remove `tokenization_context` from this class
#     tokenization_context: TokenizationContext
#     transform_chatpiece: TransformChatPieceFunc
#     decode_responses: Callable[[list[InputIds] | torch.Tensor], list[str]]

#     def encode_instructions(self, instructions: list[str]) -> torch.Tensor:
#         # transform_chatpiece = self.transform_chatpiece
#         # instructions = [
#         #     transform_chatpiece(ChatPiece(instruction, ""))[0]
#         #     for instruction in instructions
#         # ]
#         chat_pieces = [ChatPiece(instruction, "") for instruction in instructions]
#         input_ids_list, _ = self.encode_chatpieces(chat_pieces)
#         # Non-right padding is essential for batched generation.
#         # See: https://github.com/huggingface/transformers/issues/18478
#         input_ids = pad_sequences(
#             input_ids_list, self.tokenization_context.pad_token_id, "left"
#         )
#         return input_ids

#     def encode_responses(self, responses: list[str]) -> torch.Tensor:
#         return self.tokenization_context.encode_with_padding(
#             "right",
#             EncodingConfig(add_bos=False, add_eos=True),
#             responses,
#         )

#     def encode_chatpieces(
#         self, chats: list[ChatPiece]
#     ) -> tuple[list[InputIds], list[InputIds]]:
#         instructions: list[str] = []
#         responses: list[str] = []
#         for chat in chats:
#             instruction, response = self.transform_chatpiece(chat)
#             instructions.append(instruction)
#             responses.append(response)
#         # Always add bos and eos when encoding chat pieces
#         # TODO: is it correct?
#         instruction_config = EncodingConfig(add_bos=True, add_eos=False)
#         response_config = EncodingConfig(add_bos=False, add_eos=True)
#         instruction_ids = self.tokenization_context.encode(
#             instruction_config, instructions
#         )
#         response_ids = self.tokenization_context.encode(response_config, responses)
#         return instruction_ids, response_ids


# Inference side modeling
@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    top_p: float
    temperature: float
    max_length: int = field(
        default=99999999999999999,
        metadata={
            "help": "The max length of the sequence to generate, including inputs."
            "Will be considered in tandem with max_new_tokens. Whichever is more restrictive will be used."
        },
    )

    def to_transformers_generation_config(
        self, eos_token_id: int, pad_token_id: int
    ) -> TransformersGenerationConfig:
        do_sample = self.temperature != 0.0
        kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            top_p=self.top_p,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=do_sample,
        )
        if do_sample:
            kwargs["temperature"] = self.temperature
        return TransformersGenerationConfig(**kwargs)

    def with_max_new_tokens_being(self, max_new_tokens: int) -> "GenerationConfig":
        return GenerationConfig(max_new_tokens, self.top_p, self.temperature)

    @staticmethod
    def default() -> "GenerationConfig":
        return GenerationConfig(200, 1.0, 1.0)


@dataclass(frozen=True)
class Response:
    raw_inputs: torch.Tensor
    raw_outputs: torch.Tensor
    decoded_outputs: list[str]


@dataclass
class ModelContext:
    tokenization_context: TokenizationContext
    model: PreTrainedModel
    max_context_size: int

    def generate(
        self, config: GenerationConfig, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Raise ValueError when input_ids exceeds the context."""
        # NOTE: this implementation is only for decoder-only models
        # Recalculate the max number of tokens to avoid overflowing the context window
        input_len = input_ids.shape[1]
        if input_len >= self.max_context_size:
            raise ValueError(
                f"Input length {input_len} >= Context size {self.max_context_size}"
            )
        assert input_len < self.max_context_size

        max_context_size = min(
            self.max_context_size - input_len,
            config.max_new_tokens,
            config.max_length - input_len,
        )
        config = config.with_max_new_tokens_being(max_context_size)

        tf_config = config.to_transformers_generation_config(
            eos_token_id=self.tokenization_context.eos_token_id,
            pad_token_id=self.tokenization_context.pad_token_id,
        )
        attention_mask = input_ids.ne(self.tokenization_context.pad_token_id)
        # breakpoint()
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=tf_config,
        )
        # input_len = input_ids.shape[1]
        return outputs[:, input_len:]

    def complete(self, config: GenerationConfig, prompts: list[str]) -> Response:
        encoding_config = EncodingConfig(add_bos=True, add_eos=False)
        input_ids = self.tokenization_context.encode_with_padding(
            "left", encoding_config, prompts
        )
        input_ids = input_ids.to(self.model.device)
        output_ids = self.generate(config, input_ids)
        decoding_config = DecodingConfig(skip_special_tokens=True)
        output_strings = self.tokenization_context.decode(decoding_config, output_ids)
        return Response(
            raw_inputs=input_ids,
            raw_outputs=output_ids,
            decoded_outputs=output_strings,
        )

    # def respond_instructions(
    #     self, config: GenerationConfig, instructions: list[str]
    # ) -> Response:
    #     input_ids = self.chat_tokenization_context.encode_instructions(instructions)
    #     # encoding_config = EncodingConfig(add_bos=True, add_eos=False)
    #     # input_ids = self.tokenization_context.encode_with_padding(
    #     #     "left", encoding_config, instructions
    #     # )
    #     # Make sure the inputs are on the same device as the model
    #     input_ids = input_ids.to(self.model.device)
    #     outputs = self.generate(config, input_ids)
    #     responses = self.chat_tokenization_context.decode_responses(outputs)
    #     return Response(
    #         raw_inputs=input_ids,
    #         raw_outputs=outputs,
    #         decoded_outputs=responses,
    #     )


class SupportedModelKeys(Enum):
    # StarCoder-based models
    STARCODER_15B = "bigcode/starcoder"
    WIZARDCODER_STARCODER_15B = "WizardLM/WizardCoder-15B-V1.0"

    # CodeLlama-based models
    WIZARDCODER_CODELLAMA_PYTHON_7B = "WizardLM/WizardCoder-Python-7B-V1.0"
    WIZARDCODER_CODELLAMA_PYTHON_13B = "WizardLM/WizardCoder-Python-13B-V1.0"
    WIZARDCODER_CODELLAMA_PYTHON_34B = "WizardLM/WizardCoder-Python-34B-V1.0"
    CODELLAMA_PYTHON_7B = "codellama/CodeLlama-7b-Python-hf"
    CODELLAMA_PYTHON_13B = "codellama/CodeLlama-13b-Python-hf"
    CODELLAMA_PYTHON_34B = "codellama/CodeLlama-34b-Python-hf"

    # DeepSeek-Coder-based models
    DEEPSEEK_CODER_6_7B = "deepseek-ai/deepseek-coder-6.7b-base"

    @staticmethod
    def all() -> list[str]:
        return [member.value for member in SupportedModelKeys]

    @staticmethod
    def codellama_models() -> list[str]:
        return [
            SupportedModelKeys.CODELLAMA_PYTHON_7B.value,
            SupportedModelKeys.CODELLAMA_PYTHON_13B.value,
            SupportedModelKeys.CODELLAMA_PYTHON_34B.value,
            # SupportedModelKeys.WIZARDCODER_CODELLAMA_PYTHON_7B.value,
            # SupportedModelKeys.WIZARDCODER_CODELLAMA_PYTHON_13B.value,
            # SupportedModelKeys.WIZARDCODER_CODELLAMA_PYTHON_34B.value,
        ]

    @staticmethod
    def codellama_based_models() -> list[str]:
        return SupportedModelKeys.codellama_models() + [
            SupportedModelKeys.WIZARDCODER_CODELLAMA_PYTHON_7B.value,
            SupportedModelKeys.WIZARDCODER_CODELLAMA_PYTHON_13B.value,
            SupportedModelKeys.WIZARDCODER_CODELLAMA_PYTHON_34B.value,
        ]

    @staticmethod
    def starcoder_based_models() -> list[str]:
        return [
            SupportedModelKeys.STARCODER_15B.value,
            SupportedModelKeys.WIZARDCODER_STARCODER_15B.value,
        ]

    @staticmethod
    def deepseekcoder_based_models() -> list[str]:
        return [SupportedModelKeys.DEEPSEEK_CODER_6_7B.value]


def get_model_context(
    model_key: str,
    model_name_or_path: str | None = None,
    tokenization_context: TokenizationContext | None = None,
    inference_mode: bool = True,
) -> ModelContext:
    # `model_key` defines the model and the tokenizer to use, while `model_name_or_path`
    # defines where to load the weights. It can be from a local directory.
    assert model_key in SupportedModelKeys.all(), model_key
    if model_name_or_path is None:
        model_name_or_path = model_key
    if model_key in SupportedModelKeys.codellama_based_models():
        max_context_size = 16384
    elif model_key in SupportedModelKeys.starcoder_based_models():
        max_context_size = 8192
    else:
        assert model_key in SupportedModelKeys.deepseekcoder_based_models()
        max_context_size = 16384
    if tokenization_context is None:
        tokenization_context = TokenizationContext.from_model_key(model_key)
    # TODO: check if all these models use bfloat16
    dtype = torch.bfloat16
    other_kwargs = {"device_map": "auto"} if inference_mode else {}
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        **other_kwargs,
    )
    return ModelContext(tokenization_context, model, max_context_size)


def form_starcoder_infill(prefix: str, suffix: str) -> str:
    FIM_PREFIX = "<fim_prefix>"
    FIM_MIDDLE = "<fim_middle>"
    FIM_SUFFIX = "<fim_suffix>"
    prompt = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"
    return prompt


def form_codellama_infill(prefix: str, suffix: str) -> str:
    # NOTE: not using <FILL_ME> because it's treated as a special token
    # but we pass `add_special_tokens=False` to the tokenizer
    return f"▁<PRE>{prefix}▁<SUF>{suffix}▁<MID>"


def form_deepseekcoder_infill(
    tokenizer: PreTrainedTokenizer, prefix: str, suffix: str
) -> str:
    def get_str(idx: int) -> str:
        return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(idx))

    FIM_PREFIX = get_str(32016)
    FIM_MIDDLE = get_str(32015)
    FIM_SUFFIX = get_str(32017)
    assert "begin" in FIM_PREFIX and "hole" in FIM_MIDDLE and "end" in FIM_SUFFIX
    prompt = f"{FIM_PREFIX}{prefix}{FIM_MIDDLE}{suffix}{FIM_SUFFIX}"
    return prompt


def create_infilling_prompt(
    model_key: str,
    prefix: str,
    suffix: str,
    tokenizer: PreTrainedTokenizer | None = None,
) -> str:
    """TODO: how to separate magicoder from the others (magicoder now has a base
    model key. Consider change it?)"""
    if model_key in SupportedModelKeys.starcoder_based_models():
        return form_starcoder_infill(prefix, suffix)
    elif (
        model_key in SupportedModelKeys.codellama_based_models()
        and not "python" in model_key.lower()
    ):
        return form_codellama_infill(prefix, suffix)
    elif model_key in SupportedModelKeys.deepseekcoder_based_models():
        assert tokenizer is not None
        return form_deepseekcoder_infill(tokenizer, prefix, suffix)

    # TODO: other models
    assert False, f"Unsupported model key: {model_key}"
