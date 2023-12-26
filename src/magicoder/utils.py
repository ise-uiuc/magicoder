import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypeVar

import openai
import tenacity
import tiktoken

# import vertexai
# from google.api_core.exceptions import InternalServerError, ResourceExhausted
# from vertexai.preview.generative_models import FinishReason, GenerativeModel

N_CORES = 1 if (count := os.cpu_count()) is None or count == 0 else count // 2


def read_jsonl(path: str | Path) -> list[Any]:
    """Read lines of JSON from a file (including '\n')."""
    with Path(path).open("r") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: str | Path, data: Sequence[Mapping]):
    # cannot use `dict` here as it is invariant
    with Path(path).open("w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


# def reformat_python(code: str) -> str | None:
#     """Reformat Python code using Black."""

#     try:
#         return black.format_str(code, mode=black.Mode())
#     except Exception:
#         return None


_T = TypeVar("_T")


def chunked(seq: Sequence[_T], n: int) -> Iterable[Sequence[_T]]:
    """Yield successive n-sized chunks from seq."""
    return (seq[i : i + n] for i in range(0, len(seq), n))


try:
    OPENAI_CLIENT: openai.OpenAI | None = openai.OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL")
    )
except openai.OpenAIError:
    OPENAI_CLIENT = None


def retry(errors: Any, max_attempts: int = 5):
    return tenacity.retry(
        retry=tenacity.retry_if_exception_type(errors),
        wait=tenacity.wait_exponential(multiplier=1, min=5, max=20),
        stop=tenacity.stop_after_attempt(max_attempts),
        before_sleep=print,
    )


ERRORS = (
    openai.RateLimitError,
    openai.APIError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class OpenAIClient:
    def __init__(self):
        self.client = openai.OpenAI()
        self.async_client = openai.AsyncClient()

    @retry(ERRORS)
    def chat_completions_with_backoff(self, *args, **kwargs):
        return self.client.chat.completions.create(*args, **kwargs)

    @retry(ERRORS)
    async def chat_completions_with_backoff_async(self, *args, **kwargs):
        return await self.async_client.chat.completions.create(*args, **kwargs)

    async def dispatch_chat_completions(
        self,
        requests: list[dict[str, Any]],
        delay: float | None = None,
    ):
        """Dispatch chat completions requests asynchronously.
        Args:
            requests: a list of API argument names to values.
            delay: interval between requests.
        """

        async def delayed_request(request: dict[str, Any]):
            """Prevent quantized rate limit:
            https://help.openai.com/en/articles/6891753-rate-limit-advice"""
            if delay is not None:
                # synchronized sleep
                time.sleep(delay)
            return await self.chat_completions_with_backoff_async(**request)

        tasks = [delayed_request(request) for request in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)


@retry(ERRORS)
def chat_completions_with_backoff(*args, **kwargs):
    assert OPENAI_CLIENT is not None
    return OPENAI_CLIENT.chat.completions.create(*args, **kwargs)


@retry(ERRORS)
def completions_with_backoff(*args, **kwargs):
    assert OPENAI_CLIENT is not None
    return OPENAI_CLIENT.completions.create(*args, **kwargs)


# INIT = False


# @retry((IndexError, ResourceExhausted, InternalServerError))
# async def gemini_chat_completions_with_backoff(*args, **kwargs):
#     # def get_chat_response(message):
#     global INIT
#     if not INIT:
#         vertexai.init(project="thematic-axle-388805", location="us-central1")
#         INIT = True
#     assert kwargs["model"] == "gemini-pro"
#     messages = kwargs["messages"]
#     # temperature = kwargs["temperature"]
#     model = GenerativeModel(kwargs["model"])
#     if messages[0]["role"] == "system":
#         messages[1]["content"] = (
#             messages[0]["content"] + "\n\n" + messages[1]["content"]
#         )
#         messages = messages[1:]
#     prefix = """You are an excellent and informative coding assistant that offers high-quality responses. You would reason about complex problems first before you make a response.\n\n"""
#     contents = [
#         create_message(message["role"], prefix + message["content"])
#         for message in messages
#     ]
#     response = await model.generate_content_async(
#         contents,
#         generation_config={
#             "temperature": 0.0,
#             "top_p": 1.0,
#         },
#         safety_settings={0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
#     )
#     candidate = response.candidates[0]
#     if candidate.finish_reason != FinishReason.STOP:
#         print("No response generated!!!")
#         raise IndexError("No response generated")
#     return candidate.text


# def create_message(role, content):
#     return {"role": role, "parts": [{"text": content}]}


# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_string(string: str, model: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    # encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def compute_fingerprint(*args: Any, hash_length: int | None = None) -> str:
    combined = "".join(map(str, args))
    content = hashlib.sha256(combined.encode()).hexdigest()
    if hash_length is not None:
        content = content[:hash_length]
    return content
