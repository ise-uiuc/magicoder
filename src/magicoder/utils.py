import functools
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypeVar

import openai
import tiktoken

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


# OpenAI API access
# Use environment variables!
# openai.organization = "org-pQ4H2mEb8OUHqSkIkP8b50k6"
# openai.api_key = os.getenv("OPENAI_API_KEY")


def retry_with_exponential_backoff(
    errors: tuple,
    initial_delay: float = 30,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 5,
):
    """Retry a function with exponential backoff."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Initialize variables
            num_retries = 0
            delay = initial_delay

            # Loop until a successful response or max_retries is hit or an exception is raised
            while True:
                try:
                    return func(*args, **kwargs)
                # Retry on specific errors
                except errors as e:
                    print(f"Error: {e}. Retrying in {delay} seconds...")
                    # Increment retries
                    num_retries += 1
                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise Exception(
                            f"Maximum number of retries ({max_retries}) exceeded."
                        )
                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())
                    # Sleep for the delay
                    time.sleep(delay)
                    # time.sleep(60)
                # Raise exceptions for any errors not specified
                except Exception as e:
                    raise e

        return wrapper

    return decorator


ERRORS = (
    openai.RateLimitError,
    openai.APIError,
    openai.APIConnectionError,
    openai.InternalServerError,
)

try:
    OPENAI_CLIENT: openai.OpenAI | None = openai.OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL")
    )
except openai.OpenAIError:
    OPENAI_CLIENT = None


@retry_with_exponential_backoff(ERRORS)
def chat_completions_with_backoff(*args, **kwargs):
    assert OPENAI_CLIENT is not None
    return OPENAI_CLIENT.chat.completions.create(*args, **kwargs)


@retry_with_exponential_backoff(ERRORS)
def completions_with_backoff(*args, **kwargs):
    assert OPENAI_CLIENT is not None
    return OPENAI_CLIENT.completions.create(*args, **kwargs)


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
