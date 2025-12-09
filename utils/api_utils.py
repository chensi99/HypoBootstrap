import pickle
import random
import logging
import asyncio
from pathlib import Path
from typing import Awaitable

import openai


logger = logging.getLogger(__name__)


class Cache:

    # only one instance can exist
    _instance = None
    def __new__(cls, cache_mode: str, cache_file: str):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        else:
            if cache_mode == "file":
                assert cls._instance.cache_path == Path(cache_file)
            elif cache_mode == "memory":
                pass
            else:
                raise ValueError(f"Unknown cache mode: {cache_mode}")
        return cls._instance

    def __init__(self, cache_mode: str, cache_file: str):
        self.cache_mode = cache_mode
        if cache_mode == "file":
            self.cache_path = Path(cache_file)
            if self.cache_path.exists():
                with self.cache_path.open("rb") as reader:
                    self.cache = pickle.load(reader)
            else:
                # key: (model, history, temperature, n)
                # value: {"response": str, "usage": dict}
                self.cache = dict()
        elif cache_mode == "memory":
            self.cache = dict()
        else:
            raise ValueError(f"Unknown cache mode: {cache_mode}")

    def __getitem__(self, key: tuple) -> dict:
        return self.cache[key]

    def __setitem__(self, key: tuple, value: dict) -> None:
        # WARNING: latter values may overwrite former ones if two api calls
        # with the same key proceed simultaneously
        self.cache[key] = value
        if self.cache_mode == "file":
            with self.cache_path.open("wb") as writer:
                pickle.dump(self.cache, writer)
        elif self.cache_mode == "memory":
            pass
        else:
            raise ValueError(f"Unknown cache mode: {self.cache_mode}")

    def __contains__(self, key):
        return key in self.cache

    @classmethod
    def gen_key(
        cls,
        model: str,
        history: list[dict[str, str]],
        temperature: float,
        n: int
    ) -> tuple:
        return (
            model,
            tuple(tuple(h.items()) for h in history),
            temperature,
            n
        )


def retry_with_exponential_backoff(
    func: Awaitable,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (
        openai.RateLimitError,
        openai.APIConnectionError,
    ),
) -> Awaitable:
    """Retry a function with exponential backoff."""

    async def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return await func(*args, **kwargs)
            except errors as e:
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")

                logging.warning(repr(e))
                delay *= exponential_base * (1 + jitter * random.random())
                await asyncio.sleep(delay)
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
async def chat_completion_with_backoff(
    api_client: openai.AsyncOpenAI,
    model: str,
    messages: list[dict],
    temperature: float = 0,
    n: int = 1
) -> openai.types.Completion:
    completion = await api_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        n=n
    )
    return completion
