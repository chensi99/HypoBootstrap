import logging
import asyncio
from typing import Optional

import openai

from utils.api_utils import Cache, chat_completion_with_backoff


logger = logging.getLogger(__name__)


class ChatAgent:

    def __init__(
        self,
        *,
        api_client: openai.AsyncOpenAI,
        api_semaphore: asyncio.Semaphore,
        model: str,
        cache_mode: str,
        cache_file: str,
        temperature: float = 0
    ):
        self.api_client = api_client
        self.semaphore = api_semaphore
        
        self.model = model
        self.temperature = temperature

        self.cache = Cache(cache_mode, cache_file)

    async def chat_completion(
        self,
        prompt: Optional[str],
        interactions: Optional[list[dict[str, str]]] = None,
    ) -> tuple[list[dict[str, str]], dict[str, int]]:
        interactions = [] if interactions is None else interactions.copy()
        if prompt is not None:
            interactions.append({"role": "user", "content": prompt})
        else:
            assert interactions and interactions[-1]["role"] == "user"  # prompt is already in `interactions`
        
        cache_key = Cache.gen_key(self.model, interactions, self.temperature, 1)
        if cache_key in self.cache:
            cache_value = self.cache[cache_key]
            response, usage = cache_value["response"], cache_value["usage"]
        else:
            async with self.semaphore:
                completion = await chat_completion_with_backoff(self.api_client, self.model, interactions, temperature=self.temperature)
                if completion.choices[0].finish_reason != "stop":
                    logger.error(f"Interactions: {interactions}")
                    logger.error(f"Completion: {completion.choices[0].message.content}")
                    if completion.choices[0].finish_reason == "length":
                        logger.error(f"Response exceeds max length!")
                    else:
                        raise ValueError(f"Unknown stop reason: {completion.choices[0].finish_reason}")
            
            response = completion.choices[0].message.content
            _usage = completion.usage.to_dict()
            usage = {
                "prompt_tokens": _usage["prompt_tokens"],
                "completion_tokens": _usage["completion_tokens"]
            }

            self.cache[cache_key] = {"response": response, "usage": usage}

        interactions.append({"role": "assistant", "content": response})

        return interactions, usage
