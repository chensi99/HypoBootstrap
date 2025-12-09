import re
import logging

from .base import ChatAgent


logger = logging.getLogger(__name__)


class ObjectHypothesisInconsistencyEliminator(ChatAgent):


    async def __call__(
        self,
        object: str,
        object_hypothesis: list[str],
        task_name: str
    ) -> tuple[list[str], list[dict[str, str]], dict[str, int]]:
        object_hypothesis = object_hypothesis.copy()

        object_hypothesis_str = "\n".join(
            f"{j + 1}. {h}" for j, h in enumerate(object_hypothesis)
        ) if object_hypothesis else "(No pattern identified)"
        prompt = PROMPTS[task_name][self.model].format(
            object=object,
            object_hypothesis=object_hypothesis_str
        )
        interactions, usage = await self.chat_completion(prompt)

        response = interactions[-1]["content"]
        if response == "Inconsistent patterns:":
            response = "Inconsistent patterns: None"
        matcher = re.match(r"Inconsistent patterns: (None|\d+(?:, \d+)*)", response)
        if matcher:
            inconsistent_numbers = matcher.group(1)
        else:
            logger.warning(f"Unknown response: {response}")
            return object_hypothesis, interactions, usage
        
        if inconsistent_numbers == "None":
            pass
        else:
            inconsistent_numbers = map(
                lambda n: int(n),
                inconsistent_numbers.split(", ")
            )
            inconsistent_numbers = sorted(inconsistent_numbers, reverse=True)
            object_hypothesis_len = len(object_hypothesis)
            for n in inconsistent_numbers:
                if n > object_hypothesis_len:
                    continue
                object_hypothesis.pop(n - 1)
        
        return object_hypothesis, interactions, usage


PROMPTS = {

# ================================================
#   ACRE
# ================================================
"acre": {

    "gpt-4-0613":

"""Observe the data and its patterns below.

Data: {object}

Patterns:
{object_hypothesis}

Which patterns are logically inconsistent with the data and other patterns?

Please respond with pattern numbers as follows:
Inconsistent patterns: <number>, <number>, <number>, ...""",

    "deepseek-chat":

"""Observe the data and its patterns below.

Data: {object}

Patterns:
{object_hypothesis}

Which patterns are logically inconsistent with the data and other patterns?

Please respond with pattern numbers as follows. Do not output any other text.
Inconsistent patterns: <number>, <number>, <number>, ..."""

},

# ================================================
#   SCAN
# ================================================
"scan": {

    "gpt-4-0613":

"""Observe the data and its patterns below.

Data: "{object}"

Patterns:
{object_hypothesis}

Which patterns are logically inconsistent with the data and other patterns?

Please respond with pattern numbers as follows:
Inconsistent patterns: <number>, <number>, <number>, ...""",

    "deepseek-chat":

"""Observe the data and its patterns below.

Data: "{object}"

Patterns:
{object_hypothesis}

Which patterns are logically inconsistent with the data and other patterns?

Please respond with pattern numbers as follows. Do not output any other text.
Inconsistent patterns: <number>, <number>, <number>, ..."""

},

# ================================================
#   List Function
# ================================================
"list_function": {

    "gpt-4-0613":

"""Observe the data and its patterns below.

Data: {object}

Patterns:
{object_hypothesis}

Which patterns are logically inconsistent with the data and other patterns?

Please respond with pattern numbers as follows:
Inconsistent patterns: <number>, <number>, <number>, ...""",

    "deepseek-chat":

"""Observe the data and its patterns below.

Data: {object}

Patterns:
{object_hypothesis}

Which patterns are logically inconsistent with the data and other patterns?

Please respond with pattern numbers as follows. Do not output any other text.
Inconsistent patterns: <number>, <number>, <number>, ..."""

},

# ================================================
#   ARC
# ================================================
"arc": {

    "gpt-4-0613":

"""Observe the data and its patterns below.

Data: {object}

Patterns:
{object_hypothesis}

Which patterns are logically inconsistent with the data and other patterns?

Please respond with pattern numbers as follows:
Inconsistent patterns: <number>, <number>, <number>, ...""",

    "deepseek-chat":

"""Observe the data and its patterns below.

Data: {object}

Patterns:
{object_hypothesis}

Which patterns are logically inconsistent with the data and other patterns?

Please respond with pattern numbers as follows. Do not output any other text.
Inconsistent patterns: <number>, <number>, <number>, ..."""

},

}
