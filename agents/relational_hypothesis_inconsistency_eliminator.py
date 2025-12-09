import re
import logging

from .base import ChatAgent


logger = logging.getLogger(__name__)


class RelationalHypothesisInconsistencyEliminator(ChatAgent):


    async def __call__(
        self,
        input: str,
        output: str,
        relational_hypothesis: list[str],
        task_name: str
    ) -> tuple[list[str], list[dict[str, str]], dict[str, int]]:
        relational_hypothesis = relational_hypothesis.copy()

        relational_hypothesis_str = "\n".join(
            f"{j + 1}. {h}" for j, h in enumerate(relational_hypothesis)
        ) if relational_hypothesis else "(No similarities identified)"
        prompt = PROMPTS[task_name][self.model].format(
            input=input, output=output,
            relational_hypothesis=relational_hypothesis_str
        )
        interactions, usage = await self.chat_completion(prompt)

        response = interactions[-1]["content"]
        if response == "Inconsistent correlations:":
            response = "Inconsistent correlations: None"
        matcher = re.match(r"Inconsistent correlations: (None|\d+(?:, \d+)*)", response)
        if matcher:
            inconsistent_numbers = matcher.group(1)
        else:
            logger.warning(f"Unknown response: {response}")
            return relational_hypothesis, interactions, usage
        
        if inconsistent_numbers == "None":
            pass
        else:
            inconsistent_numbers = map(
                lambda n: int(n),
                inconsistent_numbers.split(", ")
            )
            inconsistent_numbers = sorted(inconsistent_numbers, reverse=True)
            relational_hypothesis_len = len(relational_hypothesis)
            for n in inconsistent_numbers:
                if n > relational_hypothesis_len:
                    continue
                relational_hypothesis.pop(n - 1)
        
        return relational_hypothesis, interactions, usage


PROMPTS = {

# ================================================
#   ACRE
# ================================================
"acre": {

    "gpt-4-0613":

"""Observe the input-output pair and the correlations between the input and the output given below.

Input: {input}
Output: {output}

Correlations:
{relational_hypothesis}

Which correlations are logically inconsistent with the input-output pair and other correlations?

Please respond with correlation numbers as follows:
Inconsistent correlations: <number>, <number>, <number>, ...""",

    "deepseek-chat":

"""Observe the input-output pair and the correlations between the input and the output given below.

Input: {input}
Output: {output}

Correlations:
{relational_hypothesis}

Which correlations are logically inconsistent with the input-output pair and other correlations?

Please respond with correlation numbers as follows. Do not output any other text.
Inconsistent correlations: <number>, <number>, <number>, ..."""

},

# ================================================
#   SCAN
# ================================================
"scan": {

    "gpt-4-0613":

"""Observe the input-output pair and the correlations between the input and the output given below.

Input: "{input}"
Output: "{output}"

Correlations:
{relational_hypothesis}

Which correlations are logically inconsistent with the input-output pair and other correlations?

Please respond with correlation numbers as follows:
Inconsistent correlations: <number>, <number>, <number>, ...""",

    "deepseek-chat":

"""Observe the input-output pair and the correlations between the input and the output given below.

Input: "{input}"
Output: "{output}"

Correlations:
{relational_hypothesis}

Which correlations are logically inconsistent with the input-output pair and other correlations?

Please respond with correlation numbers as follows. Do not output any other text.
Inconsistent correlations: <number>, <number>, <number>, ..."""

},

# ================================================
#   List Function
# ================================================
"list_function": {

    "gpt-4-0613":

"""Observe the input-output pair and the correlations between the input and the output given below.

Input: {input}
Output: {output}

Correlations:
{relational_hypothesis}

Which correlations are logically inconsistent with the input-output pair and other correlations?

Please respond with correlation numbers as follows:
Inconsistent correlations: <number>, <number>, <number>, ...""",

    "deepseek-chat":

"""Observe the input-output pair and the correlations between the input and the output given below.

Input: {input}
Output: {output}

Correlations:
{relational_hypothesis}

Which correlations are logically inconsistent with the input-output pair and other correlations?

Please respond with correlation numbers as follows. Do not output any other text.
Inconsistent correlations: <number>, <number>, <number>, ..."""

},

# ================================================
#   ARC
# ================================================
"arc": {

    "gpt-4-0613":

"""Observe the input-output pair and the correlations between the input and the output given below.

Input: {input}
Output: {output}

Correlations:
{relational_hypothesis}

Which correlations are logically inconsistent with the input-output pair and other correlations?

Please respond with correlation numbers as follows:
Inconsistent correlations: <number>, <number>, <number>, ...""",

    "deepseek-chat":

"""Observe the input-output pair and the correlations between the input and the output given below.

Input: {input}
Output: {output}

Correlations:
{relational_hypothesis}

Which correlations are logically inconsistent with the input-output pair and other correlations?

Please respond with correlation numbers as follows. Do not output any other text.
Inconsistent correlations: <number>, <number>, <number>, ..."""

},

}
