from .base import ChatAgent
from .utils import extract_list_from_response


class RelationalHypothesisGenerator(ChatAgent):


    async def __call__(
        self,
        input: str,
        output: str,
        input_object_hypothesis: str,
        output_object_hypothesis: str,
        task_name: str
    ) -> tuple[list[str], list[dict[str, str]], dict[str, int]]:
        input_object_hypothesis_str = "\n".join(
            f"{j + 1}. {h}" for j, h in enumerate(input_object_hypothesis)
        ) if input_object_hypothesis else "(No pattern identified)"
        output_object_hypothesis_str = "\n".join(
            f"{j + 1}. {h}" for j, h in enumerate(output_object_hypothesis)
        ) if output_object_hypothesis else "(No pattern identified)"
        prompt = PROMPTS[task_name][self.model].format(
            input=input, output=output,
            input_object_hypothesis=input_object_hypothesis_str,
            output_object_hypothesis=output_object_hypothesis_str
        )
        interactions, usage = await self.chat_completion(prompt)

        response = interactions[-1]["content"]
        relational_hypothesis = extract_list_from_response(response)

        return relational_hypothesis, interactions, usage


PROMPTS = {

# ================================================
#   ACRE
# ================================================
"acre": {

    "gpt-4-0613":

"""Observe the input-output pair and its patterns below. The input is a list of objects. The presence of certain objects will trigger the light to turn on. The output is either "on" or "off", indicating the state of the light.
Systematically analyze the correlations between the input and the output.

Input: {input}
Output: {output}

Patterns of the input:
{input_object_hypothesis}

Patterns of the output:
{output_object_hypothesis}

Please format your analysis as follows:

1. 
2. 
3.
...""",

    "deepseek-chat":

"""Observe the input-output pair and its patterns below. The input is a list of objects. The presence of certain objects will trigger the light to turn on. The output is either "on" or "off", indicating the state of the light.
Systematically analyze the correlations between the input and the output.

Input: {input}
Output: {output}

Patterns of the input:
{input_object_hypothesis}

Patterns of the output:
{output_object_hypothesis}

Please format your analysis as follows. One line per correlation. Do not output any other text.

1. 
2. 
3.
..."""

},

# ================================================
#   SCAN
# ================================================
"scan": {

    "gpt-4-0613":

"""Observe the input-output pair and its patterns below. Systematically analyze the correlations between the input and the output.
Try to make your analysis as minimal as possible.
Provide up to three most important correlations in your analysis.

Input: "{input}"
Output: "{output}"

Patterns of the input:
{input_object_hypothesis}

Patterns of the output:
{output_object_hypothesis}

Please format your analysis as follows:

1. 
2.
3.""",

    "deepseek-chat":

"""Observe the input-output pair and its patterns below. Systematically analyze the correlations between the input and the output.
Try to make your analysis as minimal as possible.
Provide up to three most important correlations in your analysis.

Input: "{input}"
Output: "{output}"

Patterns of the input:
{input_object_hypothesis}

Patterns of the output:
{output_object_hypothesis}

Please format your analysis as follows. One line per correlation. Do not output any other text.

1. 
2.
3."""

},

# ================================================
#   List Function
# ================================================
"list_function": {

    "gpt-4-0613":

"""Observe the input-output pair and its patterns below. Systematically analyze the correlations between the input and the output.
Provide up to three most important correlations in your analysis.
Input: {input}
Output: {output}

Patterns of the input:
{input_object_hypothesis}

Patterns of the output:
{output_object_hypothesis}

Please format your analysis as follows:

1. 
2. 
3.""",

    "deepseek-chat":

"""Observe the input-output pair and its patterns below. Systematically analyze the correlations between the input and the output.
Provide up to three most important correlations in your analysis.
Input: {input}
Output: {output}

Patterns of the input:
{input_object_hypothesis}

Patterns of the output:
{output_object_hypothesis}

Please format your analysis as follows. One line per correlation. Do not output any other text.

1. 
2. 
3."""

},

# ================================================
#   ARC
# ================================================
"arc": {

    "gpt-4-0613":

"""Observe the input-output pair and its patterns below. Systematically analyze the correlations between the input and the output.

Input: {input}
Output: {output}

Patterns of the input:
{input_object_hypothesis}

Patterns of the output:
{output_object_hypothesis}

Please format your analysis as follows:

1. 
2. 
3.
...""",

    "deepseek-chat":

"""Observe the input-output pair and its patterns below. Systematically analyze the correlations between the input and the output.

Input: {input}
Output: {output}

Patterns of the input:
{input_object_hypothesis}

Patterns of the output:
{output_object_hypothesis}

Please format your analysis as follows. One line per correlation. Do not output any other text.

1. 
2. 
3.
..."""

},

}
