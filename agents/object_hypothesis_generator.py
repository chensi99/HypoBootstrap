from .base import ChatAgent
from .utils import extract_list_from_response


class ObjectHypothesisGenerator(ChatAgent):


    async def __call__(
        self,
        object: str,
        task_name: str
    ) -> tuple[list[str], list[dict[str, str]], dict[str, int]]:
        if task_name == "scan":
            if " " not in object:
                # some LLMs (expecially old ones) may analyze the characters (instead of words)
                # if the data contains exactly one word
                return ["A single word."], [], {}
            else:
                pass
        if task_name == "acre":
            # this agent is not applicable for ACRE since the object in ACRE
            # is already symbolized
            if object in ("on", "off"):  # if the object is output
                return [f"The data is '{object}', indicating that the light is {object}."], [], {}
            else:  # if the object is input
                return [f"The data contains {object.count(", ") + 1} objects."], [], {}

        prompt = PROMPTS[task_name][self.model].format(object=object)
        interactions, usage = await self.chat_completion(prompt)

        response = interactions[-1]["content"]
        object_hypothesis = extract_list_from_response(response)

        return object_hypothesis, interactions, usage


PROMPTS = {

# ================================================
#   SCAN
# ================================================
"scan": {

    "gpt-4-0613":

"""Observe the following data, which consists of either pseudowords or colors. Systematically analyze its patterns.
If data consists of pseudowords, do not analyze its letter composition or sequence.
Try to make your analysis as minimal as possible.
Provide up to three most important patterns in your analysis.

Data: "{object}"

Please format your analysis as follows:

1. 
2.
3.""",

    "deepseek-chat":

"""Observe the following data, which consists of either pseudowords or colors. Systematically analyze its patterns.
If data consists of pseudowords, do not analyze its letter composition or sequence.
Try to make your analysis as minimal as possible.
Provide up to three most important patterns in your analysis.

Data: "{object}"

Please format your analysis as follows. One line per pattern. Do not output any other text.

1. 
2.
3."""

},

# ================================================
#   List Function
# ================================================
"list_function": {

    "gpt-4-0613":

"""Observe the following data, which consists of integers. Systematically analyze its patterns.
Provide up to three most important patterns in your analysis.
Data: {object}

Please format your analysis as follows:

1. 
2. 
3.""",

    "deepseek-chat":

"""Observe the following data, which consists of integers. Systematically analyze its patterns.
Provide up to three most important patterns in your analysis.
Data: {object}

Please format your analysis as follows. One line per pattern. Do not output any other text.

1. 
2. 
3."""

},

# ================================================
#   ARC
# ================================================
"arc": {

    "gpt-4-0613":

"""Observe the data below. Systematically analyze its patterns.

Data: {object}

Please format your analysis as follows:

1. 
2. 
3.
...""",

    "deepseek-chat":

"""Observe the data below. Systematically analyze its patterns.

Data: {object}

Please format your analysis as follows. One line per pattern. Do not output any other text.

1. 
2. 
3.
..."""

},

}
