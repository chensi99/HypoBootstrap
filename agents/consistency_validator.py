from .base import ChatAgent
from .utils import EXAMPLE_WITH_OBJECT_AND_RELATIONAL_HYPOTHESIS_TEMPLATE


class ConsistencyValidator(ChatAgent):


    async def __call__(
        self,
        functional_hypothesis: str,
        input_object_hypothesis: list[list[str]],
        output_object_hypothesis: list[list[str]],
        relational_hypothesis: list[list[str]],
        train_examples_str: list[dict[str, str]],
        task_name: str
    ):
        examples_with_object_and_relational_hypothesis = []
        for i, e in enumerate(train_examples_str):
            input_oh = "\n".join(
                f"{j + 1}. {p}" for j, p in enumerate(input_object_hypothesis[i])
            ) if input_object_hypothesis[i] else "(No pattern identified)"

            output_oh = "\n".join(
                f"{j + 1}. {p}" for j, p in enumerate(output_object_hypothesis[i])
            ) if output_object_hypothesis[i] else "(No pattern identified)"

            rh = "\n".join(
                f"{j + 1}. {s}" for j, s in enumerate(relational_hypothesis[i])
            ) if relational_hypothesis[i] else "(No similarities identified)"

            examples_with_object_and_relational_hypothesis.append(
                EXAMPLE_WITH_OBJECT_AND_RELATIONAL_HYPOTHESIS_TEMPLATE[task_name][self.model].format(
                    number=i + 1,
                    input=e["input"], output=e["output"],
                    input_object_hypothesis=input_oh, output_object_hypothesis=output_oh,
                    relational_hypothesis=rh
            ))
        examples_with_object_and_relational_hypothesis = "\n\n---\n\n".join(examples_with_object_and_relational_hypothesis)
        prompt = PROMPTS[task_name][self.model].format(
            functional_hypothesis=functional_hypothesis,
            examples_with_object_and_relational_hypothesis=examples_with_object_and_relational_hypothesis
        )

        interactions, usage = await self.chat_completion(prompt)

        response = interactions[-1]["content"]
        if response == "Yes":
            is_consistent = True
        elif response == "No":
            is_consistent = False
        else:
            raise ValueError(f"Unknown response: {response}")

        return is_consistent, interactions, usage


PROMPTS = {

# ================================================
#   ACRE
# ================================================
"acre": {

    "gpt-4-0613":

"""Is the target statement logically consistent with the input-output pairs and their patterns and correlations?

Target statement: {functional_hypothesis}

---

{examples_with_object_and_relational_hypothesis}

---

Please respond with Yes or No.""",

    "deepseek-chat":

"""Is the target statement logically consistent with the input-output pairs and their patterns and correlations?

Target statement: {functional_hypothesis}

---

{examples_with_object_and_relational_hypothesis}

---

Please respond with Yes or No. Do not output any other text."""

},

# ================================================
#   SCAN
# ================================================
"scan": {

    "gpt-4-0613":

"""Is the target statement logically consistent with the input-output pairs and their patterns and correlations?

Target statement: {functional_hypothesis}

---

{examples_with_object_and_relational_hypothesis}

---

Please respond with Yes or No.""",

    "deepseek-chat":

"""Is the target statement logically consistent with the input-output pairs and their patterns and correlations?

Target statement: {functional_hypothesis}

---

{examples_with_object_and_relational_hypothesis}

---

Please respond with Yes or No. Do not output any other text."""

},

# ================================================
#   List Function
# ================================================
"list_function": {

    "gpt-4-0613":

"""Is the target statement logically consistent with the input-output pairs and their patterns and correlations?

Target statement: {functional_hypothesis}

---

{examples_with_object_and_relational_hypothesis}

---

Please respond with Yes or No.""",

    "deepseek-chat":

"""Is the target statement logically consistent with the input-output pairs and their patterns and correlations?

Target statement: {functional_hypothesis}

---

{examples_with_object_and_relational_hypothesis}

---

Please respond with Yes or No. Do not output any other text."""

},

# ================================================
#   ARC
# ================================================
"arc": {

    "gpt-4-0613":

"""Is the target statement logically consistent with the input-output pairs and their patterns and correlations?

Target statement: {functional_hypothesis}

---

{examples_with_object_and_relational_hypothesis}

---

Please respond with Yes or No.""",

    "deepseek-chat":

"""Is the target statement logically consistent with the input-output pairs and their patterns and correlations?

Target statement: {functional_hypothesis}

---

{examples_with_object_and_relational_hypothesis}

---

Please respond with Yes or No. Do not output any other text."""

},

}
