import re
import logging
from typing import Optional

from .base import ChatAgent
from .utils import (
    EXAMPLE_WITH_OBJECT_AND_RELATIONAL_HYPOTHESIS_TEMPLATE,
    FEEDBACK_WITH_OBJECT_AND_RELATIONAL_HYPOTHESIS_TEMPLATE,
    FEEDBACK_TEMPLATE
)
from .utils import format_train_examples


logger = logging.getLogger(__name__)


class FunctionalHypothesisGenerator(ChatAgent):


    async def __call__(
        self,
        train_examples_str: list[dict[str, str]],
        input_object_hypothesis: list[str],
        output_object_hypothesis: list[str],
        relational_hypothesis: list[str],
        task_name: str,
        previous_functional_hypothesis: Optional[str] = None,
        wrong_examples: Optional[list[dict]] = None
    ) -> tuple[str, list[dict[str, str]], dict[str, int]]:
        if previous_functional_hypothesis is None:
            return await self.__call__without_feedback(
                train_examples_str,
                input_object_hypothesis,
                output_object_hypothesis,
                relational_hypothesis,
                task_name
            )
        else:
            return await self.__call__with_feedback(
                train_examples_str,
                input_object_hypothesis,
                output_object_hypothesis,
                relational_hypothesis,
                task_name,
                previous_functional_hypothesis,
                wrong_examples
            )


    async def __call__without_feedback(
        self,
        train_examples_str: list[dict[str, str]],
        input_object_hypothesis: list[str],
        output_object_hypothesis: list[str],
        relational_hypothesis: list[str],
        task_name: str
    ) -> tuple[str, list[dict[str, str]], dict[str, int]]:
        examples_with_object_and_relational_hypothesis = []
        for i, (e, input_oh, output_oh, rh) in enumerate(
            zip(train_examples_str, input_object_hypothesis, output_object_hypothesis, relational_hypothesis)
        ):
            input_oh = "\n".join(
                f"{j + 1}. {p}" for j, p in enumerate(input_oh)
            ) if input_oh else "(No pattern identified)"

            output_oh = "\n".join(
                f"{j + 1}. {p}" for j, p in enumerate(output_oh)
            ) if output_oh else "(No pattern identified)"

            rh = "\n".join(
                f"{j + 1}. {s}" for j, s in enumerate(rh)
            ) if rh else "(No similarities identified)"

            if " " in e["input"] or " " in e["output"]:  # ignore some not very useful examples to reduce token consumption
                examples_with_object_and_relational_hypothesis.append(
                    EXAMPLE_WITH_OBJECT_AND_RELATIONAL_HYPOTHESIS_TEMPLATE[task_name][self.model].format(
                        number=i + 1,
                        input=e["input"], output=e["output"],
                        input_object_hypothesis=input_oh, output_object_hypothesis=output_oh,
                        relational_hypothesis=rh
                    )
                )
            else:
                pass
        examples_with_object_and_relational_hypothesis = "\n\n---\n\n".join(examples_with_object_and_relational_hypothesis)
        prompt = PROMPTS_WITHOUT_FEEDBACK[task_name][self.model].format(
            examples=format_train_examples(train_examples_str, task_name, self.model),
            examples_with_object_and_relational_hypothesis=examples_with_object_and_relational_hypothesis,
        )

        interactions, usage = await self.chat_completion(prompt)

        response = interactions[-1]["content"]
        functional_hypothesis = self.extract_functional_hypothesis_from_response(response, task_name)

        return functional_hypothesis, interactions, usage
    

    async def __call__with_feedback(
        self,
        train_examples_str: list[dict[str, str]],
        input_object_hypothesis: list[str],
        output_object_hypothesis: list[str],
        relational_hypothesis: list[str],
        task_name: str,
        previous_functional_hypothesis: str,
        wrong_examples: list[dict]
    ) -> tuple[str, list[dict[str, str]], dict[str, int]]:
        if task_name == "scan":
            previous_functional_hypothesis = [
                f"Rule {i + 1}: {r}\nPriority {i + 1}: {priority}"
                for i, (r, priority) in enumerate(previous_functional_hypothesis)
            ]
            previous_functional_hypothesis = "\n".join(previous_functional_hypothesis)
        assert type(previous_functional_hypothesis) == str

        feedback_with_object_and_relational_hypothesis, feedback = [], []
        for we in wrong_examples:
            i = we["num"] - 1

            input_oh = "\n".join(
                f"{j + 1}. {p}" for j, p in enumerate(input_object_hypothesis[i])
            ) if input_object_hypothesis[i] else "(No pattern identified)"

            output_oh = "\n".join(
                f"{j + 1}. {p}" for j, p in enumerate(output_object_hypothesis[i])
            ) if output_object_hypothesis[i] else "(No pattern identified)"

            rh = "\n".join(
                f"{j + 1}. {s}" for j, s in enumerate(relational_hypothesis[i])
            ) if relational_hypothesis[i] else "(No correlations identified)"

            feedback_with_object_and_relational_hypothesis.append(
                FEEDBACK_WITH_OBJECT_AND_RELATIONAL_HYPOTHESIS_TEMPLATE[task_name][self.model].format(
                    input=we["input"],
                    output=we["expected_output"],
                    prediction=we["actual_output"],
                    input_object_hypothesis=input_oh,
                    output_object_hypothesis=output_oh,
                    relational_hypothesis=rh,
            ))
            feedback.append(FEEDBACK_TEMPLATE[task_name][self.model].format(
                input=we["input"],
                output=we["expected_output"],
                prediction=we["actual_output"],
            ))
        feedback_with_object_and_relational_hypothesis = "\n\n---\n\n".join(feedback_with_object_and_relational_hypothesis)
        feedback = "\n".join(feedback)
        prompt = PROMPTS_WITH_FEEDBACK[task_name][self.model].format(
            functional_hypothesis=previous_functional_hypothesis,
            feedback=feedback,
            feedback_with_object_and_relational_hypothesis=feedback_with_object_and_relational_hypothesis
        )

        interactions, usage = await self.chat_completion(prompt)

        response = interactions[-1]["content"]
        functional_hypothesis = self.extract_functional_hypothesis_from_response(response, task_name=task_name)

        return functional_hypothesis, interactions, usage
    

    def extract_functional_hypothesis_from_response(self, response: str, task_name: str) -> str:
        match task_name:
            case "list_function":
                patterns = [f"Rule:(.+)"]
                for pattern in patterns:
                    matches = re.findall(pattern, response, re.DOTALL)
                    if matches:
                        assert len(matches) == 1
                        return matches[0].strip()
                else:
                    logging.error(f"Cannot found functional hypothesis from response: {response}")
                    return None
            case "arc":
                patterns = [f"Rule:(.+)"]
                for pattern in patterns:
                    matches = re.findall(pattern, response, re.DOTALL)
                    if matches:
                        assert len(matches) == 1
                        return matches[0].strip()
                else:
                    logging.error(f"Cannot found functional hypothesis from response: {response}")
                    return None
            case "acre":
                patterns = [f"Rule:(.+)"]
                for pattern in patterns:
                    matches = re.findall(pattern, response, re.DOTALL)
                    if matches:
                        assert len(matches) == 1
                        return matches[0].strip()
                else:
                    logging.error(f"Cannot found functional hypothesis from response: {response}")
                    return None
            case "scan":
                rule_pattern = r"Rule (\d+): (.*?)(?:\n|$)"
                rule_matches = re.findall(rule_pattern, response, re.DOTALL)
                rule_matches = sorted(rule_matches, key=lambda x: int(x[0]))

                # Extract priorities
                priority_pattern = r"Priority (\d+): (\d+)"
                priority_matches = re.findall(priority_pattern, response)
                priority_matches = sorted(priority_matches, key=lambda x: int(x[0]))

                idx2rule = {rule[0]: rule[1] for rule in rule_matches}
                idx2priority = {priority[0]: priority[1] for priority in priority_matches}

                functoinal_hypothesis = []
                for idx, rule in idx2rule.items():
                    if idx not in idx2priority:
                        idx2priority[idx] = 0
                    functoinal_hypothesis.append((rule, idx2priority[idx]))
                return functoinal_hypothesis
            case _:
                raise ValueError(f"Unknown task: {task_name}")


PROMPTS_WITHOUT_FEEDBACK = {

# ================================================
#   ACRE
# ================================================
"acre": {

    "gpt-4-0613":

"""Generate a rule that maps the given inputs to their corresponding outputs. 
Each input is a list of objects. Each output is either "on" or "off", indicating the state of the light. The light turns "on" only if at least one object in the input list is a trigger. For each object, determine whether it triggers the light to turn on, does not trigger it, or if it's undetermined.

{examples}

Please format your rule as follows:

Rule: {{"object 1": <"on"/"off"/"undetermined">, "object 2": <"on"/"off"/"undetermined">, ...}}

Below, we collect the patterns and correlations of several input-output pairs mentioned above. 

---

{examples_with_object_and_relational_hypothesis}

---""",

    "deepseek-chat":

"""Generate a rule that maps the given inputs to their corresponding outputs. 
Each input is a list of objects. Each output is either "on" or "off", indicating the state of the light. The light turns "on" only if at least one object in the input list is a trigger. For each object, determine whether it triggers the light to turn on, does not trigger it, or if it's undetermined.

{examples}

Please format your rule as follows. Do not output explanations, analysis, code, or anything other than your rule.

Rule: {{"object 1": <"on"/"off"/"undetermined">, "object 2": <"on"/"off"/"undetermined">, ...}}

Below, we collect the patterns and correlations of several input-output pairs mentioned above. 

---

{examples_with_object_and_relational_hypothesis}

---"""

},

# ================================================
#   SCAN
# ================================================
"scan": {

    "gpt-4-0613":

"""Generate grammar rules that map the following inputs to their corresponding outputs. Your grammar rules should follow the format "<input> -> <output>". Use the prefix "##" to denote a nonterminal symbol. For instance, "##A twice -> ##A ##A", "##A swap ##B -> ##B ##A". The left-hand side cannot contain repetitive or adjacent nonterminal symbols; i.e., rules like "##A ##A -> ##A twice" or "##A ##B -> ##B ##A" are not allowed. Ensure that the number of unique nonterminal symbols on the left-hand side matches that on the right-hand side in your rules. 
For each rule, assign an integer as its priority. A higher priority indicates that the rule should be considered first when generating parses. Ensure that unnecessary colors are replaced by nonterminal symbols and that each rule has pseudoword(s) in the left-hand side.
Try to make your rules as minimal as possible.

{examples}

Please format your rules as follows:

Rule 1: <input> -> <output>
Priority 1: <Your priority>
...


(Below, we collect the patterns and correlations of several input-output pairs mentioned above.)

---

{examples_with_object_and_relational_hypothesis}

---""",

    "deepseek-chat":

"""Generate grammar rules that map the following inputs to their corresponding outputs. Your grammar rules should follow the format "<input> -> <output>". Use the prefix "##" to denote a nonterminal symbol. For instance, "##A twice -> ##A ##A", "##A swap ##B -> ##B ##A". The left-hand side cannot contain repetitive or adjacent nonterminal symbols; i.e., rules like "##A ##A -> ##A twice" or "##A ##B -> ##B ##A" are not allowed. Ensure that the number of unique nonterminal symbols on the left-hand side matches that on the right-hand side in your rules. 
For each rule, assign an integer as its priority. A higher priority indicates that the rule should be considered first when generating parses. Ensure that unnecessary colors are replaced by nonterminal symbols and that each rule has pseudoword(s) in the left-hand side.
Try to make your rules as minimal as possible.

{examples}

Please format your rules as follows. Do not output explanations, analysis, code, or anything other than your rule.

Rule 1: <input> -> <output>
Priority 1: <Your priority>
...


(Below, we collect the patterns and correlations of several input-output pairs mentioned above.)

---

{examples_with_object_and_relational_hypothesis}

---"""

},

# ================================================
#   List Function
# ================================================
"list_function": {

    "gpt-4-0613":

"""Generate a transformation rule that converts each of the given input lists into their corresponding output lists. 

{examples}

Please format your rule as follows:
Rule: <Your rule>

Below, we collect the patterns and correlations of several input-output pairs mentioned above. 

---

{examples_with_object_and_relational_hypothesis}

---
""",

    "deepseek-chat":

"""Generate a transformation rule that converts each of the given input lists into their corresponding output lists. 

{examples}

Please format your rule as follows. Do not output explanations, analysis, code, or anything other than your rule.
Rule: <Your rule>

Below, we collect the patterns and correlations of several input-output pairs mentioned above. 

---

{examples_with_object_and_relational_hypothesis}

---
"""

},

# ================================================
#   ARC
# ================================================
"arc": {

    "gpt-4-0613":

"""Generate a transformation rule that converts each of the given input matrices into their corresponding output matrices.

{examples}

Please format your rule as follows:
Rule: <Your rule>

Below, we collect the patterns and correlations of several input-output pairs mentioned above. 

---

{examples_with_object_and_relational_hypothesis}

---""",

    "deepseek-chat":

"""Generate a transformation rule that converts each of the given input matrices into their corresponding output matrices.

{examples}

Please format your rule as follows. Do not output explanations, analysis, code, or anything other than your rule.
Rule: <Your rule>

Below, we collect the patterns and correlations of several input-output pairs mentioned above. 

---

{examples_with_object_and_relational_hypothesis}

---"""

},

}


PROMPTS_WITH_FEEDBACK = {

# ================================================
#   ACRE
# ================================================
"acre": {

    "gpt-4-0613":

"""Based on your previous rule and the feedback, generate a revised rule that maps the given inputs to their corresponding outputs.
Each input is a list of objects. Each output is either "on" or "off", indicating the state of the light. The light turns "on" only if at least one object in the input list is a trigger. For each object, determine whether it triggers the light to turn on, does not trigger it, or if it's undetermined.

Your previous rule: {functional_hypothesis}

Feedback:
{feedback}

Please format your revised rule as follows:

Rule: {{"object 1": <"on"/"off"/"undetermined">, "object 2": <"on"/"off"/"undetermined">, ...}}

Below, we collect the patterns and correlations of several input-output pairs mentioned above. 

---

{feedback_with_object_and_relational_hypothesis}

---""",

    "deepseek-chat":

"""Based on your previous rule and the feedback, generate a revised rule that maps the given inputs to their corresponding outputs.
Each input is a list of objects. Each output is either "on" or "off", indicating the state of the light. The light turns "on" only if at least one object in the input list is a trigger. For each object, determine whether it triggers the light to turn on, does not trigger it, or if it's undetermined.

Your previous rule: {functional_hypothesis}

Feedback:
{feedback}

Please format your revised rule as follows. Do not output explanations, analysis, code, or anything other than your revised rule.

Rule: {{"object 1": <"on"/"off"/"undetermined">, "object 2": <"on"/"off"/"undetermined">, ...}}

Below, we collect the patterns and correlations of several input-output pairs mentioned above. 

---

{feedback_with_object_and_relational_hypothesis}

---"""

},

# ================================================
#   SCAN
# ================================================
"scan": {

    "gpt-4-0613":

"""Based on your previous rules and the feedback, generate revised rules that map the following inputs to their corresponding outputs. Your grammar rules should follow the format "<input> -> <output>". Use the prefix "##" to denote a nonterminal symbol. For instance, "##A twice -> ##A ##A", "##A swap ##B -> ##B ##A". The left-hand side cannot contain repetitive nonterminal symbols; i.e., rules like "##A ##A -> ##A twice" or "##A and ##A -> ##A twice" are not allowed. Ensure that the number of unique nonterminal symbols on the left-hand side matches that on the right-hand side in your rules. 
For each rule, assign an integer as its priority. A higher priority indicates that the rule should be considered first when generating parses. Ensure that unnecessary colors are replaced by nonterminal symbols and that each rule has pseudoword(s) in the left-hand side.
Try to make your rules as minimal as possible.

Your previous rules: {functional_hypothesis}

Feedback:
{feedback}

Please format your rule as follows:

Rule 1: <Your rule>
...


(Below, we collect the patterns and correlations from the input-output pairs in the feedback mentioned above.)

---

{feedback_with_object_and_relational_hypothesis}

---""",

    "deepseek-chat":

"""Based on your previous rules and the feedback, generate revised rules that map the following inputs to their corresponding outputs. Your grammar rules should follow the format "<input> -> <output>". Use the prefix "##" to denote a nonterminal symbol. For instance, "##A twice -> ##A ##A", "##A swap ##B -> ##B ##A". The left-hand side cannot contain repetitive nonterminal symbols; i.e., rules like "##A ##A -> ##A twice" or "##A and ##A -> ##A twice" are not allowed. Ensure that the number of unique nonterminal symbols on the left-hand side matches that on the right-hand side in your rules. 
For each rule, assign an integer as its priority. A higher priority indicates that the rule should be considered first when generating parses. Ensure that unnecessary colors are replaced by nonterminal symbols and that each rule has pseudoword(s) in the left-hand side.
Try to make your rules as minimal as possible.

Your previous rules: {functional_hypothesis}

Feedback:
{feedback}

Please format your rule as follows. Do not output explanations, analysis, code, or anything other than your revised rule.

Rule 1: <Your rule>
...


(Below, we collect the patterns and correlations from the input-output pairs in the feedback mentioned above.)

---

{feedback_with_object_and_relational_hypothesis}

---"""

},

# ================================================
#   List Function
# ================================================
"list_function": {

    "gpt-4-0613":

"""Based on your previous rule and the feedback, generate a revised transformation rule that converts each of the given input lists into their corresponding output lists.

Your previous rule: {functional_hypothesis}

Feedback:
{feedback}

Please format your revised rule as follows:
Rule: <Your rule>

Below, we collect the patterns and correlations of several input-output pairs mentioned above. 

---

{feedback_with_object_and_relational_hypothesis}

---""",

    "deepseek-chat":

"""Based on your previous rule and the feedback, generate a revised transformation rule that converts each of the given input lists into their corresponding output lists.

Your previous rule: {functional_hypothesis}

Feedback:
{feedback}

Please format your revised rule as follows. Do not output explanations, analysis, code, or anything other than your revised rule.
Rule: <Your rule>

Below, we collect the patterns and correlations of several input-output pairs mentioned above. 

---

{feedback_with_object_and_relational_hypothesis}

---"""

},

# ================================================
#   ARC
# ================================================
"arc": {

    "gpt-4-0613":

"""Based on your previous rule and the feedback, generate a revised transformation rule that converts each of the given input matrices into their corresponding output matrices.

Your previous rule: {functional_hypothesis}

Feedback:
{feedback}

Please format your revised rule as follows:
Rule: <Your rule>

Below, we collect the patterns and correlations of several input-output pairs mentioned above. 

---

{feedback_with_object_and_relational_hypothesis}

---""",

    "deepseek-chat":

"""Based on your previous rule and the feedback, generate a revised transformation rule that converts each of the given input matrices into their corresponding output matrices.

Your previous rule: {functional_hypothesis}

Feedback:
{feedback}

Please format your revised rule as follows. Do not output explanations, analysis, code, or anything other than your revised rule.
Rule: <Your rule>

Below, we collect the patterns and correlations of several input-output pairs mentioned above. 

---

{feedback_with_object_and_relational_hypothesis}

---"""

},

}
