import re


def extract_list_from_response(response: str) -> list[str]:
    l = []
    for line in response.splitlines():
        line = line.strip()
        if len(line) == 0:
            continue

        matcher = re.match(r"[\d+]. ([\s\S]+)", line)
        if matcher:
            l.append(matcher.group(1))
    return l


def format_train_examples(train_examples_str, task_name, model_name) -> str:
    example_strs = []
    for e in train_examples_str:
        example_strs.append(
            EXAMPLE_TEMPLATE[task_name][model_name].format(
                input=e["input"], output=e["output"]
        ))
    return "\n".join(example_strs)



EXAMPLE_TEMPLATE = {

# ================================================
#   ACRE
# ================================================
"acre": {

    "gpt-4-0613":

"""Input: {input}
Output: {output}""",

    "deepseek-chat":

"""Input: {input}
Output: {output}"""

},

# ================================================
#   SCAN
# ================================================
"scan": {

    "gpt-4-0613":

"""Input: "{input}"
Output: "{output}" """,

    "deepseek-chat":

"""Input: "{input}"
Output: "{output}" """

},

# ================================================
#   List Function
# ================================================
"list_function": {

    "gpt-4-0613":

"""Input: {input}
Output: {output}""",

    "deepseek-chat":

"""Input: {input}
Output: {output}"""

},

# ================================================
#   ARC
# ================================================
"arc": {

    "gpt-4-0613":

"""Input: {input}
Output: {output}""",

    "deepseek-chat":

"""Input: {input}
Output: {output}"""

},

}



EXAMPLE_WITH_OBJECT_AND_RELATIONAL_HYPOTHESIS_TEMPLATE = {

# ================================================
#   ACRE
# ================================================
"acre": {

    "gpt-4-0613":

"""Pair {number}
Input: {input}
Output: {output}

Patterns of the input:
{input_object_hypothesis}

Patterns of the output:
{output_object_hypothesis}

Correlations between input and output:
{relational_hypothesis}""",

    "deepseek-chat":

"""Pair {number}
Input: {input}
Output: {output}

Patterns of the input:
{input_object_hypothesis}

Patterns of the output:
{output_object_hypothesis}

Correlations between input and output:
{relational_hypothesis}"""

},

# ================================================
#   SCAN
# ================================================
"scan": {

    "gpt-4-0613":

"""Pair {number}
Input: "{input}"
Output: "{output}"

Patterns of the input:
{input_object_hypothesis}

Patterns of the output:
{output_object_hypothesis}

Correlations between input and output:
{relational_hypothesis}""",

    "deepseek-chat":

"""Pair {number}
Input: "{input}"
Output: "{output}"

Patterns of the input:
{input_object_hypothesis}

Patterns of the output:
{output_object_hypothesis}

Correlations between input and output:
{relational_hypothesis}"""

},

# ================================================
#   List Function
# ================================================
"list_function": {

    "gpt-4-0613":

"""Pair {number}
Input: {input}
Output: {output}

Patterns of the input:
{input_object_hypothesis}

Patterns of the output:
{output_object_hypothesis}

Correlations between input and output:
{relational_hypothesis}""",

    "deepseek-chat":

"""Pair {number}
Input: {input}
Output: {output}

Patterns of the input:
{input_object_hypothesis}

Patterns of the output:
{output_object_hypothesis}

Correlations between input and output:
{relational_hypothesis}"""

},

# ================================================
#   ARC
# ================================================
"arc": {

    "gpt-4-0613":

"""Pair {number}
Input: {input}
Output: {output}

Patterns of the input:
{input_object_hypothesis}

Patterns of the output:
{output_object_hypothesis}

Correlations between input and output:
{relational_hypothesis}""",

    "deepseek-chat":

"""Pair {number}
Input: {input}
Output: {output}

Patterns of the input:
{input_object_hypothesis}

Patterns of the output:
{output_object_hypothesis}

Correlations between input and output:
{relational_hypothesis}"""

},

}



FEEDBACK_WITH_OBJECT_AND_RELATIONAL_HYPOTHESIS_TEMPLATE = {

# ================================================
#   ACRE
# ================================================
"acre": {

    "gpt-4-0613":

"""Input: {input}
Expected output: {output}
Predicted output: {prediction}

Patterns of the input:
{input_object_hypothesis}

Patterns of the output:
{output_object_hypothesis}

Correlations between input and expected output:
{relational_hypothesis}""",

    "deepseek-chat":

"""Input: {input}
Expected output: {output}
Predicted output: {prediction}

Patterns of the input:
{input_object_hypothesis}

Patterns of the output:
{output_object_hypothesis}

Correlations between input and expected output:
{relational_hypothesis}"""

},

# ================================================
#   SCAN
# ================================================
"scan": {

    "gpt-4-0613":

"""Input: "{input}"
Expected output: "{output}"
Predicted output: "{prediction}"

Patterns of the input:
{input_object_hypothesis}

Patterns of the output:
{output_object_hypothesis}

Correlations between input and expected output:
{relational_hypothesis}""",

    "deepseek-chat":

"""Input: "{input}"
Expected output: "{output}"
Predicted output: "{prediction}"

Patterns of the input:
{input_object_hypothesis}

Patterns of the output:
{output_object_hypothesis}

Correlations between input and expected output:
{relational_hypothesis}"""

},

# ================================================
#   List Function
# ================================================
"list_function": {

    "gpt-4-0613":

"""Input: {input}
Expected output: {output}
Predicted output: {prediction}

Patterns of the input:
{input_object_hypothesis}

Patterns of the output:
{output_object_hypothesis}

Correlations between input and expected output:
{relational_hypothesis}""",

    "deepseek-chat":

"""Input: {input}
Expected output: {output}
Predicted output: {prediction}

Patterns of the input:
{input_object_hypothesis}

Patterns of the output:
{output_object_hypothesis}

Correlations between input and expected output:
{relational_hypothesis}"""

},

# ================================================
#   ARC
# ================================================
"arc": {

    "gpt-4-0613":

"""Input: {input}
Expected output: {output}
Predicted output: {prediction}

Patterns of the input:
{input_object_hypothesis}

Patterns of the output:
{output_object_hypothesis}

Correlations between input and expected output:
{relational_hypothesis}""",

    "deepseek-chat":

"""Input: {input}
Expected output: {output}
Predicted output: {prediction}

Patterns of the input:
{input_object_hypothesis}

Patterns of the output:
{output_object_hypothesis}

Correlations between input and expected output:
{relational_hypothesis}"""

},

}



FEEDBACK_TEMPLATE = {

# ================================================
#   ACRE
# ================================================
"acre": {

    "gpt-4-0613":

"""Input: {input}
Expected output: {output}
Predicted output: {prediction}""",

    "deepseek-chat":

"""Input: {input}
Expected output: {output}
Predicted output: {prediction}"""

},

# ================================================
#   SCAN
# ================================================
"scan": {

    "gpt-4-0613":

"""Input: "{input}"
Expected output: "{output}"
Predicted output: "{prediction}" """,

    "deepseek-chat":

"""Input: "{input}"
Expected output: "{output}"
Predicted output: "{prediction}" """

},

# ================================================
#   List Function
# ================================================
"list_function": {

    "gpt-4-0613":

"""Input: {input}
Expected output: {output}
Predicted output: {prediction}""",

    "deepseek-chat":

"""Input: {input}
Expected output: {output}
Predicted output: {prediction}"""

},

# ================================================
#   ARC
# ================================================
"arc": {

    "gpt-4-0613":

"""Input: {input}
Expected output: {output}
Predicted output: {prediction}""",

    "deepseek-chat":

"""Input: {input}
Expected output: {output}
Predicted output: {prediction}"""

},

}
