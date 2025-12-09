import ast
import logging

from .base import Task


logger = logging.getLogger(__name__)


class ListFunctionTask(Task):

    def __init__(self, *, task_name: str, examples: dict):
        super().__init__(task_name=task_name, examples=examples)
        
        self.idx = examples["idx"]
        self.train_examples_str = examples["train"]
        self.test_examples_str = examples["test"]

        def _convert_input_to_python(input: str) -> list[int]:
            input = ast.literal_eval(input)
            assert isinstance(input, list) and all(isinstance(e, int) for e in input)
            return input

        def _convert_output_to_python(output: str) -> list[int]:
            output = ast.literal_eval(output)
            assert isinstance(output, list) and all(isinstance(e, int) for e in output)
            return output

        self.train_examples = list(map(
            lambda e: {
                "input": _convert_input_to_python(e["input"]),
                "output": _convert_output_to_python(e["output"]),
            },
            self.train_examples_str
        ))
        self.test_examples = list(map(
            lambda e: {
                "input": _convert_input_to_python(e["input"]),
                "output": _convert_output_to_python(e["output"]),
            },
            self.test_examples_str
        ))
