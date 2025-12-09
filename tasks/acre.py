import logging

from .base import Task


logger = logging.getLogger(__name__)


class AcreTask(Task):

    def __init__(self, *, task_name: str, examples: dict):
        super().__init__(task_name=task_name, examples=examples)

        self.idx = examples["idx"]
        self.train_examples = examples["train"]
        self.test_examples = examples["test"]

        def format_list(list, sep=", ", bracket=False):
            rep = sep.join([str(x) for x in list])
            if bracket:
                return "[" + rep + "]"
            return rep

        self.train_examples_str = list(map(
            lambda e: {
                "input": format_list(e["input"]),
                "output": e["output"],
            },
            self.train_examples
        ))
        self.test_examples_str = list(map(
            lambda e: {
                "input": format_list(e["input"]),
                "output": e["output"],
            },
            self.test_examples
        ))
