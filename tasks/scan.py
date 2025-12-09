import copy

from .base import Task


class ScanTask(Task):

    def __init__(self, *, task_name: str, examples: dict):
        super().__init__(task_name=task_name, examples=examples)

        self.idx = examples["idx"]
        self.train_examples = examples["train"]
        self.test_examples = examples["test"]
        self.train_examples_str = copy.deepcopy(self.train_examples)
        self.test_examples_str = copy.deepcopy(self.test_examples)

