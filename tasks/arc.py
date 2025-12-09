import logging

from .base import Task


logger = logging.getLogger(__name__)


class ArcTask(Task):

    def __init__(self, *, task_name: str, examples: dict):
        super().__init__(task_name=task_name, examples=examples)
        
        self.idx = examples["idx"]
        self.train_examples = examples["train"]
        self.test_examples = examples["test"]

        def canonicalize(grid):
            if isinstance(grid, list) and isinstance(grid[0], list):
                return [[int(x) if x is not None else 0 for x in row] for row in grid]
            else:
                raise ValueError(f"Unknown grid: {grid}")

        def format_list(list, sep=", ", bracket=False):
            rep = sep.join([str(x) for x in list])
            if bracket:
                return "[" + rep + "]"
            return rep

        def format_grid(grid, row_sep="\n", sep=", "):
            return row_sep.join([format_list(row, sep, bracket=True) for row in grid])

        def format_input(input):
            return "\n" + format_grid(input, row_sep="\n", sep=", ")

        def format_output(output):
            grid = canonicalize(output)

            if isinstance(grid, list) and isinstance(grid[0], list):
                return "\n" + format_grid(grid, row_sep="\n", sep=", ")
            else:
                raise ValueError(f"Unknown grid: {grid}")

        self.train_examples_str = list(map(
            lambda e: {
                "input": format_input(e["input"]),
                "output": format_output(e["output"]),
            },
            self.train_examples
        ))
        self.test_examples_str = list(map(
            lambda e: {
                "input": format_input(e["input"]),
                "output": format_output(e["output"]),
            },
            self.test_examples
        ))
