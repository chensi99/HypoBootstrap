import re
import ast
import copy
import types
import logging
import asyncio
import concurrent
from typing import Any
from traceback import format_exc
from collections import defaultdict

import numpy as np

from utils.grammar.qcfg_rule import get_nts, get_num_nts, rule_from_string
from utils.grammar_utils import canonicalize_rule, rule_to_parses

from .base import ChatAgent


logger = logging.getLogger(__name__)


def extract_function_names(function_string: str):
    matches = re.findall(r"def (\w+?)\(", function_string)
    return matches


def execute_program_worker(program: str) -> Any:
    assert all(danger_cmd not in program for danger_cmd in ["rm -rf", "rm -f"]), program

    namespace = {}
    try:
        exec(program, namespace)
        result = namespace.get("result")
    except Exception as e:
        logger.error("Unknown error during python program execution. See below for details.\n" + \
            format_exc() + \
            f"\nError program:\n{program}"
        )
        return repr(e)
    
    if isinstance(result, types.GeneratorType):
        result = list(result)
    if isinstance(result, list):
        result = [
            list(x) if isinstance(x, types.GeneratorType) else x
            for x in result
        ]
    return result


def eval_scan_functional_hypothesis(functional_hypothesis, examples):
    rule_to_priority = {}
    for rule, priority in functional_hypothesis:
        try:
            qcfg_rule = rule_from_string(canonicalize_rule(rule))
            if len(get_nts(qcfg_rule.source)) == get_num_nts(qcfg_rule.source):
                rule_to_priority[qcfg_rule] = int(priority)
            else:
                logger.warning("Invalid rule: %s" % rule)
        except:
            logger.warning("Cannot parse rule: %s" % rule)
    all_parses = rule_to_parses(rule_to_priority, examples)
    return all_parses


class AuxiliaryConfirmer(ChatAgent):


    def __init__(
        self,
        *,
        multiprocess_executor: concurrent.futures.ProcessPoolExecutor,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.multiprocess_executor = multiprocess_executor

        if self.model in ("deepseek-chat", ):
            self.SYSTEM_ROLE = "system"
        else:
            self.SYSTEM_ROLE = "developer"


    async def __call__(
        self,
        functional_hypothesis: str,
        task_name: str,
        examples
    ):
        if task_name in ("list_function", "arc"):
            func_str, interactions, usage = await self.functional_hypothesis_to_python_program(
                functional_hypothesis, task_name        
            )

            raw_acc, accs, wrong_examples = await self.eval_python_program(func_str, examples)
            
            return func_str, interactions, usage, raw_acc, accs, wrong_examples
        elif task_name in ("acre", ):
            raw_acc, accs, wrong_examples = await self.eval_acre(functional_hypothesis, examples)
            return None, None, None, raw_acc, accs, wrong_examples
        elif task_name in ("scan", ):
            raw_acc, accs, wrong_examples = await self.eval_scan(functional_hypothesis, examples)
            return None, None, None, raw_acc, accs, wrong_examples
        else:
            raise ValueError(f"Unknown task: {task_name}")


    async def eval_python_program(self, func_str, examples):
        # execute programs and compute metrics
        inputs = copy.deepcopy(list(map(lambda e: e["input"], examples)))
        outputs = copy.deepcopy(list(map(lambda e: e["output"], examples)))

        try:
            pred_outputs = await self.execute_python_programs(func_str, inputs)
        except:
            logger.warning(f"An exception occurred during program execution,\
                           Overwrite the predictions with a list consisting of -100.")
            pred_outputs = [-100] * len(outputs)

        raw_acc, accs = self.compute_metrics(pred_outputs, outputs)

        # compute wrongly-predicted examples
        wrong_examples = []
        for num, (_input, _output, _pred) in enumerate(zip(inputs, outputs, pred_outputs)):
            if isinstance(_pred, np.ndarray):
                if (_output != _pred).any():
                    wrong_examples.append({
                        "num": num + 1,
                        "input": _input,
                        "expected_output": _output,
                        "actual_output": _pred.tolist(),
                    })
            else:
                if _output != _pred:
                    if not _pred:
                        actual_output = _pred
                    elif isinstance(_pred, int):
                        actual_output = _pred
                    elif isinstance(_pred, list) and isinstance(_pred[0], int):
                        actual_output = _pred
                    else:
                        actual_output = _pred
                    wrong_examples.append({
                        "num": num + 1,
                        "input": _input,
                        "expected_output": _output,
                        "actual_output": actual_output,
                    })

        return raw_acc, accs, wrong_examples
    

    async def eval_acre(self, functional_hypothesis, examples):
        pred_outputs = self.eval_acre_functional_hypothesis(functional_hypothesis, examples)
        
        # compute metrics
        inputs = copy.deepcopy(list(map(lambda e: e["input"], examples)))
        outputs = copy.deepcopy(list(map(lambda e: e["output"], examples)))
        raw_acc, accs = self.compute_metrics(pred_outputs, outputs)

        def format_list(list, sep=", ", bracket=False):
            rep = sep.join([str(x) for x in list])
            if bracket:
                return "[" + rep + "]"
            return rep

        # feedback
        wrong_examples = []
        for num, (_input, _output, _pred) in enumerate(zip(inputs, outputs, pred_outputs)):
            if _output != _pred:
                wrong_examples.append({
                    "num": num + 1,
                    "input": format_list(_input),
                    "expected_output": _output,
                    "actual_output": _pred,
                })

        return raw_acc, accs, wrong_examples


    def eval_acre_functional_hypothesis(self, functional_hypothesis, examples):
        matches = re.findall(r"\{[^{}]*\}", functional_hypothesis)
        if len(matches) != 1:
            raise ValueError(f"Multiple matches found: {matches}")
        functional_hypothesis = matches[-1]
        try:
            functional_hypothesis = ast.literal_eval(functional_hypothesis)
        except:
            raise ValueError(f"Failed to parse functional hypothesis: {functional_hypothesis}")
        functional_hypothesis = {k.replace("object", "").strip(): v for k, v in functional_hypothesis.items()}
        state_to_objs = defaultdict(set)
        objects = set()
        for objs, state in functional_hypothesis.items():
            objs = objs.split(",")
            for obj in objs:
                obj = obj.strip()
                state_to_objs[state].add(obj)
                objects.add(obj)
        all_objects = set()
        outputs = []
        for example in examples:
            objects = set([str(x) for x in example["input"]])
            all_objects |= objects
            if objects.intersection(state_to_objs["on"]):
                outputs.append("on")
            elif objects.intersection(state_to_objs["undetermined"]):
                outputs.append("undetermined")
            elif objects.issubset(state_to_objs["off"]):
                outputs.append("off")
            else:
                outputs.append("undetermined")
        if not all_objects.issubset(objects):
            missing_objects = all_objects - objects
            logger.warning(f"Functional hypothesis does not cover all objects: missing {missing_objects}")
        return outputs


    async def eval_scan(self, functional_hypothesis, examples):
        loop = asyncio.get_event_loop()
        pred_outputs = await asyncio.wait_for(
            loop.run_in_executor(self.multiprocess_executor, eval_scan_functional_hypothesis, functional_hypothesis, examples),
            timeout=10
        )

        for i, o in enumerate(pred_outputs):
            if isinstance(o, asyncio.TimeoutError):
                logger.warning("Timeout parsing!")
                pred_outputs[i] = [None] * len(examples)
            elif isinstance(o, Exception):
                raise o
            
        # compute metrics
        inputs = copy.deepcopy(list(map(lambda e: e["input"], examples)))
        outputs = copy.deepcopy(list(map(lambda e: e["output"], examples)))
        raw_acc, accs = self.compute_metrics(pred_outputs, outputs)

        # feedback
        wrong_examples = []
        for num, (_input, _output, _pred) in enumerate(zip(inputs, outputs, pred_outputs)):
            if _output != _pred:
                wrong_examples.append({
                    "num": num + 1,
                    "input": _input,
                    "expected_output": _output,
                    "actual_output": _pred,
                })

        return raw_acc, accs, wrong_examples


    def compute_metrics(self, preds, targets) -> tuple[float, list[int]]:
        accs = []
        for pred, target in zip(preds, targets):
            if isinstance(pred, int):
                accs.append(int(pred == target))
            elif isinstance(pred, list):
                accs.append(int(pred == target))
            elif isinstance(pred, np.ndarray):
                accs.append(int((pred == target).all()))
            elif isinstance(pred, str):  # possibly error message
                accs.append(int(pred == target))
            elif pred is None:
                accs.append(0)
            else:
                raise TypeError(f"Unknown type {type(pred)}")
        raw_acc = np.mean(accs)
        return raw_acc, accs
    

    async def execute_python_programs(self, func_str: str, inputs: list, timeout=10):
        programs = []
        func_names = extract_function_names(func_str)
        fn_name = "fn" if "fn" in func_names else func_names[-1]

        for inp in inputs:
            program = f"{func_str}\nresult = {fn_name}({inp})"
            programs.append(program)

        loop = asyncio.get_event_loop()
        tasks = [
            asyncio.wait_for(
                loop.run_in_executor(self.multiprocess_executor, execute_program_worker, program),
                timeout=timeout
            )
            for program in programs
        ]
        outputs = await asyncio.gather(*tasks)

        return outputs


    async def functional_hypothesis_to_python_program(
        self,
        functional_hypothesis: str,
        task_name: str
    ) -> tuple[str, list[dict[str, str]], dict[str, int]]:
        interactions, usage = await self.chat_completion(
            NATURAL_LANGUAGE_TO_PYTHON_PROGRAM_PROMPTS[task_name][self.model].format(functional_hypothesis=functional_hypothesis),
            interactions=[{"role": self.SYSTEM_ROLE, "content": "You are an expert Python programmer."}]
        )
        func_str = self.extract_func_str(interactions[-1]["content"])
        return func_str, interactions, usage
    

    def extract_func_str(self, response: str):
        pattern = r"```python\s*([\s\S]+?)\s*```"
        matches = re.findall(pattern, response)
        matches = list(filter(lambda m: "def" in m, matches))
        if matches:
            return "\n".join(matches)
        else:
            logger.warning(f"Cannot extract program from response: {response}")
            return ""


NATURAL_LANGUAGE_TO_PYTHON_PROGRAM_PROMPTS = {

# ================================================
#   List Function
# ================================================
"list_function": {

    "gpt-4-0613":

"""Write a Python function `fn` for the following rule, where the input must be a list of integers and the output must also be a list of integers.

Rule: {functional_hypothesis}""",

    "deepseek-chat":

"""Write a Python function `fn` for the following rule, where the input must be a list of integers and the output must also be a list of integers.

Rule: {functional_hypothesis}

Do not output anything other than the Python functions."""

},

# ================================================
#   ARC
# ================================================
"arc": {

    "gpt-4-0613":

"""Write a Python function `fn` for the following rule, where the input must be a nested list that represents a 2D grid of integers and the output must also a nested list that represents a 2D grid of integers.

Rule: {functional_hypothesis}""",

    "deepseek-chat":

"""Write a Python function `fn` for the following rule, where the input must be a nested list that represents a 2D grid of integers and the output must also a nested list that represents a 2D grid of integers.

Rule: {functional_hypothesis}

Do not output anything other than the Python functions."""

},

}
