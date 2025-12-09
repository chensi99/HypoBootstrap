import os
import time
import json
import logging
import asyncio
import concurrent
from pathlib import Path
from typing import Optional
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np
from tqdm.asyncio import tqdm_asyncio
import openai

# To disable the warning "None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used."
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
from transformers import HfArgumentParser

from utils.data_utils import read_data
from agents import (
    ObjectHypothesisGenerator, ObjectHypothesisInconsistencyEliminator,
    RelationalHypothesisGenerator, RelationalHypothesisInconsistencyEliminator,
    FunctionalHypothesisGenerator,
    AuxiliaryConfirmer,
    ConsistencyValidator
)
from tasks import Task


logger = logging.getLogger(__name__)


@dataclass
class Args:
    """
    Arguments.
    """

    task: str = field(
        metadata={
            "help": "Task name. Available tasks include `acre`, `scan`, `list_function`, and `arc`.",
            "choices": ["acre", "scan", "list_function", "arc"],
        },
    )
    data_file: str = field(
        metadata={"help": "Path to data file (in jsonl format). One task per line."},
    )
    model: str = field(
        metadata={
            "help": "Model name. Available models include `gpt-4-0613` and `deepseek-chat` (DeepSeek-V3).",
            "choices": ["gpt-4-0613", "deepseek-chat"],
        },
    )
    base_url: str = field(
        default=None,
        metadata={
            "help": "`base_url` for openai.AsyncOpenAI. Default (None) refers to OpenAI API."
        },
    )

    n_tasks: Optional[int] = field(
        default=None,
        metadata={"help": "Only evaluate the first `n_tasks` tasks in the `data_file`."},
    )
    max_iter: int = field(
        default=1,
        metadata={"help": "Maximum number of refinement iterations ($T$)."},
    )
    no_consistency_check: bool = field(
        default=False,
        metadata={"help": "Disable ObjectHypothesisInconsistencyEliminator, RelationalHypothesisInconsistencyEliminator, and ConsistencyValidator."},
    )

    max_workers: Optional[int] = field(
        default=None,
        metadata={"help": "Number of multiprocessing workers used to execute LLM-generated python programs."},
    )
    concurrent_api_calls: int = field(
        default=500,
        metadata={"help": "Maximum number of concurrent api calls."},
    )

    log_file: Optional[str] = field(
        default=None,
        metadata={"help": "Log file path. Default to `{task}_{model}_iter{max_iter}_{datetime.now}.log`."},
    )
    output_file: Optional[str] = field(
        default=None,
        metadata={"help": "Output file path. Default to `{task}_{model}_iter{max_iter}_{datetime.now}.json`."},
    )
    cache_mode: str = field(
        default="file",
        metadata={
            "help": "Cache in local file or in memory. WARNING: If interrupted, the file caching mechanism is error-prone and the file cache may be broken.",
            "choices": ["file", "memory"],
        },
    )
    cache_file: Optional[str] = field(
        default=None,
        metadata={"help": "Cache file path if `cache_mode` is `file`. Default to `cache.pkl`."},
    )

    def __post_init__(self):
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if not self.no_consistency_check:
            name = f"{self.task}_{self.model}_iter{self.max_iter}_{now}"
        else:
            name = f"{self.task}_{self.model}_iter{self.max_iter}_noconsistency_{now}"
        if self.log_file is None:
            self.log_file = f"{name}.log"
        if self.output_file is None:
            self.output_file = f"{name}.json"

        if Path(self.output_file).exists():
            raise FileExistsError(f"Output file exists: {self.output_file}")
        
        if self.cache_mode == "file" and self.cache_file is None:
            self.cache_file = "cache.pkl"


async def run_one_task(
    max_iter: int,
    task: Task,
    object_hypothesis_generator: ObjectHypothesisGenerator,
    object_hypothesis_inconsistency_eliminator: ObjectHypothesisInconsistencyEliminator,
    relational_hypothesis_generator: RelationalHypothesisGenerator,
    relational_hypothesis_inconsistency_eliminator: RelationalHypothesisInconsistencyEliminator,
    functional_hypothesis_generator: FunctionalHypothesisGenerator,
    auxiliary_confirmer: AuxiliaryConfirmer,
    consistency_validator: ConsistencyValidator,
    no_consistency_check: bool = False
) -> dict:
    records = []
    all_train_raw_acc, all_train_example_accs = [], []
    all_test_raw_acc, all_test_example_accs = [], []
    last_iter_test_raw_acc_without_consistency_validation = None


    # ==================================================================================================
    #   Generate object hypothesis via ObjectHypothesisGenerator
    # ==================================================================================================
    logging.info(f"Task {task.idx}: generating object hypothesis via ObjectHypothesisGenerator")
    aws = [
        object_hypothesis_generator(e["input"], task.name)
        for e in task.train_examples_str
    ] + [
        object_hypothesis_generator(e["output"], task.name)
        for e in task.train_examples_str
    ]
    results = await asyncio.gather(*aws)
    object_hypothesis, interactions, usage = zip(*results)
    input_object_hypothesis = object_hypothesis[: len(object_hypothesis) // 2]
    output_object_hypothesis = object_hypothesis[len(object_hypothesis) // 2: ]
    records.append({
        "agent": "ObjectHypothesisGenerator",
        "input object hypothesis": input_object_hypothesis,
        "output object hypothesis": output_object_hypothesis,
        "interactions": interactions,
        "usage": usage,
    })


    # ==================================================================================================
    #   Eliminate inconsistency in object hypothesis via ObjectHypothesisInconsistencyEliminator
    # ==================================================================================================
    if not no_consistency_check:
        logging.info(f"Task {task.idx}: eliminating inconsistency in object hypothesis via ObjectHypothesisInconsistencyEliminator")
        aws = [
            object_hypothesis_inconsistency_eliminator(
                task.train_examples_str[i]["input"], h, task.name
            )
            for i, h in enumerate(input_object_hypothesis)
        ] + [
            object_hypothesis_inconsistency_eliminator(
                task.train_examples_str[i]["output"], h, task.name
            )
            for i, h in enumerate(output_object_hypothesis)
        ]
        results = await asyncio.gather(*aws)
        object_hypothesis, interactions, usage = zip(*results)
        input_object_hypothesis = object_hypothesis[: len(object_hypothesis) // 2]
        output_object_hypothesis = object_hypothesis[len(object_hypothesis) // 2: ]
        records.append({
            "agent": "ObjectHypothesisInconsistencyEliminator",
            "consistent input object hypothesis": input_object_hypothesis,
            "consistent output object hypothesis": output_object_hypothesis,
            "interactions": interactions,
            "usage": usage,
        })


    # ==================================================================================================
    #   Generate relational hypothesis via RelationalHypothesisGenerator
    # ==================================================================================================
    logging.info(f"Task {task.idx}: generating relational hypothesis via RelationalHypothesisGenerator")
    results = await asyncio.gather(*[
        relational_hypothesis_generator(
            e["input"], e["output"],
            input_object_hypothesis[i], output_object_hypothesis[i],
            task.name
        )
        for i, e in enumerate(task.train_examples_str)
    ])
    relational_hypothesis, interactions, usage = zip(*results)
    records.append({
        "agent": "RelationalHypothesisGenerator",
        "relational hypothesis": relational_hypothesis,
        "interactions": interactions,
        "usage": usage,
    })


    # ==================================================================================================
    #   Eliminate inconsistency in relational hypothesis via RelationalHypothesisInconsistencyEliminator
    # ==================================================================================================
    if not no_consistency_check:
        logging.info(f"Task {task.idx}: eliminating inconsistency in relational hypothesis via RelationalHypothesisInconsistencyEliminator")
        results = await asyncio.gather(*[
            relational_hypothesis_inconsistency_eliminator(
                task.train_examples_str[i]["input"], task.train_examples_str[i]["output"],
                h, task.name
            )
            for i, h in enumerate(relational_hypothesis)
        ])
        relational_hypothesis, interactions, usage = zip(*results)
        records.append({
            "agent": "RelationalHypothesisInconsistencyEliminator",
            "consistent relational_hypothesis": relational_hypothesis,
            "interactions": interactions,
            "usage": usage,
        })


    # ==================================================================================================
    #   Refinement Loop
    # ==================================================================================================
    for i in range(1, max_iter + 1):
        if i == 1:
            # ==================================================================================================
            #   Generate functional hypothesis via FunctionalHypothesisGenerator
            # ==================================================================================================
            logging.info(f"Task {task.idx}: generating functional hypothesis via FunctionalHypothesisGenerator")
            functional_hypothesis, interactions, usage = await functional_hypothesis_generator(
                task.train_examples_str,
                input_object_hypothesis, output_object_hypothesis,
                relational_hypothesis,
                task.name
            )
            records.append({
                "agent": "FunctionalHypothesisGenerator",
                "functional hypothesis": functional_hypothesis,
                "interactions": interactions,
                "usage": usage, 
            })
        else:
            # ==================================================================================================
            #   Generate functional hypothesis via FunctionalHypothesisGenerator with feedback
            # ==================================================================================================
            logging.info(f"Task {task.idx}: generating functional hypothesis via FunctionalHypothesisGenerator with feedback")
            functional_hypothesis, interactions, usage = await functional_hypothesis_generator(
                task.train_examples_str,
                input_object_hypothesis, output_object_hypothesis,
                relational_hypothesis,
                task.name,
                previous_functional_hypothesis=functional_hypothesis,
                wrong_examples=train_wrong_examples,
            )
            records.append({
                "agent": "FunctionalHypothesisGenerator",
                "functional hypothesis": functional_hypothesis,
                "interactions": interactions,
                "usage": usage, 
            })
            if functional_hypothesis is None:
                break


        # ==================================================================================================
        #   Evaluate performance on train examples via AuxiliaryConfirmer
        # ==================================================================================================
        logging.info(f"Task {task.idx}: evaluating performance on train examples via AuxiliaryConfirmer")
        func_str, interactions, usage, train_raw_acc, train_example_accs, train_wrong_examples = await auxiliary_confirmer(
            functional_hypothesis, task.name, task.train_examples
        )
        records.append({
            "agent": "AuxiliaryConfirmer",
            "train_raw_accuracy": train_raw_acc,
            "train_example_accuracy": train_example_accs,
            "func_str": func_str,
            "train_wrong_examples": train_wrong_examples,
            "interactions": interactions,
            "usage": usage,
        })
        all_train_raw_acc.append(train_raw_acc)
        all_train_example_accs.append(train_example_accs)


        # ==================================================================================================
        #   Evaluate performance on test examples via AuxiliaryConfirmer
        #   (intermediate test performance is for analysis only, not for selecting the best iteration)
        # ==================================================================================================
        logging.info(f"Task {task.idx}: evaluating performance on test examples via AuxiliaryConfirmer")
        func_str, interactions, usage, test_raw_acc, test_example_accs, test_wrong_examples = await auxiliary_confirmer(
            functional_hypothesis, task.name, task.test_examples
        )
        records.append({
            "agent": "AuxiliaryConfirmer",
            "test_raw_accuracy": test_raw_acc,
            "test_example_accuracy": test_example_accs,
            "func_str": func_str,
            "test_wrong_examples": test_wrong_examples,
            "interactions": interactions,
            "usage": usage,
        })
        all_test_raw_acc.append(test_raw_acc)
        all_test_example_accs.append(test_example_accs)
        if last_iter_test_raw_acc_without_consistency_validation is None and train_raw_acc == 1.0:
            last_iter_test_raw_acc_without_consistency_validation = test_raw_acc


        if not train_wrong_examples:
            # ==================================================================================================
            #   Check consistency via ConsistencyValidator
            # ==================================================================================================
            if not no_consistency_check:
                logging.info(f"Task {task.idx}: checking consistency via ConsistencyValidator")
                is_consistent, interactions, usage = await consistency_validator(
                    functional_hypothesis, input_object_hypothesis, output_object_hypothesis, relational_hypothesis,
                    task.train_examples_str, task.name
                )
                records.append({
                    "agent": "ConsistencyValidator",
                    "is_consistent": is_consistent,
                    "interactions": interactions,
                    "usage": usage,
                })

                if is_consistent:
                    break
            else:
                break


    # ==================================================================================================
    #   Return results
    # ==================================================================================================
    best_iter = all_train_raw_acc.index(max(all_train_raw_acc))
    if last_iter_test_raw_acc_without_consistency_validation is None:
        last_iter_test_raw_acc_without_consistency_validation = test_raw_acc
    return {
        "best_accuracy": all_test_raw_acc[best_iter],
        "last_iter_test_raw_acc_without_consistency_validation": last_iter_test_raw_acc_without_consistency_validation,
        "train": {
            "raw_accuracy": all_train_raw_acc,
            "example_accuracy": all_train_example_accs
        },
        "test": {
            "raw_accuracy": all_test_raw_acc,
            "example_accuracy": all_test_example_accs
        },
        "records": records
    }


async def run_all_tasks(args: Args) -> dict:
    # ================================================
    #   Load data
    # ================================================
    tasks_examples: list[dict] = read_data(args.data_file)
    if args.n_tasks is not None:
        logger.info(f"Use the first {args.n_tasks}/{len(tasks_examples)} tasks")
        tasks_examples = tasks_examples[: args.n_tasks]
    else:
        logger.info(f"Use all {len(tasks_examples)} tasks")


    # ================================================
    #   Initialize
    # ================================================
    api_client = openai.AsyncOpenAI(base_url=args.base_url)
    api_semaphore = asyncio.Semaphore(args.concurrent_api_calls)
    multiprocess_executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers)

    object_hypothesis_generator = ObjectHypothesisGenerator(api_client=api_client, api_semaphore=api_semaphore, model=args.model, cache_mode=args.cache_mode, cache_file=args.cache_file, temperature=0)
    object_hypothesis_inconsistency_eliminator = ObjectHypothesisInconsistencyEliminator(api_client=api_client, api_semaphore=api_semaphore, model=args.model, cache_mode=args.cache_mode, cache_file=args.cache_file, temperature=0)
    relational_hypothesis_generator = RelationalHypothesisGenerator(api_client=api_client, api_semaphore=api_semaphore, model=args.model, cache_mode=args.cache_mode, cache_file=args.cache_file, temperature=0)
    relational_hypothesis_inconsistency_eliminator = RelationalHypothesisInconsistencyEliminator(api_client=api_client, api_semaphore=api_semaphore, model=args.model, cache_mode=args.cache_mode, cache_file=args.cache_file, temperature=0)
    functional_hypothesis_generator = FunctionalHypothesisGenerator(api_client=api_client, api_semaphore=api_semaphore, model=args.model, cache_mode=args.cache_mode, cache_file=args.cache_file, temperature=0)
    auxiliary_confirmer = AuxiliaryConfirmer(multiprocess_executor=multiprocess_executor, api_client=api_client, api_semaphore=api_semaphore, model=args.model, cache_mode=args.cache_mode, cache_file=args.cache_file, temperature=0)
    consistency_validator = ConsistencyValidator(api_client=api_client, api_semaphore=api_semaphore, model=args.model, cache_mode=args.cache_mode, cache_file=args.cache_file, temperature=0)


    # ================================================
    #   Run
    # ================================================
    aws = []
    for examples in tasks_examples:
        task = Task(task_name=args.task, examples=examples)
        aws.append(run_one_task(
            args.max_iter,
            task=task,
            object_hypothesis_generator=object_hypothesis_generator,
            object_hypothesis_inconsistency_eliminator=object_hypothesis_inconsistency_eliminator,
            relational_hypothesis_generator=relational_hypothesis_generator,
            relational_hypothesis_inconsistency_eliminator=relational_hypothesis_inconsistency_eliminator,
            functional_hypothesis_generator=functional_hypothesis_generator,
            auxiliary_confirmer=auxiliary_confirmer,
            consistency_validator=consistency_validator,
            no_consistency_check=args.no_consistency_check
        ))
    outputs_per_task = await tqdm_asyncio.gather(
        *aws,
        desc="Finished Tasks",
        file=Path(args.log_file).open("a")
    )


    # ================================================
    #   Post process outputs
    # ================================================
    best_accuracy = [o["best_accuracy"] for o in outputs_per_task]
    last_iter_test_raw_acc_without_consistency_validation = [o["last_iter_test_raw_acc_without_consistency_validation"] for o in outputs_per_task]
    all_train_raw_accuracy = [
        [o["train"]["raw_accuracy"][min(i, len(o["train"]["raw_accuracy"]) - 1)] for o in outputs_per_task]
        for i in range(args.max_iter)
    ]
    all_test_raw_accuracy = [
        [o["test"]["raw_accuracy"][min(i, len(o["test"]["raw_accuracy"]) - 1)] for o in outputs_per_task]
        for i in range(args.max_iter)
    ]

    prompt_tokens, completion_tokens = 0, 0
    for o in outputs_per_task:
        for rec in o["records"]:
            if "usage" in rec and rec["usage"] is not None:
                usage = rec["usage"]
                if isinstance(usage, dict):
                    prompt_tokens += usage["prompt_tokens"]
                    completion_tokens += usage["completion_tokens"]
                elif isinstance(usage[0], dict):
                    prompt_tokens += sum(u["prompt_tokens"] for u in usage if u)
                    completion_tokens += sum(u["completion_tokens"] for u in usage if u)
                else:
                    prompt_tokens += sum(u2["prompt_tokens"] for u1 in usage for u2 in u1)
                    completion_tokens += sum(u2["completion_tokens"] for u1 in usage for u2 in u1)

    output = {
        "task": args.task,
        "model": args.model,
        "max_iter": args.max_iter,
        "n_tasks": len(tasks_examples),
        "max_workers": multiprocess_executor._max_workers,
        "total_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        },
        "task_accuracy": (np.array(best_accuracy) == 1).mean(),
        "raw_accuracy": np.mean(best_accuracy),
        "last_iter_task_accuracy_without_consistency_validation": (np.array(last_iter_test_raw_acc_without_consistency_validation) == 1).mean(),
        "last_iter_raw_accuracy_without_consistency_validation": np.mean(last_iter_test_raw_acc_without_consistency_validation),
        "train": {
            "task_accuracy": [
                (np.array(all_train_raw_accuracy[i]) == 1).mean()
                for i in range(args.max_iter)
            ],
            "raw_accuracy": [np.mean(all_train_raw_accuracy[i]) for i in range(args.max_iter)],
            "example_accuracy": [o["train"]["example_accuracy"] for o in outputs_per_task]
        },
        "test": {
            "task_accuracy": [
                (np.array(all_test_raw_accuracy[i]) == 1).mean()
                for i in range(args.max_iter)
            ],
            "raw_accuracy": [np.mean(all_test_raw_accuracy[i]) for i in range(args.max_iter)],
            "example_accuracy": [o["test"]["example_accuracy"] for o in outputs_per_task]
        },
        "records": [o["records"] for o in outputs_per_task]
    }


    # ================================================
    #   Finalize
    # ================================================
    await api_client.close()
    multiprocess_executor.shutdown(wait=True)


    return output


def main():
    # ================================================
    #   Parse arguments
    # ================================================
    parser = HfArgumentParser((Args, ))
    args, = parser.parse_args_into_dataclasses()


    # ================================================
    #   Set up logging
    # ================================================
    log_file = Path(args.log_file)
    if log_file.exists():
        logger.info(f"New log is appended to {log_file} as it already exists.")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="[%(levelname)s] [%(asctime)s] [PID=%(process)d] [%(name)s:%(lineno)d] %(message)s",
    )
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)  # disable logging for normal HTTP request


    # ================================================
    #   Run
    # ================================================
    outputs = asyncio.run(run_all_tasks(args))


    # ================================================
    #   Write results
    # ================================================
    outputs["train"]["example_accuracy"] = list(map(
        lambda accs: json.dumps(accs),
        outputs["train"]["example_accuracy"]
    ))
    outputs["test"]["example_accuracy"] = list(map(
        lambda accs: json.dumps(accs),
        outputs["test"]["example_accuracy"]
    ))
    for records in outputs["records"]:
        for rec in records:
            if "usage" in rec and rec["usage"] is not None:
                usage = rec["usage"]
                if isinstance(usage, dict):
                    rec["usage"] = ", ".join(f"{k}: {v}" for k, v in usage.items())
                elif isinstance(usage[0], dict):
                    rec["usage"] = list(rec["usage"])
                    for i, u in enumerate(usage):
                        rec["usage"][i] = ", ".join(f"{k}: {v}" for k, v in u.items())
                else:
                    rec["usage"] = list(rec["usage"])
                    for i, u1 in enumerate(usage):
                        for j, u2 in enumerate(u1):
                            rec["usage"][i][j] = ", ".join(f"{k}: {v}" for k, v in u2.items())
            if "train_example_accuracy" in rec:
                rec["train_example_accuracy"] = json.dumps(rec["train_example_accuracy"])
            if "test_example_accuracy" in rec:
                rec["test_example_accuracy"] = json.dumps(rec["test_example_accuracy"])

    if Path(args.output_file).exists():
        logger.info(json.dumps(outputs, indent=4, ensure_ascii=False))
        raise FileExistsError(f"Output file exists: {args.output_file}")
    else:
        with Path(args.output_file).open("w") as writer:
            json.dump(outputs, writer, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
