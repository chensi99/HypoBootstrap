# HypoBootstrap





This repository contains the code necessary to reproduce HypoBootstrap, an LLM-based inductive reasoning framework introduced in our NIPS 2025 paper:

**HypoBootstrap: A Bootstrapping Framework for Inductive Reasoning**





# Installation

The code is tested with `Ubuntu 22.04.3 LTS` and `Python 3.12.8`.

You can install the dependencies in a conda environment with the following instructions:
```shell
conda create -n HypoBootstrap python=3.12.8
conda activate HypoBootstrap
pip install -r requirements.txt
```





# Using cache

Since several LLMs are (or will be) legacy or unavailable from online API service, we provide full intermediate results of our experiments to ensure reproducibility.
```shell
python scripts/gen_cache_from_outputs.py --output_dir ./outputs --cache_file cache.pkl
```
Then, the agents will only read LLM responses from cache and do not call API service.
Otherwise, to call LLM API, `OPENAI_API_KEY` should be set as environment variable.





# Run



To reproduce $\texttt{HB}, \texttt{HB}^*$ with `GPT-4`, run

($T=1$)
```shell
# List Function
python run.py --task=list_function --data_file=./data/list_function.jsonl --model=gpt-4-0613 --max_iter=1 --cache_file=cache.pkl
# ARC
python run.py --task=arc --data_file=./data/miniarc.jsonl --model=gpt-4-0613 --max_iter=1 --cache_file=cache.pkl
# ACRE
python run.py --task=acre --data_file=./data/acre.jsonl --model=gpt-4-0613 --max_iter=1 --cache_file=cache.pkl
# SCAN
python run.py --task=scan --data_file=./data/miniscan.jsonl --model=gpt-4-0613 --max_iter=1 --cache_file=cache.pkl
```

After execution, the program will produce corresponding result files (.json), which should be (approximately) the same with the result files we provide in `./outputs` (if using cache).

- To experiment with `DeepSeek-V3`, replace `--model=gpt-4-0613` with `--model=deepseek-chat` and add `--base_url=https://api.deepseek.com`.
- To experiment with `T=3`, replace `---max_iter=1` with `--max_iter=3`.
- To reproduce $\widetilde{\texttt{HB}}, \widetilde{\texttt{HB}}^*$, add `--no_consistency_check`.


> Note: The cache to reproduce `ARC` task with `DeepSeek-V3` was broken, probably because the caching mechanism we use did not correctly hit the cache when we conducted experiments. Therefore, the corresponding result files are missing.





# Result file

```json
    "task":  # task name
    "model":  # model name
    "max_iter":  # number of refinement iterations
    "n_tasks":  # number of tasks
    "max_workers":  # number of multiprocessing workers
    "total_usage": {
        "prompt_tokens":  # number of prompt tokens
        "completion_tokens":  # number of completion tokens
    },
    "task_accuracy":  # task accuracy (HB*)
    "raw_accuracy":  # raw_accuracy (HB*)
    "last_iter_task_accuracy_without_consistency_validation":  # task accuracy (HB)
    "last_iter_raw_accuracy_without_consistency_validation":  # raw accuracy (HB)
    "train": {
        "task_accuracy": [
            # train task accuracy per iteration
        ],
        "raw_accuracy": [
            # train raw accuracy per iteration
        ],
        "example_accuracy":  # exhaustive train example accuracy
    },
    "test": {
        "task_accuracy": [
            # test task accuracy per iteration
        ],
        "raw_accuracy": [
            # test raw accuracy per iteration
        ],
        "example_accuracy":  # exhaustive test example accuracy
    },
    "records":  # full intermediate records
```




# Acknowlegment

Several parts of our code are derived from [this repository](https://github.com/linlu-qiu/lm-inductive-reasoning).
We acknowledge their detailed open-source code.


