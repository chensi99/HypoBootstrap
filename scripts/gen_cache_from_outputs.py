import os
import re
import sys
import json
import pickle
import argparse
from glob import glob
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from utils.api_utils import Cache


def create_cache(file):
    with open(file, "r") as reader:
        outputs = json.load(reader)
    model = outputs["model"]
    temperature = 0
    n = 1

    cache = {}
    for records in outputs["records"]:
        for rec in records:
            if "interactions" in rec and rec["interactions"] is not None:
                if isinstance(rec["interactions"][0], list):
                    for num, inter in enumerate(rec["interactions"]):
                        if len(inter) == 0:
                            continue
                        dial = inter[-1]
                        history = inter[: -1]
                        assert dial["role"] == "assistant"
                        key = Cache.gen_key(model, history, temperature, n)
                        matcher = re.match(
                            r"prompt_tokens: (\d+), completion_tokens: (\d+)",
                            rec["usage"][num]
                        )
                        usage = {
                            "prompt_tokens": int(matcher.group(1)),
                            "completion_tokens": int(matcher.group(2))
                        }
                        value = {"response": dial["content"], "usage": usage}
                        cache[key] = value
                else:
                    inter = rec["interactions"]
                    if len(inter) == 0:
                        continue
                    dial = inter[-1]
                    history = inter[: -1]
                    assert dial["role"] == "assistant"
                    key = Cache.gen_key(model, history, temperature, n)
                    matcher = re.match(
                        r"prompt_tokens: (\d+), completion_tokens: (\d+)",
                        rec["usage"]
                    )
                    usage = {
                        "prompt_tokens": int(matcher.group(1)),
                        "completion_tokens": int(matcher.group(2))
                    }
                    value = {"response": dial["content"], "usage": usage}
                    cache[key] = value

    print(f"Created cache of size {len(cache)} for {file}")
    return cache


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cache_file", type=str, default="cache.pkl")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert not os.path.exists(args.cache_file)
    files = glob(f"{args.output_dir}/*.json")
    cache = {}
    for file in files:
        new_cache = create_cache(file)
        cache.update(new_cache)
    print(f"\nTotal number of files: {len(files)}")
    print(f"Total cache size: {len(cache)}")

    assert not os.path.exists(args.cache_file)
    with open(args.cache_file, "wb") as f:
        pickle.dump(cache, f)
    print(f"Saved cache to {args.cache_file}")


if __name__ == "__main__":
    main()
