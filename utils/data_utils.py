import json
from pathlib import Path


def read_data(data_file: str) -> list:
    data = []
    with Path(data_file).open("r") as reader:
        for line in reader:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data
