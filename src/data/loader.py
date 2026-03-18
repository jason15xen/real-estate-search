import json
from pathlib import Path

from src.models.property import Property


def load_properties(file_path: str = "mockup.json") -> list[Property]:
    path = Path(file_path)
    with open(path) as f:
        raw_data = json.load(f)
    return [Property(**item) for item in raw_data]
