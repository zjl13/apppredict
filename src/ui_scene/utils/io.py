from __future__ import annotations

import json
from pathlib import Path


def read_json(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)

