from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class SampleRecord:
    sample_id: str
    label: str
    split: str
    image_path: Path
    json_path: Path


@dataclass(slots=True)
class ModelSample:
    sample_id: str
    label: str
    image_path: Path
    json_path: Path
    tree_text: str

