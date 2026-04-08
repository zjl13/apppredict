from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ProjectPaths:
    data_root: Path = Path("data_qwen_split")
    manifest_root: Path = Path("data_qwen_split/manifests")
    output_root: Path = Path("outputs")

