from __future__ import annotations

import json
from pathlib import Path

from .schema import SampleRecord


class ManifestDataset:
    """Lightweight manifest reader for image+json paired samples."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.records = self._load_manifest()

    def _load_manifest(self) -> list[SampleRecord]:
        records: list[SampleRecord] = []
        with self.manifest_path.open("r", encoding="utf-8-sig") as fp:
            for line in fp:
                row = json.loads(line)
                records.append(
                    SampleRecord(
                        sample_id=row["sample_id"],
                        label=row["label"],
                        split=row["split"],
                        image_path=Path(row["image_path"]),
                        json_path=Path(row["json_path"]),
                    )
                )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> SampleRecord:
        return self.records[index]


def load_manifest_records(manifest_path: str | Path) -> list[SampleRecord]:
    return ManifestDataset(manifest_path).records


def build_label_to_index(records: list[SampleRecord]) -> dict[str, int]:
    labels = sorted({record.label for record in records})
    return {label: index for index, label in enumerate(labels)}
