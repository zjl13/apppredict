from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .schema import SampleRecord


class ImageClassificationDataset(Dataset):
    """Manifest-driven dataset for image-only classification."""

    def __init__(
        self,
        records: list[SampleRecord],
        label_to_index: dict[str, int],
        transform=None,
    ) -> None:
        self.records = records
        self.label_to_index = label_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image = self._load_image(record.image_path)
        if self.transform is not None:
            image = self.transform(image)

        return {
            "sample_id": record.sample_id,
            "image": image,
            "label": self.label_to_index[record.label],
            "label_name": record.label,
        }

    @staticmethod
    def _load_image(image_path: str | Path) -> Image.Image:
        image_path = Path(image_path)
        with Image.open(image_path) as image:
            return image.convert("RGB")
