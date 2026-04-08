from __future__ import annotations

from torch.utils.data import Dataset

from ui_scene.preprocess.tree_linearizer import linearize_tree

from .image_dataset import ImageClassificationDataset
from .schema import SampleRecord


class MultimodalClassificationDataset(Dataset):
    """Manifest-driven dataset for image + UI tree classification."""

    def __init__(
        self,
        records: list[SampleRecord],
        label_to_index: dict[str, int],
        image_transform=None,
        max_tree_nodes: int = 256,
        tree_profile: str = 'legacy',
    ) -> None:
        self.image_dataset = ImageClassificationDataset(records, label_to_index, image_transform)
        self.records = records
        self.max_tree_nodes = max_tree_nodes
        self.tree_profile = tree_profile

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        image_sample = self.image_dataset[index]
        record = self.records[index]

        return {
            'sample_id': image_sample['sample_id'],
            'image': image_sample['image'],
            'label': image_sample['label'],
            'label_name': image_sample['label_name'],
            'tree_text': linearize_tree(
                record.json_path,
                max_nodes=self.max_tree_nodes,
                profile=self.tree_profile,
            ),
        }
