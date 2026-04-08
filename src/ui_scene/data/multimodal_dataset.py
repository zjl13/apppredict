from __future__ import annotations

from multiprocessing import Pool

import torch
from torch.utils.data import Dataset

from ui_scene.preprocess.tree_linearizer import linearize_tree

from .image_dataset import ImageClassificationDataset
from .schema import SampleRecord


def _linearize_record(payload: tuple[str, int, str]) -> str:
    json_path, max_tree_nodes, tree_profile = payload
    return linearize_tree(json_path, max_nodes=max_tree_nodes, profile=tree_profile)


class MultimodalClassificationDataset(Dataset):
    """Manifest-driven dataset for image + UI tree classification."""

    def __init__(
        self,
        records: list[SampleRecord],
        label_to_index: dict[str, int],
        image_transform=None,
        max_tree_nodes: int = 256,
        tree_profile: str = 'legacy',
        cache_tree_text: bool = True,
        tree_cache_workers: int = 0,
    ) -> None:
        self.image_dataset = ImageClassificationDataset(records, label_to_index, image_transform)
        self.records = records
        self.max_tree_nodes = max_tree_nodes
        self.tree_profile = tree_profile
        self.cache_tree_text = cache_tree_text
        self.tree_cache_workers = max(int(tree_cache_workers), 0)
        self.tree_texts = self._build_tree_texts() if cache_tree_text else None
        self.tree_feature_matrix = None

    def _build_tree_texts(self) -> list[str]:
        payloads = [
            (str(record.json_path), self.max_tree_nodes, self.tree_profile)
            for record in self.records
        ]
        if self.tree_cache_workers <= 1 or len(payloads) < 512:
            return [_linearize_record(payload) for payload in payloads]

        with Pool(processes=self.tree_cache_workers) as pool:
            return list(pool.imap(_linearize_record, payloads, chunksize=64))

    def set_tree_feature_matrix(self, tree_feature_matrix) -> None:
        self.tree_feature_matrix = tree_feature_matrix

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        image_sample = self.image_dataset[index]
        record = self.records[index]
        tree_text = self.tree_texts[index] if self.tree_texts is not None else linearize_tree(
            record.json_path,
            max_nodes=self.max_tree_nodes,
            profile=self.tree_profile,
        )
        tree_features = None
        if self.tree_feature_matrix is not None:
            tree_features = torch.from_numpy(self.tree_feature_matrix[index]).to(torch.float32)

        return {
            'sample_id': image_sample['sample_id'],
            'image': image_sample['image'],
            'label': image_sample['label'],
            'label_name': image_sample['label_name'],
            'tree_text': tree_text,
            'tree_features': tree_features,
        }
