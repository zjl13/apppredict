from __future__ import annotations

import hashlib
import pickle
from multiprocessing import Pool
from pathlib import Path

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
        cache_root: str | Path | None = None,
        split_name: str | None = None,
    ) -> None:
        self.image_dataset = ImageClassificationDataset(records, label_to_index, image_transform)
        self.records = records
        self.max_tree_nodes = max_tree_nodes
        self.tree_profile = tree_profile
        self.cache_tree_text = cache_tree_text
        self.tree_cache_workers = max(int(tree_cache_workers), 0)
        self.split_name = split_name or self._infer_split_name()
        self.cache_root = Path(cache_root) if cache_root is not None else None
        self.cache_key = self._build_cache_key()
        self.tree_texts = self._build_tree_texts() if cache_tree_text else None
        self.tree_feature_matrix = None

    def _infer_split_name(self) -> str:
        if not self.records:
            return 'unknown'
        split_names = {record.split for record in self.records}
        if len(split_names) == 1:
            return next(iter(split_names))
        return 'mixed'

    def _build_cache_key(self) -> str:
        digest = hashlib.sha1()
        digest.update(self.split_name.encode('utf-8'))
        digest.update(str(self.max_tree_nodes).encode('utf-8'))
        digest.update(self.tree_profile.encode('utf-8'))
        digest.update(str(len(self.records)).encode('utf-8'))
        for record in self.records:
            digest.update(record.sample_id.encode('utf-8'))
            digest.update(str(record.json_path).encode('utf-8'))
        return digest.hexdigest()[:16]

    def _tree_text_cache_path(self) -> Path | None:
        if self.cache_root is None:
            return None
        return self.cache_root / 'tree_texts' / f'{self.split_name}_{self.cache_key}.pkl'

    def _build_tree_texts(self) -> list[str]:
        cache_path = self._tree_text_cache_path()
        if cache_path is not None and cache_path.exists():
            with cache_path.open('rb') as fp:
                return pickle.load(fp)

        payloads = [
            (str(record.json_path), self.max_tree_nodes, self.tree_profile)
            for record in self.records
        ]
        if self.tree_cache_workers <= 1 or len(payloads) < 512:
            tree_texts = [_linearize_record(payload) for payload in payloads]
        else:
            with Pool(processes=self.tree_cache_workers) as pool:
                tree_texts = list(pool.imap(_linearize_record, payloads, chunksize=64))

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open('wb') as fp:
                pickle.dump(tree_texts, fp, protocol=pickle.HIGHEST_PROTOCOL)
        return tree_texts

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
            tree_features = torch.tensor(self.tree_feature_matrix[index], dtype=torch.float32)

        return {
            'sample_id': image_sample['sample_id'],
            'image': image_sample['image'],
            'label': image_sample['label'],
            'label_name': image_sample['label_name'],
            'tree_text': tree_text,
            'tree_features': tree_features,
        }
