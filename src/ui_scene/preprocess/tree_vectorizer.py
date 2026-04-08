from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


def _to_range(value: object, default: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return default


def _transform_chunk(payload: tuple[dict, list[str]]) -> np.ndarray:
    config, texts = payload
    vectorizer = TreeTextVectorizer(**config)
    return vectorizer.transform(texts)


class TreeTextVectorizer:
    """Batch vectorizer for linearized UI tree text."""

    def __init__(
        self,
        output_dim: int = 512,
        strategy: str = 'word_hashing',
        word_dim: int | None = None,
        char_dim: int | None = None,
        word_ngram_range: tuple[int, int] = (1, 1),
        char_ngram_range: tuple[int, int] = (3, 5),
    ) -> None:
        self.output_dim = int(output_dim)
        self.strategy = str(strategy).lower()
        self.word_ngram_range = word_ngram_range
        self.char_ngram_range = char_ngram_range

        if self.strategy == 'word_hashing':
            self.word_vectorizer = HashingVectorizer(
                n_features=self.output_dim,
                alternate_sign=False,
                norm='l2',
                lowercase=True,
                ngram_range=word_ngram_range,
            )
            self.word_dim = self.output_dim
            self.char_dim = 0
            self.char_vectorizer = None
            return

        if self.strategy == 'hybrid_hashing':
            if word_dim is None and char_dim is None:
                word_dim = self.output_dim // 2
                char_dim = self.output_dim - word_dim
            elif word_dim is None:
                char_dim = int(char_dim)
                word_dim = max(self.output_dim - char_dim, 1)
            elif char_dim is None:
                word_dim = int(word_dim)
                char_dim = max(self.output_dim - word_dim, 1)

            self.word_dim = max(int(word_dim), 1)
            self.char_dim = max(int(char_dim), 1)
            total_dim = self.word_dim + self.char_dim
            if total_dim != self.output_dim:
                self.char_dim += self.output_dim - total_dim
                self.char_dim = max(self.char_dim, 1)
                self.output_dim = self.word_dim + self.char_dim

            self.word_vectorizer = HashingVectorizer(
                n_features=self.word_dim,
                alternate_sign=False,
                norm=None,
                lowercase=True,
                ngram_range=word_ngram_range,
            )
            self.char_vectorizer = HashingVectorizer(
                n_features=self.char_dim,
                alternate_sign=False,
                norm=None,
                lowercase=True,
                analyzer='char_wb',
                ngram_range=char_ngram_range,
            )
            return

        raise ValueError(f'Unsupported tree vectorizer strategy: {strategy}')

    @classmethod
    def from_config(cls, train_cfg: dict) -> 'TreeTextVectorizer':
        output_dim = int(train_cfg.get('tree_input_dim', 512))
        vectorizer_cfg = train_cfg.get('tree_vectorizer', {}) or {}
        if isinstance(vectorizer_cfg, str):
            vectorizer_cfg = {'name': vectorizer_cfg}

        return cls(
            output_dim=output_dim,
            strategy=str(vectorizer_cfg.get('name', 'word_hashing')),
            word_dim=vectorizer_cfg.get('word_dim'),
            char_dim=vectorizer_cfg.get('char_dim'),
            word_ngram_range=_to_range(vectorizer_cfg.get('word_ngram_range'), (1, 1)),
            char_ngram_range=_to_range(vectorizer_cfg.get('char_ngram_range'), (3, 5)),
        )

    def to_config(self) -> dict:
        return {
            'output_dim': self.output_dim,
            'strategy': self.strategy,
            'word_dim': self.word_dim,
            'char_dim': self.char_dim,
            'word_ngram_range': self.word_ngram_range,
            'char_ngram_range': self.char_ngram_range,
        }

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.strategy == 'word_hashing':
            return self.word_vectorizer.transform(texts).toarray().astype(np.float32, copy=False)

        word_features = self.word_vectorizer.transform(texts).toarray().astype(np.float32, copy=False)
        char_features = self.char_vectorizer.transform(texts).toarray().astype(np.float32, copy=False)
        features = np.concatenate([word_features, char_features], axis=1)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return (features / norms).astype(np.float32, copy=False)

    def transform_parallel(
        self,
        texts: list[str],
        num_workers: int = 0,
        chunk_size: int = 2048,
    ) -> np.ndarray:
        if num_workers <= 1 or len(texts) <= chunk_size:
            return self.transform(texts)

        config = self.to_config()
        chunks = [texts[index:index + chunk_size] for index in range(0, len(texts), chunk_size)]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            arrays = list(executor.map(_transform_chunk, [(config, chunk) for chunk in chunks]))
        return np.concatenate(arrays, axis=0) if arrays else np.empty((0, self.output_dim), dtype=np.float32)
