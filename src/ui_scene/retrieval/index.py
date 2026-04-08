from __future__ import annotations

import numpy as np


def cosine_similarity(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    query = query / (np.linalg.norm(query) + 1e-8)
    gallery = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-8)
    return gallery @ query

