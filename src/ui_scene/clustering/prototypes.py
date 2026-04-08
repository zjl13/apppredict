from __future__ import annotations

import numpy as np


def nearest_to_centroid(embeddings: np.ndarray, centroid: np.ndarray) -> int:
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    return int(np.argmin(distances))

