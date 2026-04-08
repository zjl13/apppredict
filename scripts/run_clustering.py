from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui_scene.clustering.prototypes import nearest_to_centroid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster exported embeddings and select prototypes.")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to .npz embeddings file.")
    parser.add_argument("--num-clusters", type=int, default=22)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embedding_path = Path(args.embeddings)
    payload = np.load(embedding_path, allow_pickle=True)

    embeddings = payload["embeddings"]
    labels = payload["labels"]
    label_names = payload["label_names"]
    sample_ids = payload["sample_ids"]

    if len(embeddings) == 0:
        raise ValueError("Embedding file is empty.")
    if len(embeddings) < args.num_clusters:
        raise ValueError("Number of samples is smaller than the requested cluster count.")

    kmeans = KMeans(n_clusters=args.num_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings)

    nmi = float(normalized_mutual_info_score(labels, cluster_ids))
    ari = float(adjusted_rand_score(labels, cluster_ids))
    sil = float(silhouette_score(embeddings, cluster_ids)) if len(set(cluster_ids)) > 1 else 0.0

    cluster_members: dict[int, list[int]] = defaultdict(list)
    for index, cluster_id in enumerate(cluster_ids):
        cluster_members[int(cluster_id)].append(index)

    prototypes: list[dict] = []
    for cluster_id, member_indices in sorted(cluster_members.items()):
        member_embeddings = embeddings[member_indices]
        centroid = kmeans.cluster_centers_[cluster_id]
        local_idx = nearest_to_centroid(member_embeddings, centroid)
        global_idx = member_indices[local_idx]
        label_counter = Counter(label_names[member_indices])

        prototypes.append(
            {
                "cluster_id": cluster_id,
                "prototype_sample_id": str(sample_ids[global_idx]),
                "prototype_label": str(label_names[global_idx]),
                "cluster_size": len(member_indices),
                "top_labels": dict(label_counter.most_common(5)),
            }
        )

    output_dir = embedding_path.parent / "cluster_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"{embedding_path.stem}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "embedding_file": str(embedding_path),
                "num_samples": int(len(embeddings)),
                "embedding_dim": int(embeddings.shape[1]),
                "num_clusters": int(args.num_clusters),
                "nmi": nmi,
                "ari": ari,
                "silhouette_score": sil,
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )

    prototype_path = output_dir / f"{embedding_path.stem}_prototypes.json"
    with prototype_path.open("w", encoding="utf-8") as fp:
        json.dump(prototypes, fp, ensure_ascii=False, indent=2)

    assignment_path = output_dir / f"{embedding_path.stem}_assignments.jsonl"
    with assignment_path.open("w", encoding="utf-8") as fp:
        for index, cluster_id in enumerate(cluster_ids):
            fp.write(
                json.dumps(
                    {
                        "sample_id": str(sample_ids[index]),
                        "label": str(label_names[index]),
                        "label_id": int(labels[index]),
                        "cluster_id": int(cluster_id),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved prototypes to: {prototype_path}")


if __name__ == "__main__":
    main()
