from __future__ import annotations

from collections import Counter

from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import f1_score


def accuracy(predictions: list[int], labels: list[int]) -> float:
    if not labels:
        return 0.0
    correct = sum(int(p == y) for p, y in zip(predictions, labels))
    return correct / len(labels)


def label_distribution(labels: list[str]) -> dict[str, int]:
    return dict(Counter(labels))


def macro_f1(predictions: list[int], labels: list[int]) -> float:
    if not labels:
        return 0.0
    return float(f1_score(labels, predictions, average="macro"))


def confusion_matrix(
    predictions: list[int],
    labels: list[int],
    num_labels: int | None = None,
) -> list[list[int]]:
    if not labels:
        return []
    known_labels = list(range(num_labels)) if num_labels is not None else None
    return sk_confusion_matrix(labels, predictions, labels=known_labels).tolist()
