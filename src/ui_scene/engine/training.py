from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import torch
from torch import nn
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import WeightedRandomSampler

from ui_scene.data.schema import SampleRecord


def count_labels(records: list[SampleRecord], label_to_index: dict[str, int]) -> dict[str, int]:
    counts = {label: 0 for label in label_to_index}
    for record in records:
        counts[record.label] += 1
    return counts


def build_class_weights(records: list[SampleRecord], label_to_index: dict[str, int]) -> torch.Tensor:
    counts = count_labels(records, label_to_index)
    total = sum(counts.values())
    weights = []
    for label, index in sorted(label_to_index.items(), key=lambda item: item[1]):
        del index
        weights.append(total / max(counts[label], 1))
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    return weight_tensor / weight_tensor.mean()


def build_weighted_sampler(
    records: list[SampleRecord],
    label_to_index: dict[str, int],
) -> WeightedRandomSampler:
    counts = count_labels(records, label_to_index)
    sample_weights = [1.0 / max(counts[record.label], 1) for record in records]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_criterion(
    loss_name: str,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> nn.Module:
    normalized_name = loss_name.lower()
    if normalized_name != 'cross_entropy':
        raise ValueError(f'Unsupported loss: {loss_name}')
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
) -> Optimizer:
    normalized_name = optimizer_name.lower()
    if normalized_name == 'adam':
        return Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if normalized_name == 'adamw':
        return AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f'Unsupported optimizer: {optimizer_name}')


def build_scheduler(optimizer: Optimizer, scheduler_cfg: dict | None, epochs: int):
    scheduler_cfg = scheduler_cfg or {}
    normalized_name = str(scheduler_cfg.get('name', 'none')).lower()
    if normalized_name in {'', 'none'}:
        return None
    if normalized_name == 'cosine':
        t_max = int(scheduler_cfg.get('t_max', epochs))
        eta_min = float(scheduler_cfg.get('eta_min', 1e-6))
        return CosineAnnealingLR(optimizer, T_max=max(t_max, 1), eta_min=eta_min)
    raise ValueError(f'Unsupported scheduler: {normalized_name}')


def build_loader_kwargs(
    num_workers: int,
    use_cuda: bool,
    collate_fn=None,
) -> dict:
    loader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': use_cuda,
    }
    if collate_fn is not None:
        loader_kwargs['collate_fn'] = collate_fn
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2
    return loader_kwargs
