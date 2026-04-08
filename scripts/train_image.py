from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui_scene.data.dataset import build_label_to_index, load_manifest_records
from ui_scene.data.image_dataset import ImageClassificationDataset
from ui_scene.engine.metrics import accuracy, confusion_matrix, macro_f1
from ui_scene.engine.training import (
    build_class_weights,
    build_criterion,
    build_loader_kwargs,
    build_optimizer,
    build_scheduler,
    build_weighted_sampler,
)
from ui_scene.utils.config import load_config
from ui_scene.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train image-only scene classifier.')
    parser.add_argument('--config', type=str, default='configs/train_image.yaml', help='Path to training config.')
    parser.add_argument('--epochs', type=int, default=None, help='Override training epochs.')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size.')
    parser.add_argument('--limit-train', type=int, default=None, help='Limit number of training samples for smoke tests.')
    parser.add_argument('--limit-val', type=int, default=None, help='Limit number of validation samples for smoke tests.')
    parser.add_argument('--resume-from', type=str, default=None, help='Optional checkpoint to resume model weights from.')
    return parser.parse_args()


def build_transforms(image_size: int, augmentation: str = 'legacy') -> tuple[transforms.Compose, transforms.Compose]:
    augmentation = augmentation.lower()
    if augmentation == 'ui_light':
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
                transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.98, 1.02)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    elif augmentation == 'none':
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def build_model(backbone_name: str, num_classes: int) -> nn.Module:
    if backbone_name == 'mobilenet_v3_small':
        try:
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        except Exception:
            model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if backbone_name == 'resnet18':
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f'Unsupported backbone: {backbone_name}')


def save_snapshot(
    run_dir: Path,
    model: nn.Module,
    label_to_index: dict[str, int],
    config: dict,
    val_metrics: dict,
    index_to_label: dict[int, str],
) -> None:
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'label_to_index': label_to_index,
            'config': config,
        },
        run_dir / 'best_model.pt',
    )
    with (run_dir / 'best_val_metrics.json').open('w', encoding='utf-8') as fp:
        json.dump(
            {
                'accuracy': val_metrics['accuracy'],
                'macro_f1': val_metrics['macro_f1'],
                'loss': val_metrics['loss'],
                'confusion_matrix': val_metrics['confusion_matrix'],
                'index_to_label': index_to_label,
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    criterion: nn.Module,
    amp_enabled: bool,
) -> dict:
    model.eval()
    non_blocking = device.type == 'cuda'
    predictions: list[int] = []
    labels: list[int] = []
    losses: list[float] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device, non_blocking=non_blocking)
            target = batch['label'].to(device, non_blocking=non_blocking)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, target)

            losses.append(float(loss.item()))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(target.cpu().tolist())

    avg_loss = sum(losses) / max(len(losses), 1)
    return {
        'loss': avg_loss,
        'accuracy': accuracy(predictions, labels),
        'macro_f1': macro_f1(predictions, labels),
        'confusion_matrix': confusion_matrix(predictions, labels, num_labels=num_classes),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = int(config['project']['seed'])
    set_seed(seed)

    train_cfg = config['train']
    model_cfg = config['model']
    path_cfg = config['paths']

    epochs = args.epochs if args.epochs is not None else int(train_cfg['epochs'])
    batch_size = args.batch_size if args.batch_size is not None else int(train_cfg['batch_size'])
    num_workers = int(train_cfg.get('num_workers', 0))
    image_size = int(train_cfg['image_size'])
    learning_rate = float(train_cfg['learning_rate'])
    weight_decay = float(train_cfg.get('weight_decay', 0.0))
    augmentation = str(train_cfg.get('augmentation', 'legacy'))
    sampler_name = str(train_cfg.get('sampler', 'none')).lower()
    label_smoothing = float(train_cfg.get('label_smoothing', 0.0))
    optimizer_cfg = train_cfg.get('optimizer', {})
    scheduler_cfg = train_cfg.get('scheduler', {})
    optimizer_name = str(optimizer_cfg.get('name', 'adam')) if isinstance(optimizer_cfg, dict) else str(optimizer_cfg)

    manifest_root = Path(path_cfg['manifest_root'])
    output_root = Path(path_cfg['output_root'])
    run_name = str(train_cfg.get('output_name', 'train_image'))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_root / 'runs' / f'{run_name}_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)

    full_train_records = load_manifest_records(manifest_root / 'train.jsonl')
    full_val_records = load_manifest_records(manifest_root / 'val.jsonl')

    label_to_index = build_label_to_index(full_train_records)
    index_to_label = {index: label for label, index in label_to_index.items()}

    train_records = full_train_records
    val_records = full_val_records

    if args.limit_train is not None:
        train_records = train_records[: args.limit_train]
    if args.limit_val is not None:
        val_records = val_records[: args.limit_val]

    train_transform, eval_transform = build_transforms(image_size, augmentation=augmentation)
    train_dataset = ImageClassificationDataset(train_records, label_to_index, train_transform)
    val_dataset = ImageClassificationDataset(val_records, label_to_index, eval_transform)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    train_sampler = None
    if sampler_name == 'weighted_random':
        train_sampler = build_weighted_sampler(train_records, label_to_index)

    loader_kwargs = build_loader_kwargs(num_workers=num_workers, use_cuda=use_cuda)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    device = torch.device('cuda' if use_cuda else 'cpu')
    model = build_model(model_cfg['image_backbone'], len(label_to_index)).to(device)

    resume_from = args.resume_from if args.resume_from is not None else train_cfg.get('resume_from')
    if resume_from:
        checkpoint = torch.load(resume_from, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Resumed model weights from: {resume_from}')

    class_weights = None
    if bool(train_cfg.get('use_class_weights', False)):
        class_weights = build_class_weights(train_records, label_to_index).to(device)

    criterion = build_criterion(
        loss_name=str(train_cfg.get('loss_name', 'cross_entropy')),
        class_weights=class_weights,
        label_smoothing=label_smoothing,
    )
    optimizer = build_optimizer(
        model.parameters(),
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = build_scheduler(optimizer, scheduler_cfg if isinstance(scheduler_cfg, dict) else {'name': scheduler_cfg}, epochs)

    amp_enabled = bool(train_cfg.get('use_amp', use_cuda)) and use_cuda
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    best_val_acc = -1.0
    history: list[dict] = []

    print(f'Device: {device}')
    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    print(f'Classes: {len(label_to_index)}')
    print(f'Optimizer: {optimizer_name}')
    print(f'Sampler: {sampler_name}')
    non_blocking = device.type == 'cuda'

    if resume_from:
        initial_metrics = evaluate(
            model,
            val_loader,
            device,
            num_classes=len(label_to_index),
            criterion=criterion,
            amp_enabled=amp_enabled,
        )
        best_val_acc = initial_metrics['accuracy']
        history.append(
            {
                'epoch': 0,
                'train': None,
                'val': {
                    'loss': initial_metrics['loss'],
                    'accuracy': initial_metrics['accuracy'],
                    'macro_f1': initial_metrics['macro_f1'],
                },
            }
        )
        save_snapshot(run_dir, model, label_to_index, config, initial_metrics, index_to_label)
        print(
            'Resume eval | '
            f"val_loss={initial_metrics['loss']:.4f} "
            f"val_acc={initial_metrics['accuracy']:.4f} "
            f"val_macro_f1={initial_metrics['macro_f1']:.4f}"
        )

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        predictions: list[int] = []
        labels: list[int] = []

        for batch in train_loader:
            images = batch['image'].to(device, non_blocking=non_blocking)
            target = batch['label'].to(device, non_blocking=non_blocking)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, target)

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += float(loss.item())
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            labels.extend(target.cpu().tolist())

        if scheduler is not None:
            scheduler.step()

        train_metrics = {
            'loss': running_loss / max(len(train_loader), 1),
            'accuracy': accuracy(predictions, labels),
            'macro_f1': macro_f1(predictions, labels),
        }
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            num_classes=len(label_to_index),
            criterion=criterion,
            amp_enabled=amp_enabled,
        )

        epoch_result = {
            'epoch': epoch,
            'train': train_metrics,
            'val': {
                'loss': val_metrics['loss'],
                'accuracy': val_metrics['accuracy'],
                'macro_f1': val_metrics['macro_f1'],
            },
        }
        history.append(epoch_result)

        print(
            f'Epoch {epoch}/{epochs} | '
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            save_snapshot(run_dir, model, label_to_index, config, val_metrics, index_to_label)

    with (run_dir / 'history.json').open('w', encoding='utf-8') as fp:
        json.dump(history, fp, ensure_ascii=False, indent=2)

    with (run_dir / 'label_to_index.json').open('w', encoding='utf-8') as fp:
        json.dump(label_to_index, fp, ensure_ascii=False, indent=2)

    print(f'Training finished. Outputs saved to: {run_dir}')


if __name__ == '__main__':
    main()
