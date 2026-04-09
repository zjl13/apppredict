from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui_scene.data.dataset import build_label_to_index, load_manifest_records
from ui_scene.data.multimodal_dataset import MultimodalClassificationDataset
from ui_scene.engine.metrics import accuracy, confusion_matrix, macro_f1
from ui_scene.engine.training import (
    ModelEMA,
    build_class_weights,
    build_criterion,
    build_loader_kwargs,
    build_optimizer,
    build_scheduler,
    build_weighted_sampler,
)
from ui_scene.models.multimodal_model import MultimodalSceneClassifier
from ui_scene.preprocess.tree_vectorizer import TreeTextVectorizer
from ui_scene.utils.config import load_config
from ui_scene.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train multimodal scene classifier.')
    parser.add_argument('--config', type=str, default='configs/train_multimodal.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--limit-train', type=int, default=None)
    parser.add_argument('--limit-val', type=int, default=None)
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


def build_collate_fn(tree_vectorizer: TreeTextVectorizer):
    def collate_fn(batch: list[dict]) -> dict:
        images = torch.stack([item['image'] for item in batch], dim=0)
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        tree_texts = [item['tree_text'] for item in batch]
        if batch and batch[0].get('tree_features') is not None:
            tree_features = torch.stack([item['tree_features'] for item in batch], dim=0)
        else:
            tree_features = torch.tensor(tree_vectorizer.transform(tree_texts), dtype=torch.float32)
        return {
            'sample_id': [item['sample_id'] for item in batch],
            'image': images,
            'label': labels,
            'label_name': [item['label_name'] for item in batch],
            'tree_text': tree_texts,
            'tree_features': tree_features,
        }

    return collate_fn


def build_tree_feature_cache_path(
    cache_root: Path,
    dataset: MultimodalClassificationDataset,
    tree_vectorizer: TreeTextVectorizer,
) -> Path:
    return cache_root / 'tree_features' / (
        f'{dataset.split_name}_{dataset.cache_key}_{tree_vectorizer.config_hash()}.npy'
    )


def build_optimizer_parameters(
    model: MultimodalSceneClassifier,
    learning_rate: float,
    backbone_lr_scale: float,
):
    if abs(backbone_lr_scale - 1.0) < 1e-9:
        return model.parameters()

    backbone_params = []
    head_params = []
    backbone_param_ids = {id(parameter) for parameter in model.image_backbone.parameters()}
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in backbone_param_ids:
            backbone_params.append(parameter)
        else:
            head_params.append(parameter)

    return [
        {'params': backbone_params, 'lr': learning_rate * backbone_lr_scale},
        {'params': head_params, 'lr': learning_rate},
    ]


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
            tree_features = batch['tree_features'].to(device, non_blocking=non_blocking)
            target = batch['label'].to(device, non_blocking=non_blocking)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits, _ = model(images, tree_features)
                loss = criterion(logits, target)

            losses.append(float(loss.item()))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(target.cpu().tolist())

    return {
        'loss': sum(losses) / max(len(losses), 1),
        'accuracy': accuracy(predictions, labels),
        'macro_f1': macro_f1(predictions, labels),
        'confusion_matrix': confusion_matrix(predictions, labels, num_labels=num_classes),
    }


def set_backbone_trainable(model: MultimodalSceneClassifier, enabled: bool) -> None:
    for parameter in model.image_backbone.parameters():
        parameter.requires_grad = enabled


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_seed(int(config['project']['seed']))

    train_cfg = config['train']
    model_cfg = config['model']
    path_cfg = config['paths']

    epochs = args.epochs if args.epochs is not None else int(train_cfg['epochs'])
    batch_size = args.batch_size if args.batch_size is not None else int(train_cfg['batch_size'])
    image_size = int(train_cfg['image_size'])
    num_workers = int(train_cfg.get('num_workers', 0))
    max_tree_tokens = int(train_cfg.get('max_tree_tokens', 256))
    tree_profile = str(train_cfg.get('tree_profile', 'legacy'))
    learning_rate = float(train_cfg['learning_rate'])
    weight_decay = float(train_cfg.get('weight_decay', 0.0))
    augmentation = str(train_cfg.get('augmentation', 'legacy'))
    sampler_name = str(train_cfg.get('sampler', 'none')).lower()
    label_smoothing = float(train_cfg.get('label_smoothing', 0.0))
    optimizer_cfg = train_cfg.get('optimizer', {})
    scheduler_cfg = train_cfg.get('scheduler', {})
    optimizer_name = str(optimizer_cfg.get('name', 'adam')) if isinstance(optimizer_cfg, dict) else str(optimizer_cfg)
    freeze_backbone_epochs = int(train_cfg.get('freeze_backbone_epochs', 0))
    precompute_tree_features = bool(train_cfg.get('precompute_tree_features', False))
    tree_cache_workers = int(train_cfg.get('tree_cache_workers', 0))
    tree_feature_chunk_size = int(train_cfg.get('tree_feature_chunk_size', 2048))
    aux_cfg = train_cfg.get('auxiliary_loss', {}) or {}
    image_aux_weight = float(aux_cfg.get('image_weight', 0.0))
    tree_aux_weight = float(aux_cfg.get('tree_weight', 0.0))
    ema_cfg = train_cfg.get('ema', {}) or {}
    ema_enabled = bool(ema_cfg.get('use', False))
    ema_decay = float(ema_cfg.get('decay', 0.999))
    backbone_lr_scale = float(optimizer_cfg.get('backbone_lr_scale', 1.0))

    manifest_root = Path(path_cfg['manifest_root'])
    output_root = Path(path_cfg['output_root'])
    cache_root = Path(path_cfg.get('cache_root', output_root / 'cache'))
    run_name = str(train_cfg.get('output_name', 'train_multimodal'))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_root / 'runs' / f'{run_name}_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)

    full_train_records = load_manifest_records(manifest_root / 'train.jsonl')
    full_val_records = load_manifest_records(manifest_root / 'val.jsonl')
    label_to_index = build_label_to_index(full_train_records)
    index_to_label = {index: label for label, index in label_to_index.items()}

    train_records = full_train_records[: args.limit_train] if args.limit_train is not None else full_train_records
    val_records = full_val_records[: args.limit_val] if args.limit_val is not None else full_val_records

    train_transform, eval_transform = build_transforms(image_size, augmentation=augmentation)
    train_dataset = MultimodalClassificationDataset(
        train_records,
        label_to_index,
        image_transform=train_transform,
        max_tree_nodes=max_tree_tokens,
        tree_profile=tree_profile,
        tree_cache_workers=tree_cache_workers,
        cache_root=cache_root,
    )
    val_dataset = MultimodalClassificationDataset(
        val_records,
        label_to_index,
        image_transform=eval_transform,
        max_tree_nodes=max_tree_tokens,
        tree_profile=tree_profile,
        tree_cache_workers=tree_cache_workers,
        cache_root=cache_root,
    )
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    train_sampler = None
    if sampler_name == 'weighted_random':
        train_sampler = build_weighted_sampler(train_records, label_to_index)

    tree_vectorizer = TreeTextVectorizer.from_config(train_cfg)
    if precompute_tree_features:
        print('Precomputing train tree features...')
        train_dataset.set_tree_feature_matrix(
            tree_vectorizer.transform_parallel(
                train_dataset.tree_texts,
                num_workers=tree_cache_workers,
                chunk_size=tree_feature_chunk_size,
                cache_path=build_tree_feature_cache_path(cache_root, train_dataset, tree_vectorizer),
            )
        )
        print('Precomputing val tree features...')
        val_dataset.set_tree_feature_matrix(
            tree_vectorizer.transform_parallel(
                val_dataset.tree_texts,
                num_workers=tree_cache_workers,
                chunk_size=tree_feature_chunk_size,
                cache_path=build_tree_feature_cache_path(cache_root, val_dataset, tree_vectorizer),
            )
        )
    loader_kwargs = build_loader_kwargs(
        num_workers=num_workers,
        use_cuda=use_cuda,
        collate_fn=build_collate_fn(tree_vectorizer),
    )
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
    model = MultimodalSceneClassifier(
        backbone_name=model_cfg['image_backbone'],
        image_dim=int(model_cfg.get('image_dim', model_cfg.get('fusion_dim', 256))),
        tree_input_dim=tree_vectorizer.output_dim,
        tree_dim=int(model_cfg.get('tree_hidden_dim', 128)),
        fusion_dim=int(model_cfg.get('fusion_dim', 256)),
        num_classes=len(label_to_index),
        fusion_mode=str(model_cfg.get('fusion_mode', 'concat')),
        tree_encoder_type=str(model_cfg.get('tree_encoder_type', 'simple')),
        dropout=float(model_cfg.get('dropout', 0.1)),
        use_aux_heads=bool(model_cfg.get('use_aux_heads', False)),
        branch_dropout=float(model_cfg.get('branch_dropout', 0.0)),
    ).to(device)

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
        build_optimizer_parameters(model, learning_rate, backbone_lr_scale),
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = build_scheduler(optimizer, scheduler_cfg if isinstance(scheduler_cfg, dict) else {'name': scheduler_cfg}, epochs)

    amp_enabled = bool(train_cfg.get('use_amp', use_cuda)) and use_cuda
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    best_val_acc = -1.0
    history: list[dict] = []
    ema_helper = ModelEMA(model, decay=ema_decay) if ema_enabled else None

    print(f'Device: {device}')
    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    print(f'Classes: {len(label_to_index)}')
    print(f'Optimizer: {optimizer_name}')
    print(f'Sampler: {sampler_name}')
    print(f'Tree profile: {tree_profile}')
    print(f'Tree vectorizer: {tree_vectorizer.strategy} ({tree_vectorizer.output_dim} dims)')
    print(f'Precompute tree features: {precompute_tree_features}')
    print(f'Tree cache workers: {tree_cache_workers}')
    print(f'Fusion mode: {model_cfg.get("fusion_mode", "concat")}')
    print(f'Freeze backbone epochs: {freeze_backbone_epochs}')
    print(f'Cache root: {cache_root}')
    print(f'Backbone LR scale: {backbone_lr_scale}')
    print(f'EMA enabled: {ema_enabled}')
    if ema_enabled:
        print(f'EMA decay: {ema_decay}')
    if image_aux_weight > 0.0 or tree_aux_weight > 0.0:
        print(f'Auxiliary loss weights: image={image_aux_weight} tree={tree_aux_weight}')
    non_blocking = device.type == 'cuda'

    backbone_frozen = False
    if freeze_backbone_epochs > 0 and hasattr(model, 'image_backbone'):
        set_backbone_trainable(model, False)
        backbone_frozen = True
        print('Image backbone frozen for warmup epochs.')

    if resume_from:
        initial_metrics = evaluate(
            ema_helper.ema_model if ema_helper is not None else model,
            val_loader,
            device,
            len(label_to_index),
            criterion,
            amp_enabled,
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
        save_snapshot(
            run_dir,
            ema_helper.ema_model if ema_helper is not None else model,
            label_to_index,
            config,
            initial_metrics,
            index_to_label,
        )
        print(
            'Resume eval | '
            f"val_loss={initial_metrics['loss']:.4f} "
            f"val_acc={initial_metrics['accuracy']:.4f} "
            f"val_macro_f1={initial_metrics['macro_f1']:.4f}"
        )

    use_aux_losses = bool(model_cfg.get('use_aux_heads', False)) and (image_aux_weight > 0.0 or tree_aux_weight > 0.0)

    for epoch in range(1, epochs + 1):
        if backbone_frozen and epoch == freeze_backbone_epochs + 1:
            set_backbone_trainable(model, True)
            backbone_frozen = False
            print('Image backbone unfrozen.')

        model.train()
        predictions: list[int] = []
        labels: list[int] = []
        running_loss = 0.0

        for batch in train_loader:
            images = batch['image'].to(device, non_blocking=non_blocking)
            tree_features = batch['tree_features'].to(device, non_blocking=non_blocking)
            target = batch['label'].to(device, non_blocking=non_blocking)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                if use_aux_losses:
                    logits, _, aux_outputs = model(images, tree_features, return_aux=True)
                else:
                    logits, _ = model(images, tree_features)
                    aux_outputs = {}

                loss = criterion(logits, target)
                if image_aux_weight > 0.0 and 'image_logits' in aux_outputs:
                    loss = loss + image_aux_weight * criterion(aux_outputs['image_logits'], target)
                if tree_aux_weight > 0.0 and 'tree_logits' in aux_outputs:
                    loss = loss + tree_aux_weight * criterion(aux_outputs['tree_logits'], target)

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if ema_helper is not None:
                ema_helper.update(model)

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
            ema_helper.ema_model if ema_helper is not None else model,
            val_loader,
            device,
            len(label_to_index),
            criterion,
            amp_enabled,
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
            save_snapshot(
                run_dir,
                ema_helper.ema_model if ema_helper is not None else model,
                label_to_index,
                config,
                val_metrics,
                index_to_label,
            )

    with (run_dir / 'history.json').open('w', encoding='utf-8') as fp:
        json.dump(history, fp, ensure_ascii=False, indent=2)

    with (run_dir / 'label_to_index.json').open('w', encoding='utf-8') as fp:
        json.dump(label_to_index, fp, ensure_ascii=False, indent=2)

    print(f'Training finished. Outputs saved to: {run_dir}')


if __name__ == '__main__':
    main()
